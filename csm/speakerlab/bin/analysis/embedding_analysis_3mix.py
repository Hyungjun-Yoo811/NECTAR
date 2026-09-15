"""
Reproduces paper Figure 5 (Sec 5.2): draws --num-triplets random speaker
triplets (default 10), grouped by embedding proximity, and checks via a
shared t-SNE/PCA fit that each mixture group's actual centroid matches the
unweighted average of its constituents' centroids (paper: mean cos_sim=0.89,
MSE=4e-4). --plot-types selects the figures; a metrics bar chart is always saved.

Usage:
    python speakerlab/bin/analysis/embedding_analysis_3mix.py
    python speakerlab/bin/analysis/embedding_analysis_3mix.py --num-triplets 20
    python speakerlab/bin/analysis/embedding_analysis_3mix.py --all-groups --plot-types pca scatter
"""

import argparse
import csv
import itertools
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_REPO_ROOT)

from speakerlab.dataset.dataset import _crop_or_pad, _load_mono, build_speaker_index
from speakerlab.models.campplus.csm import load_pretrained_campplus
from speakerlab.process.processor import FBank
from _superposition_plot import plot_all_triplets, speaker_colors

DEFAULT_LIBRISPEECH_ROOT = os.path.join(_PROJECT_ROOT, "dataset/data/LibriSpeech")
DEFAULT_CAMPPLUS_CKPT = os.path.expanduser(
    "~/.cache/modelscope/models/"
    "iic--speech_campplus_sv_zh_en_16k-common_advanced/"
    "snapshots/v1.0.0/campplus_cn_en_common.pt"
)
DEFAULT_OUTPUT_DIR = os.path.join(_REPO_ROOT, "figures/embedding_analysis_3mix")

# key -> which of the triplet's (s1, s2, s3) indices get mixed together.
_TRIANGLE_GROUPS = [
    ("s1", (0,)),
    ("s2", (1,)),
    ("s3", (2,)),
    ("s1+s2", (0, 1)),
    ("s2+s3", (1, 2)),
    ("s3+s1", (2, 0)),
    ("s1+s2+s3", (0, 1, 2)),
]




def _mix_all_with_sir(wavs, sir_range):
    """wavs[0] is the reference (0 dB); every other wav is independently
    scaled so its power relative to wavs[0] hits a fresh random SIR (dB)
    drawn from sir_range, then all are summed -- generalizes
    speakerlab.dataset.dataset._mix_with_sir to N speakers."""
    mix = wavs[0].clone()
    power_ref = wavs[0].pow(2).mean().clamp_min(1e-8)
    for wav in wavs[1:]:
        power_wav = wav.pow(2).mean().clamp_min(1e-8)
        sir_db = random.uniform(*sir_range)
        scale = torch.sqrt(power_ref / power_wav / (10 ** (sir_db / 10)))
        mix = mix + scale * wav
    return mix


def sample_combos(paths_lists, n):
    """n distinct tuples, one path independently drawn from each list in
    paths_lists (2 lists -> pairwise-mixture combos, 3 lists -> 3-way-mixture
    combos). Falls back to every combination if the full product has <= n
    entries."""
    total = 1
    for paths in paths_lists:
        total *= len(paths)
    if total <= n:
        return list(itertools.product(*paths_lists))
    combos = set()
    while len(combos) < n:
        combos.add(tuple(random.choice(paths) for paths in paths_lists))
    return list(combos)


@torch.no_grad()
def _embed_batch(campp, fbanks, device):
    """fbanks: list of [T, 80] tensors (all the same T). Returns L2-normalized
    embeddings, [N, D] ndarray."""
    batch = torch.stack(fbanks).to(device)
    return F.normalize(campp(batch), p=2, dim=-1).cpu().numpy()


def embed_utterances(campp, paths, sample_rate, num_samples, fbank_fn, device, batch_size):
    """Raw (unit-norm) CAM++ embeddings for every path in `paths`. [N, D] ndarray."""
    chunks = []
    for i in range(0, len(paths), batch_size):
        fbanks = [
            fbank_fn(_crop_or_pad(_load_mono(p, sample_rate), num_samples))
            for p in paths[i:i + batch_size]
        ]
        chunks.append(_embed_batch(campp, fbanks, device))
    return np.concatenate(chunks)


def embed_mixture_combos(campp, combos, wav_cache, sample_rate, num_samples,
                          fbank_fn, sir_range, device, batch_size):
    """Raw (unit-norm) CAM++ embeddings for every combo in `combos` (each a
    tuple of 2 or 3 utterance paths -- one per speaker being mixed),
    synthesized via _mix_all_with_sir at fresh random SIRs. wav_cache caches
    each file's crop across combos so repeated utterances aren't re-decoded."""

    def get_crop(path):
        wav = wav_cache.get(path)
        if wav is None:
            wav = _crop_or_pad(_load_mono(path, sample_rate), num_samples)
            wav_cache[path] = wav
        return wav

    chunks = []
    for i in range(0, len(combos), batch_size):
        fbanks = []
        for combo in combos[i:i + batch_size]:
            mix_wav = _mix_all_with_sir([get_crop(p) for p in combo], sir_range)
            fbanks.append(fbank_fn(mix_wav))
        chunks.append(_embed_batch(campp, fbanks, device))
    return np.concatenate(chunks)


def _add_points(embeddings, meta, arr, kind, role, color, triplet_idx, label=None):
    """Append arr's rows to the flat embeddings/meta accumulators that will
    become one shared t-SNE fit. `color` encodes which triplet a point
    belongs to; `role` drives marker shape. `triplet_idx` is kept explicit
    (not just inferred from color) so replot_embedding_analysis_3mix.py's
    --palette can recolor without relying on the original colors."""
    embeddings.append(arr)
    for _ in range(arr.shape[0]):
        meta.append({"kind": kind, "role": role, "color": color, "label": label, "triplet_idx": triplet_idx})


def cos_distance(a, b, eps=1e-8):
    """1 - cosine_similarity(a, b), for magnitude-carrying (non-normalized)
    vectors a, b."""
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def group_by_proximity(speakers, centroid_of):
    """Greedily partition `speakers` into consecutive triplets of mutually
    close speakers via nearest-neighbor chaining (pop an anchor, pop its 2
    closest still-unassigned speakers by cos_distance, repeat) instead of an
    arbitrary grouping. len(speakers) must be a multiple of 3."""
    remaining = list(speakers)
    triplets = []
    while remaining:
        anchor = remaining.pop(0)
        remaining.sort(key=lambda s: cos_distance(centroid_of[anchor], centroid_of[s]))
        mate1, mate2 = remaining.pop(0), remaining.pop(0)
        triplets.append((anchor, mate1, mate2))
    return triplets


def plot_metrics(labels, cos_dists, mses, output_path):
    """Bar chart of estimated-vs-oracle mixture centroid error, one bar per
    (triplet, mixture group) -- cosine distance (direction) on top, MSE
    (direction + magnitude) below."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(labels) * 0.35), 7), sharex=True)
    x = np.arange(len(labels))

    ax1.bar(x, cos_dists, color="tab:blue")
    ax1.axhline(float(np.mean(cos_dists)), color="black", linewidth=1.0, linestyle="dashed",
                label=f"mean = {np.mean(cos_dists):.4f}")
    ax1.set_ylabel("Cosine distance\n(1 - cos_sim)")
    ax1.set_title("Estimated vs. oracle mixture centroid: superposition error per (triplet, mixture group)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x, mses, color="tab:red")
    ax2.axhline(float(np.mean(mses)), color="black", linewidth=1.0, linestyle="dashed",
                label=f"mean = {np.mean(mses):.4f}")
    ax2.set_ylabel("MSE")
    ax2.set_xlabel("Mixture group")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_metrics_csv(triplet_idxs, triplets, labels, cos_dists, mses, csv_path):
    """One row per (triplet, mixture group) -- same rows plot_metrics bars,
    plus the triplet's speaker ids for traceability back to which triplet
    a row came from."""
    fieldnames = ["triplet_idx", "speakers", "mixture_group", "cosine_distance", "mse"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for tri_idx, label, cos_dist, mse in zip(triplet_idxs, labels, cos_dists, mses):
            writer.writerow({
                "triplet_idx": tri_idx,
                "speakers": "+".join(triplets[tri_idx]),
                "mixture_group": label,
                "cosine_distance": cos_dist,
                "mse": mse,
            })


def plot_scatter(est_pool, actual_pool, output_path):
    """Predicted-vs-actual scatter, pooled across every mixture group and
    embedding dimension: x = mix_est[d] (unweighted average of constituent
    speakers' centroids), y = mix_centroid[d] (the group's actual centroid).
    If superposition holds, points cluster along y=x. Returns (r_squared, rmse)."""
    est_flat = np.asarray(est_pool).reshape(-1)
    actual_flat = np.asarray(actual_pool).reshape(-1)

    slope, intercept = np.polyfit(est_flat, actual_flat, 1)
    r_squared = float(np.corrcoef(est_flat, actual_flat)[0, 1] ** 2)
    rmse = float(np.sqrt(np.mean((actual_flat - est_flat) ** 2)))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    hb = ax.hexbin(est_flat, actual_flat, gridsize=60, cmap="viridis", mincnt=1, bins="log")
    fig.colorbar(hb, ax=ax, label="log10(count)")

    lo = float(min(est_flat.min(), actual_flat.min()))
    hi = float(max(est_flat.max(), actual_flat.max()))
    xs = np.array([lo, hi])
    ax.plot(xs, xs, color="red", linewidth=1.5, linestyle="dashed", label="y = x (perfect superposition)")
    ax.plot(xs, slope * xs + intercept, color="orange", linewidth=1.5,
             label=f"linear fit: y = {slope:.3f}x + {intercept:.3f}")

    ax.set_xlabel("Estimated mixture centroid component -- sum of constituent speakers' centroids")
    ax.set_ylabel("Actual mixture centroid component")
    ax.set_title(
        f"Predicted vs. actual mixture centroid components (pooled over all groups/dims, N={est_flat.size})\n"
        f"R² = {r_squared:.4f}, RMSE = {rmse:.4f}"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return r_squared, rmse


def save_embedding_cache(embeddings, meta, triplets, args, npz_path, json_path):
    """Saves the pooled embeddings feeding the t-SNE/PCA fit, plus everything
    plot_all_triplets needs to redraw from them, so
    replot_embedding_analysis_3mix.py can reproduce or restyle the figure
    without re-running CAM++. Plain numpy (.npz) + json, no torch/pickle."""
    np.savez(npz_path, embeddings=embeddings)
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "triplets": triplets, "seed": args.seed, "n": args.n,
                    "triple_only": args.triple_only}, f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campplus-ckpt", default=DEFAULT_CAMPPLUS_CKPT)
    parser.add_argument("--emb-dim", type=int, default=192)
    parser.add_argument("--librispeech-root", default=DEFAULT_LIBRISPEECH_ROOT)
    parser.add_argument("--subsets", nargs="+", default=["train-100"],
                         help="e.g. train-100, train-360, dev")
    parser.add_argument("--num-triplets", type=int, default=10,
                         help="Number of random speaker triplets (default 10, matching the paper figure).")
    parser.add_argument("--n", type=int, default=100,
                         help="Sample count per group (utterances for s1/s2/s3, combos for mixtures).")
    parser.add_argument("--num-samples-per-group", type=int, default=10,
                         help="Sample points plotted per group; ignored under --triple-only (the default).")
    parser.add_argument("--triple-only", dest="triple_only", action="store_true", default=True,
                         help="Plot only single-speaker + s1+s2+s3 centroids (default); skips pairwise groups.")
    parser.add_argument("--all-groups", dest="triple_only", action="store_false",
                         help="Restore all 7 groups (3 single + 3 pairwise + 1 triple) plus sample points.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--enroll-len", type=float, default=3.0, help="Crop length (s) for s1/s2/s3 utterances.")
    parser.add_argument("--mix-len", type=float, default=3.0, help="Crop length (s) for synthesized mixtures.")
    parser.add_argument("--sir-range", type=float, nargs=2, default=[-5.0, 5.0],
                         help="Random per-speaker SIR (dB) range, relative to the first speaker in the mixture.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--plot-types", nargs="+", choices=["tsne", "pca", "scatter"],
                         default=["tsne"],
                         help="Which superposition figures to generate.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="superposition_tsne_all_triplets_5pair.png",
                         help="t-SNE figure filename (used if 'tsne' is in --plot-types).")
    parser.add_argument("--pca-output-name", default="superposition_pca_all_triplets.png",
                         help="PCA figure filename (used if 'pca' is in --plot-types).")
    parser.add_argument("--scatter-output-name", default="superposition_scatter.png",
                         help="Pooled predicted-vs-actual scatter filename (used if 'scatter' is in --plot-types).")
    parser.add_argument("--metrics-output-name", default="superposition_metrics.png",
                         help="Per-group cosine-distance/MSE bar chart filename.")
    parser.add_argument("--metrics-csv-name", default="superposition_metrics.csv",
                         help="CSV of the same per-group rows as --metrics-output-name.")
    parser.add_argument("--cache-name", default="superposition_cache",
                         help="Basename for the pooled-embeddings cache (<name>.npz + .json) this run saves.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1234,
                         help="Fixes which speaker triplets/utterances/SIRs get drawn.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    campp = load_pretrained_campplus(args.campplus_ckpt, args.emb_dim, device)
    fbank_fn = FBank(80, sample_rate=args.sample_rate, mean_nor=True)
    enroll_samples = int(round(args.enroll_len * args.sample_rate))
    mix_samples = int(round(args.mix_len * args.sample_rate))

    speaker_index = build_speaker_index(args.librispeech_root, args.subsets)
    # --n utterances are drawn WITH replacement (see main loop below), so
    # eligibility only requires at least 1 utterance, not >= args.n.
    eligible = sorted(spk for spk, paths in speaker_index.items() if len(paths) >= 1)
    needed = 3 * args.num_triplets
    if len(eligible) < needed:
        raise ValueError(
            f"Only {len(eligible)} speakers in {args.subsets}; need {needed} for --num-triplets "
            f"{args.num_triplets} non-overlapping triplets"
        )
    chosen = random.sample(eligible, needed)

    # --n utterances per speaker, drawn WITH replacement (so this works even
    # for a speaker with fewer than --n utterances), embedded ONCE up front
    # for ALL 30 chosen speakers -- both to unify the sample count with every
    # mixture group's combo count below (all at exactly --n), and so the
    # triplet grouping right after can use these centroids' proximity.
    speaker_raw, speaker_centroid = {}, {}
    for spk in chosen:
        speaker_raw[spk] = embed_utterances(
            campp, [random.choice(speaker_index[spk]) for _ in range(args.n)],
            args.sample_rate, enroll_samples, fbank_fn, device, args.batch_size,
        )
        speaker_centroid[spk] = speaker_raw[spk].mean(axis=0)

    # Grouped by embedding-space proximity (see group_by_proximity), NOT an
    # arbitrary chunking of the random draw -- keeps each triplet's triangle
    # compact in the shared t-SNE/PCA figure instead of crossing others'.
    triplets = group_by_proximity(chosen, speaker_centroid)
    print(f"Selected {len(triplets)} speaker triplets, grouped by embedding proximity "
          f"(seed={args.seed}): {triplets}")

    # One color per TRIPLET (not per speaker) -- see speaker_colors/plot_all_triplets.
    triplet_colors = speaker_colors(len(triplets))

    os.makedirs(args.output_dir, exist_ok=True)

    embeddings, meta = [], []
    metric_triplet_idxs, metric_labels, cos_dists, mses = [], [], [], []
    mix_est_pool, mix_centroid_pool = [], []
    for tri_idx, spk_ids in enumerate(triplets):
        paths = [speaker_index[spk] for spk in spk_ids]
        raw = [speaker_raw[spk] for spk in spk_ids]
        actual_centroid = [speaker_centroid[spk] for spk in spk_ids]
        color = triplet_colors[tri_idx]
        print(f"  triplet {tri_idx + 1}/{len(triplets)}: {spk_ids} (n={args.n} per speaker, "
              f"{', '.join(str(len(p)) for p in paths)} available utts)")

        wav_cache = {}
        for key, member_idxs in _TRIANGLE_GROUPS:
            if len(member_idxs) == 1:
                i = member_idxs[0]
                if not args.triple_only:
                    samples = raw[i][random.sample(range(len(raw[i])), args.num_samples_per_group)]
                    _add_points(embeddings, meta, samples, "sample", "single", color, tri_idx)
                _add_points(embeddings, meta, actual_centroid[i][None, :], "centroid", "single",
                            color, tri_idx, label=spk_ids[i])
            else:
                if len(member_idxs) == 2 and args.triple_only:
                    # --triple-only never embeds or plots pairwise mixtures.
                    continue

                # UNIFIED combo count: exactly --n synthesized mixtures for
                # every mixture group (pairwise or triple) alike.
                combos = sample_combos([paths[i] for i in member_idxs], args.n)
                print(f"    {key}: -> {len(combos)} mixtures")

                mix_samples_emb = embed_mixture_combos(
                    campp, combos, wav_cache, args.sample_rate, mix_samples, fbank_fn, args.sir_range,
                    device, args.batch_size,
                )
                mix_centroid = mix_samples_emb.mean(axis=0)
                # Unweighted average, not raw sum -- see module docstring.
                mix_est = np.mean([actual_centroid[i] for i in member_idxs], axis=0)
                group_label = "+".join(spk_ids[i] for i in member_idxs)

                metric_triplet_idxs.append(tri_idx)
                metric_labels.append(f"{group_label} ({key})")
                cos_dists.append(cos_distance(mix_est, mix_centroid))
                mses.append(float(np.mean((mix_est - mix_centroid) ** 2)))
                mix_est_pool.append(mix_est)
                mix_centroid_pool.append(mix_centroid)

                # Centroid/metrics use every combo above; only a small subsample
                # gets plotted as points (skipped entirely under --triple-only).
                if not args.triple_only:
                    plot_n = min(args.num_samples_per_group, len(mix_samples_emb))
                    plotted_mix = mix_samples_emb[random.sample(range(len(mix_samples_emb)), plot_n)]
                    _add_points(embeddings, meta, plotted_mix, "sample", "mix", color, tri_idx)
                _add_points(embeddings, meta, mix_centroid[None, :], "centroid", "mix",
                            color, tri_idx, label=group_label)
                _add_points(embeddings, meta, mix_est[None, :], "estimated", "mix", color, tri_idx,
                            label=group_label)

    embeddings = np.concatenate(embeddings, axis=0)

    cache_npz_path = os.path.join(args.output_dir, args.cache_name + ".npz")
    cache_json_path = os.path.join(args.output_dir, args.cache_name + ".json")
    save_embedding_cache(embeddings, meta, triplets, args, cache_npz_path, cache_json_path)
    print(f"Saved {cache_npz_path} and {cache_json_path} "
          f"(for replot_embedding_analysis_3mix.py -- no CAM++ needed)")

    if "tsne" in args.plot_types:
        perplexity = min(args.tsne_perplexity, (embeddings.shape[0] - 1) / 3)
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=args.seed)
        coords = tsne.fit_transform(embeddings)
        output_path = os.path.join(args.output_dir, args.output_name)
        plot_all_triplets(coords, meta, triplets, output_path, "t-SNE", "t-SNE dim 1", "t-SNE dim 2")
        print(f"Saved {output_path}")

    if "pca" in args.plot_types:
        pca = PCA(n_components=2, random_state=args.seed)
        coords_pca = pca.fit_transform(embeddings)
        var1, var2 = pca.explained_variance_ratio_[:2]
        output_path = os.path.join(args.output_dir, args.pca_output_name)
        plot_all_triplets(coords_pca, meta, triplets, output_path, "PCA",
                           f"PC1 ({var1:.1%} var)", f"PC2 ({var2:.1%} var)")
        print(f"Saved {output_path}")

    if "scatter" in args.plot_types:
        scatter_path = os.path.join(args.output_dir, args.scatter_output_name)
        r_squared, rmse = plot_scatter(np.array(mix_est_pool), np.array(mix_centroid_pool), scatter_path)
        print(f"Saved {scatter_path} (R² = {r_squared:.4f}, RMSE = {rmse:.4f})")

    metrics_path = os.path.join(args.output_dir, args.metrics_output_name)
    plot_metrics(metric_labels, cos_dists, mses, metrics_path)
    print(f"Saved {metrics_path}")

    csv_path = os.path.join(args.output_dir, args.metrics_csv_name)
    save_metrics_csv(metric_triplet_idxs, triplets, metric_labels, cos_dists, mses, csv_path)
    print(f"Saved {csv_path}")

    print(f"Mean cosine distance = {np.mean(cos_dists):.4f}, mean MSE = {np.mean(mses):.4f}")


if __name__ == "__main__":
    main()
