"""
Reproduces paper Figure 1: at a fixed centroid size N, sweeps the
interpolation weight alpha (cos_sim(alpha*C_s1+(1-alpha)*C_s2, C_mix)) and
the scale beta (MSE(2*beta*avg(C_s1,C_s2), C_mix)) across random speaker
pairs, for CAM++ and/or ECAPA-TDNN, validating the midpoint relationship
c_mix ~= 0.5*(c_s1+c_s2) (peak near alpha=0.5, trough near beta=0.5).

Usage:
    python speakerlab/bin/analysis/param_sweep.py                       # both backbones
    python speakerlab/bin/analysis/param_sweep.py --backbone campplus
    python speakerlab/bin/analysis/param_sweep.py --num-pairs 30 --n 100 --device cuda:5
"""

import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_PROJECT_ROOT = os.path.dirname(_REPO_ROOT)
_PLOTTING_DIR = os.path.join(os.path.dirname(_THIS_DIR), "plotting")
if _PLOTTING_DIR not in sys.path:
    sys.path.insert(0, _PLOTTING_DIR)

from speakerlab.dataset.dataset import (
    _crop_or_pad,
    _load_mono,
    _mix_with_sir,
    build_speaker_index,
    list_unique_pairs,
)
from speakerlab.models.campplus.csm import load_pretrained_campplus
from speakerlab.models.ecapa_tdnn.csm import load_pretrained_ecapa
from speakerlab.process.processor import FBank
# Shared with replot_param_sweep.py so a fresh run and a from-CSV replot
# render identically (one plotting implementation, not two drifting apart).
from replot_param_sweep import plot_alpha_vs_cos_sim, plot_beta_vs_mse

DEFAULT_DATA_ROOT = os.path.join(_PROJECT_ROOT, "dataset/data/Libri2Mix/wav16k/min")
DEFAULT_LIBRISPEECH_ROOT = os.path.join(_PROJECT_ROOT, "dataset/data/LibriSpeech")
DEFAULT_CAMPPLUS_CKPT = os.path.expanduser(
    "~/.cache/modelscope/models/"
    "iic--speech_campplus_sv_zh_en_16k-common_advanced/"
    "snapshots/v1.0.0/campplus_cn_en_common.pt"
)
DEFAULT_ECAPA_CKPT = os.path.expanduser(
    "~/.cache/modelscope/models/"
    "iic--speech_ecapa-tdnn_sv_en_voxceleb_16k/"
    "snapshots/v1.0.1/ecapa_tdnn.bin"
)
DEFAULT_OUTPUT_DIR = os.path.join(_REPO_ROOT, "figures/param_sweep")
DEFAULT_N = 100
DEFAULT_NUM_PAIRS = 30
DEFAULT_NUM_POINTS = 11  # alpha/beta grid resolution: 0.0, 0.1, 0.2, ..., 1.0

# CAM++ solid, ECAPA-TDNN dashed -- both filled markers (color alone would
# be enough with only 2 categories, but linestyle reinforces it).
BACKBONE_LINESTYLES = {"campplus": "-", "ecapa": "-"}
BACKBONE_LABELS = {"campplus": "CAM++", "ecapa": "ECAPA-TDNN"}
BACKBONE_COLORS = {"campplus": "tab:blue", "ecapa": "tab:orange"}


def _len_or_full(value):
    """argparse type for --enroll-len/--mix-len: 'full' -> None (full length,
    unbatched), anything else -> float seconds (fixed crop, batched)."""
    return None if value.lower() == "full" else float(value)


def load_backbone(name, campplus_ckpt, ecapa_ckpt, emb_dim, device):
    if name == "campplus":
        return load_pretrained_campplus(campplus_ckpt, emb_dim, device)
    if name == "ecapa":
        return load_pretrained_ecapa(ecapa_ckpt, emb_dim, device)
    raise ValueError(f"Unknown backbone {name!r}")


@torch.no_grad()
def _embed_batch(model, fbanks, device):
    """fbanks: list of [T, 80] tensors (all the same T). Returns L2-normalized
    embeddings, [B, D] ndarray."""
    batch = torch.stack(fbanks).to(device)
    return F.normalize(model(batch), p=2, dim=-1).cpu().numpy()


@torch.no_grad()
def _embed_one(model, fbank, device):
    """fbank: [T, 80] tensor. Returns a single L2-normalized embedding, [D]
    ndarray. Used for variable-length (full-length) inputs -- CAM++/ECAPA-
    TDNN's pooling is length-sensitive, so these can't share one padded batch
    tensor without corrupting the pooled statistics."""
    emb = model(fbank.unsqueeze(0).to(device))
    return F.normalize(emb, p=2, dim=-1).squeeze(0).cpu().numpy()


def embed_solo_samples(model, paths, sample_rate, num_samples, fbank_fn, device, batch_size, wav_cache):
    """L2-normalized embeddings for each path in `paths` (may repeat --
    sampling is WITH replacement), in order. [len(paths), D] ndarray.
    num_samples=None -> each utterance's own full length (embedded one at a
    time, unbatched); otherwise a fixed crop/pad length (batched, faster)."""

    def get_wav(path):
        wav = wav_cache.get(path)
        if wav is None:
            wav = _load_mono(path, sample_rate)
            if num_samples is not None:
                wav = _crop_or_pad(wav, num_samples)
            wav_cache[path] = wav
        return wav

    if num_samples is None:
        return np.stack([_embed_one(model, fbank_fn(get_wav(p)), device) for p in paths])

    chunks = []
    for i in range(0, len(paths), batch_size):
        fbanks = [fbank_fn(get_wav(p)) for p in paths[i:i + batch_size]]
        chunks.append(_embed_batch(model, fbanks, device))
    return np.concatenate(chunks)


def embed_mixture_samples(model, combos, sample_rate, num_samples, fbank_fn, sir_range, device, batch_size, wav_cache, rng):
    """L2-normalized embeddings for each (path_a, path_b) combo in `combos`,
    each synthesized at a fresh random SIR (drawn from `rng`). [len(combos), D]
    ndarray. num_samples=None mixes each combo at its own (shorter-of-two)
    length and embeds one at a time; otherwise a fixed crop/pad, batched."""

    def get_wav(path):
        wav = wav_cache.get(path)
        if wav is None:
            wav = _load_mono(path, sample_rate)
            if num_samples is not None:
                wav = _crop_or_pad(wav, num_samples)
            wav_cache[path] = wav
        return wav

    if num_samples is None:
        embeds = []
        for path_a, path_b in combos:
            wav_a, wav_b = get_wav(path_a), get_wav(path_b)
            n = min(wav_a.shape[-1], wav_b.shape[-1])
            mix_wav = _mix_with_sir(wav_a[:n], wav_b[:n], rng.uniform(*sir_range))
            embeds.append(_embed_one(model, fbank_fn(mix_wav), device))
        return np.stack(embeds)

    chunks = []
    for i in range(0, len(combos), batch_size):
        fbanks = []
        for path_a, path_b in combos[i:i + batch_size]:
            mix_wav = _mix_with_sir(get_wav(path_a), get_wav(path_b), rng.uniform(*sir_range))
            fbanks.append(fbank_fn(mix_wav))
        chunks.append(_embed_batch(model, fbanks, device))
    return np.concatenate(chunks)


def cos_sim_grid(v_grid, m, eps=1e-8):
    """v_grid: [G, P, D]; m: [P, D] (broadcasts against v_grid's P, D dims).
    Returns [G, P] cosine similarities."""
    dot = np.sum(v_grid * m[None, :, :], axis=-1)
    return dot / (np.linalg.norm(v_grid, axis=-1) * np.linalg.norm(m, axis=-1)[None, :] + eps)


def mse_grid_fn(v_grid, m):
    """v_grid: [G, P, D]; m: [P, D] (broadcasts). Returns [G, P] per-pair MSE."""
    return np.mean((v_grid - m[None, :, :]) ** 2, axis=-1)


def compute_curve_inputs(model, chosen_pairs, speaker_index, args, fbank_fn, enroll_samples, mix_samples, device, backbone_name, n_mix):
    """Per-pair C_s1/C_s2/C_mix, same convention/seeding as before -- shared
    verbatim across backbones (the (seed, key) rng draws the SAME
    utterances/SIRs regardless of backbone, so only the embeddings differ).
    C_mix always uses n_mix combos. C_s1/C_s2 use every utterance of that
    speaker when args.enroll_all_utts, else args.n draws with replacement."""
    c_s1_all = np.zeros((len(chosen_pairs), args.emb_dim))
    c_s2_all = np.zeros((len(chosen_pairs), args.emb_dim))
    c_mix_all = np.zeros((len(chosen_pairs), args.emb_dim))

    for pair_idx, key in enumerate(tqdm(chosen_pairs, desc=f"pairs ({backbone_name})", unit="pair")):
        s1, s2 = key.split("_")
        paths_1, paths_2 = speaker_index[s1], speaker_index[s2]

        rng = random.Random(f"{args.seed}:{key}")
        combos = [(rng.choice(paths_1), rng.choice(paths_2)) for _ in range(n_mix)]
        if args.enroll_all_utts:
            solo_1, solo_2 = paths_1, paths_2
        else:
            solo_1 = [rng.choice(paths_1) for _ in range(args.n)]
            solo_2 = [rng.choice(paths_2) for _ in range(args.n)]

        wav_cache = {}
        mix_embeds = embed_mixture_samples(
            model, combos, args.sample_rate, mix_samples, fbank_fn, args.sir_range,
            device, args.batch_size, wav_cache, rng,
        )
        s1_embeds = embed_solo_samples(
            model, solo_1, args.sample_rate, enroll_samples, fbank_fn, device, args.batch_size, wav_cache,
        )
        s2_embeds = embed_solo_samples(
            model, solo_2, args.sample_rate, enroll_samples, fbank_fn, device, args.batch_size, wav_cache,
        )

        c_mix_all[pair_idx] = mix_embeds.mean(axis=0)
        c_s1_all[pair_idx] = s1_embeds.mean(axis=0)
        c_s2_all[pair_idx] = s2_embeds.mean(axis=0)

    return c_s1_all, c_s2_all, c_mix_all


def plot_combined(results, output_path, num_pairs, enroll_desc, mix_desc):
    """Both curves overlaid on one shared [0, 1] x-axis -- cos_sim vs alpha
    (blue, left y-axis) and MSE vs beta (red, right y-axis, twinx). With
    multiple backbones, each contributes one solid/dashed pair of curves."""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    for name, r in results.items():
        ls = BACKBONE_LINESTYLES[name]
        a = 1.0 if name == "campplus" else 0.75
        ax1.plot(r["alphas"], r["mean_cos_sim"], marker="o", color="tab:blue", linewidth=2,
                  linestyle=ls, alpha=a,
                  label=f"cos_sim vs alpha [{BACKBONE_LABELS[name]}]")
        ax2.plot(r["betas"], r["mean_mse"], marker="o", color="tab:red", linewidth=2,
                  linestyle=ls, alpha=a,
                  label=f"mse vs beta [{BACKBONE_LABELS[name]}]")

    ax1.set_xlim(0.0, 1.0)
    ax1.set_xticks(next(iter(results.values()))["alphas"])
    ax1.set_xlabel("alpha / beta")
    ax1.set_ylabel("cosine similarity", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_ylabel("MSE", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.axvline(0.5, color="black", linewidth=1.0, linestyle="dotted", alpha=0.5)
    ax1.set_title(f"cos_sim vs alpha & MSE vs beta (N_enroll={enroll_desc}, N_mix={mix_desc}, {num_pairs} speaker pairs)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=7)
    ax1.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_csv(results, csv_path):
    """Long format: one row per (backbone, grid point)."""
    fieldnames = ["backbone", "alpha", "mean_cos_sim", "beta", "mean_mse"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, r in results.items():
            for a, cs, b, m in zip(r["alphas"], r["mean_cos_sim"], r["betas"], r["mean_mse"]):
                writer.writerow({"backbone": name, "alpha": a, "mean_cos_sim": cs, "beta": b, "mean_mse": m})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", choices=["campplus", "ecapa", "both_100"], default="both",
                         help="Which verification backbone(s) to sweep; 'both' overlays both on the same figures.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Libri2Mix wav16k/min root")
    parser.add_argument("--librispeech-root", default=DEFAULT_LIBRISPEECH_ROOT)
    parser.add_argument("--subsets", nargs="+", default=["train-100"],
                         help="Shared Libri2Mix/LibriSpeech subset name(s), e.g. train-100 or dev")
    parser.add_argument("--mode", default="mix_clean")
    parser.add_argument("--campplus-ckpt", default=DEFAULT_CAMPPLUS_CKPT)
    parser.add_argument("--ecapa-ckpt", default=DEFAULT_ECAPA_CKPT)
    parser.add_argument("--emb-dim", type=int, default=192)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--enroll-len", type=_len_or_full, default=3.0,
                         help="Crop length (s) for s1/s2 utterances; 'full' uses each utterance's own length.")
    parser.add_argument("--mix-len", type=_len_or_full, default=3.0,
                         help="Crop length (s) for synthesized mixtures; 'full' crops to the shorter utterance.")
    parser.add_argument("--sir-range", type=float, nargs=2, default=[-5.0, 5.0])
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                         help="Fixed sample count for C_s1/C_s2 (with replacement); also the default for --n-mix.")
    parser.add_argument("--enroll-all-utts", action="store_true",
                         help="Build C_s1/C_s2 from every utterance of that speaker instead of --n draws.")
    parser.add_argument("--n-mix", type=int, default=None,
                         help="Sample count for C_mix combos, decoupled from --n. Defaults to --n.")
    parser.add_argument("--num-pairs", type=int, default=DEFAULT_NUM_PAIRS,
                         help="Number of random Libri2Mix speaker pairs to average over.")
    parser.add_argument("--num-points", type=int, default=DEFAULT_NUM_POINTS,
                         help="Resolution of the alpha/beta sweep grids over [0, 1].")
    parser.add_argument("--batch-size", type=int, default=256,
                         help="Only used when --enroll-len/--mix-len set a fixed crop length (batched mode).")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cos-sim-output-name", default="alpha_vs_cos_sim.png")
    parser.add_argument("--mse-output-name", default="beta_vs_mse.png")
    parser.add_argument("--combined-output-name", default="combined.png")
    parser.add_argument("--csv-name", default="param_sweep_100.csv")
    parser.add_argument("--device", default="cuda:4" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1234,
                         help="Fixes which pairs/utterances/SIRs get drawn.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    fbank_fn = FBank(80, sample_rate=args.sample_rate, mean_nor=True)
    enroll_samples = None if args.enroll_len is None else int(round(args.enroll_len * args.sample_rate))
    mix_samples = None if args.mix_len is None else int(round(args.mix_len * args.sample_rate))

    speaker_index = build_speaker_index(args.librispeech_root, args.subsets)

    mix_dirs = [os.path.join(args.data_root, s) for s in args.subsets]
    all_pairs = list_unique_pairs(mix_dirs, mode=args.mode)
    valid_pairs = [key for key in all_pairs if all(spk in speaker_index for spk in key.split("_"))]
    print(f"{len(valid_pairs)}/{len(all_pairs)} pairs have both speakers in the LibriSpeech index")
    if len(valid_pairs) < args.num_pairs:
        raise ValueError(f"Only {len(valid_pairs)} valid pairs available, need --num-pairs {args.num_pairs}")
    # Sampled ONCE, backbone-independent, so --backbone both compares the
    # SAME pairs/utterances/SIRs across backbones (only the embeddings differ).
    chosen_pairs = random.sample(valid_pairs, args.num_pairs)

    backbone_names = ["campplus", "ecapa"] if args.backbone == "both" else [args.backbone]
    alphas = np.linspace(0.0, 1.0, args.num_points)
    betas = np.linspace(0.0, 1.0, args.num_points)
    n_mix = args.n_mix if args.n_mix is not None else args.n
    enroll_desc = "all utts" if args.enroll_all_utts else str(args.n)

    results = {}
    for name in backbone_names:
        model = load_backbone(name, args.campplus_ckpt, args.ecapa_ckpt, args.emb_dim, device)
        c_s1_all, c_s2_all, c_mix_all = compute_curve_inputs(
            model, chosen_pairs, speaker_index, args, fbank_fn, enroll_samples, mix_samples, device, name, n_mix,
        )

        # --- 1. cos_sim vs alpha (2*beta is scale-invariant for cos_sim, dropped) ---
        interp_grid = alphas[:, None, None] * c_s1_all[None, :, :] + (1 - alphas[:, None, None]) * c_s2_all[None, :, :]
        mean_cos_sim = cos_sim_grid(interp_grid, c_mix_all).mean(axis=1)  # [num_points]

        # --- 2. MSE vs beta, alpha fixed at 0.5 ---
        avg_all = 0.5 * c_s1_all + 0.5 * c_s2_all  # [P, D], alpha_star = 0.5
        scaled_grid = (2 * betas)[:, None, None] * avg_all[None, :, :]
        mean_mse = mse_grid_fn(scaled_grid, c_mix_all).mean(axis=1)  # [num_points]

        argmax_alpha = alphas[np.argmax(mean_cos_sim)]
        argmin_beta = betas[np.argmin(mean_mse)]
        print(f"[{BACKBONE_LABELS[name]}] cos_sim peak: alpha={argmax_alpha:.4f} (mean cos_sim={mean_cos_sim.max():.4f})")
        print(f"[{BACKBONE_LABELS[name]}] MSE trough : beta={argmin_beta:.4f} (mean mse={mean_mse.min():.6f})")

        results[name] = dict(alphas=alphas, mean_cos_sim=mean_cos_sim, betas=betas, mean_mse=mean_mse)

    os.makedirs(args.output_dir, exist_ok=True)
    cos_sim_path = os.path.join(args.output_dir, args.cos_sim_output_name)
    mse_path = os.path.join(args.output_dir, args.mse_output_name)
    combined_path = os.path.join(args.output_dir, args.combined_output_name)
    plot_alpha_vs_cos_sim(results, cos_sim_path)
    plot_beta_vs_mse(results, mse_path)
    plot_combined(results, combined_path, args.num_pairs, enroll_desc, n_mix)
    print(f"Saved {cos_sim_path}")
    print(f"Saved {mse_path}")
    print(f"Saved {combined_path}")

    csv_path = os.path.join(args.output_dir, args.csv_name)
    save_csv(results, csv_path)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
