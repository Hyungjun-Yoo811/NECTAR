"""
Reproduces paper Figure 2 (centroid deviation vs. centroid size N, and CSM's
improvement over N=1): sweeps scale beta (pseudo_c_s2(N) = beta*c_mix(N) -
c_s1(N), beta=1.0 is the "gs" clue formula) across N for CAM++/ECAPA-TDNN,
and separately scores the trained CSM head at N=1, both against the oracle
interferer centroid. Data-generation only; replot_beta_figures.py plots the CSV.

Usage:
    python speakerlab/bin/analysis/beta_convergence.py --backbone campplus
    python speakerlab/bin/analysis/beta_convergence.py --backbone ecapa
    python speakerlab/bin/analysis/beta_convergence.py --backbone campplus --num-pairs 30 --device cuda:5  # quick check
    python speakerlab/bin/analysis/beta_convergence.py --backbone campplus --gpus 4,5,6,7
"""

import argparse
import csv
import os
import random
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_PROJECT_ROOT = os.path.dirname(_REPO_ROOT)

from speakerlab.dataset.dataset import (
    _crop_or_pad,
    _load_mono,
    _mix_with_sir,
    build_speaker_index,
    list_unique_pairs,
)
from speakerlab.models.campplus.csm import (
    CAMPlusCSM,
    ModelCheckpointProxy,
    load_pretrained_campplus,
)
from speakerlab.models.ecapa_tdnn.csm import (
    ECAPACSM,
    ModelCheckpointProxy as EcapaModelCheckpointProxy,
    load_pretrained_ecapa,
)
from speakerlab.process.processor import FBank

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
# Per-backbone defaults -- subsets/oracle cache/estimator run dir all have to
# agree with each other (see csm_ecapa.yaml's header for
# why an ECAPA centroid target is not comparable to a CAM++ one).
BACKBONE_DEFAULTS = {
    "campplus": dict(
        subsets=["train-100"],
        oracle_centroid_cache=os.path.join(
            _PROJECT_ROOT, "dataset/data/centroid_cache/without_norm/train_speaker_centroids.pt"),
        estimator_run_dir=os.path.join(_REPO_ROOT, "exp/csm/csm"),
    ),
    "ecapa": dict(
        subsets=["train-100", "train-360"],
        oracle_centroid_cache=os.path.join(
            _PROJECT_ROOT,
            "dataset/data/centroid_cache/without_norm/train_speaker_centroids_100_360_ecapa.pt"),
        estimator_run_dir=os.path.join(_REPO_ROOT, "exp/csm/csm_ecapa"),
    ),
}
DEFAULT_OUTPUT_DIR = os.path.join(_REPO_ROOT, "figures/beta_convergence")
DEFAULT_N_VALUES = [1, 3, 5, 10, 15, 20, 30, 40, 50, 100]
DEFAULT_BETA_VALUES = [0.5, 1.0, 2.0]
DEFAULT_NUM_PAIRS = 30


def _len_or_full(value):
    return None if value.lower() == "full" else float(value)


def _samples_or_full(len_seconds, sample_rate):
    return None if len_seconds is None else int(round(len_seconds * sample_rate))


def _maybe_crop_or_pad(wav, num_samples):
    return wav if num_samples is None else _crop_or_pad(wav, num_samples)


def load_backbone(backbone, campplus_ckpt, ecapa_ckpt, emb_dim, device):
    if backbone == "campplus":
        return load_pretrained_campplus(campplus_ckpt, emb_dim, device)
    if backbone == "ecapa":
        return load_pretrained_ecapa(ecapa_ckpt, emb_dim, device)
    raise ValueError(f"Unknown backbone {backbone!r}")


def load_csm(run_dir, device, ckpt_name="CKPT+best"):
    """Build the trained centroid estimator and load its weights. Dispatches
    on model.backbone ("campplus", the default, or "ecapa"), not model.impl
    (always "v1" so far). Both backbones' pretrained-checkpoint path lives
    under the config's "campplus" key (a naming leftover from before ECAPA
    support); "ecapa"/"backbone_frame_dim" are honored too, if present."""
    with open(os.path.join(run_dir, "config.yaml")) as f:
        ce_cfg = yaml.safe_load(f)
    mcfg = ce_cfg["model"]
    impl = mcfg.get("impl", "v1")
    backbone = mcfg.get("backbone", "campplus")

    if backbone == "ecapa":
        backbone_ckpt_path = ce_cfg["ecapa"]["ckpt_path"] if "ecapa" in ce_cfg else ce_cfg["campplus"]["ckpt_path"]
        est_backbone = load_pretrained_ecapa(backbone_ckpt_path, mcfg["embedding_dim"], device)
        estimator = ECAPACSM(
            ecapa=est_backbone,
            ecapa_frame_dim=mcfg.get("backbone_frame_dim", mcfg.get("campp_frame_dim", 3072)),
            transformer_dim=mcfg.get("transformer_dim", 256),
            embedding_dim=mcfg["embedding_dim"],
            num_layers=mcfg.get("num_layers", 2),
            num_heads=mcfg.get("num_heads", 8),
            ff_dim=mcfg.get("ff_dim", 512),
            dropout=mcfg.get("dropout", 0.1),
            freeze_ecapa=True,
            use_conv=mcfg.get("use_conv", False),
            conv_kernel_size=mcfg.get("conv_kernel_size", 3),
            use_transformer=mcfg.get("use_transformer", True),
        ).to(device)
        checkpoint_proxy_cls = EcapaModelCheckpointProxy
    elif backbone == "campplus" and impl == "v1":
        est_campp = load_pretrained_campplus(ce_cfg["campplus"]["ckpt_path"], mcfg["embedding_dim"], device)
        estimator = CAMPlusCSM(
            campp=est_campp,
            campp_frame_dim=mcfg.get("campp_frame_dim", 512),
            transformer_dim=mcfg.get("transformer_dim", 256),
            embedding_dim=mcfg["embedding_dim"],
            num_layers=mcfg.get("num_layers", 2),
            num_heads=mcfg.get("num_heads", 8),
            ff_dim=mcfg.get("ff_dim", 512),
            dropout=mcfg.get("dropout", 0.1),
            freeze_campp=True,
            use_conv=mcfg.get("use_conv", False),
            conv_kernel_size=mcfg.get("conv_kernel_size", 3),
            use_transformer=mcfg.get("use_transformer", True),
        ).to(device)
        checkpoint_proxy_cls = ModelCheckpointProxy
    else:
        raise ValueError(
            f"{run_dir}/config.yaml has model.backbone={backbone!r}, model.impl={impl!r}; "
            "expected backbone 'campplus' (impl 'v1') or backbone 'ecapa'"
        )

    estimator.eval()
    ckpt_path = os.path.join(run_dir, "checkpoints", ckpt_name, "model.ckpt")
    checkpoint_proxy_cls(estimator).load(ckpt_path, device)
    print(f"Loaded centroid estimator: {run_dir} [{ckpt_name}] (backbone={backbone}, impl={impl})")
    return estimator, ce_cfg


def shard(items, num_shards, shard_index):
    if num_shards is None:
        return items
    if not (0 <= shard_index < num_shards):
        raise ValueError(f"--shard-index must be in [0, {num_shards})")
    return items[shard_index::num_shards]


def shard_results_path(output_dir, output_name, num_shards, shard_index):
    stem, _ = os.path.splitext(output_name)
    return os.path.join(output_dir, f".{stem}_shard{shard_index}of{num_shards}.pt")


def _strip_flag(argv, flag):
    out, skip_next = [], False
    for tok in argv:
        if skip_next:
            skip_next = False
            continue
        if tok == flag:
            skip_next = True
            continue
        if tok.startswith(flag + "="):
            continue
        out.append(tok)
    return out


def dispatch_multi_gpu(gpu_ids, args):
    child_argv = _strip_flag(sys.argv[1:], "--gpus")
    num_shards = len(gpu_ids)

    procs = []
    for i, gpu_id in enumerate(gpu_ids):
        cmd = [sys.executable, os.path.abspath(__file__)] + child_argv + [
            "--num-shards", str(num_shards),
            "--shard-index", str(i),
            "--device", f"cuda:{gpu_id}",
            "--tqdm-position", str(i),
        ]
        procs.append(subprocess.Popen(cmd))

    failed = [i for i, p in enumerate(procs) if p.wait() != 0]
    if failed:
        raise RuntimeError(f"Worker shard(s) {failed} failed; see their output above.")

    all_mse_beta, all_mse_cm = [], []
    all_pairs = []
    for i in range(num_shards):
        shard_path = shard_results_path(args.output_dir, args.csv_name, num_shards, i)
        shard_data = torch.load(shard_path, map_location="cpu")
        all_mse_beta.append(shard_data["mse_beta"].numpy())
        all_mse_cm.append(shard_data["mse_cm"].numpy())
        all_pairs.extend(shard_data["pair_keys"])
        os.remove(shard_path)
    mse_beta = np.concatenate(all_mse_beta, axis=0)
    mse_cm = np.concatenate(all_mse_cm, axis=0)
    print(f"[multi-gpu] merged {len(all_pairs)} pairs from {num_shards} GPU shards")
    finalize(mse_beta, mse_cm, args.n_values, args.beta_values,
             args.output_dir, args.csv_name, len(all_pairs))


def save_csv(n_values, beta_values, mean_mse_beta, mean_mse_cm, csv_path):
    fieldnames = ["N"] + [f"mse_beta_{b:g}" for b in beta_values] + ["mse_cm"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, n in enumerate(n_values):
            row = {"N": n, "mse_cm": mean_mse_cm}
            for j, b in enumerate(beta_values):
                row[f"mse_beta_{b:g}"] = mean_mse_beta[i, j]
            writer.writerow(row)


def finalize(mse_beta, mse_cm, n_values, beta_values, output_dir, csv_name, num_pairs):
    mean_mse_beta = mse_beta.mean(axis=0)  # [len(n_values), len(beta_values)]
    mean_mse_cm = float(mse_cm.mean())

    for n_idx, n in enumerate(n_values):
        row_str = "  ".join(f"beta={b:g}: {mean_mse_beta[n_idx, j]:.5f}" for j, b in enumerate(beta_values))
        print(f"  N={n:4d}  {row_str}")
    print(f"  CM: {mean_mse_cm:.5f} (N-independent)")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, csv_name)
    save_csv(n_values, beta_values, mean_mse_beta, mean_mse_cm, csv_path)
    print(f"Saved {csv_path} ({num_pairs} pairs)")


@torch.no_grad()
def _embed_batch(model, fbanks, device):
    batch = torch.stack(fbanks).to(device)
    return F.normalize(model(batch), p=2, dim=-1).cpu().numpy()


@torch.no_grad()
def _embed_one(model, fbank, device):
    emb = model(fbank.unsqueeze(0).to(device))
    return F.normalize(emb, p=2, dim=-1).squeeze(0).cpu().numpy()


def embed_solo_samples(model, paths, sample_rate, num_samples, fbank_fn, device, batch_size, wav_cache):
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


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", choices=["campplus", "ecapa"], default="campplus")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--librispeech-root", default=DEFAULT_LIBRISPEECH_ROOT)
    parser.add_argument("--subsets", nargs="+", default=None,
                         help="Defaults to the backbone's own matching subset(s) (see BACKBONE_DEFAULTS).")
    parser.add_argument("--mode", default="mix_clean")
    parser.add_argument("--oracle-centroid-cache", default=None,
                         help="Defaults to the backbone's own matching cache (see BACKBONE_DEFAULTS).")
    parser.add_argument("--campplus-ckpt", default=DEFAULT_CAMPPLUS_CKPT)
    parser.add_argument("--ecapa-ckpt", default=DEFAULT_ECAPA_CKPT)
    parser.add_argument("--emb-dim", type=int, default=192)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--enroll-len", type=_len_or_full, default=3.0)
    parser.add_argument("--mix-len", type=_len_or_full, default=3.0)
    parser.add_argument("--sir-range", type=float, nargs=2, default=[-5.0, 5.0])
    parser.add_argument("--n-values", type=int, nargs="+", default=DEFAULT_N_VALUES)
    parser.add_argument("--beta-values", type=float, nargs="+", default=DEFAULT_BETA_VALUES)
    parser.add_argument("--num-pairs", type=int, default=DEFAULT_NUM_PAIRS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--estimator-run-dir", default=None,
                         help="Defaults to the backbone's own matching CM checkpoint dir (see BACKBONE_DEFAULTS).")
    parser.add_argument("--estimator-ckpt-name", default="CKPT+best")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--csv-name", default=None,
                         help="Defaults to 'beta_convergence_<backbone>.csv'.")
    parser.add_argument("--device", default="cuda:4" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--tqdm-position", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    bd = BACKBONE_DEFAULTS[args.backbone]
    if args.subsets is None:
        args.subsets = bd["subsets"]
    if args.oracle_centroid_cache is None:
        args.oracle_centroid_cache = bd["oracle_centroid_cache"]
    if args.estimator_run_dir is None:
        args.estimator_run_dir = bd["estimator_run_dir"]
    if args.csv_name is None:
        args.csv_name = f"beta_convergence_{args.backbone}.csv"

    if (args.num_shards is None) != (args.shard_index is None):
        raise ValueError("--num-shards and --shard-index must be given together")
    if args.gpus is not None:
        if args.num_shards is not None:
            raise ValueError("--gpus and --num-shards/--shard-index are mutually exclusive")
        gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
        dispatch_multi_gpu(gpu_ids, args)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    max_n = max(args.n_values)

    model = load_backbone(args.backbone, args.campplus_ckpt, args.ecapa_ckpt, args.emb_dim, device)
    fbank_fn = FBank(80, sample_rate=args.sample_rate, mean_nor=True)
    enroll_samples = None if args.enroll_len is None else int(round(args.enroll_len * args.sample_rate))
    mix_samples = None if args.mix_len is None else int(round(args.mix_len * args.sample_rate))

    estimator, ce_cfg = load_csm(args.estimator_run_dir, device, args.estimator_ckpt_name)
    estimator_enroll_samples = _samples_or_full(ce_cfg["data"]["enroll_len"], args.sample_rate)
    estimator_mix_samples = _samples_or_full(ce_cfg["data"]["mix_len"], args.sample_rate)

    speaker_index = build_speaker_index(args.librispeech_root, args.subsets)
    oracle_centroids = torch.load(args.oracle_centroid_cache, map_location="cpu")

    mix_dirs = [os.path.join(args.data_root, s) for s in args.subsets]
    all_pairs = list_unique_pairs(mix_dirs, mode=args.mode)
    valid_pairs = [
        key for key in all_pairs
        if all(spk in speaker_index and spk in oracle_centroids for spk in key.split("_"))
    ]
    print(f"{len(valid_pairs)}/{len(all_pairs)} pairs have both speakers in the LibriSpeech index "
          f"and the oracle centroid cache")
    if len(valid_pairs) < args.num_pairs:
        raise ValueError(f"Only {len(valid_pairs)} valid pairs available, need --num-pairs {args.num_pairs}")
    chosen_pairs = random.sample(valid_pairs, args.num_pairs)
    shard_pairs = shard(chosen_pairs, args.num_shards, args.shard_index)

    # mse_beta[pair_idx][n_idx][beta_idx]; mse_cm[pair_idx] (N-independent).
    mse_beta = np.zeros((len(shard_pairs), len(args.n_values), len(args.beta_values)))
    mse_cm = np.zeros(len(shard_pairs))

    for pair_idx, key in enumerate(tqdm(shard_pairs, desc="pairs", unit="pair", position=args.tqdm_position)):
        s1, s2 = key.split("_")
        paths_1, paths_2 = speaker_index[s1], speaker_index[s2]
        oracle_s2 = oracle_centroids[s2].numpy()

        rng = random.Random(f"{args.seed}:{key}")
        combos = [(rng.choice(paths_1), rng.choice(paths_2)) for _ in range(max_n)]
        solo_1 = [rng.choice(paths_1) for _ in range(max_n)]

        wav_cache = {}
        mix_embeds = embed_mixture_samples(
            model, combos, args.sample_rate, mix_samples, fbank_fn, args.sir_range,
            device, args.batch_size, wav_cache, rng,
        )
        s1_embeds = embed_solo_samples(
            model, solo_1, args.sample_rate, enroll_samples, fbank_fn, device, args.batch_size, wav_cache,
        )

        mix_cumsum = np.cumsum(mix_embeds, axis=0)
        s1_cumsum = np.cumsum(s1_embeds, axis=0)

        for n_idx, n in enumerate(args.n_values):
            c_mix_n = mix_cumsum[n - 1] / n
            c_s1_n = s1_cumsum[n - 1] / n
            for beta_idx, beta in enumerate(args.beta_values):
                pseudo_s2 = beta * c_mix_n - c_s1_n
                mse_beta[pair_idx, n_idx, beta_idx] = mse(pseudo_s2, oracle_s2)

        # CM baseline: trained centroid estimator on a single N=1 draw, its own
        # independent rng stream so it doesn't perturb the beta-sweep draws above.
        est_rng = random.Random(f"{args.seed}:{key}:estimator")
        est_mix_path_a, est_mix_path_b = combos[0]
        est_sir_db = est_rng.uniform(*args.sir_range)
        est_wav_a = _maybe_crop_or_pad(_load_mono(est_mix_path_a, args.sample_rate), estimator_mix_samples)
        est_wav_b = _maybe_crop_or_pad(_load_mono(est_mix_path_b, args.sample_rate), estimator_mix_samples)
        est_mix_wav = _mix_with_sir(est_wav_a, est_wav_b, est_sir_db)
        est_enroll_wav = _maybe_crop_or_pad(_load_mono(solo_1[0], args.sample_rate), estimator_enroll_samples)
        with torch.no_grad():
            c_hat_mix = estimator(
                fbank_fn(est_mix_wav).unsqueeze(0).to(device)
            )["centroid"].squeeze(0).cpu().numpy()
            c_hat_s1 = estimator(
                fbank_fn(est_enroll_wav).unsqueeze(0).to(device)
            )["centroid"].squeeze(0).cpu().numpy()
        pseudo_s2_cm = c_hat_mix - c_hat_s1
        mse_cm[pair_idx] = mse(pseudo_s2_cm, oracle_s2)

    if args.num_shards is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        shard_path = shard_results_path(args.output_dir, args.csv_name, args.num_shards, args.shard_index)
        torch.save({
            "mse_beta": torch.from_numpy(mse_beta),
            "mse_cm": torch.from_numpy(mse_cm),
            "pair_keys": shard_pairs,
        }, shard_path)
        print(f"[shard {args.shard_index}/{args.num_shards}] saved {len(shard_pairs)} pairs -> {shard_path}")
        return

    finalize(mse_beta, mse_cm, args.n_values, args.beta_values,
             args.output_dir, args.csv_name, len(chosen_pairs))


if __name__ == "__main__":
    main()
