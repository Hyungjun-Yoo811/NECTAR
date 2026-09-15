"""
Offline precompute of per-speaker reference centroids (paper eq. 5-6: average
+ normalize many utterance-level embeddings from a frozen, pretrained CAM++
or ECAPA-TDNN) for every LibriSpeech speaker in the given subset(s). These
centroids are the AP-loss training target for the enrollment branch, and the
ground-truth used for validation/eval.

Usage:
    python speakerlab/bin/preprocessing/precompute_centroids.py
    python speakerlab/bin/preprocessing/precompute_centroids.py --gpus 0,1,2,3
"""

import argparse
import os
import random
import subprocess
import sys

import torch
import torchaudio
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_PROJECT_ROOT = os.path.dirname(_REPO_ROOT)

from speakerlab.dataset.dataset import build_speaker_index
from speakerlab.models.campplus.csm import (
    compute_speaker_centroid,
    load_pretrained_campplus,
)
from speakerlab.models.ecapa_tdnn.csm import load_pretrained_ecapa
from speakerlab.process.processor import FBank

DEFAULT_LIBRISPEECH_ROOT = os.path.join(_PROJECT_ROOT, "dataset/data/LibriSpeech")
DEFAULT_CAMPPLUS_CKPT = os.path.expanduser(
    "~/.cache/modelscope/models/"
    "iic--speech_campplus_sv_zh_en_16k-common_advanced/"
    "snapshots/v1.0.0/campplus_cn_en_common.pt"
)
DEFAULT_ECAPA_CKPT = os.path.expanduser(
    "~/.cache/modelscope/models/"
    "iic--speech_ecapa-tdnn_sv_en_voxceleb_16k/"
    "snapshots/master/ecapa_tdnn.bin"
)
# compute_speaker_centroid keeps the averaged centroid's magnitude
# (final_normalize=False): these caches are the "without_norm" targets.
DEFAULT_OUTPUT = os.path.join(
    _PROJECT_ROOT, "dataset/data/centroid_cache/without_norm/train_speaker_centroids.pt")


def shard(items, num_shards, shard_index):
    if num_shards is None:
        return items
    if not (0 <= shard_index < num_shards):
        raise ValueError(f"--shard-index must be in [0, {num_shards})")
    return items[shard_index::num_shards]


def shard_output_path(output, num_shards, shard_index):
    if num_shards is None:
        return output
    stem, ext = os.path.splitext(output)
    return f"{stem}_shard{shard_index}of{num_shards}{ext}"


def _strip_flag(argv, flag):
    """Remove '--flag value' or '--flag=value' occurrences from argv."""
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


def dispatch_multi_gpu(gpu_ids, output):
    """Spawn one worker subprocess per GPU (this same script, sharded), wait
    for all of them, then merge their shard outputs into `output`."""
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

    merged = {}
    for i in range(num_shards):
        shard_path = shard_output_path(output, num_shards, i)
        merged.update(torch.load(shard_path, map_location="cpu"))
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    torch.save(merged, output)
    print(f"[multi-gpu] merged {len(merged)} entries from {num_shards} GPU shards -> {output}")


def load_fbank(path, sample_rate, fbank):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(0)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return fbank(wav)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--librispeech-root", default=DEFAULT_LIBRISPEECH_ROOT)
    parser.add_argument("--subsets", nargs="+", default=["train-100"],
                         help="e.g. train-100, train-360, dev")
    parser.add_argument("--backbone", choices=["campplus", "ecapa"], default="campplus",
                         help="Frozen verification backbone to embed with.")
    parser.add_argument("--campplus-ckpt", default=DEFAULT_CAMPPLUS_CKPT)
    parser.add_argument("--ecapa-ckpt", default=DEFAULT_ECAPA_CKPT)
    parser.add_argument("--emb-dim", type=int, default=192)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--num-utts-per-speaker", type=int, default=None,
                         help="Cap utterances per speaker (random sample); default uses all.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-speakers", type=int, default=None,
                         help="Cap number of speakers, for smoke testing.")
    parser.add_argument("--num-shards", type=int, default=None,
                         help="Split speakers into N shards for parallel runs.")
    parser.add_argument("--shard-index", type=int, default=None,
                         help="Which shard (0-indexed) this process computes.")
    parser.add_argument("--gpus", default=None,
                         help="Comma-separated GPU ids; spawns one worker per GPU and merges the result.")
    parser.add_argument("--tqdm-position", type=int, default=0,
                         help=argparse.SUPPRESS)  # set internally by --gpus workers
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cpu-threads", type=int, default=16,
                         help="Cap torch's CPU intra-op thread count (avoids thrashing on a shared box).")
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    if (args.num_shards is None) != (args.shard_index is None):
        raise ValueError("--num-shards and --shard-index must be given together")

    output_base = args.output
    if args.backbone == "ecapa" and output_base == DEFAULT_OUTPUT:
        # Auto-suffix so a bare `--backbone ecapa` run can never silently
        # overwrite the CAM++ cache at DEFAULT_OUTPUT. Computed BEFORE the
        # --gpus dispatch branch below so the parent's merge target and each
        # spawned worker's own shard path (an independent re-parse of the
        # same argv) always agree.
        stem, ext = os.path.splitext(output_base)
        output_base = f"{stem}_ecapa{ext}"

    if args.gpus is not None:
        if args.num_shards is not None:
            raise ValueError("--gpus and --num-shards/--shard-index are mutually exclusive")
        gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
        dispatch_multi_gpu(gpu_ids, output_base)
        return

    random.seed(args.seed)
    device = torch.device(args.device)

    if args.backbone == "campplus":
        campp = load_pretrained_campplus(args.campplus_ckpt, args.emb_dim, device)
    else:
        campp = load_pretrained_ecapa(args.ecapa_ckpt, args.emb_dim, device)
    fbank = FBank(80, sample_rate=args.sample_rate, mean_nor=True)

    speaker_index = build_speaker_index(args.librispeech_root, args.subsets)
    speakers = sorted(speaker_index.keys())
    if args.limit_speakers is not None:
        speakers = speakers[:args.limit_speakers]
    speakers = shard(speakers, args.num_shards, args.shard_index)
    output = shard_output_path(output_base, args.num_shards, args.shard_index)
    print(f"Computing centroids for {len(speakers)} speakers "
          f"from subsets {args.subsets} -> {output}")

    centroids = {}
    for spk in tqdm(speakers, desc="speakers", unit="spk", position=args.tqdm_position):
        files = speaker_index[spk]
        if args.num_utts_per_speaker is None:
            chosen = files
        else:
            chosen = random.sample(files, min(len(files), args.num_utts_per_speaker))
        fbanks = [load_fbank(p, args.sample_rate, fbank) for p in chosen]
        centroids[spk] = compute_speaker_centroid(campp, fbanks, device)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    torch.save(centroids, output)
    print(f"Saved {len(centroids)} speaker centroids to {output}")


if __name__ == "__main__":
    main()
