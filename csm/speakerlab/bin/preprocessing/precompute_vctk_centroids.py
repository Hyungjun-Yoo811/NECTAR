"""
Offline precompute of per-speaker reference centroids for the VCTK corpus
(paper eq. 5-6, magnitude-carrying), for the centroid estimator only -- these
caches are never used by BSRNN. Train/valid speakers come from explicit list
files in the corpus root; there is no test split. Writes
<output-dir>/{train,valid}_speaker_centroids.pt.

Usage:
    python speakerlab/bin/preprocessing/precompute_vctk_centroids.py
    python speakerlab/bin/preprocessing/precompute_vctk_centroids.py --gpus 4,5,6,7
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

from speakerlab.dataset.dataset import (
    DEFAULT_VCTK_ROOT,
    build_vctk_speaker_index,
    vctk_speaker_split,
)
from speakerlab.models.campplus.csm import (
    compute_speaker_centroid,
    load_pretrained_campplus,
)
from speakerlab.process.processor import FBank

DEFAULT_CAMPPLUS_CKPT = os.path.expanduser(
    "~/.cache/modelscope/models/"
    "iic--speech_campplus_sv_zh_en_16k-common_advanced/"
    "snapshots/v1.0.0/campplus_cn_en_common.pt"
)
DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "dataset/data/centroid_cache/vctk/without_norm")
SPLITS = ("train", "valid")


def split_of_speaker(split_map):
    """Invert split_map -> {speaker_id: split_name}."""
    return {spk: split for split, spks in split_map.items() for spk in spks}


def shard(items, num_shards, shard_index):
    if num_shards is None:
        return items
    if not (0 <= shard_index < num_shards):
        raise ValueError(f"--shard-index must be in [0, {num_shards})")
    return items[shard_index::num_shards]


def shard_output_path(output_dir, split, num_shards, shard_index):
    stem = os.path.join(output_dir, f"{split}_speaker_centroids")
    if num_shards is None:
        return f"{stem}.pt"
    return f"{stem}_shard{shard_index}of{num_shards}.pt"


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


def dispatch_multi_gpu(gpu_ids, output_dir):
    """Spawn one worker subprocess per GPU (this same script, sharded), wait
    for all of them, then merge their per-split shard files into one cache per
    split."""
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

    os.makedirs(output_dir, exist_ok=True)
    for split in SPLITS:
        merged = {}
        for i in range(num_shards):
            shard_path = shard_output_path(output_dir, split, num_shards, i)
            merged.update(torch.load(shard_path, map_location="cpu"))
        out_path = shard_output_path(output_dir, split, None, None)
        torch.save(merged, out_path)
        print(f"[multi-gpu] {split}: merged {len(merged)} speakers from "
              f"{num_shards} shards -> {out_path}")


def load_fbank(path, sample_rate, fbank):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(0)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return fbank(wav)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vctk-root", default=DEFAULT_VCTK_ROOT)
    parser.add_argument("--wav-subdir", default="wav48",
                         help="Subdir of --vctk-root holding <spk>/<utt> files.")
    parser.add_argument("--ext", default=".wav", help="Audio file extension, e.g. .wav or .flac")
    parser.add_argument("--mic", default=None,
                         help="Keep only files whose stem ends with _<mic>, for the VCTK 0.92 dual-mic layout.")
    parser.add_argument("--campplus-ckpt", default=DEFAULT_CAMPPLUS_CKPT)
    parser.add_argument("--emb-dim", type=int, default=192)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--train-list", default=None,
                         help="Speaker-id list file for the train split (default: <vctk-root>/train-speakers.txt).")
    parser.add_argument("--val-list", default=None,
                         help="Speaker-id list file for the valid split (default: <vctk-root>/val-speakers.txt).")
    parser.add_argument("--num-utts-per-speaker", type=int, default=None,
                         help="Cap utterances per speaker (random sample); default uses all.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-shards", type=int, default=None,
                         help="Split speakers into N shards for parallel runs.")
    parser.add_argument("--shard-index", type=int, default=None,
                         help="Which shard (0-indexed) this process computes.")
    parser.add_argument("--gpus", default=None,
                         help="Comma-separated GPU ids; spawns one worker per GPU and merges the result.")
    parser.add_argument("--tqdm-position", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if (args.num_shards is None) != (args.shard_index is None):
        raise ValueError("--num-shards and --shard-index must be given together")
    if args.gpus is not None:
        if args.num_shards is not None:
            raise ValueError("--gpus and --num-shards/--shard-index are mutually exclusive")
        gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
        dispatch_multi_gpu(gpu_ids, args.output_dir)
        return

    random.seed(args.seed)
    device = torch.device(args.device)
    is_main = args.shard_index in (None, 0)

    speaker_index = build_vctk_speaker_index(args.vctk_root, args.wav_subdir, args.ext, args.mic)
    all_speakers = sorted(speaker_index.keys())
    split_map = vctk_speaker_split(
        args.vctk_root, available_speakers=all_speakers,
        train_list=args.train_list, val_list=args.val_list,
    )
    spk_split = split_of_speaker(split_map)
    considered = sorted(spk_split)  # train + valid speakers that have audio
    if is_main:
        dropped = sorted(set(all_speakers) - set(considered))
        print(f"VCTK speakers: {len(all_speakers)} with audio -> "
              f"train={len(split_map['train'])}, valid={len(split_map['valid'])} "
              f"(dropped {len(dropped)} not in either list: {dropped})")
        print(f"  valid: {split_map['valid']}")

    # Shard only the considered (train+valid) speakers for balanced parallel work.
    my_speakers = shard(considered, args.num_shards, args.shard_index)

    campp = load_pretrained_campplus(args.campplus_ckpt, args.emb_dim, device)
    fbank = FBank(80, sample_rate=args.sample_rate, mean_nor=True)

    centroids = {split: {} for split in SPLITS}
    for spk in tqdm(my_speakers, desc="speakers", unit="spk", position=args.tqdm_position):
        files = speaker_index[spk]
        if args.num_utts_per_speaker is not None:
            files = random.sample(files, min(len(files), args.num_utts_per_speaker))
        fbanks = [load_fbank(p, args.sample_rate, fbank) for p in files]
        # final_normalize=False (the compute_speaker_centroid default): keep the
        # centroid's magnitude, matching centroid_cache/without_norm/.
        centroids[spk_split[spk]][spk] = compute_speaker_centroid(campp, fbanks, device)

    os.makedirs(args.output_dir, exist_ok=True)
    for split in SPLITS:
        out_path = shard_output_path(args.output_dir, split, args.num_shards, args.shard_index)
        torch.save(centroids[split], out_path)
        print(f"Saved {len(centroids[split])} {split} speaker centroids to {out_path}")


if __name__ == "__main__":
    main()
