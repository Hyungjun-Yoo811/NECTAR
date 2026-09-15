"""
Offline precompute of per-speaker-pair "mixture oracle centroid" caches: for
each speaker pair in a Libri2Mix subset, synthesizes random-utterance
mixtures, embeds them with a frozen CAM++/ECAPA-TDNN, and averages
(without_norm convention, see precompute_centroids.py) into one centroid per
pair -- the training target for the centroid estimator's mixture branch.

Usage:
    python speakerlab/bin/preprocessing/precompute_mixture_centroids.py
    python speakerlab/bin/preprocessing/precompute_mixture_centroids.py --gpus 4,5,6,7
    python speakerlab/bin/preprocessing/precompute_mixture_centroids.py --subsets train-360 \\
        --merge-with .../mixture_pair_centroids_train.pt --output .../mixture_pair_centroids_train.pt
"""

import argparse
import itertools
import os
import random
import subprocess
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_PROJECT_ROOT = os.path.dirname(_REPO_ROOT)

from speakerlab.dataset.dataset import build_speaker_index, list_unique_pairs, pair_key
from speakerlab.models.campplus.csm import load_pretrained_campplus
from speakerlab.models.ecapa_tdnn.csm import load_pretrained_ecapa
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
    "snapshots/master/ecapa_tdnn.bin"
)
DEFAULT_OUTPUT = os.path.join(
    _PROJECT_ROOT, "dataset/data/centroid_cache/without_norm/mixture_pair_centroids_train.pt")


def _len_or_full(value):
    """argparse type for --audio-length: 'full' -> None (full length,
    unbatched, min-length-trimmed mixing), anything else -> float seconds
    (fixed crop/pad, batched)."""
    return None if value.lower() == "full" else float(value)


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


def dispatch_multi_gpu(gpu_ids, output, merge_with):
    """Spawn one worker subprocess per GPU (this same script, sharded), wait
    for all of them, then merge their shard outputs (and, if given,
    --merge-with's existing cache) into `output`."""
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
    saved = _save_merged(merged, output, merge_with)
    suffix = f" (merged with existing {merge_with})" if merge_with else ""
    print(f"[multi-gpu] merged {len(merged)} entries from {num_shards} GPU shards "
          f"-> {len(saved)} total -> {output}{suffix}")


def checkpoint_path_for(output):
    """Resumable-checkpoint path for a given --output (or shard output) path
    -- a distinct file per --output, so single-process runs and each --gpus
    shard worker (which already have distinct, shard-specific `output`
    values via shard_output_path) never collide on the same checkpoint."""
    return f"{output}.ckpt.pt"


def _atomic_torch_save(obj, path):
    """torch.save with an atomic rename -- if the process is killed mid-
    write, only the .tmp file is left corrupted; `path` itself (the
    checkpoint a resumed run reads) is never partially written."""
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _save_merged(centroids, output, merge_with):
    """Save `centroids` to `output`, first merging them on top of an existing
    cache at `merge_with` if given -- lets a run that only computed a new
    --subsets slice extend an already-precomputed cache without recomputing
    it. `output` may equal `merge_with` to update the cache in place."""
    if merge_with:
        existing = torch.load(merge_with, map_location="cpu")
        overlap = set(existing) & set(centroids)
        if overlap:
            print(f"  [merge] {len(overlap)} pair(s) already in {merge_with} are "
                  f"being overwritten with this run's newly computed values")
        existing.update(centroids)
        centroids = existing
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    torch.save(centroids, output)
    return centroids


def _load_mono(path, target_sr):
    import torchaudio
    wav, sr = torchaudio.load(path)
    wav = wav.mean(0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def _crop_or_pad(wav, num_samples):
    length = wav.shape[-1]
    if length < num_samples:
        return torch.nn.functional.pad(wav, (0, num_samples - length))
    start = random.randint(0, length - num_samples)
    return wav[start:start + num_samples]


def _mix_with_sir(wav_a, wav_b, sir_db):
    """Scale wav_b so that 10*log10(power(wav_a) / power(scaled wav_b)) == sir_db
    -- matches Libri2Mix's own random-SIR mixing convention, rather than a
    plain unweighted sum, so the synthesized mixtures are representative of
    what CAM++ actually sees at train/inference time."""
    power_a = wav_a.pow(2).mean().clamp_min(1e-8)
    power_b = wav_b.pow(2).mean().clamp_min(1e-8)
    scale = torch.sqrt(power_a / power_b / (10 ** (sir_db / 10)))
    return wav_a + scale * wav_b


def _get_cached_wav(path, sample_rate, num_samples, cache):
    """Load+resample a file once per process and reuse it on repeat picks --
    avoids redundant disk I/O/decode. num_samples=None caches the full
    waveform untouched (min-length trimming for mixing happens per-combo in
    _pair_centroid); otherwise crops/pads to num_samples once and caches that."""
    cached = cache.get(path)
    if cached is not None:
        return cached
    wav = _load_mono(path, sample_rate)
    if num_samples is not None:
        wav = _crop_or_pad(wav, num_samples)
    cache[path] = wav
    return wav


def _iter_full_combos(spk_a, spk_b, speaker_index, limit_combos_per_pair):
    """Every (utt_a, utt_b) combination for this pair (--full-combos mode)."""
    combos = itertools.product(speaker_index[spk_a], speaker_index[spk_b])
    if limit_combos_per_pair is not None:
        combos = itertools.islice(combos, limit_combos_per_pair)
    return combos


def _iter_sampled_combos(spk_a, spk_b, speaker_index, num_samples_per_pair):
    """num_samples_per_pair random (utt_a, utt_b) combinations for this pair
    (default mode) -- independent draws WITH replacement, so this works even
    when a speaker has fewer utterances than num_samples_per_pair."""
    files_a, files_b = speaker_index[spk_a], speaker_index[spk_b]
    for _ in range(num_samples_per_pair):
        yield random.choice(files_a), random.choice(files_b)


def _pair_centroid(
    key, speaker_index, campp, fbank_fn, device, sample_rate, crop_samples,
    sir_range, batch_size, wav_cache, combos, progress,
):
    """Reduce `combos` ((utt_a_path, utt_b_path) pairs) through CAM++/ECAPA to
    a running sum (never materializes a per-mixture embedding list), so memory
    stays bounded regardless of combo count. crop_samples=None mixes each
    combo at its own (shorter-of-two) length and embeds one at a time (CAM++'s
    stats pooling is length-sensitive); a fixed length instead batches."""
    embed_sum = None
    n = 0

    if crop_samples is None:
        with torch.no_grad():
            for path_a, path_b in combos:
                wav_a = _get_cached_wav(path_a, sample_rate, None, wav_cache)
                wav_b = _get_cached_wav(path_b, sample_rate, None, wav_cache)
                m = min(wav_a.shape[-1], wav_b.shape[-1])
                mix_wav = _mix_with_sir(wav_a[:m], wav_b[:m], random.uniform(*sir_range))
                feat = fbank_fn(mix_wav).unsqueeze(0).to(device)
                embed = F.normalize(campp(feat), p=2, dim=-1).cpu().squeeze(0)
                embed_sum = embed if embed_sum is None else embed_sum + embed
                n += 1
                progress.update(1)
    else:
        batch_feats = []

        def flush(embed_sum, n):
            if not batch_feats:
                return embed_sum, n
            feats = torch.stack(batch_feats, dim=0).to(device)  # [B, T, 80]
            with torch.no_grad():
                embeds = F.normalize(campp(feats), p=2, dim=-1).cpu()
            batch_sum = embeds.sum(dim=0)
            embed_sum = batch_sum if embed_sum is None else embed_sum + batch_sum
            n += embeds.shape[0]
            progress.update(embeds.shape[0])
            batch_feats.clear()
            return embed_sum, n

        for path_a, path_b in combos:
            wav_a = _get_cached_wav(path_a, sample_rate, crop_samples, wav_cache)
            wav_b = _get_cached_wav(path_b, sample_rate, crop_samples, wav_cache)
            mix_wav = _mix_with_sir(wav_a, wav_b, random.uniform(*sir_range))
            batch_feats.append(fbank_fn(mix_wav))
            if len(batch_feats) >= batch_size:
                embed_sum, n = flush(embed_sum, n)
        embed_sum, n = flush(embed_sum, n)

    if n == 0:
        raise ValueError(f"Pair {key} has zero utterance combinations (empty speaker file list?).")
    # without_norm convention: mean of per-mixture L2-normalized embeddings,
    # WITHOUT a final re-normalize, so the cache stays magnitude-carrying
    # (matches precompute_centroids.py / BSRNN's other oracle-centroid
    # caches; see module docstring).
    return embed_sum / n


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Libri2Mix wav16k/min root")
    parser.add_argument("--librispeech-root", default=DEFAULT_LIBRISPEECH_ROOT)
    parser.add_argument("--subsets", nargs="+", default=["train-100"],
                         help="Shared Libri2Mix/LibriSpeech subset name(s), e.g. train-100 or dev")
    parser.add_argument("--mode", default="mix_clean")
    parser.add_argument("--backbone", choices=["campplus", "ecapa"], default="campplus",
                         help="Frozen verification backbone to embed with.")
    parser.add_argument("--campplus-ckpt", default=DEFAULT_CAMPPLUS_CKPT)
    parser.add_argument("--ecapa-ckpt", default=DEFAULT_ECAPA_CKPT)
    parser.add_argument("--emb-dim", type=int, default=192)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--audio-length", type=_len_or_full, default="full",
                         help="Crop/pad length in seconds per mixture; 'full' embeds unbatched at each pair's own length.")
    parser.add_argument("--num-samples-per-pair", type=int, default=50,
                         help="Random utterance-combination samples per pair (Monte-Carlo average).")
    parser.add_argument("--full-combos", action="store_true",
                         help="Use the full utterance cross product per pair instead of sampling.")
    parser.add_argument("--sir-range", type=float, nargs=2, default=[-5.0, 5.0])
    parser.add_argument("--batch-size", type=int, default=8192,
                         help="Mixtures per GPU forward call, batched within one pair.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--merge-with", default=None,
                         help="Merge this run's newly computed pairs into an existing cache before saving.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-pairs", type=int, default=None,
                         help="Cap number of pairs, for smoke testing.")
    parser.add_argument("--limit-combos-per-pair", type=int, default=None,
                         help="--full-combos mode only: cap combinations per pair, for smoke testing.")
    parser.add_argument("--num-shards", type=int, default=None,
                         help="Split pairs into N shards for parallel runs.")
    parser.add_argument("--shard-index", type=int, default=None,
                         help="Which shard (0-indexed) this process computes.")
    parser.add_argument("--gpus", default=None,
                         help="Comma-separated GPU ids; spawns one worker per GPU and merges the result.")
    parser.add_argument("--tqdm-position", type=int, default=0,
                         help=argparse.SUPPRESS)  # set internally by --gpus workers
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--checkpoint-every", type=int, default=50,
                         help="Save a resumable checkpoint every N completed pairs (0 disables it).")
    parser.add_argument("--cpu-threads", type=int, default=16,
                         help="Cap torch's CPU intra-op thread count (avoids thrashing on a shared box).")
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    if (args.num_shards is None) != (args.shard_index is None):
        raise ValueError("--num-shards and --shard-index must be given together")
    if args.limit_combos_per_pair is not None and not args.full_combos:
        raise ValueError("--limit-combos-per-pair only applies to --full-combos mode; "
                          "use --num-samples-per-pair to cap sampled mode instead.")
    if args.merge_with is not None and not os.path.exists(args.merge_with):
        raise FileNotFoundError(f"--merge-with cache not found: {args.merge_with}")

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
        dispatch_multi_gpu(gpu_ids, output_base, args.merge_with)
        return

    random.seed(args.seed)
    device = torch.device(args.device)

    if args.backbone == "campplus":
        campp = load_pretrained_campplus(args.campplus_ckpt, args.emb_dim, device)
    else:
        campp = load_pretrained_ecapa(args.ecapa_ckpt, args.emb_dim, device)
    fbank_fn = FBank(80, sample_rate=args.sample_rate, mean_nor=True)

    mix_dirs = [os.path.join(args.data_root, s) for s in args.subsets]
    pairs = list_unique_pairs(mix_dirs, mode=args.mode)
    if args.limit_pairs is not None:
        pairs = pairs[:args.limit_pairs]
    pairs = shard(pairs, args.num_shards, args.shard_index)
    output = shard_output_path(output_base, args.num_shards, args.shard_index)

    # Resume support: reload any partial result a previous (interrupted) run
    # of this EXACT --output/shard already checkpointed, and skip those pairs
    # below instead of recomputing them.
    checkpoint_path = checkpoint_path_for(output)
    centroids = {}
    if args.checkpoint_every > 0 and os.path.exists(checkpoint_path):
        centroids = torch.load(checkpoint_path, map_location="cpu")
        print(f"Resuming from checkpoint {checkpoint_path}: {len(centroids)} pair(s) already computed.")

    speaker_index = build_speaker_index(args.librispeech_root, args.subsets)
    crop_samples = None if args.audio_length is None else int(args.audio_length * args.sample_rate)

    valid_pairs, skipped, resumed = [], 0, 0
    for key in pairs:
        spk_a, spk_b = key.split("_")
        if spk_a not in speaker_index or spk_b not in speaker_index:
            skipped += 1
        elif key in centroids:
            resumed += 1
        else:
            valid_pairs.append(key)
    if skipped:
        print(f"  skipping {skipped} pair(s) with a speaker missing from the LibriSpeech index")
    if resumed:
        print(f"  skipping {resumed} pair(s) already completed in the checkpoint")

    if args.full_combos:
        def _pair_combo_count(key):
            n = len(speaker_index[key.split("_")[0]]) * len(speaker_index[key.split("_")[1]])
            return n if args.limit_combos_per_pair is None else min(n, args.limit_combos_per_pair)

        def _combos_for(key):
            spk_a, spk_b = key.split("_")
            return _iter_full_combos(spk_a, spk_b, speaker_index, args.limit_combos_per_pair)

        total_combos = sum(_pair_combo_count(k) for k in valid_pairs)
        mode_desc = "full cross product" + (
            "" if args.limit_combos_per_pair is None else f", capped at {args.limit_combos_per_pair}/pair"
        )
    else:
        def _combos_for(key):
            spk_a, spk_b = key.split("_")
            return _iter_sampled_combos(spk_a, spk_b, speaker_index, args.num_samples_per_pair)

        total_combos = len(valid_pairs) * args.num_samples_per_pair
        mode_desc = f"{args.num_samples_per_pair} random samples/pair"

    print(f"Computing mixture-pair centroids for {len(valid_pairs)} speaker pairs "
          f"({total_combos:,} total utterance combinations, {mode_desc}) "
          f"from {mix_dirs} -> {output}")

    wav_cache = {}
    progress = tqdm(total=total_combos, desc="combos", unit="combo", position=args.tqdm_position)
    since_checkpoint = 0
    for key in valid_pairs:
        centroids[key] = _pair_centroid(
            key, speaker_index, campp, fbank_fn, device, args.sample_rate, crop_samples,
            args.sir_range, args.batch_size, wav_cache, _combos_for(key), progress,
        )
        since_checkpoint += 1
        if args.checkpoint_every > 0 and since_checkpoint >= args.checkpoint_every:
            _atomic_torch_save(centroids, checkpoint_path)
            since_checkpoint = 0
    progress.close()

    # --merge-with only applies to a true final save -- a manually-sharded
    # (--num-shards/--shard-index, no --gpus) worker always writes its own
    # unmerged shard file, merged later via merge_shards.py.
    merge_with = args.merge_with if args.num_shards is None else None
    saved = _save_merged(centroids, output, merge_with)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    suffix = f" (merged with existing {merge_with}, {len(saved)} total)" if merge_with else ""
    print(f"Saved {len(centroids)} mixture-pair centroids to {output}{suffix}")


if __name__ == "__main__":
    main()
