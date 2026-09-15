"""
Merge "{output}_shard{i}of{N}.pt" files produced by running
precompute_centroids.py / precompute_mixture_centroids.py with
--num-shards/--shard-index across multiple terminals/GPUs into one cache file.

Usage:
    python speakerlab/bin/preprocessing/merge_shards.py \
        --shards centroid_cache/train-100_shard*of4.pt \
        --output centroid_cache/train-100.pt
"""

import argparse
import glob
import os

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", required=True,
                         help="Shard file paths, or glob patterns.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    shard_paths = []
    for pattern in args.shards:
        matches = sorted(glob.glob(pattern))
        shard_paths.extend(matches if matches else [pattern])

    if not shard_paths:
        raise ValueError(f"No shard files matched {args.shards}")

    merged = {}
    for path in shard_paths:
        shard = torch.load(path, map_location="cpu")
        overlap = set(shard) & set(merged)
        if overlap:
            raise ValueError(f"{path} duplicates {len(overlap)} key(s) already "
                              f"merged, e.g. {sorted(overlap)[:5]} -- shards "
                              f"should be disjoint (produced by the same "
                              f"--num-shards run).")
        merged.update(shard)
        print(f"  + {path}: {len(shard)} entries (total {len(merged)})")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(merged, args.output)
    print(f"Saved {len(merged)} merged entries to {args.output}")


if __name__ == "__main__":
    main()
