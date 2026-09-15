"""
Merge several per-speaker centroid caches ({speaker_id: tensor} .pt files) into
one, for mixed-corpus training/eval (e.g. LibriSpeech + VCTK) so a single cache
covers every speaker in a mixed train_subset. Later inputs override earlier
ones on a key clash (error by default; pass --allow-overlap to merge anyway).

Usage:
    python speakerlab/bin/preprocessing/merge_caches.py \
        --inputs a.pt b.pt [c.pt ...] --output merged.pt
"""

import argparse
import os
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True,
                         help="Two or more {speaker_id: tensor} .pt caches to merge.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-overlap", action="store_true",
                         help="Permit speaker ids shared across inputs (last input wins). "
                              "Default: any overlap is an error.")
    args = parser.parse_args()

    if len(args.inputs) < 2:
        print("Note: fewer than 2 inputs -- nothing to merge, just copying.", file=sys.stderr)

    merged = {}
    for path in args.inputs:
        cache = torch.load(path, map_location="cpu")
        overlap = set(cache) & set(merged)
        if overlap and not args.allow_overlap:
            raise ValueError(
                f"{len(overlap)} speaker id(s) in {path} already present from an "
                f"earlier input, e.g. {sorted(overlap)[:5]}. Pass --allow-overlap "
                f"to merge anyway (last input wins)."
            )
        merged.update(cache)
        print(f"  + {path}: {len(cache)} speakers (running total {len(merged)})")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(merged, args.output)
    print(f"Saved {len(merged)} merged speaker centroids to {args.output}")


if __name__ == "__main__":
    main()
