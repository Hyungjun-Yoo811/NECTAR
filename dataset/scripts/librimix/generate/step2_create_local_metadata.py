"""
Step 2 of 4: copies the mixture metadata CSVs produced by LibriMix generation
into per-split local directories for step3a/step3b to read.
Adapted from the Asteroid project's create_local_metadata.py (MIT License).
"""

import os
import shutil
import argparse
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument(
    "--librimix_dir", type=str, required=True,
    help="Path to the Libri2Mix root directory"
)


def create_local_metadata(librimix_dir):
    # Find every Libri2Mix/wav??k/{min,max}/metadata/ directory
    md_dirs = [f for f in glob(os.path.join(librimix_dir, "*/*/*")) if f.endswith("metadata")]
    for md_dir in md_dirs:
        # Only mixture_*.csv files (skip metrics_*.csv)
        md_files = [f for f in os.listdir(md_dir) if f.startswith("mix")]
        for md_file in md_files:
            # Extract split name from filename: mixture_train-360_mix_clean.csv -> train-360
            subset = md_file.split("_")[1]
            local_path = os.path.join(
                "data", os.path.relpath(md_dir, librimix_dir), subset
            ).replace("/metadata", "")
            os.makedirs(local_path, exist_ok=True)
            shutil.copy(os.path.join(md_dir, md_file), local_path)
            print(f"Copied: {md_file} → {local_path}/")


if __name__ == "__main__":
    args = parser.parse_args()
    create_local_metadata(args.librimix_dir)
