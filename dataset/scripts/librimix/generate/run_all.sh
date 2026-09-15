#!/bin/bash
# =============================================================================
# Runs the full Libri2Mix pipeline from scratch: step1 downloads LibriSpeech +
# WHAM noise and builds mixtures, step2 copies metadata CSVs, step3a/3b build
# enrollment CSVs. Requires ~600GB free space and soundfile/scipy/tqdm/pysndfx.
# =============================================================================

set -eu

STORAGE_DIR=${1:-"/data"}   # Root directory to store audio files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../metadata"  # Metadata output location

echo "=== Step 1: Generate LibriMix (download + mix) ==="
bash "$SCRIPT_DIR/step1_generate_librimix.sh" "$STORAGE_DIR"
LIBRIMIX_DIR="$STORAGE_DIR/Libri2Mix"

echo "=== Step 2: Copy local metadata CSVs ==="
cd "$DATA_DIR"
python "$SCRIPT_DIR/step2_create_local_metadata.py" \
    --librimix_dir "$LIBRIMIX_DIR"

echo "=== Step 3a: Generate train enrollment CSV ==="
python "$SCRIPT_DIR/step3a_enrollment_train.py" \
    "$DATA_DIR/wav16k/min/train-100/mixture_train-100_mix_both.csv" \
    "$DATA_DIR/wav16k/min/train-100/mixture2enrollment.csv"
python "$SCRIPT_DIR/step3a_enrollment_train.py" \
    "$DATA_DIR/wav16k/min/train-360/mixture_train-360_mix_both.csv" \
    "$DATA_DIR/wav16k/min/train-360/mixture2enrollment.csv"

echo "=== Step 3b: Generate dev/test enrollment CSV (requires fixed mapping) ==="
# The map_mixture2enrollment files come from the speakerbeam repo's release
python "$SCRIPT_DIR/step3b_enrollment_eval.py" \
    "$DATA_DIR/wav16k/min/dev/mixture_dev_mix_both.csv" \
    "$DATA_DIR/wav16k/min/dev/map_mixture2enrollment" \
    "$DATA_DIR/wav16k/min/dev/mixture2enrollment.csv"
python "$SCRIPT_DIR/step3b_enrollment_eval.py" \
    "$DATA_DIR/wav16k/min/test/mixture_test_mix_both.csv" \
    "$DATA_DIR/wav16k/min/test/map_mixture2enrollment" \
    "$DATA_DIR/wav16k/min/test/mixture2enrollment.csv"

echo "=== Done! ==="
echo "Audio:    $LIBRIMIX_DIR/wav16k/min/{train-360,train-100,dev,test}/"
echo "Metadata: $DATA_DIR/wav16k/min/{split}/mixture*.csv"
