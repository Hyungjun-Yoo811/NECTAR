#!/bin/bash
# =============================================================================
# Step 1 of 4: downloads LibriSpeech + WHAM noise, augments WHAM train noise,
# and calls create_librimix_from_metadata.py to generate the Libri2Mix
# mixture wavs. Usage: bash step1_generate_librimix.sh /path/to/storage
# Output: LibriSpeech/, wham_noise/, Libri2Mix/wav{8k,16k}/{min,max}/{split}/
# =============================================================================

set -eu

storage_dir=$1
librispeech_dir=$storage_dir/LibriSpeech
wham_dir=$storage_dir/wham_noise
librimix_outdir=$storage_dir/

# --- Download each LibriSpeech split (if not already present) ---
function LibriSpeech_dev_clean() {
    if ! test -e $librispeech_dir/dev-clean; then
        echo "Downloading LibriSpeech/dev-clean..."
        wget -c --tries=0 --read-timeout=20 \
            http://www.openslr.org/resources/12/dev-clean.tar.gz -P $storage_dir
        tar -xzf $storage_dir/dev-clean.tar.gz -C $storage_dir
        rm -f $storage_dir/dev-clean.tar.gz
    fi
}

function LibriSpeech_test_clean() {
    if ! test -e $librispeech_dir/test-clean; then
        echo "Downloading LibriSpeech/test-clean..."
        wget -c --tries=0 --read-timeout=20 \
            http://www.openslr.org/resources/12/test-clean.tar.gz -P $storage_dir
        tar -xzf $storage_dir/test-clean.tar.gz -C $storage_dir
        rm -f $storage_dir/test-clean.tar.gz
    fi
}

function LibriSpeech_clean100() {
    if ! test -e $librispeech_dir/train-clean-100; then
        echo "Downloading LibriSpeech/train-clean-100..."
        wget -c --tries=0 --read-timeout=20 \
            http://www.openslr.org/resources/12/train-clean-100.tar.gz -P $storage_dir
        tar -xzf $storage_dir/train-clean-100.tar.gz -C $storage_dir
        rm -f $storage_dir/train-clean-100.tar.gz
    fi
}

function LibriSpeech_clean360() {
    if ! test -e $librispeech_dir/train-clean-360; then
        echo "Downloading LibriSpeech/train-clean-360..."
        wget -c --tries=0 --read-timeout=20 \
            http://www.openslr.org/resources/12/train-clean-360.tar.gz -P $storage_dir
        tar -xzf $storage_dir/train-clean-360.tar.gz -C $storage_dir
        rm -f $storage_dir/train-clean-360.tar.gz
    fi
}

function wham() {
    if ! test -e $wham_dir; then
        echo "Downloading WHAM noise..."
        wget -c --tries=0 --read-timeout=20 \
            https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip \
            -P $storage_dir
        unzip -qn $storage_dir/wham_noise.zip -d $storage_dir
        rm -f $storage_dir/wham_noise.zip
    fi
}

# Download all splits in parallel
LibriSpeech_dev_clean &
LibriSpeech_test_clean &
LibriSpeech_clean100 &
LibriSpeech_clean360 &
wham &
wait

# WHAM train noise augmentation: 20k files -> 60k files (speed perturbation).
# Automatically skipped if already done (checked inside augment_train_noise.py).
# Requires a clone of https://github.com/JorisCos/LibriMix; override the
# location with the LIBRIMIX_REPO env var, or it defaults under storage_dir.
LIBRIMIX_REPO="${LIBRIMIX_REPO:-$storage_dir/LibriMix}"
python $LIBRIMIX_REPO/scripts/augment_train_noise.py --wham_dir $wham_dir

# Generate Libri2Mix
# --freqs 16k: generate 16kHz wav files
# --modes min: trim to the shorter utterance length (max pads to the longer one)
# --types mix_clean mix_both: generate both the noise-free (clean) and noisy (both) versions
python $LIBRIMIX_REPO/scripts/create_librimix_from_metadata.py \
    --librispeech_dir $librispeech_dir \
    --wham_dir $wham_dir \
    --metadata_dir $LIBRIMIX_REPO/metadata/Libri2Mix \
    --librimix_outdir $librimix_outdir \
    --n_src 2 \
    --freqs 16k \
    --modes min \
    --types mix_clean mix_both

echo "Done. Output at: ${librimix_outdir}Libri2Mix/"
