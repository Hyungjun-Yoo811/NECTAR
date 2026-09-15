"""
Datasets and dataloaders for training CSM (paper Sec. 3.2): the enrollment
branch (EnrollmentCentroidDataset, recovering a speaker's centroid from one
utterance) and the mixture branch (RealMixtureDataset, over real Libri2Mix
mixtures). Also supports VCTK and optional per-batch crop-length / noise
augmentation.
"""

import glob
import os
import random

import numpy as np
import pyloudnorm as pyln
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset, Sampler

from speakerlab.process.processor import FBank

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_SUBSET_MAP = {
    "train-100": "train-clean-100",
    "train-360": "train-clean-360",
    "dev": "dev-clean",
    "test": "test-clean",
}

# VCTK support (CSM training/eval only; never used by BSRNN). Train/valid
# speakers come from explicit list files (train-speakers.txt /
# val-speakers.txt); there is no test split. The same lists drive both this
# dataloader and precompute_vctk_centroids.py, so the split can't drift.
DEFAULT_VCTK_ROOT = os.path.join(_PROJECT_ROOT, "dataset/data/vctk/VCTK-Corpus")
DEFAULT_VCTK_TRAIN_LIST = "train-speakers.txt"
DEFAULT_VCTK_VAL_LIST = "val-speakers.txt"
VCTK_SPLITS = ("train", "valid")


def build_vctk_speaker_index(vctk_root=DEFAULT_VCTK_ROOT, wav_subdir="wav48", ext=".wav", mic=None):
    """{speaker_id: [utterance_paths]} for every VCTK speaker under
    <vctk_root>/<wav_subdir>/<spk>/*<ext>. If `mic` is given (e.g. "mic1"),
    only files whose stem ends with f"_{mic}" are kept (VCTK 0.92 layout)."""
    base = os.path.join(vctk_root, wav_subdir)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"VCTK wav dir not found: {base}")
    index: dict[str, list[str]] = {}
    for spk in sorted(os.listdir(base)):
        spk_dir = os.path.join(base, spk)
        if not os.path.isdir(spk_dir):
            continue
        files = sorted(glob.glob(os.path.join(spk_dir, f"*{ext}")))
        if mic is not None:
            suffix = f"_{mic}"
            files = [f for f in files if os.path.splitext(os.path.basename(f))[0].endswith(suffix)]
        if files:
            index[spk] = files
    return index


def _read_speaker_list(path):
    """Speaker ids from a text file, one per line; blanks and #-comments skipped."""
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]


def vctk_speaker_split(vctk_root=DEFAULT_VCTK_ROOT, available_speakers=None,
                       train_list=None, val_list=None):
    """{"train": [...], "valid": [...]} (each sorted) read from explicit
    speaker-list files (<vctk_root>/train-speakers.txt and val-speakers.txt by
    default). No test split. If available_speakers is given, list entries with
    no audio under it are dropped."""
    train_path = train_list or os.path.join(vctk_root, DEFAULT_VCTK_TRAIN_LIST)
    val_path = val_list or os.path.join(vctk_root, DEFAULT_VCTK_VAL_LIST)
    train = _read_speaker_list(train_path)
    valid = _read_speaker_list(val_path)
    overlap = set(train) & set(valid)
    if overlap:
        raise ValueError(f"Speaker(s) listed in BOTH VCTK train and val lists: {sorted(overlap)}")
    if available_speakers is not None:
        avail = set(available_speakers)
        train = [s for s in train if s in avail]
        valid = [s for s in valid if s in avail]
    return {"train": sorted(train), "valid": sorted(valid)}


def build_speaker_index(librispeech_root, subsets, vctk_root=DEFAULT_VCTK_ROOT):
    """Scan one or more datasets and return {spk_id: [audio_file_paths]}.

    Supports LibriSpeech subset tokens (train-100/train-360/dev/test, resolved
    under `librispeech_root`) and VCTK tokens:
      - "vctk-train" / "vctk-valid" -- VCTK's train/valid speaker lists
        (train-speakers.txt / val-speakers.txt; see vctk_speaker_split),
        resolved under `vctk_root`. There is no vctk-test.
      - "vctk" -- all listed (train+valid) VCTK speakers.
    Tokens may be mixed in one list (e.g. ["train-100", "train-360",
    "vctk-train"]); VCTK ids are "pXXX" so they never collide with LibriSpeech's
    numeric ids."""
    if isinstance(subsets, str):
        subsets = [subsets]
    spk_dict: dict[str, list[str]] = {}
    vctk_index = None
    vctk_split = None
    for subset in subsets:
        if subset == "vctk" or subset.startswith("vctk-"):
            if vctk_index is None:
                vctk_index = build_vctk_speaker_index(vctk_root)
                vctk_split = vctk_speaker_split(vctk_root, available_speakers=sorted(vctk_index))
            if subset == "vctk":
                chosen = sorted(set(vctk_split["train"]) | set(vctk_split["valid"]))
            else:
                split = subset.split("-", 1)[1]
                if split not in vctk_split:
                    raise KeyError(
                        f"Unknown VCTK subset {subset!r}; use vctk-train / "
                        f"vctk-valid, or vctk (train+valid)."
                    )
                chosen = vctk_split[split]
            for spk in chosen:
                spk_dict.setdefault(spk, []).extend(vctk_index[spk])
            continue

        ls_subset = _SUBSET_MAP[subset]
        subset_dir = os.path.join(librispeech_root, ls_subset)
        for spk in os.listdir(subset_dir):
            spk_dir = os.path.join(subset_dir, spk)
            if not os.path.isdir(spk_dir):
                continue
            files = []
            for chapter in os.listdir(spk_dir):
                chapter_dir = os.path.join(spk_dir, chapter)
                if not os.path.isdir(chapter_dir):
                    continue
                for f in os.listdir(chapter_dir):
                    if f.endswith(".flac") or f.endswith(".wav"):
                        files.append(os.path.join(chapter_dir, f))
            if files:
                spk_dict.setdefault(spk, []).extend(files)
    return spk_dict


def list_noise_files(noise_dir):
    """Flat list of noise clip paths under a directory (e.g. Libri2Mix's
    wav16k/min/<subset>/noise/ -- WHAM! clips, one per Libri2Mix mixture but
    used here as a generic noise pool, unpaired from any specific mixture)."""
    files = [
        os.path.join(noise_dir, f)
        for f in os.listdir(noise_dir)
        if f.endswith(".wav") or f.endswith(".flac")
    ]
    if not files:
        raise ValueError(f"No noise clips found under {noise_dir}")
    return files


def parse_pair_from_filename(mixture_filename):
    """'1089-134686-0000_1221-135766-0001.wav' -> ('1089', '1221')."""
    stem = mixture_filename[:-4] if mixture_filename.endswith(".wav") else mixture_filename
    s1_name, s2_name = stem.split("_")
    return s1_name.split("-")[0], s2_name.split("-")[0]


def pair_key(spk_a, spk_b):
    """Order-independent cache key: mixture audio doesn't depend on which
    speaker is later treated as target vs. interference."""
    return "_".join(sorted([spk_a, spk_b]))


def list_unique_pairs(mix_dirs, mode="mix_clean"):
    """Scan Libri2Mix mixture filenames (no audio I/O) for the set of unique
    (order-independent) speaker pairs present, keyed via pair_key()."""
    if isinstance(mix_dirs, str):
        mix_dirs = [mix_dirs]
    pairs = set()
    for mix_dir in mix_dirs:
        audio_dir = os.path.join(mix_dir, mode)
        for f in os.listdir(audio_dir):
            if f.endswith(".wav"):
                spk_a, spk_b = parse_pair_from_filename(f)
                pairs.add(pair_key(spk_a, spk_b))
    return sorted(pairs)


def _load_mono(path, target_sr):
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
    """Scale wav_b so that 10*log10(power(wav_a) / power(scaled wav_b)) == sir_db.
    Same formula for a second speaker (SIR) or a noise clip (SNR) -- both are
    just "how loud is this interfering signal relative to wav_a"."""
    power_a = wav_a.pow(2).mean().clamp_min(1e-8)
    power_b = wav_b.pow(2).mean().clamp_min(1e-8)
    scale = torch.sqrt(power_a / power_b / (10 ** (sir_db / 10)))
    return wav_a + scale * wav_b


_LOUDNESS_METERS: dict[int, "pyln.Meter"] = {}


def _get_loudness_meter(sample_rate: int) -> "pyln.Meter":
    meter = _LOUDNESS_METERS.get(sample_rate)
    if meter is None:
        meter = pyln.Meter(sample_rate)
        _LOUDNESS_METERS[sample_rate] = meter
    return meter


def _normalize_to_loudness(wav, target_lufs, sample_rate):
    """Scale wav so its ITU-R BS.1770 integrated loudness (pyloudnorm) hits
    target_lufs dB. No-op if wav is too quiet/short for pyloudnorm to return
    a finite loudness (integrated_loudness returns -inf for near-silence, or
    raises/degenerates on clips shorter than its 400ms gating block)."""
    meter = _get_loudness_meter(sample_rate)
    level = meter.integrated_loudness(wav.numpy().astype(np.float64))
    if not np.isfinite(level):
        return wav
    gain_db = target_lufs - level
    return wav * (10 ** (gain_db / 20.0))


def _sample_num_samples(fixed_len, crop_len_range, sample_rate):
    """fixed_len: float seconds, used as-is when crop_len_range is None.
    crop_len_range: (min_sec, max_sec) -- if given, overrides fixed_len with
    a single uniformly-drawn random duration. Call once per BATCH, not per
    item, so every item in the batch shares one length and no padding is
    needed (padding would otherwise leak into valid frames near each item's
    boundary through the backbone's conv receptive field)."""
    if crop_len_range is None:
        return int(fixed_len * sample_rate)
    lo, hi = crop_len_range
    return int(random.uniform(lo, hi) * sample_rate)


def _maybe_add_noise(
    wav,
    num_samples,
    noise_files,
    noise_prob,
    snr_range,
    sample_rate: int = 16000,
    snr_loudness: bool = False,
    speech_lufs_range: tuple[float, float] = (-33.0, -25.0),
    noise_lufs_range: tuple[float, float] = (-38.0, -30.0),
):
    """With probability noise_prob, mix in a random noise clip (cropped/padded
    to match wav's length). No-op otherwise (including when noise_files is
    None, i.e. augmentation disabled).

    snr_loudness=True matches Libri2Mix/WHAM's own mix_both mechanism: wav
    and the noise clip are each independently normalized (ITU-R BS.1770
    loudness, via pyloudnorm) to a LUFS value drawn from speech_lufs_range /
    noise_lufs_range, then summed -- the resulting SNR is an emergent side
    effect, not directly controlled (snr_range is unused in this mode). If
    False (default), snr_range is hit directly via _mix_with_sir's raw
    power ratio."""
    if not noise_files or noise_prob <= 0.0 or random.random() >= noise_prob:
        return wav
    noise_wav = _crop_or_pad(_load_mono(random.choice(noise_files), sample_rate), num_samples)
    if snr_loudness:
        wav = _normalize_to_loudness(wav, random.uniform(*speech_lufs_range), sample_rate)
        noise_wav = _normalize_to_loudness(noise_wav, random.uniform(*noise_lufs_range), sample_rate)
        return wav + noise_wav
    return _mix_with_sir(wav, noise_wav, random.uniform(*snr_range))


class EnrollmentCentroidDataset(Dataset):
    """
    One LibriSpeech utterance per item, returned as raw (uncropped) audio --
    cropping/noise/fbank all happen batch-side, in the collate_fn built by
    make_enrollment_collate() below, so every item in a batch shares one
    crop length with no padding needed.
    """

    def __init__(self, speaker_index):
        self.speaker_index = speaker_index
        self.items = [
            (path, spk)
            for spk, paths in speaker_index.items()
            for path in paths
        ]
        if not self.items:
            raise ValueError("speaker_index is empty -- no utterances found.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, spk = self.items[idx]
        return {"path": path, "spk": spk}


class UniqueSpeakerBatchSampler(Sampler):
    """
    Yields batches of dataset indices with distinct speakers per batch (each
    batch draws `batch_size` distinct speakers, sampled without replacement
    per batch but independently across batches, and one random utterance
    per speaker). Avoids in-batch same-speaker collisions for losses like
    InBatchCentroidLoss that treat each batch item as its own class.

    Requires batch_size <= number of distinct speakers in speaker_index.
    """

    def __init__(self, speaker_index, batch_size, num_batches):
        if batch_size > len(speaker_index):
            raise ValueError(
                f"batch_size ({batch_size}) must be <= number of speakers "
                f"({len(speaker_index)}) so every batch can use distinct speakers."
            )
        self.batch_size = batch_size
        self.num_batches = num_batches

        self.spk_to_indices: dict[str, list[int]] = {}
        idx = 0
        for spk, paths in speaker_index.items():
            self.spk_to_indices[spk] = list(range(idx, idx + len(paths)))
            idx += len(paths)

    def __iter__(self):
        speakers = list(self.spk_to_indices.keys())
        for _ in range(self.num_batches):
            batch_speakers = random.sample(speakers, self.batch_size)
            yield [random.choice(self.spk_to_indices[spk]) for spk in batch_speakers]

    def __len__(self):
        return self.num_batches


def _worker_init_fn(worker_id):
    """DataLoader workers are forked processes that inherit an identical copy
    of Python's stdlib `random` state; PyTorch only auto-reseeds its own
    torch RNG per worker, not stdlib `random`, which the collate functions
    below rely on for their crop offset/length. Without this, crop windows
    can correlate across workers instead of being independently random."""
    random.seed(torch.initial_seed() % 2**32)


def make_enrollment_collate(
    fbank,
    sample_rate: int = 16000,
    enroll_len: float = 3.0,
    crop_len_range: tuple[float, float] | None = None,
    noise_prob: float = 0.0,
    noise_dir: str | None = None,
    snr_range: tuple[float, float] = (0.0, 15.0),
    snr_loudness: bool = False,
    speech_lufs_range: tuple[float, float] = (-33.0, -25.0),
    noise_lufs_range: tuple[float, float] = (-38.0, -30.0),
    variable_length: bool = False,
    max_len: float | None = None,
):
    """Builds a collate_fn for EnrollmentCentroidDataset. Returns
    (fbank, lengths, spk_ids).

    Fixed mode (default): one crop length is drawn per batch (fixed enroll_len,
    or crop_len_range), applied to every item, so the batch is uniform-length
    and lengths is None (the model's original, mask-free pooling path runs).

    Variable mode (variable_length=True, i.e. native full-length training):
    each utterance keeps its OWN length -- no per-batch crop -- optionally
    capped at max_len seconds; per-item fbanks are zero-padded to the batch's
    longest and `lengths` (valid fbank-frame count per item) is returned so the
    model builds a padding mask and pools over valid frames only
    (_masked_statistics_pooling). Matches full-length enrollment at test."""
    noise_files = list_noise_files(noise_dir) if noise_prob > 0.0 else None
    max_samples = int(round(max_len * sample_rate)) if max_len else None

    def collate_fn_enrollment(samples):
        if variable_length:
            feats = []
            for s in samples:
                wav = _load_mono(s["path"], sample_rate)
                if max_samples is not None and wav.shape[-1] > max_samples:
                    wav = _crop_or_pad(wav, max_samples)
                n = wav.shape[-1]
                wav = _maybe_add_noise(
                    wav, n, noise_files, noise_prob, snr_range, sample_rate, snr_loudness,
                    speech_lufs_range, noise_lufs_range,
                )
                feats.append(fbank(wav))
            lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
            t_max = int(lengths.max())
            padded = feats[0].new_zeros((len(feats), t_max, feats[0].shape[1]))
            for i, f in enumerate(feats):
                padded[i, : f.shape[0]] = f
            return padded, lengths, [s["spk"] for s in samples]

        num_samples = _sample_num_samples(enroll_len, crop_len_range, sample_rate)
        feats = []
        for s in samples:
            wav = _crop_or_pad(_load_mono(s["path"], sample_rate), num_samples)
            wav = _maybe_add_noise(
                wav, num_samples, noise_files, noise_prob, snr_range, sample_rate, snr_loudness,
                speech_lufs_range, noise_lufs_range,
            )
            feats.append(fbank(wav))
        return torch.stack(feats), None, [s["spk"] for s in samples]

    return collate_fn_enrollment


def get_enrollment_dataloaders(
    batch_size,
    num_workers,
    librispeech_root,
    enroll_len: float | None = 3.0,
    sample_rate: int = 16000,
    train_subset="train-100",
    valid_subset: str = "dev",
    batches_per_epoch: int | None = None,
    val_batch_size: int | None = None,
    crop_len_range: tuple[float, float] | None = None,
    noise_prob: float = 0.0,
    noise_dir: str | None = None,
    snr_range: tuple[float, float] = (0.0, 15.0),
    snr_loudness: bool = False,
    speech_lufs_range: tuple[float, float] = (-33.0, -25.0),
    noise_lufs_range: tuple[float, float] = (-38.0, -30.0),
    max_enroll_len: float | None = None,
    fbank_dim: int = 80,
):
    """
    val_batch_size defaults to `batch_size` if not given. Kept separate
    because UniqueSpeakerBatchSampler requires batch_size <= number of
    distinct speakers, and dev-clean only has 40 -- far fewer than a
    paper-scale training batch_size like 128.

    enroll_len=None selects NATIVE full-length ("variable") training: each
    utterance keeps its own length (optionally capped at max_enroll_len
    seconds), padded per batch with a lengths tensor + padding mask -- both
    train AND val use it, so the metric reflects the same full-length regime
    the model is evaluated under. crop_len_range is ignored in that mode.

    crop_len_range/noise_prob/noise_dir/snr_range/snr_loudness/
    speech_lufs_range/noise_lufs_range apply to the TRAIN split only -- the
    val split stays clean (and, in fixed mode, fixed-length enroll_len) for a
    stable, comparable metric across epochs.
    """
    val_batch_size = val_batch_size or batch_size
    variable_length = enroll_len is None
    fbank = FBank(fbank_dim, sample_rate=sample_rate, mean_nor=True)

    train_subsets = [train_subset] if isinstance(train_subset, str) else list(train_subset)
    train_speaker_index = build_speaker_index(librispeech_root, train_subsets)
    train_set = EnrollmentCentroidDataset(speaker_index=train_speaker_index)
    if batches_per_epoch is None:
        batches_per_epoch = len(train_set) // batch_size
    train_loader = DataLoader(
        train_set,
        batch_sampler=UniqueSpeakerBatchSampler(
            train_speaker_index, batch_size, batches_per_epoch,
        ),
        num_workers=num_workers,
        collate_fn=make_enrollment_collate(
            fbank, sample_rate, enroll_len, crop_len_range, noise_prob, noise_dir, snr_range, snr_loudness,
            speech_lufs_range=speech_lufs_range, noise_lufs_range=noise_lufs_range,
            variable_length=variable_length, max_len=max_enroll_len,
        ),
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
    )

    val_subsets = [valid_subset] if isinstance(valid_subset, str) else list(valid_subset)
    val_speaker_index = build_speaker_index(librispeech_root, val_subsets)
    val_set = EnrollmentCentroidDataset(speaker_index=val_speaker_index)
    val_batches_per_epoch = len(val_set) // val_batch_size
    val_loader = DataLoader(
        val_set,
        batch_sampler=UniqueSpeakerBatchSampler(
            val_speaker_index, val_batch_size, val_batches_per_epoch,
        ),
        num_workers=num_workers,
        collate_fn=make_enrollment_collate(
            fbank, sample_rate, enroll_len,
            variable_length=variable_length, max_len=max_enroll_len,
        ),
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
    )

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Real Libri2Mix mixture branch (superposition target): loads the actual
# pre-mixed Libri2Mix mix_clean/*.wav files (one file == one example) and
# parses the speaker pair from each filename. The training target C1 + C2 is
# built downstream (see train.py: gather_mixture_reference).
# ─────────────────────────────────────────────────────────────────────────────
class RealMixtureDataset(Dataset):
    """One real Libri2Mix mixture .wav per item. Scans <mix_dir>/<mix_mode>/ for
    every mixture file and returns its audio path plus the order-independent
    speaker pair_key parsed from the filename (see parse_pair_from_filename /
    pair_key). Cropping + fbank happen batch-side in make_real_mixture_collate."""

    def __init__(self, mix_dirs, mix_mode="mix_clean"):
        if isinstance(mix_dirs, str):
            mix_dirs = [mix_dirs]
        self.items = []  # list[(audio_path, pair_key)]
        for mix_dir in mix_dirs:
            audio_dir = os.path.join(mix_dir, mix_mode)
            for f in sorted(os.listdir(audio_dir)):
                if f.endswith(".wav"):
                    spk_a, spk_b = parse_pair_from_filename(f)
                    self.items.append((os.path.join(audio_dir, f), pair_key(spk_a, spk_b)))
        if not self.items:
            raise ValueError(f"No .wav mixtures found under {mix_dirs} (mode={mix_mode!r})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, pk = self.items[idx]
        return {"path": path, "pair_key": pk}


def make_real_mixture_collate(fbank, sample_rate: int = 16000, mix_len: float = 3.0):
    """collate_fn for RealMixtureDataset. Each mixture is cropped/padded to a
    fixed mix_len (random offset via _crop_or_pad), so the batch is uniform-
    length and lengths is None (mask-free pooling, matching the enrollment
    fixed-length path). Returns (fbank, None, pair_keys)."""
    num_samples = int(round(mix_len * sample_rate))

    def collate_fn_real_mixture(samples):
        feats = [fbank(_crop_or_pad(_load_mono(s["path"], sample_rate), num_samples)) for s in samples]
        return torch.stack(feats), None, [s["pair_key"] for s in samples]

    return collate_fn_real_mixture


def get_real_mixture_dataloaders(
    batch_size,
    num_workers,
    train_mix_dirs,
    dev_mix_dirs,
    mix_mode: str = "mix_clean",
    mix_len: float = 3.0,
    sample_rate: int = 16000,
    val_batch_size: int | None = None,
    fbank_dim: int = 80,
):
    """Train/val loaders over REAL Libri2Mix mixtures. One mixture file == one
    example; a plain shuffled DataLoader is used (the mixture branch trains on
    MSE against the C1+C2 sum, so it needs no UniqueSpeakerBatchSampler / in-
    batch-AP class structure). val uses the dev mixtures, unshuffled."""
    val_batch_size = val_batch_size or batch_size
    fbank = FBank(fbank_dim, sample_rate=sample_rate, mean_nor=True)

    train_set = RealMixtureDataset(train_mix_dirs, mix_mode)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=make_real_mixture_collate(fbank, sample_rate, mix_len),
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
    )

    val_set = RealMixtureDataset(dev_mix_dirs, mix_mode)
    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=make_real_mixture_collate(fbank, sample_rate, mix_len),
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
    )

    return train_loader, val_loader
