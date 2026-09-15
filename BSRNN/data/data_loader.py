"""LibriMix data loader that returns enrollment waveforms for on-the-fly Cam++ extraction."""

import os
import random
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

torch.manual_seed(42)
np.random.seed(42)

_SUBSET_MAP = {
    "train-100": "train-clean-100",
    "train-360": "train-clean-360",
    "dev": "dev-clean",
    "test": "test-clean",
}


def build_enroll_index(librispeech_root, subsets):
    """Scan one or more LibriSpeech directories and return {spk_id: [audio_file_paths]}.

    `subsets` may be a single Libri2Mix-style subset name (e.g. "train-100")
    or a list of them (e.g. ["train-100", "train-360"]) -- speakers found in
    more than one subset get their file lists merged.
    """
    if isinstance(subsets, str):
        subsets = [subsets]
    spk_dict: dict[str, list[str]] = {}
    for subset in subsets:
        ls_subset = _SUBSET_MAP[subset]
        subset_dir = os.path.join(librispeech_root, ls_subset)
        n_before = len(spk_dict)
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
        print(f"Built enrollment index: scanned {subset_dir} "
              f"(+{len(spk_dict) - n_before} new speakers)")
    print(f"Built enrollment index: {len(spk_dict)} total speakers from {subsets}")
    return spk_dict


class LibriMixCamppDataset(Dataset):
    """
    Returns (mixture, s1_enroll, s2_enroll, s1_target, s2_target) per item.
    collate_fn_campp duplicates each mixture for both speakers.
    """

    def __init__(
        self,
        mix_dirs,
        enroll_spk_dict,
        max_audio=4,
        subset="train",
        sample_rate=16000,
        enroll_len=3.0,
        mode="mix_clean",
    ):
        assert subset in ["train", "dev", "test"]
        if isinstance(mix_dirs, str):
            mix_dirs = [mix_dirs]
        self.enroll_spk_dict = enroll_spk_dict
        self.subset = subset
        self.sr = sample_rate
        self.mode = mode
        self.seg_samples = int(max_audio * sample_rate)
        # enroll_len=None -> no crop/pad, use each enrollment utterance's own
        # full length ("long enrollment").
        self.enroll_samples = int(enroll_len * sample_rate) if enroll_len is not None else None

        # (mix_path, label_dir) pairs -- label_dir is tracked per-item so s1/s2
        # lookups use the same subset root the mixture came from, letting
        # multiple Libri2Mix subsets (e.g. train-100 + train-360) be combined.
        self.items = []
        for mix_dir in mix_dirs:
            audio_dir = os.path.join(mix_dir, mode)
            paths = sorted(
                os.path.join(audio_dir, f)
                for f in os.listdir(audio_dir)
                if f.endswith(".wav")
            )
            assert len(paths) > 0, f"No wavs found in {audio_dir}"
            print(f"[{subset}] {len(paths)} mixtures from {audio_dir}")
            self.items.extend((p, mix_dir) for p in paths)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        mix_path, label_dir = self.items[idx]
        aud_name = os.path.basename(mix_path)

        mix_wav = self._load(mix_path)
        s1_wav = self._load(os.path.join(label_dir, "s1", aud_name))
        s2_wav = self._load(os.path.join(label_dir, "s2", aud_name))
        # Only mix_both has a noise/ subdir alongside s1/s2 -- mix_clean's
        # "mixture" is exactly s1+s2, no separate noise component to load.
        # Used to build a noisy-oracle negative clue (target_speaker's
        # in-mixture reference + this exact noise clip); zeros elsewhere so
        # the dict shape (and collate_fn_campp's unconditional stacking) stay
        # uniform regardless of mode.
        has_noise = self.mode == "mix_both"
        noise_wav = self._load(os.path.join(label_dir, "noise", aud_name)) if has_noise else torch.zeros_like(mix_wav)

        # Aligned crop/pad for mixture, both targets, and the noise clip
        if self.subset != "test":
            n = self.seg_samples
            length = mix_wav.shape[-1]
            if length < n:
                left = (n - length) // 2
                right = n - length - left
                mix_wav = torch.nn.functional.pad(mix_wav, (left, right))
                s1_wav = torch.nn.functional.pad(s1_wav, (left, right))
                s2_wav = torch.nn.functional.pad(s2_wav, (left, right))
                noise_wav = torch.nn.functional.pad(noise_wav, (left, right))
            else:
                start = torch.randint(0, length - n + 1, (1,)).item()
                mix_wav = mix_wav[start:start + n]
                s1_wav = s1_wav[start:start + n]
                s2_wav = s2_wav[start:start + n]
                noise_wav = noise_wav[start:start + n]

        # Parse speaker IDs from filename like "1089-134686-0000_1221-135766-0001.wav"
        stem = aud_name[:-4]  # strip .wav
        s1_name, s2_name = stem.split("_")
        s1_spk = s1_name.split("-")[0]
        s2_spk = s2_name.split("-")[0]

        s1_enroll = self._load_enroll(s1_spk, s1_name)
        s2_enroll = self._load_enroll(s2_spk, s2_name)

        return {
            "mixture": mix_wav,
            "s1_enroll": s1_enroll,
            "s2_enroll": s2_enroll,
            "s1_target": s1_wav,
            "s2_target": s2_wav,
            "s1_spk": s1_spk,
            "s2_spk": s2_spk,
            "noise": noise_wav,
        }

    def _load(self, path):
        wav, sr = torchaudio.load(path)
        wav = wav.mean(0)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        return wav

    def _load_enroll(self, spk_id, exclude_stem):
        files = self.enroll_spk_dict.get(spk_id, [])
        assert files, f"No enrollment files for speaker {spk_id}"

        # Try to pick an utterance different from the mixture utterance
        path = random.choice(files)
        for _ in range(10):
            if os.path.splitext(os.path.basename(path))[0] != exclude_stem:
                break
            path = random.choice(files)

        wav, sr = torchaudio.load(path)
        wav = wav.mean(0)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)

        if self.enroll_samples is None:
            return wav

        n = self.enroll_samples
        if wav.shape[-1] < n:
            wav = torch.nn.functional.pad(wav, (0, n - wav.shape[-1]))
        else:
            start = random.randint(0, wav.shape[-1] - n)
            wav = wav[start:start + n]
        return wav


def collate_fn_campp(samples):
    """Duplicates each sample for s1 and s2. Returns mix_wavs (B*2, L);
    enroll_wavs, a list (variable-length under enroll_len=None); interference_
    enroll_wavs, the other speaker's separate enrollment utterance (clue_mode
    "v2"'s negative half); interference_target_wavs (B*2, L), the other
    speaker's own clip actually summed into this mixture; target_wavs (B*2,
    L); interference_spk_ids/target_spk_ids, for centroid-cache lookups; and
    noise_wavs (B*2, L), all-zero under mix_clean."""
    mix_wavs, enroll_wavs, interference_enroll_wavs, interference_target_wavs, target_wavs = [], [], [], [], []
    interference_spk_ids, target_spk_ids, noise_wavs = [], [], []
    for s in samples:
        for speaker, interference in (("s1", "s2"), ("s2", "s1")):
            mix_wavs.append(s["mixture"])
            enroll_wavs.append(s[f"{speaker}_enroll"])
            interference_enroll_wavs.append(s[f"{interference}_enroll"])
            interference_target_wavs.append(s[f"{interference}_target"])
            target_wavs.append(s[f"{speaker}_target"])
            interference_spk_ids.append(s[f"{interference}_spk"])
            target_spk_ids.append(s[f"{speaker}_spk"])
            noise_wavs.append(s.get("noise", torch.zeros_like(s["mixture"])))
    return (
        torch.stack(mix_wavs),
        enroll_wavs,
        interference_enroll_wavs,
        torch.stack(interference_target_wavs),
        torch.stack(target_wavs),
        interference_spk_ids,
        target_spk_ids,
        torch.stack(noise_wavs),
    )


def list_noise_files(noise_dir):
    """Flat list of noise clip paths under a directory (e.g. Libri2Mix's
    wav16k/min/<subset>/noise/ -- WHAM! clips, used here as a generic noise
    pool for DSM's on-the-fly mixing, unpaired from any specific mixture)."""
    files = [
        os.path.join(noise_dir, f)
        for f in os.listdir(noise_dir)
        if f.endswith(".wav") or f.endswith(".flac")
    ]
    if not files:
        raise ValueError(f"No noise clips found under {noise_dir}")
    return files


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


def _count_mix_files(data_root, subset, mode):
    """Number of pre-mixed Libri2Mix files under <data_root>/<subset>/<mode>
    -- used only as DSMLibriMixDataset's default steps_per_epoch, so a DSM
    run's per-epoch step budget matches the equivalent static run's exactly
    (same #steps/epoch -> comparable LR-schedule/early-stopping timing),
    even though DSM itself never reads these files' audio."""
    audio_dir = os.path.join(data_root, subset, mode)
    return len([f for f in os.listdir(audio_dir) if f.endswith(".wav")])


class DSMLibriMixDataset(Dataset):
    """
    Dynamic Speaker Mixing (WeSep, arXiv:2409.15799, Algorithm 1): instead of
    reading LibriMixCamppDataset's fixed, pre-mixed Libri2Mix files, each
    item synthesizes a FRESH (target, interference) pair -- two distinct
    LibriSpeech speakers, one random utterance each -- mixed on the fly at a
    random SIR (plus, if noise_files is given, a random-SNR WHAM! clip on
    top). This removes the fixed-file cap on (speaker-pair, SNR) diversity a
    static Libri2Mix subset has: every epoch sees new combinations instead of
    cycling the same ~14k pre-baked mixtures.

    __getitem__ ignores idx entirely (every call redraws fresh) -- __len__ is
    just an epoch-size knob (see _count_mix_files), not a real dataset size.
    Returns the same dict shape as LibriMixCamppDataset (minus "noise"), so
    collate_fn_campp is reused unchanged.
    """

    def __init__(
        self,
        speaker_index,
        max_audio=4.0,
        sample_rate=16000,
        enroll_len=None,
        sir_range=(-5.0, 5.0),
        noise_files=None,
        noise_prob=0.0,
        snr_range=(0.0, 15.0),
        steps_per_epoch=1000,
    ):
        self.speaker_index = speaker_index
        self.speakers = sorted(speaker_index.keys())
        if len(self.speakers) < 2:
            raise ValueError("DSM needs at least 2 speakers to draw a pair from.")
        self.sr = sample_rate
        self.seg_samples = int(max_audio * sample_rate)
        self.enroll_samples = int(enroll_len * sample_rate) if enroll_len is not None else None
        self.sir_range = sir_range
        self.noise_files = noise_files
        self.noise_prob = noise_prob
        self.snr_range = snr_range
        self.steps_per_epoch = steps_per_epoch

    def __len__(self):
        return self.steps_per_epoch

    def _load(self, path):
        wav, sr = torchaudio.load(path)
        wav = wav.mean(0)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        return wav

    def _pick_target(self, spk):
        path = random.choice(self.speaker_index[spk])
        wav = _crop_or_pad(self._load(path), self.seg_samples)
        return wav, path

    def _pick_enroll(self, spk, exclude_path):
        """Same "try to avoid the exact target utterance" pattern as
        LibriMixCamppDataset._load_enroll, applied to the target utterance
        just drawn for this item (not the mixture file, since DSM has none)."""
        files = self.speaker_index[spk]
        path = random.choice(files)
        for _ in range(10):
            if path != exclude_path:
                break
            path = random.choice(files)
        wav = self._load(path)
        if self.enroll_samples is None:
            return wav
        return _crop_or_pad(wav, self.enroll_samples)

    def __getitem__(self, idx):
        spk_a, spk_b = random.sample(self.speakers, 2)
        s1_target, s1_path = self._pick_target(spk_a)
        s2_target, s2_path = self._pick_target(spk_b)

        sir = random.uniform(*self.sir_range)
        mixture = _mix_with_sir(s1_target, s2_target, sir)
        if self.noise_files and self.noise_prob > 0.0 and random.random() < self.noise_prob:
            noise_wav = _crop_or_pad(self._load(random.choice(self.noise_files)), self.seg_samples)
            mixture = _mix_with_sir(mixture, noise_wav, random.uniform(*self.snr_range))

        return {
            "mixture": mixture,
            "s1_enroll": self._pick_enroll(spk_a, s1_path),
            "s2_enroll": self._pick_enroll(spk_b, s2_path),
            "s1_target": s1_target,
            "s2_target": s2_target,
            "s1_spk": spk_a,
            "s2_spk": spk_b,
        }


def get_dataloader_campp_dsm(
    batch_size,
    num_workers,
    data_root,
    librispeech_root,
    audio_length=4,
    sample_rate=16000,
    mode="mix_both",
    train_subset="train-100",
    valid_subset="dev",
    enroll_len=None,
    distributed=False,
    sir_range=(-5.0, 5.0),
    noise_dir=None,
    noise_prob=0.0,
    snr_range=(0.0, 15.0),
    steps_per_epoch=None,
    **kwargs,
):
    """The TRAIN loader draws fresh on-the-fly mixtures (DSMLibriMixDataset,
    2 random LibriSpeech speakers + a random SIR every step) instead of
    reading pre-mixed Libri2Mix files. The VAL loader is the static Libri2Mix
    dev set (LibriMixCamppDataset), unaffected by DSM -- so SI-SDRi stays
    comparable across runs on that same fixed dev set.
    """
    train_subsets = [train_subset] if isinstance(train_subset, str) else list(train_subset)
    train_speaker_index = build_enroll_index(librispeech_root, train_subsets)

    if steps_per_epoch is None:
        steps_per_epoch = sum(_count_mix_files(data_root, s, mode) for s in train_subsets)

    noise_files = list_noise_files(noise_dir) if (noise_dir and noise_prob > 0.0) else None

    train_set = DSMLibriMixDataset(
        speaker_index=train_speaker_index,
        max_audio=audio_length,
        sample_rate=sample_rate,
        enroll_len=enroll_len,
        sir_range=tuple(sir_range),
        noise_files=noise_files,
        noise_prob=noise_prob,
        snr_range=tuple(snr_range),
        steps_per_epoch=steps_per_epoch,
    )

    train_sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        drop_last=True,
        sampler=train_sampler,
        collate_fn=collate_fn_campp,
        pin_memory=True,
    )

    # Val stays the static Libri2Mix dev set (LibriMixCamppDataset), untouched by DSM.
    val_mix_dir = os.path.join(data_root, valid_subset)
    val_enroll = build_enroll_index(librispeech_root, "dev")
    val_set = LibriMixCamppDataset(
        mix_dirs=val_mix_dir,
        enroll_spk_dict=val_enroll,
        max_audio=audio_length,
        subset="dev",
        sample_rate=sample_rate,
        enroll_len=enroll_len,
        mode=mode,
    )
    val_sampler = DistributedSampler(val_set, shuffle=False) if distributed else None
    val_loader = DataLoader(
        val_set,
        batch_size=max(batch_size, 8),
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        sampler=val_sampler,
        collate_fn=collate_fn_campp,
        pin_memory=True,
    )

    return train_loader, val_loader
