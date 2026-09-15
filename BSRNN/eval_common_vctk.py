"""
VCTK-2Mix variant of eval_common.py, for the out-of-domain siblings of
eval_baseline.py / eval_nectar.py / eval_mixture_clue.py (Table 3). Only the
dataset/loader differs -- VCTK-2Mix mixture ids are "<spk1>_<utt1>_<spk2>_
<utt2>" (4 fields), incompatible with Libri2Mix's 2-field convention, so
VCTKMixDataset below is a from-scratch equivalent that returns the same
sample dict shape as eval_common.py's EvalDataset.
"""

import csv
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import yaml

from losses import singlesrc_neg_sisdr
from speakerlab.dataset.dataset import DEFAULT_VCTK_ROOT, _crop_or_pad, _load_mono, build_vctk_speaker_index

try:
    from pesq import pesq as pesq_fn
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False

try:
    from pystoi import stoi as stoi_fn
    HAS_STOI = True
except ImportError:
    HAS_STOI = False

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)

DEFAULT_MIX_WINDOW = "full"
# create_VCTKmix_from_metadata.py appends "VCTK{n_src}Mix" onto whatever
# --VCTKmix_outdir it's given, so the generated mixtures land one level
# deeper, under "VCTK2Mix/".
DEFAULT_VCTK_MIX_DIR = os.path.join(_REPO_ROOT, "dataset/data/vctk/VCTK2Mix/wav16k/min/VCTK_md")


def mix_window_type(value):
    """argparse type for --mix_window: 'full' -> None (no crop, each
    mixture's own natural length), anything else -> float seconds (fixed
    crop/pad)."""
    return None if value.lower() == "full" else float(value)


OPTIONAL_METRICS = {"pesq", "estoi"}
DEFAULT_METRICS = "pesq,estoi"


def metrics_type(value):
    """argparse type for --metrics: comma-separated subset of {'pesq',
    'estoi'} to compute (SI-SDR/SI-SDRi are always computed)."""
    metrics = {m.strip() for m in value.split(",") if m.strip()}
    invalid = metrics - OPTIONAL_METRICS
    if invalid:
        raise ValueError(f"unknown metric(s) {sorted(invalid)}, choose from {sorted(OPTIONAL_METRICS)}")
    return metrics


class VCTKMixDataset(torch.utils.data.Dataset):
    """One item per VCTK-2Mix mixture (mix_clean/<stem>.wav, with matching
    s1/<stem>.wav, s2/<stem>.wav clean targets). Returns the same sample dict
    shape as eval_common.py's EvalDataset."""

    def __init__(self, mix_dir, vctk_speaker_index, sample_rate=16000,
                 mix_window=None, enroll_window=None, mode="mix_clean", limit=None):
        self.mix_dir = mix_dir
        self.speaker_index = vctk_speaker_index
        self.sr = sample_rate
        self.mix_samples = None if mix_window is None else int(round(mix_window * sample_rate))
        self.enroll_samples = None if enroll_window is None else int(round(enroll_window * sample_rate))

        audio_dir = os.path.join(mix_dir, mode)
        if not os.path.isdir(audio_dir):
            raise FileNotFoundError(
                f"{audio_dir} not found -- generate VCTK-2Mix first (see "
                f"https://github.com/JorisCos/VCTK-2Mix)."
            )
        self.paths = sorted(
            os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")
        )
        if not self.paths:
            raise ValueError(f"No wavs found in {audio_dir}")
        if limit is not None:
            self.paths = self.paths[:limit]

    def __len__(self):
        return len(self.paths)

    def _load(self, path):
        return _load_mono(path, self.sr)

    def _load_enroll(self, spk_id, exclude_stem):
        files = self.speaker_index.get(spk_id, [])
        if not files:
            raise ValueError(f"No VCTK enrollment files for speaker {spk_id!r}")
        path = random.choice(files)
        for _ in range(10):
            if os.path.splitext(os.path.basename(path))[0] != exclude_stem:
                break
            path = random.choice(files)
        wav = self._load(path)
        return wav if self.enroll_samples is None else _crop_or_pad(wav, self.enroll_samples)

    def __getitem__(self, idx):
        mix_path = self.paths[idx]
        stem = os.path.splitext(os.path.basename(mix_path))[0]
        # VCTK-2Mix mixture ids are "<spk1>_<utt1>_<spk2>_<utt2>" (e.g.
        # "p374_150_p265_081") -- 4 fields, unlike Libri2Mix's 2-field
        # convention (see module docstring).
        s1_spk, s1_utt, s2_spk, s2_utt = stem.split("_")

        mix_wav = self._load(mix_path)
        s1_wav = self._load(os.path.join(self.mix_dir, "s1", f"{stem}.wav"))
        s2_wav = self._load(os.path.join(self.mix_dir, "s2", f"{stem}.wav"))
        noise_wav = torch.zeros_like(mix_wav)  # mix_clean has no noise/ subdir

        n = self.mix_samples
        if n is not None:
            length = mix_wav.shape[-1]
            if length < n:
                left = (n - length) // 2
                right = n - length - left
                crop = lambda w: F.pad(w, (left, right))
            else:
                start = random.randint(0, length - n)
                crop = lambda w: w[start:start + n]
            mix_wav, s1_wav, s2_wav, noise_wav = crop(mix_wav), crop(s1_wav), crop(s2_wav), crop(noise_wav)

        s1_enroll = self._load_enroll(s1_spk, f"{s1_spk}_{s1_utt}")
        s2_enroll = self._load_enroll(s2_spk, f"{s2_spk}_{s2_utt}")

        return {
            "mixture": mix_wav, "s1_target": s1_wav, "s2_target": s2_wav, "noise": noise_wav,
            "s1_enroll": s1_enroll, "s2_enroll": s2_enroll,
            "s1_spk": s1_spk, "s2_spk": s2_spk, "key": stem,
        }


def collate_fn_eval(samples):
    """Duplicates each sample for s1/s2 (target/interference swapped). Carries
    both interference_target_wavs (the other speaker's own clip from this
    mixture) and interference_enroll_wavs (a separate enrollment utterance of
    the other speaker, clue_mode "v2")."""
    mix_wavs, enroll_wavs, interference_enroll_wavs, interference_target_wavs, target_wavs = [], [], [], [], []
    keys, interference_spk_ids, target_spk_ids = [], [], []
    for s in samples:
        for speaker, interference in (("s1", "s2"), ("s2", "s1")):
            mix_wavs.append(s["mixture"])
            enroll_wavs.append(s[f"{speaker}_enroll"])
            interference_enroll_wavs.append(s[f"{interference}_enroll"])
            interference_target_wavs.append(s[f"{interference}_target"])
            target_wavs.append(s[f"{speaker}_target"])
            keys.append(f"{s['key']}_{speaker}")
            interference_spk_ids.append(s[f"{interference}_spk"])
            target_spk_ids.append(s[f"{speaker}_spk"])
    return (
        torch.stack(mix_wavs),
        enroll_wavs,
        interference_enroll_wavs,
        torch.stack(interference_target_wavs),
        torch.stack(target_wavs),
        keys,
        interference_spk_ids,
        target_spk_ids,
    )


def build_test_loader(cfg, mix_window, num_workers, enroll_window=None,
                       vctk_mix_dir=DEFAULT_VCTK_MIX_DIR, vctk_root=DEFAULT_VCTK_ROOT, limit=None):
    """Only cfg["data"]["sample_rate"] is read from the backbone config;
    vctk_mix_dir/vctk_root are VCTK-specific paths, overridable via CLI."""
    dc = cfg["data"]
    speaker_index = build_vctk_speaker_index(vctk_root, wav_subdir="wav48")
    test_set = VCTKMixDataset(
        vctk_mix_dir, speaker_index, sample_rate=dc.get("sample_rate", 16000),
        mix_window=mix_window, enroll_window=enroll_window, limit=limit,
    )
    # batch_size=1: collate_fn_eval expands each item into an (s1, s2) pair.
    return torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_eval,
    )


def compute_metrics(est_wavs, target_wavs, mix_wavs, sr, metrics=OPTIONAL_METRICS):
    sisdr_est = -singlesrc_neg_sisdr(est_wavs, target_wavs)
    sisdr_mix = -singlesrc_neg_sisdr(mix_wavs, target_wavs)
    sisdri = sisdr_est - sisdr_mix
    est_np = est_wavs.detach().cpu().numpy()
    ref_np = target_wavs.detach().cpu().numpy()
    mix_np = mix_wavs.detach().cpu().numpy()

    rows = []
    for b in range(est_wavs.shape[0]):
        row = {"SI-SDR": float(sisdr_est[b]), "SI-SDRi": float(sisdri[b]), "SI-SDR_mix": float(sisdr_mix[b])}
        if HAS_PESQ and "pesq" in metrics:
            try:
                row["PESQ"] = float(pesq_fn(sr, ref_np[b], est_np[b], "wb"))
            except Exception:
                row["PESQ"] = float("nan")
            try:
                row["PESQ_mix"] = float(pesq_fn(sr, ref_np[b], mix_np[b], "wb"))
            except Exception:
                row["PESQ_mix"] = float("nan")
        if HAS_STOI and "estoi" in metrics:
            try:
                row["eSTOI"] = float(stoi_fn(ref_np[b], est_np[b], sr, extended=True))
            except Exception:
                row["eSTOI"] = float("nan")
            try:
                row["eSTOI_mix"] = float(stoi_fn(ref_np[b], mix_np[b], sr, extended=True))
            except Exception:
                row["eSTOI_mix"] = float("nan")
        rows.append(row)
    return rows


def save_audio_sample(path, wav, sr):
    torchaudio.save(path, wav.detach().cpu().unsqueeze(0), sr)


def save_case(rows, output_dir, checkpoint, extra=None):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "results_test.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    numeric_keys = [k for k, v in rows[0].items() if k != "key" and isinstance(v, (int, float))]
    agg = {}
    for m in numeric_keys:
        vals = [r[m] for r in rows if m in r and not (isinstance(r[m], float) and np.isnan(r[m]))]
        if vals:
            agg[m] = float(np.mean(vals))
    sisdri_vals = np.array([r["SI-SDRi"] for r in rows])
    agg["NSR"] = float(np.mean(sisdri_vals < 0))
    agg["Acc"] = float(np.mean(sisdri_vals > 1))

    with open(os.path.join(output_dir, "summary_test.yml"), "w") as f:
        yaml.dump({"checkpoint": checkpoint, **agg, **(extra or {})}, f, default_flow_style=False)
    print(f"[{output_dir}]")
    for k, v in agg.items():
        print(f"  {k:8s}: {v * 100:.2f}%" if k in ("NSR", "Acc") else f"  {k:8s}: {v:.4f}")
    return agg


def print_and_save_comparison(results, comparison_output):
    """results: list of agg dicts (each from save_case, plus a 'case' key) --
    prints + writes a combined table when more than one case ran."""
    if len(results) <= 1:
        return
    print("\n── Comparison ────────────────────────────────")
    cols = ["case", "SI-SDR", "SI-SDRi", "SI-SDR_mix", "PESQ", "PESQ_mix", "eSTOI", "eSTOI_mix", "NSR", "Acc"]
    cols = [c for c in cols if any(c in r for r in results)]
    header = " | ".join(f"{c:>10s}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in results:
        line = []
        for c in cols:
            v = r.get(c)
            if v is None:
                line.append(f"{'':>10s}")
            elif c == "case":
                line.append(f"{v:>10s}")
            elif c in ("NSR", "Acc"):
                line.append(f"{v * 100:9.2f}%")
            else:
                line.append(f"{v:10.4f}")
        print(" | ".join(line))

    os.makedirs(os.path.dirname(os.path.abspath(comparison_output)), exist_ok=True)
    with open(comparison_output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in results:
            writer.writerow({c: r.get(c) for c in cols})
    print(f"\nSaved comparison table to {comparison_output}")
