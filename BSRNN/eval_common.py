"""
Shared test-set/metrics plumbing for eval_baseline.py / eval_nectar.py /
eval_mixture_clue.py. `build_test_loader`'s `mix_window` (mixture/target
SI-SDR window) and `enroll_window` (enrollment crop) are independent: each
defaults to None ("full" length, no crop/pad) unless a caller passes a fixed
number of seconds.
"""

import csv
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import yaml

from data.data_loader import LibriMixCamppDataset, build_enroll_index
from losses import singlesrc_neg_sisdr

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

DEFAULT_MIX_WINDOW = "full"


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


class EvalDataset(LibriMixCamppDataset):
    """Test-split dataset: mixture/s1/s2/noise cropped or center-padded to
    `mix_window` seconds (None = natural length), enrollment likewise to
    `enroll_window`. Also returns a `key` (mixture stem) for logging."""

    def __init__(self, mix_dirs, enroll_spk_dict, sample_rate, mode, mix_window, enroll_window=None):
        super().__init__(
            mix_dirs=mix_dirs, enroll_spk_dict=enroll_spk_dict, subset="test",
            sample_rate=sample_rate, enroll_len=enroll_window, mode=mode,
        )
        self.mix_window = mix_window
        self.mix_samples = None if mix_window is None else int(round(mix_window * sample_rate))

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        n = self.mix_samples
        if n is not None:
            length = sample["mixture"].shape[-1]
            if length < n:
                left = (n - length) // 2
                right = n - length - left
                crop = lambda w: F.pad(w, (left, right))
            else:
                start = random.randint(0, length - n)
                crop = lambda w: w[start:start + n]
            for key in ("mixture", "s1_target", "s2_target", "noise"):
                sample[key] = crop(sample[key])
        mix_path, _ = self.items[idx]
        sample["key"] = os.path.splitext(os.path.basename(mix_path))[0]
        return sample


def collate_fn_eval(samples):
    """Duplicates each sample for s1/s2 (target/interference swapped). Carries
    both interference_target_wavs (the other speaker's own clip from this
    mixture, used by oracle-utterance eval cases) and interference_enroll_wavs
    (a separate enrollment utterance of the other speaker, clue_mode "v2")."""
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


def build_test_loader(cfg, mix_window, num_workers, enroll_window=None):
    dc = cfg["data"]
    test_mix_dir = os.path.join(dc["data_root"], "test")
    test_enroll = build_enroll_index(dc["librispeech_root"], "test")
    test_set = EvalDataset(
        mix_dirs=test_mix_dir, enroll_spk_dict=test_enroll,
        sample_rate=dc.get("sample_rate", 16000), mode=dc.get("mode", "mix_clean"),
        mix_window=mix_window, enroll_window=enroll_window,
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
