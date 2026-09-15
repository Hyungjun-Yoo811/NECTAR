"""
Evaluates a Baseline checkpoint (Table 1): positive clue only, pos =
norm(raw(target_enroll)), no negative clue. --mix_window/--enroll_window
default to "full" (each utterance's own natural length); pass a number of
seconds for a fixed crop/pad instead. Usage: `cd BSRNN && python
eval_baseline.py [--mix_window 3.0] [--enroll_window 3.0]`.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_CSM_DIR = os.path.join(_REPO_ROOT, "csm")
if _CSM_DIR not in sys.path:
    sys.path.insert(0, _CSM_DIR)
sys.path.insert(0, _THIS_DIR)

from models.bsrnn.bsrnn import BSRNN
from train import build_verification_model, extract_embeddings
from eval_common import (
    DEFAULT_METRICS, build_test_loader, compute_metrics, metrics_type, mix_window_type, save_audio_sample, save_case,
)

DEFAULT_CFG = "configs/baseline_noisy.yml"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfg", default=DEFAULT_CFG, help="Path to the training config used for this checkpoint.")
    p.add_argument("--checkpoint", default=None, help="Override <exp_dir>/best.ckpt")
    p.add_argument("--mix_window", type=mix_window_type, default="full",
                   help="Mixture/target SI-SDR eval window in seconds, or 'full' for no crop.")
    p.add_argument("--enroll_window", type=mix_window_type, default="full",
                   help="Enrollment crop in seconds, or 'full' for each utterance's own length.")
    p.add_argument("--num_audio_samples", type=int, default=5,
                   help="Save this many (mix, target, est) wav triples for listening; 0 disables.")
    p.add_argument("--metrics", type=metrics_type, default=DEFAULT_METRICS,
                   help=f"Comma-separated subset of {{'pesq','estoi'}} to also compute (default: {DEFAULT_METRICS}).")
    args = p.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    if cfg["model"].get("clue_mode") is not None:
        raise ValueError(f"{args.cfg}: model.clue_mode={cfg['model']['clue_mode']!r} -- "
                          f"eval_baseline.py is for the clue_mode-absent (positive-only) baseline; "
                          f"use eval_nectar.py for negative-clue checkpoints.")

    exp_dir = cfg["training"]["exp_dir"]
    checkpoint = args.checkpoint or os.path.join(exp_dir, "best.ckpt")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint} -- train {args.cfg} first.")

    gpu_id = cfg.get("eval", {}).get("gpu_id", 0)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    campplus, fbank = build_verification_model(cfg, device)

    model_cfg = dict(cfg["model"])
    model_cfg.pop("clue_mode", None)
    model_cfg.pop("enroll_source", None)
    model_cfg.pop("normalize_emb", None)
    model = BSRNN(**model_cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    epoch_info = f"epoch {ckpt['epoch']}" if isinstance(ckpt, dict) and "epoch" in ckpt else "n/a"
    print(f"Config     : {args.cfg}")
    print(f"Checkpoint : {checkpoint} ({epoch_info})")
    print(f"Device     : {device}")
    mix_window_desc = "full-length (no crop)" if args.mix_window is None else f"{args.mix_window}s"
    enroll_window_desc = "full-length (no crop)" if args.enroll_window is None else f"{args.enroll_window}s"
    print(f"Mix window : {mix_window_desc}")
    print(f"Enroll win.: {enroll_window_desc}")

    sr = cfg["data"].get("sample_rate", 16000)
    test_loader = build_test_loader(
        cfg, args.mix_window, cfg["data"].get("num_workers", 4), enroll_window=args.enroll_window)
    print(f"Test mixtures: {len(test_loader)}")

    audio_dir = os.path.join(exp_dir, "eval_baseline", "audio_samples")
    audio_saved = 0
    if args.num_audio_samples > 0:
        os.makedirs(audio_dir, exist_ok=True)

    rows_all = []
    run_sum = {}
    run_cnt = {}

    with torch.no_grad():
        for (mix_wavs, enroll_wavs, interference_enroll_wavs, interference_target_wavs,
             target_wavs, keys, interference_spk_ids, target_spk_ids) in tqdm(test_loader, desc="baseline"):
            mix_wavs = mix_wavs.to(device)
            target_wavs = target_wavs.to(device)

            pos_embs_raw = extract_embeddings(enroll_wavs, campplus, fbank, device)
            spk_embs = F.normalize(pos_embs_raw, dim=-1)

            est_wavs = model(mix_wavs, spk_embs)
            rows = compute_metrics(est_wavs, target_wavs, mix_wavs, sr, metrics=args.metrics)

            for b, key in enumerate(keys):
                rows[b]["key"] = key
                parts = []
                for m, v in rows[b].items():
                    if m == "key":
                        continue
                    if v == v:  # not NaN
                        run_sum[m] = run_sum.get(m, 0.0) + v
                        run_cnt[m] = run_cnt.get(m, 0) + 1
                    avg = run_sum[m] / run_cnt[m] if run_cnt.get(m) else float("nan")
                    parts.append(f"{m}={v:.2f}(avg={avg:.2f})")
                tqdm.write(f"{key}  " + " ".join(parts))
                if audio_saved < args.num_audio_samples:
                    save_audio_sample(os.path.join(audio_dir, f"{key}_mix.wav"), mix_wavs[b], sr)
                    save_audio_sample(os.path.join(audio_dir, f"{key}_target.wav"), target_wavs[b], sr)
                    save_audio_sample(os.path.join(audio_dir, f"{key}_est.wav"), est_wavs[b], sr)
                    audio_saved += 1

            rows_all.extend(rows)

    if audio_saved:
        print(f"\nSaved {audio_saved} audio sample(s) to {audio_dir}")

    print("\n── Results ──────────────────────────────────")
    save_case(
        rows_all, os.path.join(exp_dir, "eval_baseline"), checkpoint,
        extra={"cfg": args.cfg, "mix_window": args.mix_window, "enroll_window": args.enroll_window},
    )


if __name__ == "__main__":
    main()
