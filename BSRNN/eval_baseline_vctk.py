"""
Out-of-domain (VCTK-2Mix, Table 3) sibling of eval_baseline.py: same
checkpoint and positive-only condition, but the mixture/targets/enrollment
come from VCTK-2Mix instead of Libri2Mix's test split. Enrollment is always
full-length; --mix_window defaults to "full" (each mixture's own length).
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
from eval_common_vctk import (
    DEFAULT_METRICS,
    DEFAULT_VCTK_MIX_DIR,
    DEFAULT_VCTK_ROOT,
    build_test_loader,
    compute_metrics,
    metrics_type,
    mix_window_type,
    save_audio_sample,
    save_case,
)

DEFAULT_CFG = "configs/baseline.yml"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfg", default=DEFAULT_CFG, help="Path to the training config used for this checkpoint.")
    p.add_argument("--checkpoint", default=None, help="Override <exp_dir>/best.ckpt")
    p.add_argument("--vctk_mix_dir", default=DEFAULT_VCTK_MIX_DIR,
                   help="VCTK-2Mix output dir (contains mix_clean/, s1/, s2/).")
    p.add_argument("--vctk_root", default=DEFAULT_VCTK_ROOT,
                   help="VCTK-Corpus root, for enrollment utterances (drawn from wav48/).")
    p.add_argument("--mix_window", type=mix_window_type, default="full",
                   help="Mixture/target SI-SDR eval window in seconds, or 'full' for no crop.")
    p.add_argument("--limit", type=int, default=None, help="Cap number of VCTK-2Mix mixtures, for a quick check.")
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
    print(f"Mix window : {mix_window_desc} (enrollment: full-length)")
    print(f"VCTK mixes : {args.vctk_mix_dir}")

    sr = cfg["data"].get("sample_rate", 16000)
    test_loader = build_test_loader(
        cfg, args.mix_window, cfg["data"].get("num_workers", 4),
        vctk_mix_dir=args.vctk_mix_dir, vctk_root=args.vctk_root, limit=args.limit,
    )
    print(f"Test mixtures: {len(test_loader)}")

    audio_dir = os.path.join(exp_dir, "eval_baseline_vctk", "audio_samples")
    audio_saved = 0
    if args.num_audio_samples > 0:
        os.makedirs(audio_dir, exist_ok=True)

    rows_all = []
    run_sum = {"SI-SDR": 0.0, "SI-SDRi": 0.0}
    run_n = 0

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
                si, sii = rows[b]["SI-SDR"], rows[b]["SI-SDRi"]
                run_sum["SI-SDR"] += si
                run_sum["SI-SDRi"] += sii
                run_n += 1
                tqdm.write(
                    f"{key}  SI-SDR={si:.2f} SI-SDRi={sii:.2f} "
                    f"(avg SI-SDR={run_sum['SI-SDR'] / run_n:.2f} SI-SDRi={run_sum['SI-SDRi'] / run_n:.2f})"
                )
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
        rows_all, os.path.join(exp_dir, "eval_baseline_vctk"), checkpoint,
        extra={"cfg": args.cfg, "mix_window": args.mix_window, "enrollment": "full-length",
               "dataset": "VCTK-2Mix (out-of-domain)"},
    )


if __name__ == "__main__":
    main()
