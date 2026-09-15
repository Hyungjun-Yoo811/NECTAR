"""
Out-of-domain (VCTK-2Mix, Table 3) sibling of eval_nectar.py: same NECTAR
checkpoint (configs/nectar.yml) and negative-clue cases, evaluated over
VCTK-2Mix instead of Libri2Mix. Only "oracle_utterance" and "estimator" are
offered (the other eval_nectar.py cases need LibriSpeech-only oracle centroid
caches with no VCTK counterpart). Usage: `cd BSRNN && python
eval_nectar_vctk.py [--mix_window 3.0] [--enroll_window 3.0] [--cases ...]`.
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
from train import (
    build_verification_model,
    build_spk_embs,
    compute_pseudo_interference_centroid,
    extract_embeddings,
    load_csm,
    run_csm,
)
from eval_common_vctk import (
    DEFAULT_METRICS,
    DEFAULT_VCTK_MIX_DIR,
    DEFAULT_VCTK_ROOT,
    build_test_loader,
    compute_metrics,
    metrics_type,
    mix_window_type,
    print_and_save_comparison,
    save_audio_sample,
    save_case,
)

DEFAULT_CFG = "configs/nectar.yml"
DEFAULT_ESTIMATOR_RUN_DIRS = [os.path.join(_REPO_ROOT, "csm/exp/csm/csm")]
DEFAULT_CKPT_NAME = "CKPT+best"
CASE_GROUPS = [
    "oracle_utterance",
    "estimator",
]


def read_estimator_training_lengths(run_dir):
    """(enroll_len, mix_len) from this CSM checkpoint's own training config,
    used to warn if --enroll_window/--mix_window mismatch what it trained on."""
    with open(os.path.join(run_dir, "config.yaml")) as f:
        ce_cfg = yaml.safe_load(f)
    enroll_len = ce_cfg.get("data", {}).get("enroll_len")
    mix_len = ce_cfg.get("data", {}).get("mix_len")
    return enroll_len, mix_len


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfg", default=DEFAULT_CFG, help="Fixed backbone config (checkpoint + model/data).")
    p.add_argument("--checkpoint", default=None, help="Override <exp_dir>/best.ckpt")
    p.add_argument("--vctk_mix_dir", default=DEFAULT_VCTK_MIX_DIR,
                   help="VCTK-2Mix output dir (contains mix_clean/, s1/, s2/).")
    p.add_argument("--vctk_root", default=DEFAULT_VCTK_ROOT,
                   help="VCTK-Corpus root, for enrollment utterances (drawn from wav48/).")
    p.add_argument("--limit", type=int, default=None, help="Cap number of VCTK-2Mix mixtures, for a quick check.")
    p.add_argument("--mix_window", type=mix_window_type, default="full",
                   help="Mixture/target SI-SDR eval window in seconds, or 'full' for no crop.")
    p.add_argument("--enroll_window", type=mix_window_type, default="full",
                   help="Enrollment crop in seconds, or 'full' for each utterance's own length.")
    p.add_argument("--cases", nargs="+", default=CASE_GROUPS, choices=CASE_GROUPS,
                   help="Which case group(s) to evaluate (default: all).")
    p.add_argument("--estimator_run_dirs", nargs="*", default=DEFAULT_ESTIMATOR_RUN_DIRS,
                   help="One 'estimator' case per trained CSM run dir; pass nothing to skip that case.")
    p.add_argument("--estimator_ckpt_name", default=DEFAULT_CKPT_NAME, help="Checkpoint subdir name to load.")
    p.add_argument("--num_audio_samples", type=int, default=5,
                   help="Save this many (mix, target, est_<case>) wav sets for listening; 0 disables.")
    p.add_argument("--comparison_output", default=None,
                   help="Default: <exp_dir>/eval_nectar_vctk/comparison_3.0.csv")
    p.add_argument("--metrics", type=metrics_type, default=DEFAULT_METRICS,
                   help=f"Comma-separated subset of {{'pesq','estoi'}} to also compute (default: {DEFAULT_METRICS}).")
    args = p.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    exp_dir = cfg["training"]["exp_dir"]
    checkpoint = args.checkpoint or os.path.join(exp_dir, "best.ckpt")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint} -- train {args.cfg} first.")

    gpu_id = cfg.get("eval", {}).get("gpu_id", 0)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    campplus, fbank = build_verification_model(cfg, device)

    model_cfg = dict(cfg["model"])
    backbone_clue_mode = model_cfg.pop("clue_mode", None)
    model_cfg.pop("enroll_source", None)
    model_cfg.pop("normalize_emb", None)
    model = BSRNN(**model_cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    epoch_info = f"epoch {ckpt['epoch']}" if isinstance(ckpt, dict) and "epoch" in ckpt else "n/a"
    print(f"Backbone   : {args.cfg} (trained clue_mode={backbone_clue_mode!r})")
    print(f"Checkpoint : {checkpoint} ({epoch_info})")
    print(f"Device     : {device}")
    mix_window_desc = "full-length (no crop)" if args.mix_window is None else f"{args.mix_window}s"
    enroll_window_desc = "full-length (no crop)" if args.enroll_window is None else f"{args.enroll_window}s"
    print(f"Mix window : {mix_window_desc}")
    print(f"Enroll win.: {enroll_window_desc}")

    # ── Case 1: oracle utterance -- computed directly below, no resources to
    # load. ("oracle_centroid"/"oracle_superposition" from eval_nectar.py
    # are NOT offered here -- they need precomputed LibriSpeech-only oracle
    # centroid caches BSRNN has never used for VCTK.)
    run_oracle_utterance = "oracle_utterance" in args.cases
    if run_oracle_utterance:
        print("Case       : oracle_utterance (norm(raw(interference_target)))")

    # ── Case 2: one per --estimator_run_dirs, formula fixed at
    # C_hat_mix - C_hat_s1 (see read_estimator_training_lengths) -- skipped
    # (no checkpoints loaded at all) unless 'estimator' is selected.
    estimator_cases = []
    if "estimator" in args.cases:
        for run_dir in args.estimator_run_dirs:
            enroll_len, mix_len = read_estimator_training_lengths(run_dir)
            estimator = load_csm(run_dir, campplus, device, args.estimator_ckpt_name)
            name = os.path.basename(os.path.normpath(run_dir))
            print(f"Case       : estimator:{name} (formula: C_hat_mix - C_hat_s1)")
            if enroll_len is not None and enroll_len != args.enroll_window:
                fed = "full-length" if args.enroll_window is None else f"{args.enroll_window}s"
                print(f"  [warn] {run_dir}'s enroll branch was trained with enroll_len={enroll_len}, "
                      f"but --enroll_window feeds it {fed} -- a train/eval length mismatch on this "
                      f"case's negative half specifically. Pass --enroll_window {enroll_len} to match "
                      f"its training convention exactly.")
            if mix_len is not None and mix_len != args.mix_window:
                fed = "full-length" if args.mix_window is None else f"{args.mix_window}s"
                print(f"  [warn] {run_dir}'s mixture branch was trained with mix_len={mix_len}, but "
                      f"--mix_window feeds it {fed} -- a train/eval length mismatch on this case's "
                      f"C_hat_mix specifically. Pass --mix_window {mix_len} to match its training "
                      f"convention exactly.")
            estimator_cases.append({"name": f"estimator_{name}", "estimator": estimator})

    sample_rate = cfg["data"].get("sample_rate", 16000)
    print(f"VCTK mixes : {args.vctk_mix_dir}")
    test_loader = build_test_loader(
        cfg, args.mix_window, cfg["data"].get("num_workers", 4), enroll_window=args.enroll_window,
        vctk_mix_dir=args.vctk_mix_dir, vctk_root=args.vctk_root, limit=args.limit,
    )
    print(f"Test mixtures: {len(test_loader)}")

    case_names = (["oracle_utterance"] if run_oracle_utterance else []) + [c["name"] for c in estimator_cases]
    if not case_names:
        raise ValueError("--cases selected nothing to evaluate.")
    audio_dir = os.path.join(exp_dir, "eval_nectar_vctk", "audio_samples")
    audio_saved = 0
    if args.num_audio_samples > 0:
        os.makedirs(audio_dir, exist_ok=True)

    sr = sample_rate
    all_rows = {name: [] for name in case_names}
    run_sum = {name: {"SI-SDR": 0.0, "SI-SDRi": 0.0} for name in case_names}
    run_n = {name: 0 for name in case_names}

    with torch.no_grad():
        for (mix_wavs, enroll_wavs, interference_enroll_wavs, interference_target_wavs,
             target_wavs, keys, interference_spk_ids, target_spk_ids) in tqdm(test_loader, desc="eval_nectar_vctk"):
            mix_wavs = mix_wavs.to(device)
            target_wavs = target_wavs.to(device)
            interference_target_wavs = interference_target_wavs.to(device)

            # Shared by every case: the target's own raw enrollment embedding
            # (cropped per --enroll_window) -- pos is always norm(pos_embs_raw).
            pos_embs_raw = extract_embeddings(enroll_wavs, campplus, fbank, device)
            pos = F.normalize(pos_embs_raw, dim=-1)

            spk_embs = {}
            if run_oracle_utterance:
                # Interference speaker's own clean source clip actually mixed
                # into this mixture, L2-normalized (same convention as `pos`).
                neg = F.normalize(
                    extract_embeddings(interference_target_wavs, campplus, fbank, device), dim=-1)
                spk_embs["oracle_utterance"] = build_spk_embs(pos, neg)
            for c in estimator_cases:
                # mix_wavs is already mix_window-cropped by EvalDataset --
                # matches these checkpoints' own mix_len: 3.0 training
                # convention, no extra crop needed here.
                ce_pos = run_csm(enroll_wavs, c["estimator"], fbank, device)
                ce_mix = run_csm(mix_wavs, c["estimator"], fbank, device)
                neg = compute_pseudo_interference_centroid(ce_mix, ce_pos)
                spk_embs[c["name"]] = build_spk_embs(pos, neg)

            est = {name: model(mix_wavs, spk_embs[name]) for name in case_names}
            rows = {name: compute_metrics(est[name], target_wavs, mix_wavs, sr, metrics=args.metrics)
                    for name in case_names}

            for b, key in enumerate(keys):
                for name in case_names:
                    rows[name][b]["key"] = key
                parts = []
                for name in case_names:
                    si, sii = rows[name][b]["SI-SDR"], rows[name][b]["SI-SDRi"]
                    run_sum[name]["SI-SDR"] += si
                    run_sum[name]["SI-SDRi"] += sii
                    run_n[name] += 1
                    avg_si = run_sum[name]["SI-SDR"] / run_n[name]
                    avg_sii = run_sum[name]["SI-SDRi"] / run_n[name]
                    parts.append(f"{name}: SI-SDR={si:.2f} SI-SDRi={sii:.2f} (avg={avg_si:.2f}/{avg_sii:.2f})")
                tqdm.write(f"{key}  " + " | ".join(parts))
                if audio_saved < args.num_audio_samples:
                    save_audio_sample(os.path.join(audio_dir, f"{key}_mix.wav"), mix_wavs[b], sr)
                    save_audio_sample(os.path.join(audio_dir, f"{key}_target.wav"), target_wavs[b], sr)
                    for name in case_names:
                        save_audio_sample(os.path.join(audio_dir, f"{key}_est_{name}.wav"), est[name][b], sr)
                    audio_saved += 1

            for name in case_names:
                all_rows[name].extend(rows[name])

    if audio_saved:
        print(f"\nSaved {audio_saved} audio sample(s) to {audio_dir}")

    print("\n── Results ──────────────────────────────────")
    results = []
    for name in case_names:
        agg = save_case(
            all_rows[name], os.path.join(exp_dir, "eval_nectar_vctk", name), checkpoint,
            extra={"backbone_cfg": args.cfg, "backbone_clue_mode": backbone_clue_mode, "case": name,
                   "dataset": "VCTK-2Mix (out-of-domain)"},
        )
        agg["case"] = name
        results.append(agg)

    comparison_output = args.comparison_output or os.path.join(exp_dir, "eval_nectar_vctk", "comparison_3.0.csv")
    print_and_save_comparison(results, comparison_output)


if __name__ == "__main__":
    main()
