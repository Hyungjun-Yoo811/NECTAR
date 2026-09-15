"""
Reports params/FLOPs for the full NECTAR inference pipeline -- CAM++ (full,
for the positive clue) + CSM (2 passes, for the negative clue) + BSRNN
separator -- supporting the paper's "negligible computational overhead"
claim. CAM++ is counted once in params (one shared checkpoint) but its FLOPs
add across all passes. See lstm_flops_from_hooks for why BSRNN's LSTM FLOPs
need a manual correction.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import yaml
from torch.utils.flop_counter import FlopCounterMode

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_CSM_DIR = os.path.join(os.path.dirname(_THIS_DIR), "csm")
if _CSM_DIR not in sys.path:
    sys.path.insert(0, _CSM_DIR)
_CSM_ANALYSIS_DIR = os.path.join(_CSM_DIR, "speakerlab", "bin", "analysis")
if _CSM_ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _CSM_ANALYSIS_DIR)

from models.bsrnn.bsrnn import BSRNN
from speakerlab.models.campplus.DTDNN import CAMPPlus
from speakerlab.process.processor import FBank
from model_cost import build_model as build_csm, count_flops, count_params, human  # csm/speakerlab/bin/analysis/model_cost.py

DEFAULT_BSRNN_CFG = os.path.join(_THIS_DIR, "configs", "nectar.yml")
DEFAULT_ESTIMATOR_RUN_DIR = os.path.join(_CSM_DIR, "exp/csm/csm")
DEFAULT_CAMPPLUS_CKPT = os.path.expanduser(
    "~/.cache/modelscope/models/"
    "iic--speech_campplus_sv_zh_en_16k-common_advanced/"
    "snapshots/v1.0.0/campplus_cn_en_common.pt"
)
DEFAULT_ECAPA_CKPT = os.path.expanduser(
    "~/.cache/modelscope/models/"
    "iic--speech_ecapa-tdnn_sv_en_voxceleb_16k/"
    "snapshots/v1.0.1/ecapa_tdnn.bin"
)


def build_campplus(ckpt_path, emb_dim, device):
    """Full (untruncated, incl. stats+dense) frozen CAM++ -- train.py's own
    build_campplus, reproduced here rather than imported to avoid pulling in
    train.py's wandb/torch.distributed/argparse module-level dependencies
    for a measurement-only script."""
    model = CAMPPlus(feat_dim=80, embedding_size=emb_dim)
    state = torch.load(os.path.expanduser(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def lstm_flops_from_hooks(model, forward_fn):
    """Hooks every nn.LSTM in `model`, runs `forward_fn()`, and returns the
    summed closed-form LSTM FLOPs. torch's FlopCounterMode always reports 0
    for nn.LSTM (it dispatches to an opaque cuDNN/MKL-DNN kernel with no
    registered flop formula), so this fills that gap manually: 4 gates x
    (input*hidden + hidden*hidden) MACs per step per direction per layer,
    x2 for FLOPs."""
    captured = []

    def hook(mod, inp, out):
        captured.append((mod, tuple(inp[0].shape)))

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.LSTM)]
    try:
        forward_fn()
    finally:
        for h in handles:
            h.remove()

    total = 0
    for mod, (batch, seq, in_size) in captured:
        hid = mod.hidden_size
        directions = 2 if mod.bidirectional else 1
        layer_in = in_size
        for _ in range(mod.num_layers):
            macs = 4 * (layer_in * hid + hid * hid) * seq * batch * directions
            total += 2 * macs
            layer_in = hid * directions
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bsrnn-cfg", default=DEFAULT_BSRNN_CFG, help="BSRNN training config to measure.")
    parser.add_argument("--estimator-run-dir", default=DEFAULT_ESTIMATOR_RUN_DIR, help="Trained CSM run dir.")
    parser.add_argument("--campplus-ckpt", default=DEFAULT_CAMPPLUS_CKPT, help="Frozen CAM++ checkpoint for the CSM.")
    parser.add_argument("--ecapa-ckpt", default=DEFAULT_ECAPA_CKPT, help="Frozen ECAPA-TDNN checkpoint for the CSM.")
    parser.add_argument("--duration", type=float, default=3.0, help="Input length (s) for every forward pass.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate in Hz.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="torch device.")
    args = parser.parse_args()

    device = torch.device(args.device)
    with open(args.bsrnn_cfg) as f:
        bsrnn_cfg = yaml.safe_load(f)
    mcfg = bsrnn_cfg["model"]

    fbank_fn = FBank(80, sample_rate=args.sample_rate, mean_nor=True)
    num_samples = int(round(args.duration * args.sample_rate))
    fbank = fbank_fn(torch.zeros(num_samples)).unsqueeze(0).to(device)  # [1, T, 80]

    # 1. CAM++, full (positive-clue raw embedder).
    campp_full = build_campplus(bsrnn_cfg["campplus"]["ckpt_path"], bsrnn_cfg["campplus"]["emb_dim"], device)
    campp_params = count_params(campp_full)
    campp_flops = count_flops(campp_full, fbank)

    # 2/3. CSM (its own truncated-CAM++ backbone + head) -- 2 forward passes.
    with open(os.path.join(args.estimator_run_dir, "config.yaml")) as f:
        ce_cfg = yaml.safe_load(f)
    ce_model, ce_backbone = build_csm(ce_cfg, args.campplus_ckpt, args.ecapa_ckpt, device)
    ce_backbone_params = count_params(ce_backbone)
    ce_head_params = count_params(ce_model) - ce_backbone_params
    ce_backbone_flops_1x = count_flops(ce_model.extract_campp_frame_features, fbank)
    ce_total_flops_1x = count_flops(ce_model, fbank)
    ce_head_flops_1x = ce_total_flops_1x - ce_backbone_flops_1x

    # 4. BSRNN separator.
    bsrnn = BSRNN(
        spk_emb_dim=mcfg["spk_emb_dim"], sr=mcfg["sr"], win=mcfg["win"], stride=mcfg["stride"],
        feature_dim=mcfg["feature_dim"], num_repeat=mcfg["num_repeat"],
        use_spk_transform=mcfg["use_spk_transform"], spk_fuse_type=mcfg["spk_fuse_type"],
    ).to(device)
    bsrnn.eval()
    bsrnn_params = count_params(bsrnn)

    mix_wav = torch.zeros(1, num_samples, device=device)
    clue = torch.zeros(1, mcfg["spk_emb_dim"], device=device)

    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        bsrnn(mix_wav, clue)
    bsrnn_flops_no_lstm = fc.get_total_flops()
    bsrnn_lstm_flops = lstm_flops_from_hooks(bsrnn, lambda: bsrnn(mix_wav, clue))
    bsrnn_flops = bsrnn_flops_no_lstm + bsrnn_lstm_flops

    # CAM++ counted once in params (one shared checkpoint); its FLOPs still
    # add across all 3 passes that touch it (once full, twice inside the CSM).
    total_params = campp_params + ce_head_params + bsrnn_params
    total_flops = campp_flops + 2 * ce_total_flops_1x + bsrnn_flops

    print(f"BSRNN cfg        : {args.bsrnn_cfg}")
    print(f"estimator run_dir: {args.estimator_run_dir}")
    print(f"duration: {args.duration:g}s ({fbank.shape[1]} fbank frames), device: {device}")
    print()
    print(f"{'':22s}{'params':>14s}{'FLOPs':>16s}{'x per utt':>12s}")
    print(f"{'CAM++ (full, shared)':22s}{human(campp_params):>14s}{human(campp_flops):>16s}{'1':>12s}")
    print(f"{'  CSM backbone*':22s}{'(shared)':>14s}{human(ce_backbone_flops_1x):>16s}{'x2':>12s}")
    print(f"{'CSM head':22s}{human(ce_head_params):>14s}{human(ce_head_flops_1x):>16s}{'x2':>12s}")
    print(f"{'BSRNN separator':22s}{human(bsrnn_params):>14s}{human(bsrnn_flops):>16s}{'1':>12s}")
    print(f"{'-' * 66}")
    print(f"{'TOTAL':22s}{human(total_params):>14s}{human(total_flops):>16s}")
    print()
    print("* CSM backbone is CAM++'s own weights again (same checkpoint as the 'CAM++' row "
          "above, truncated before stats+dense) -- NOT double-counted in params, only its FLOPs "
          "(genuinely run again, twice: once for the enrollment, once for the mixture) are added.")
    print()
    print(f"campp_params={campp_params} ce_head_params={ce_head_params} bsrnn_params={bsrnn_params} "
          f"total_params={total_params}")
    print(f"campp_flops={campp_flops} ce_total_flops_1x={ce_total_flops_1x} bsrnn_flops={bsrnn_flops} "
          f"total_flops={total_flops}")


if __name__ == "__main__":
    main()
