"""
Reports parameter count and FLOPs (via torch's FlopCounterMode) for a trained
CSM run, broken into backbone (frozen CAM++/ECAPA-TDNN) vs. head (the
trainable Transformer+ASP stack) vs. total -- supports the paper's claim that
the CSM head adds negligible computational overhead over the frozen backbone.
FLOPs are for one forward pass over a single --duration-second utterance.

Usage:
    python speakerlab/bin/analysis/model_cost.py --run-dir exp/csm/csm
    python speakerlab/bin/analysis/model_cost.py --run-dir exp/csm/csm_ecapa
"""

import argparse
import os
import sys

import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from speakerlab.models.campplus.csm import (
    CAMPlusCSM, load_pretrained_campplus,
)
from speakerlab.models.ecapa_tdnn.csm import ECAPACSM, load_pretrained_ecapa
from speakerlab.process.processor import FBank

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


def build_model(cfg, campplus_ckpt, ecapa_ckpt, device):
    """Same config -> model dispatch as beta_convergence.py's load_csm, minus
    the trained-checkpoint load -- params/FLOPs depend only on architecture,
    not learned weight values. Returns (model, backbone)."""
    mcfg = cfg["model"]
    backbone_name = mcfg.get("backbone", "campplus")

    if backbone_name == "ecapa":
        ckpt_path = cfg["ecapa"]["ckpt_path"] if "ecapa" in cfg else ecapa_ckpt
        backbone = load_pretrained_ecapa(ckpt_path, mcfg["embedding_dim"], device)
        model = ECAPACSM(
            ecapa=backbone,
            ecapa_frame_dim=mcfg.get("backbone_frame_dim", mcfg.get("campp_frame_dim", 3072)),
            transformer_dim=mcfg.get("transformer_dim", 256),
            embedding_dim=mcfg["embedding_dim"],
            num_layers=mcfg.get("num_layers", 2),
            num_heads=mcfg.get("num_heads", 8),
            ff_dim=mcfg.get("ff_dim", 512),
            dropout=mcfg.get("dropout", 0.1),
            freeze_ecapa=True,
            use_conv=mcfg.get("use_conv", False),
            conv_kernel_size=mcfg.get("conv_kernel_size", 3),
            use_transformer=mcfg.get("use_transformer", True),
        ).to(device)
    elif backbone_name == "campplus":
        backbone = load_pretrained_campplus(cfg["campplus"]["ckpt_path"], mcfg["embedding_dim"], device)
        model = CAMPlusCSM(
            campp=backbone,
            campp_frame_dim=mcfg.get("campp_frame_dim", 512),
            transformer_dim=mcfg.get("transformer_dim", 256),
            embedding_dim=mcfg["embedding_dim"],
            num_layers=mcfg.get("num_layers", 2),
            num_heads=mcfg.get("num_heads", 8),
            ff_dim=mcfg.get("ff_dim", 512),
            dropout=mcfg.get("dropout", 0.1),
            freeze_campp=True,
            use_conv=mcfg.get("use_conv", False),
            conv_kernel_size=mcfg.get("conv_kernel_size", 3),
            use_transformer=mcfg.get("use_transformer", True),
        ).to(device)
    else:
        raise ValueError(f"Unknown model.backbone {backbone_name!r}")

    model.eval()
    return model, backbone


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def count_flops(fn, *args, **kwargs):
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        fn(*args, **kwargs)
    return fc.get_total_flops()


def human(n):
    for unit, div in [("G", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(n) >= div:
            return f"{n / div:.3f}{unit}"
    return str(n)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--campplus-ckpt", default=DEFAULT_CAMPPLUS_CKPT)
    parser.add_argument("--ecapa-ckpt", default=DEFAULT_ECAPA_CKPT)
    parser.add_argument("--duration", type=float, default=None,
                         help="Input length (s) for the FLOPs forward pass; defaults to config.data.enroll_len.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    with open(os.path.join(args.run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    duration = args.duration if args.duration is not None else cfg["data"]["enroll_len"]
    model, backbone = build_model(cfg, args.campplus_ckpt, args.ecapa_ckpt, device)

    total_params = count_params(model)
    backbone_params = count_params(backbone)
    head_params = total_params - backbone_params

    fbank_fn = FBank(80, sample_rate=args.sample_rate, mean_nor=True)
    num_samples = int(round(duration * args.sample_rate))
    dummy_wav = torch.zeros(num_samples)
    fbank = fbank_fn(dummy_wav).unsqueeze(0).to(device)  # [1, T, 80]

    backbone_flops = count_flops(model.extract_campp_frame_features, fbank) \
        if hasattr(model, "extract_campp_frame_features") else count_flops(backbone, fbank)
    total_flops = count_flops(model, fbank)
    head_flops = total_flops - backbone_flops

    print(f"run_dir: {args.run_dir}")
    print(f"backbone: {cfg['model'].get('backbone', 'campplus')}, "
          f"duration: {duration:g}s ({fbank.shape[1]} frames), device: {device}")
    print()
    print(f"{'':10s}{'params':>14s}{'FLOPs (1 fwd)':>18s}")
    print(f"{'backbone':10s}{human(backbone_params):>14s}{human(backbone_flops):>18s}")
    print(f"{'head':10s}{human(head_params):>14s}{human(head_flops):>18s}")
    print(f"{'total':10s}{human(total_params):>14s}{human(total_flops):>18s}")
    print()
    print(f"backbone_params={backbone_params}  head_params={head_params}  total_params={total_params}")
    print(f"backbone_flops={backbone_flops}  head_flops={head_flops}  total_flops={total_flops}")


if __name__ == "__main__":
    main()
