"""
Trains the BSRNN target-speaker-extraction backbone on Libri2Mix, conditioned
on a frozen, pretrained CAM++ speaker embedding. `model.clue_mode` selects
the auxiliary clue concatenated onto the target enrollment embedding: absent
-> Baseline (Table 1), "v2" -> NECTAR's training-time oracle negative
enrollment (evaluated at inference with the CSM-estimated clue instead, see
eval_nectar.py), "v5" -> Mixture Clue. Usage: `cd BSRNN && python train.py
--cfg configs/baseline.yml`.
"""

import argparse
import gc
import math
import os
import sys
import time

import numpy as np
import torch

# Cap CPU threads: unset, torch grabs every core for small per-sample fbank
# ops, starving both the GPU and the DataLoader workers.
torch.set_num_threads(4)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import yaml
from tqdm import tqdm

# Add csm/ to sys.path so its speakerlab package (CAM++, FBank, CSM) is importable.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_CSM_DIR = os.path.join(_REPO_ROOT, "csm")
if _CSM_DIR not in sys.path:
    sys.path.insert(0, _CSM_DIR)
sys.path.insert(0, _THIS_DIR)

from data.data_loader import get_dataloader_campp_dsm
from losses import singlesrc_neg_sisdr
from speakerlab.dataset.dataset import pair_key
from speakerlab.models.campplus.DTDNN import CAMPPlus
from speakerlab.models.campplus.csm import (
    ModelCheckpointProxy,
    compute_pseudo_interference_centroid,
)
from speakerlab.process.processor import FBank

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


def build_campplus(ckpt_path, emb_dim, device):
    """Load pretrained Cam++ and freeze it."""
    model = CAMPPlus(feat_dim=80, embedding_size=emb_dim)
    state = torch.load(os.path.expanduser(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device)
    fbank = FBank(80, sample_rate=16000, mean_nor=True)
    print(f"Loaded Cam++ from {ckpt_path} (frozen)")
    return model, fbank


def build_verification_model(cfg, device):
    """Loads the frozen CAM++ speaker encoder used by every BSRNN config (the
    CSM's own training separately supports ECAPA-TDNN too, see csm/)."""
    vc = cfg["campplus"]
    emb_dim = vc.get("emb_dim", cfg["model"]["spk_emb_dim"])
    return build_campplus(vc["ckpt_path"], emb_dim, device)


@torch.no_grad()
def extract_embeddings(enroll_wavs, campplus, fbank, device):
    """enroll_wavs: list[Tensor] of len B, each (T_b,) -- lengths may differ.
    Returns (B, D) CAM++ embeddings. Same-length inputs are batched; mixed
    lengths are embedded one at a time (CAM++'s stats pooling is length-
    sensitive, so padding would corrupt it)."""
    lengths = {wav.shape[-1] for wav in enroll_wavs}
    if len(lengths) == 1:
        feats = torch.stack([fbank(wav.cpu()) for wav in enroll_wavs]).to(device)
        return campplus(feats)
    embeddings = [campplus(fbank(wav.cpu()).unsqueeze(0).to(device)).squeeze(0) for wav in enroll_wavs]
    return torch.stack(embeddings)


def load_speaker_centroids(path):
    """dict {speaker id: (D,) tensor}, from precompute_centroids.py."""
    return torch.load(path, map_location="cpu")


def lookup_centroid_embs(spk_ids, centroid_dict, device):
    """spk_ids: list[str], len B -> (B, D) tensor of precomputed centroids."""
    return torch.stack([centroid_dict[s] for s in spk_ids]).to(device)


def load_pair_centroids(path):
    """dict {pair_key(spk_a, spk_b): (D,) tensor}, the averaged mixture
    embedding for that speaker pair, from precompute_mixture_centroids.py."""
    return torch.load(path, map_location="cpu")


def lookup_pair_centroid_embs(spk_ids_a, spk_ids_b, pair_centroid_dict, device):
    """spk_ids_a/spk_ids_b: list[str], len B -> (B, D) mixture-pair centroids,
    looked up by the unordered (a, b) key."""
    return torch.stack([
        pair_centroid_dict[pair_key(a, b)] for a, b in zip(spk_ids_a, spk_ids_b)
    ]).to(device)


def batch_fbank(wavs, fbank_fn):
    """wavs (variable-length tensors) -> zero-padded (B, T'_max, 80) fbank
    batch + per-item valid frame lengths, for CAMPlusCSM's length-aware
    pooling."""
    feats = [fbank_fn(w.cpu()) for w in wavs]
    lengths = torch.tensor([f.shape[0] for f in feats])
    max_len = int(lengths.max())
    padded = torch.stack([
        f if f.shape[0] == max_len else F.pad(f, (0, 0, 0, max_len - f.shape[0]))
        for f in feats
    ])
    return padded, lengths


def load_csm(run_dir, campp, device, ckpt_name="CKPT+best"):
    """Build a CAMPlusCSM from <run_dir>/config.yaml and load its trained,
    frozen weights from <run_dir>/checkpoints/<ckpt_name>/model.ckpt, reusing
    the already-loaded frozen CAM++ backbone `campp`."""
    with open(os.path.join(run_dir, "config.yaml")) as f:
        ce_cfg = yaml.safe_load(f)
    mcfg = ce_cfg["model"]
    from speakerlab.models.campplus.csm import CAMPlusCSM

    estimator = CAMPlusCSM(
        campp=campp,
        campp_frame_dim=mcfg.get("campp_frame_dim", 512),
        transformer_dim=mcfg.get("transformer_dim", 256),
        embedding_dim=mcfg["embedding_dim"],
        num_layers=mcfg.get("num_layers", 2),
        num_heads=mcfg.get("num_heads", 8),
        ff_dim=mcfg.get("ff_dim", 512),
        dropout=mcfg.get("dropout", 0.1),
        use_conv=mcfg.get("use_conv", False),
        conv_kernel_size=mcfg.get("conv_kernel_size", 3),
        use_transformer=mcfg.get("use_transformer", True),
        freeze_campp=True,
    ).to(device)

    estimator.eval()
    for p in estimator.parameters():
        p.requires_grad_(False)

    ckpt_path = os.path.join(run_dir, "checkpoints", ckpt_name, "model.ckpt")
    ModelCheckpointProxy(estimator).load(ckpt_path, device)
    print(f"CSM: {run_dir} [{ckpt_name}] (embedding_dim={mcfg['embedding_dim']})")
    return estimator


@torch.no_grad()
def run_csm(wavs, csm, fbank, device):
    """wavs: list[Tensor] or stacked (B, T) tensor, all sharing one length
    (enroll_len-cropped/padded) -> (B, D) magnitude-carrying centroid_raw
    estimates."""
    feats, lengths = batch_fbank(wavs, fbank)
    out = csm(feats.to(device), lengths=lengths.to(device))
    return out["centroid_raw"]


def build_spk_embs(pos_embs, neg_embs):
    """cat([pos_embs, neg_embs]) -- spk_emb_dim = 2x the raw CAM++ dim."""
    return torch.cat([pos_embs, neg_embs], dim=-1)


def build_clue_embs(clue_mode, pos_embs_raw, interference_enroll_wavs,
                     campplus, fbank, device, normalize_emb=False, mix_wavs=None):
    """Dispatch on model.clue_mode (see module docstring) to build BSRNN's
    conditioning vector from the target's raw CAM++ embedding (pos_embs_raw)
    and, for "v2"/"v5", a negative half. normalize_emb L2-normalizes each
    half independently before concatenating."""
    if clue_mode is None:
        return F.normalize(pos_embs_raw, dim=-1) if normalize_emb else pos_embs_raw
    if clue_mode == "v2":
        pos = pos_embs_raw
        neg = extract_embeddings(interference_enroll_wavs, campplus, fbank, device)
    elif clue_mode == "v5":
        if mix_wavs is None:
            raise ValueError("clue_mode v5 needs mix_wavs (the mixture waveform).")
        pos = pos_embs_raw
        # Raw CAM++ embedding of the actual mixture. mix_wavs is a (B, T)
        # tensor; extract_embeddings iterates its rows, embedding each mixture
        # one at a time (same per-sample path as variable-length enrollment).
        neg = extract_embeddings(mix_wavs, campplus, fbank, device)
    else:
        raise ValueError(f"Unknown clue_mode: {clue_mode!r}")
    if normalize_emb:
        pos = F.normalize(pos, dim=-1)
        neg = F.normalize(neg, dim=-1)
    return build_spk_embs(pos, neg)


def get_model(cfg, device):
    from models.bsrnn.bsrnn import BSRNN

    model_cfg = dict(cfg["model"])
    model_cfg.pop("clue_mode", None)  # train.py-only flag, not a BSRNN kwarg
    model_cfg.pop("enroll_source", None)  # train.py-only flag, not a BSRNN kwarg
    model_cfg.pop("normalize_emb", None)  # train.py-only flag, not a BSRNN kwarg
    return BSRNN(**model_cfg).to(device)


def print_model_summary(model):
    """Print per-submodule + total trainable parameter counts."""
    print("BSRNN parameter breakdown:")
    total = 0
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total += n
        print(f"  {name:20s} {n:>12,}")
    print(f"  {'TOTAL':20s} {total:>12,}")


def train(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    exp_dir = cfg["training"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "cfg.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print("=" * 60)
    print(f"Config ({exp_dir}):")
    print(yaml.dump(cfg, default_flow_style=False))
    print("=" * 60)

    # ── W&B ───────────────────────────────────────────────────────────────────
    wb_cfg = cfg.get("wandb", {})
    use_wandb = wb_cfg.get("enabled", True)
    if use_wandb:
        wandb.init(
            project=wb_cfg.get("project", "BSRNN-CAM"),
            name=wb_cfg.get("name", cfg["training"].get("exp_dir", "run").replace("/", "_")),
            config=cfg,
        )

    # Verification backbone (frozen CAM++) — lives outside the training
    # graph. Its embedding_size is always the raw CAM++ dim (192),
    # independent of model.spk_emb_dim (BSRNN's conditioning-vector size:
    # 2x that once model.clue_mode is set).
    campplus, fbank = build_verification_model(cfg, device)

    # BSRNN
    model = get_model(cfg, device)
    clue_mode = cfg["model"].get("clue_mode")
    valid_clue_modes = ("v2", "v5")
    if clue_mode is not None and clue_mode not in valid_clue_modes:
        raise ValueError(f"model.clue_mode must be one of {valid_clue_modes} (or absent), got {clue_mode!r}")
    normalize_emb = cfg["model"].get("normalize_emb", False)
    print(f"clue_mode: {clue_mode}  normalize_emb: {normalize_emb}")
    print_model_summary(model)

    # ── Optim settings ────────────────────────────────────────────────────────
    opt_cfg = cfg["optim"]
    clip_grad = cfg["grad_clipping"].get("clip_grad", 5.0)
    max_epoch = opt_cfg.get("max_epoch", 150)
    final_lr = opt_cfg.get("final_lr", 2e-5)
    stop_patience = opt_cfg.get("stop_patience", 5)
    # Exponential LR decay, updated every optimizer step: current_lr =
    # initial_lr * exp((step / max_iter) * log(final_lr / initial_lr)),
    # reaching final_lr exactly at the last training step.
    lr_schedule = opt_cfg.get("lr_schedule", "exponential")
    if lr_schedule != "exponential":
        raise ValueError(f"train.py only supports optim.lr_schedule: 'exponential', got {lr_schedule!r}")
    initial_lr = opt_cfg["lr"]
    eval_interval = cfg["training"].get("eval_interval", 1)
    num_avg = cfg["training"].get("num_avg", 5)
    accum_grad = cfg["training"].get("accum_grad", 1)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg.get("weight_decay", 0.0),
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val = float("inf")
    valid_losses = []
    early_stop_flag = 0

    resume = cfg["training"].get("resume", False)
    resumed = False
    if resume:
        ckpt_file = os.path.join(exp_dir, "last.ckpt")
        if os.path.exists(ckpt_file):
            ckpt = torch.load(ckpt_file, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"]
            valid_losses = ckpt.get("valid_losses", [])
            best_val = min(valid_losses) if valid_losses else float("inf")
            resumed = True
            print(f"Resumed from epoch {start_epoch} ({ckpt_file})")
        else:
            print(f"Checkpoint not found: {ckpt_file}, starting fresh")

    # ── Data ──────────────────────────────────────────────────────────────────
    if not cfg["data"].get("dsm", False):
        raise ValueError("train.py only supports data.dsm: true (dynamic speaker mixing).")
    train_loader, val_loader = get_dataloader_campp_dsm(**cfg["data"])
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    # max_iter is the step horizon the LR schedule anneals over. On resume we
    # keep the checkpoint's original horizon (not a new one from the current
    # max_epoch), so extending training just holds at final_lr instead of
    # restretching the decay curve.
    if resumed:
        global_step = ckpt.get("global_step", start_epoch * len(train_loader))
        max_iter = ckpt.get("max_iter", max(start_epoch * len(train_loader), 1))
    else:
        global_step = 0
        max_iter = max_epoch * len(train_loader)

    step_log_interval = wb_cfg.get("step_log_interval", 500)

    # ── Epoch loop ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, max_epoch):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        t0 = time.time()
        train_loss_sum = 0.0
        optimizer.zero_grad()

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch:03d} [train]",
            disable=False,
            leave=True,
        )
        for batch_id, (mix_wavs, enroll_wavs, interference_enroll_wavs, interference_target_wavs,
                       target_wavs, interference_spk_ids, target_spk_ids, noise_wavs) in pbar:
            mix_wavs = mix_wavs.to(device)
            target_wavs = target_wavs.to(device)
            # enroll_wavs/interference_enroll_wavs are list[Tensor] (possibly
            # variable-length under enroll_len=None) -- extract_embeddings
            # moves each to `device` itself, per-sample.

            with torch.no_grad():
                pos_embs_raw = extract_embeddings(enroll_wavs, campplus, fbank, device)
                spk_embs = build_clue_embs(
                    clue_mode, pos_embs_raw, interference_enroll_wavs,
                    campplus, fbank, device, normalize_emb=normalize_emb, mix_wavs=mix_wavs,
                )

            est_wavs = model(mix_wavs, spk_embs)
            loss = singlesrc_neg_sisdr(est_wavs, target_wavs).mean()

            if accum_grad > 1:
                loss = loss / accum_grad
            if torch.isnan(loss):
                continue

            loss.backward()

            if ((batch_id + 1) % accum_grad == 0) or ((batch_id + 1) == len(train_loader)):
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                optimizer.zero_grad()

                # Clamp at 1.0 so LR holds at final_lr past max_iter instead
                # of continuing to decay.
                frac = min(global_step / max_iter, 1.0)
                step_lr = initial_lr * math.exp(frac * math.log(final_lr / initial_lr))
                for pg in optimizer.param_groups:
                    pg["lr"] = step_lr

            train_loss_sum += loss.item() * accum_grad
            running_avg = train_loss_sum / (batch_id + 1)
            pbar.set_postfix(loss=f"{running_avg:.4f}", si_sdr=f"{-running_avg:.2f}dB")

            global_step += 1
            if use_wandb and global_step % step_log_interval == 0:
                step_loss = loss.item() * accum_grad
                wandb.log(
                    {
                        "train/step_loss": step_loss,
                        "train/step_si_sdr": -step_loss,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

        train_avg_loss = train_loss_sum / (batch_id + 1)
        elapsed = (time.time() - t0) / 60

        # ── Validation ────────────────────────────────────────────────────────
        val_avg_loss = None
        val_sdri_mean = val_nsr = val_pesq_mean = val_stoi_mean = None
        if epoch % eval_interval == 0 and epoch != 0:
            model.eval()
            val_loss_sum = 0.0
            val_sdri_all: list[float] = []
            val_pesq_all: list[float] = []
            val_stoi_all: list[float] = []
            with torch.no_grad():
                vbar = tqdm(
                    enumerate(val_loader),
                    total=len(val_loader),
                    desc=f"Epoch {epoch:03d} [val]  ",
                    disable=False,
                    leave=True,
                )
                for vbid, (mix_wavs, enroll_wavs, interference_enroll_wavs, interference_target_wavs,
                           target_wavs, interference_spk_ids, target_spk_ids, noise_wavs) in vbar:
                    mix_wavs = mix_wavs.to(device)
                    target_wavs = target_wavs.to(device)

                    pos_embs_raw = extract_embeddings(enroll_wavs, campplus, fbank, device)
                    spk_embs = build_clue_embs(
                        clue_mode, pos_embs_raw, interference_enroll_wavs,
                        campplus, fbank, device, normalize_emb=normalize_emb, mix_wavs=mix_wavs,
                    )
                    est_wavs = model(mix_wavs, spk_embs)

                    per_sample_neg_sisdr = singlesrc_neg_sisdr(est_wavs, target_wavs)  # (B,)
                    loss = per_sample_neg_sisdr.mean()
                    val_loss_sum += loss.item()
                    running_val = val_loss_sum / (vbid + 1)
                    vbar.set_postfix(loss=f"{running_val:.4f}", si_sdr=f"{-running_val:.2f}dB")

                    # SI-SDRi = SI-SDR(est) - SI-SDR(mixture) -- same "unprocessed
                    # input" baseline used everywhere else in this repo.
                    mix_neg_sisdr = singlesrc_neg_sisdr(mix_wavs, target_wavs)  # (B,)
                    sdri_batch = (-per_sample_neg_sisdr) - (-mix_neg_sisdr)
                    val_sdri_all.extend(sdri_batch.detach().cpu().tolist())

                    if HAS_PESQ or HAS_STOI:
                        est_np = est_wavs.detach().cpu().numpy()
                        ref_np = target_wavs.detach().cpu().numpy()
                        for b in range(est_np.shape[0]):
                            if HAS_PESQ:
                                try:
                                    val_pesq_all.append(
                                        float(pesq_fn(cfg["data"]["sample_rate"], ref_np[b], est_np[b], "wb"))
                                    )
                                except Exception:
                                    pass
                            if HAS_STOI:
                                try:
                                    val_stoi_all.append(
                                        float(stoi_fn(ref_np[b], est_np[b], cfg["data"]["sample_rate"], extended=False))
                                    )
                                except Exception:
                                    pass

            val_avg_loss = val_loss_sum / (vbid + 1)

            if val_sdri_all:
                val_sdri_mean = float(np.mean(val_sdri_all))
                val_nsr = float(np.mean(np.array(val_sdri_all) < 0))
            if val_pesq_all:
                val_pesq_mean = float(np.mean(val_pesq_all))
            if val_stoi_all:
                val_stoi_mean = float(np.mean(val_stoi_all))

        # ── Log ───────────────────────────────────────────────────────────────
        current_lr = optimizer.param_groups[0]["lr"]
        log = (
            f"epoch:{epoch}->{epoch+1}"
            f"|time:{elapsed:.2f}min"
            f"|train_loss:{train_avg_loss:.4f}"
            f"|lr:{current_lr:.6f}"
        )
        if val_avg_loss is not None:
            log += f"|val_loss:{val_avg_loss:.4f}|si_sdr:{-val_avg_loss:.2f}dB"
            if val_sdri_mean is not None:
                log += f"|si_sdri:{val_sdri_mean:+.2f}dB|nsr:{val_nsr*100:.1f}%"
            if val_pesq_mean is not None:
                log += f"|pesq:{val_pesq_mean:.3f}"
            if val_stoi_mean is not None:
                log += f"|stoi:{val_stoi_mean:.3f}"
        print(log)

        if use_wandb:
            wb_log = {
                "train/loss": train_avg_loss,
                "train/si_sdr": -train_avg_loss,
                "train/lr": current_lr,
                "train/epoch_time_min": elapsed,
            }
            if val_avg_loss is not None:
                wb_log["val/loss"] = val_avg_loss
                wb_log["val/si_sdr"] = -val_avg_loss
            if val_sdri_mean is not None:
                wb_log["val/si_sdri"] = val_sdri_mean
                wb_log["val/nsr"] = val_nsr
            if val_pesq_mean is not None:
                wb_log["val/pesq"] = val_pesq_mean
            if val_stoi_mean is not None:
                wb_log["val/stoi"] = val_stoi_mean
            wb_log["train/epoch"] = epoch + 1
            wandb.log(wb_log, step=global_step)

        # ── Checkpoint & early stopping ───────────────────────────────────────
        if val_avg_loss is not None:
            valid_losses.append(val_avg_loss)

            is_better = len(valid_losses) <= 1
            if len(valid_losses) > 1:
                top_n_threshold = sorted(valid_losses)[: num_avg][-1]
                is_better = val_avg_loss < top_n_threshold

            if is_better:
                early_stop_flag = 0
                best_val = val_avg_loss
                _ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_val": best_val,
                    "valid_losses": valid_losses,
                    "global_step": global_step,
                    "max_iter": max_iter,
                }
                best_p = os.path.join(exp_dir, "best.ckpt")
                torch.save(_ckpt, best_p)
                print(f"  → saved best checkpoint (epoch {epoch+1}, SI-SDR={-val_avg_loss:.2f}dB)")
            else:
                early_stop_flag += 1

            # Also save last
            last_ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_val": best_val,
                "valid_losses": valid_losses,
                "global_step": global_step,
                "max_iter": max_iter,
            }
            torch.save(last_ckpt, os.path.join(exp_dir, "last.ckpt"))

        gc.collect()

        if early_stop_flag >= stop_patience:
            print(f"Early stop at epoch {epoch + 1} (early_stop_flag={early_stop_flag})")
            break

    print(f"Training finished. Best val SI-SDR = {-best_val:.2f} dB")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    gpus = cfg["distribute"].get("gpu_ids", [0])
    if len(gpus) > 1:
        raise ValueError(
            f"train.py only supports single-GPU training (distribute.gpu_ids), got {gpus!r}"
        )
    train(cfg)
