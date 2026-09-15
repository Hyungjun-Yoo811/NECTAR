"""
Trains CSM (paper Sec. 3.2) on a single GPU, jointly on the enrollment
branch (target: the speaker's oracle centroid) and the real-mixture branch
(target: C1 + C2, the sum of the pair's oracle centroids, per the
midpoint relationship in paper Sec. 3.1). Loss is MSE against these
targets on both branches.

Usage:
    python speakerlab/bin/train.py --config <path/to.yaml>
"""

import argparse
import itertools
import math
import os
import shutil
import sys
import time

import torch
import torch.nn.functional as F
import wandb
import yaml
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from speakerlab.dataset.dataset import (
    _crop_or_pad,
    _load_mono,
    build_speaker_index,
    get_enrollment_dataloaders,
    get_real_mixture_dataloaders,
)
from speakerlab.models.campplus.csm import ModelCheckpointProxy, load_pretrained_campplus
from speakerlab.models.ecapa_tdnn.csm import (
    ECAPACSM,
    ModelCheckpointProxy as EcapaModelCheckpointProxy,
    load_pretrained_ecapa,
)
from speakerlab.process.processor import FBank
from speakerlab.utils.checkpoint import Checkpointer, METAFNAME
from speakerlab.utils.config import build_config
from speakerlab.utils.utils import AverageMeters, get_logger, set_seed

parser = argparse.ArgumentParser(description="CSM Training")
parser.add_argument("--config", required=True, type=str)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--seed", default=1234, type=int)
parser.add_argument("--gpu", default=0, type=int,
                     help="Single-GPU device id. Ignored if the config sets training.gpus.")


def load_centroid_cache(path, device):
    cache = torch.load(path, map_location=device)
    return {spk: centroid.to(device) for spk, centroid in cache.items()}


def gather_reference(cache, spk_ids, device):
    missing = [spk for spk in spk_ids if spk not in cache]
    if missing:
        raise KeyError(f"{len(missing)} speaker(s) missing from centroid cache, "
                        f"e.g. {missing[:5]}")
    return torch.stack([cache[spk] for spk in spk_ids]).to(device)


def gather_mixture_reference(cache, pair_keys, device):
    """Superposition target for a real 2-speaker mixture: the plain sum C1 +
    C2 of the pair's two (non-normalized, magnitude-carrying) per-speaker
    centroids -- algebraic, independent of the mixture's own SIR (paper
    Sec. 3.2: this sum approximates 2*c_mix). pair_key is "<spkA>_<spkB>"
    (order-independent, see dataset.pair_key). A model trained against this
    target recovers the interference speaker's centroid via
    hat_c_mix - hat_c_s1 at inference (compute_pseudo_interference_centroid)."""
    missing = set()
    for pk in pair_keys:
        a, b = pk.split("_")
        for spk in (a, b):
            if spk not in cache:
                missing.add(spk)
    if missing:
        raise KeyError(f"{len(missing)} mixture speaker(s) missing from centroid cache, "
                        f"e.g. {sorted(missing)[:5]}")
    refs = [cache[a] + cache[b] for pk in pair_keys for a, b in [pk.split("_")]]
    return torch.stack(refs).to(device)


def build_lr_scheduler(optimizer, config, train_loader, total_epochs):
    """training.lr_schedule: "cosine_warmup" (linear warmup for
    training.warmup_steps, then cosine decay to 0 over the remaining steps),
    "exponential" (smooth exponential decay from training.lr to
    training.final_lr over all training steps), or omitted (no scheduler,
    constant LR). Stepped once per optimizer.step() (see train_one_epoch),
    not per epoch, so warmup_steps/total_steps are in units of optimizer
    steps."""
    lr_schedule = config.training.get("lr_schedule")
    if lr_schedule is None:
        return None

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * total_epochs

    if lr_schedule == "cosine_warmup":
        warmup_steps = config.training.get("warmup_steps", 0)

        def _cosine_warmup(step):
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, _cosine_warmup)

    if lr_schedule == "exponential":
        base_lr = config.training["lr"]
        final_lr = config.training["final_lr"]
        decay_ratio = final_lr / base_lr

        def _exponential(step):
            progress = min(step / max(1, total_steps - 1), 1.0)
            return decay_ratio ** progress

        return torch.optim.lr_scheduler.LambdaLR(optimizer, _exponential)

    raise ValueError(
        f"training.lr_schedule must be 'cosine_warmup', 'exponential', or omitted, got {lr_schedule!r}"
    )


def save_named_checkpoint(checkpointer, name, epoch, meta=None):
    """Overwrite the "last"/"best" checkpoint slot in place, instead of
    Checkpointer's default of accumulating one directory per epoch (which
    would otherwise re-save the frozen ~27MB CAM++ backbone every epoch)."""
    ckpt_dir = checkpointer._custom_checkpoint_dirpath(name)
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    checkpointer.save_checkpoint(meta=meta or {}, name=name, epoch=epoch)


def load_best_val_loss(checkpointer):
    """Recover the "best" checkpoint's val loss (stored in its meta file) so
    --resume doesn't reset the best-tracking threshold to +inf and overwrite a
    genuinely-better "best" checkpoint with a worse one."""
    best_dir = checkpointer._custom_checkpoint_dirpath("best")
    meta_path = best_dir / METAFNAME
    if not meta_path.exists():
        return float("inf")
    with open(meta_path) as f:
        meta = yaml.load(f, Loader=yaml.Loader)
    return meta.get("val_loss", float("inf"))


def main():
    args = parser.parse_args()
    config = build_config(args.config, None, copy=True)
    set_seed(getattr(config, "seed", args.seed))

    gpu_ids = config.training.get("gpus", None)
    single_gpu = gpu_ids[0] if gpu_ids else args.gpu
    device = torch.device(f"cuda:{single_gpu}" if torch.cuda.is_available() else "cpu")

    os.makedirs(config.exp_dir, exist_ok=True)
    logger = get_logger(os.path.join(config.exp_dir, "train.log"))
    wandb_cfg = getattr(config, "wandb", {}) or {}
    wandb.init(
        project=wandb_cfg.get("project", "csm"),
        name=wandb_cfg.get("name", os.path.basename(config.exp_dir.rstrip("/"))),
        mode=wandb_cfg.get("mode", "offline"),
        dir=config.exp_dir,
        config=vars(config),
    )

    # CAM++ by default, ECAPA-TDNN if config.model.backbone: 'ecapa' (then
    # config.campplus["ckpt_path"] points at the ECAPA-TDNN checkpoint
    # instead; section name kept as `campplus:` for backward compat).
    backbone = config.model.get("backbone", "campplus")
    if backbone == "campplus":
        campp = load_pretrained_campplus(config.campplus["ckpt_path"], config.model["embedding_dim"], device)

        from speakerlab.models.campplus.csm import CAMPlusCSM

        model = CAMPlusCSM(
            campp=campp,
            campp_frame_dim=config.model.get("campp_frame_dim", 512),
            transformer_dim=config.model.get("transformer_dim", 256),
            embedding_dim=config.model["embedding_dim"],
            num_layers=config.model.get("num_layers", 2),
            num_heads=config.model.get("num_heads", 8),
            ff_dim=config.model.get("ff_dim", 512),
            dropout=config.model.get("dropout", 0.1),
            use_conv=config.model.get("use_conv", False),
            conv_kernel_size=config.model.get("conv_kernel_size", 3),
            use_transformer=config.model.get("use_transformer", True),
            freeze_campp=True,
        ).to(device)
        checkpoint_proxy_cls = ModelCheckpointProxy
    elif backbone == "ecapa":
        if config.model.get("impl", "v1") != "v1":
            raise ValueError("backbone: 'ecapa' only supports model.impl 'v1' (no residual "
                              "variant exists for ECAPA-TDNN).")
        ecapa = load_pretrained_ecapa(config.campplus["ckpt_path"], config.model["embedding_dim"], device)

        model = ECAPACSM(
            ecapa=ecapa,
            ecapa_frame_dim=config.model.get("campp_frame_dim", 3072),
            transformer_dim=config.model.get("transformer_dim", 256),
            embedding_dim=config.model["embedding_dim"],
            num_layers=config.model.get("num_layers", 2),
            num_heads=config.model.get("num_heads", 8),
            ff_dim=config.model.get("ff_dim", 512),
            dropout=config.model.get("dropout", 0.1),
            freeze_ecapa=True,
            use_conv=config.model.get("use_conv", False),
            conv_kernel_size=config.model.get("conv_kernel_size", 3),
            use_transformer=config.model.get("use_transformer", True),
        ).to(device)
        checkpoint_proxy_cls = EcapaModelCheckpointProxy
    else:
        raise ValueError(f"model.backbone={backbone!r}, expected 'campplus' or 'ecapa'")

    # Combined training: every optimizer step draws one enroll batch and one
    # real Libri2Mix mixture batch, sums their losses (mixture term weighted
    # by training.mix_loss_weight), and applies one combined backward/step.
    train_branches = config.training.get("train_branches", ["enroll", "mixture"])
    if set(train_branches) != {"enroll", "mixture"}:
        raise ValueError(
            f"train.py only supports combined training (train_branches: "
            f"['enroll', 'mixture']), got {train_branches!r}"
        )

    mixture_train_loader, mixture_val_loader = get_real_mixture_dataloaders(
        batch_size=config.data["batch_size"],
        num_workers=config.data["num_workers"],
        train_mix_dirs=config.data["train_mix_dirs"],
        dev_mix_dirs=config.data["dev_mix_dirs"],
        mix_mode=config.data.get("mix_mode", "mix_clean"),
        mix_len=config.data.get("mix_len", 3.0),
        sample_rate=config.data["sample_rate"],
        val_batch_size=config.data.get("val_batch_size"),
    )

    train_loader, val_loader = get_enrollment_dataloaders(
        batch_size=config.data["batch_size"],
        num_workers=config.data["num_workers"],
        librispeech_root=config.data["librispeech_root"],
        enroll_len=config.data.get("enroll_len", 3.0),
        sample_rate=config.data["sample_rate"],
        train_subset=config.data["train_subset"],
        valid_subset=config.data.get("valid_subset", "dev"),
        val_batch_size=config.data.get("val_batch_size"),
    )

    dev_centroids = load_centroid_cache(config.data["dev_centroid_cache"], device)
    train_centroids = load_centroid_cache(config.data["train_centroid_cache"], device)

    # gs-residual diagnostic (evaluate_gs_residual): needs the model's own
    # enroll-branch prediction (hat_c_s1) alongside its mixture-branch
    # prediction. Built once here (fixed dev speaker_index + one
    # deterministic enrollment utterance per dev speaker) and re-evaluated
    # through the current model weights every eval epoch -- a monitoring
    # metric, not a reproduction of the training preprocessing.
    dev_speaker_index = build_speaker_index(config.data["librispeech_root"], config.data.get("valid_subset", "dev"))
    gs_fbank_fn = FBank(80, sample_rate=config.data["sample_rate"], mean_nor=True)
    gs_enroll_len = config.data.get("enroll_len") or 3.0
    compute_dev_enroll_centroids_fn = lambda m: compute_dev_enroll_centroids(
        m, dev_speaker_index, gs_fbank_fn, config.data["sample_rate"], gs_enroll_len, device,
    )

    mixture_train_reference_fn = lambda keys: gather_mixture_reference(train_centroids, keys, device)
    mixture_val_reference_fn = lambda keys: gather_mixture_reference(dev_centroids, keys, device)
    train_reference_fn = lambda keys: gather_reference(train_centroids, keys, device)
    val_reference_fn = lambda keys: gather_reference(dev_centroids, keys, device)

    # The C1+C2 mixture target only makes sense with magnitude, so both
    # branches train MSE-only (no AP loss, no cosine alignment -- those
    # discard magnitude by re-normalizing).
    align_loss_type = config.training.get("align_loss_type", "mse")
    if align_loss_type != "mse":
        raise ValueError(f"train.py only supports training.align_loss_type: 'mse', got {align_loss_type!r}")
    if config.training.get("use_ap_loss", False):
        raise ValueError("train.py no longer supports training.use_ap_loss: true "
                          "(the in-batch AP loss was only used by enroll-only training, "
                          "which no surviving config uses).")
    mix_loss_weight = config.training.get("mix_loss_weight", 1.0)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.training["lr"],
        weight_decay=config.training.get("weight_decay", 0.0),
    )

    total_epochs = config.training["num_epoch"]
    scheduler = build_lr_scheduler(optimizer, config, train_loader, total_epochs)

    recoverables = {"model": checkpoint_proxy_cls(model), "optimizer": optimizer}
    if scheduler is not None:
        recoverables["scheduler"] = scheduler
    checkpointer = Checkpointer(
        checkpoints_dir=os.path.join(config.exp_dir, "checkpoints"),
        recoverables=recoverables,
    )
    best_val_loss = float("inf")
    if args.resume:
        checkpointer.recover_if_possible(device=device)
        best_val_loss = load_best_val_loss(checkpointer)

    log_batch_freq = config.training.get("log_batch_freq", 100)
    save_epoch_freq = config.training.get("save_epoch_freq", 1)
    save_every_epoch = config.training.get("save_every_epoch", False)
    eval_epoch_freq = config.training.get("eval_epoch_freq", 1)
    early_stop_patience = config.training.get("early_stop_patience", None)
    epochs_without_improvement = 0

    for epoch in range(1, total_epochs + 1):
        train_one_epoch(
            train_loader, mixture_train_loader, model, optimizer, epoch,
            train_reference_fn, mixture_train_reference_fn, device, logger,
            log_batch_freq, scheduler, mix_loss_weight,
        )

        should_stop = False
        if epoch % eval_epoch_freq == 0:
            val_loss, _ = evaluate(
                val_loader, mixture_val_loader, model, train_reference_fn=val_reference_fn,
                mixture_reference_fn=mixture_val_reference_fn, device=device, logger=logger,
                epoch=epoch, mix_loss_weight=mix_loss_weight,
                compute_dev_enroll_centroids_fn=compute_dev_enroll_centroids_fn,
                dev_centroids=dev_centroids,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                save_named_checkpoint(checkpointer, "best", epoch, meta={"val_loss": val_loss})
            else:
                epochs_without_improvement += 1
                if early_stop_patience is not None and epochs_without_improvement >= early_stop_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch}: val loss hasn't "
                        f"improved on best={best_val_loss:.4e} for {early_stop_patience} eval(s)."
                    )
                    should_stop = True

        if epoch % save_epoch_freq == 0:
            save_named_checkpoint(checkpointer, "last", epoch)
            if save_every_epoch:
                checkpointer.save_checkpoint(meta={}, epoch=epoch)

        if should_stop:
            break

    wandb.finish()


def train_one_epoch(
    train_loader, mixture_loader, model, optimizer, epoch,
    reference_fn, mixture_reference_fn, device, logger, log_batch_freq, scheduler, mix_loss_weight,
):
    """One training epoch. Every step draws one enroll batch from
    train_loader and one mixture batch from mixture_loader (cycled with
    itertools.cycle if shorter than train_loader), sums their MSE alignment
    losses (mixture term weighted by mix_loss_weight), and applies a single
    backward()/optimizer.step()."""
    stats = AverageMeters()
    stats.add("Time", ":6.3f")
    stats.add("MSE", ":.4e")
    stats.add("MixAlignLoss", ":.4e")

    model.train()
    end = time.time()

    mixture_iter = itertools.cycle(mixture_loader)

    pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}", leave=False)
    for i, (enroll_fbank, lengths, target_spk) in enumerate(pbar):
        enroll_fbank = enroll_fbank.to(device)
        lengths = lengths.to(device) if lengths is not None else None
        reference = reference_fn(target_spk)

        enroll_out = model(enroll_fbank, lengths=lengths)
        align_loss = F.mse_loss(enroll_out["centroid"], reference)
        batch_n = enroll_fbank.size(0)

        mix_fbank, mix_lengths, mix_keys = next(mixture_iter)
        mix_fbank = mix_fbank.to(device)
        mix_lengths = mix_lengths.to(device) if mix_lengths is not None else None
        mix_reference = mixture_reference_fn(mix_keys)
        mix_out = model(mix_fbank, lengths=mix_lengths)
        mix_align_loss = F.mse_loss(mix_out["centroid"], mix_reference)
        loss = align_loss + mix_loss_weight * mix_align_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        stats.update("MSE", align_loss.item(), batch_n)
        stats.update("MixAlignLoss", mix_align_loss.item(), mix_fbank.size(0))
        stats.update("Time", time.time() - end)
        end = time.time()

        pbar.set_postfix(mse=f"{align_loss.item():.4f}", mix_align_loss=f"{mix_align_loss.item():.4f}")
        if i % log_batch_freq == 0:
            with torch.no_grad():
                cos_sim = F.cosine_similarity(enroll_out["centroid"], reference, dim=-1).mean().item()
            wandb.log({
                "train/cos_sim": cos_sim,
                "train/loss": loss.item(),
                "train/mix_align_loss": mix_align_loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            })

    logger.info(
        f"Epoch [{epoch}] done.  avg MSE={stats.avg('MSE'):.4e}  avg MixAlignLoss={stats.avg('MixAlignLoss'):.4e}"
    )


@torch.no_grad()
def compute_dev_enroll_centroids(model, dev_speaker_index, fbank_fn, sample_rate, enroll_len, device, batch_size=64):
    """One deterministic enrollment utterance per dev speaker (the first path
    build_speaker_index lists) -> the model's predicted enroll-branch
    centroid, batched. Returns {spk: [D] tensor}, recomputed fresh every eval
    epoch. Used only by evaluate_gs_residual below (not the primary enroll
    branch's own val/cos_sim, which uses a random per-epoch utterance draw
    instead): fixing the utterance file keeps this diagnostic's epoch-to-
    epoch trend readable (its crop offset still varies, but that's a much
    smaller noise source than a fresh utterance draw)."""
    num_samples = int(round(enroll_len * sample_rate))
    spks = sorted(dev_speaker_index)
    centroids = {}
    for i in range(0, len(spks), batch_size):
        batch_spks = spks[i:i + batch_size]
        feats = torch.stack([
            fbank_fn(_crop_or_pad(_load_mono(dev_speaker_index[spk][0], sample_rate), num_samples))
            for spk in batch_spks
        ]).to(device)
        pred = model(feats, lengths=None)["centroid"]
        for spk, emb in zip(batch_spks, pred):
            centroids[spk] = emb
    return centroids


@torch.no_grad()
def evaluate_gs_residual(mixture_val_loader, model, dev_enroll_centroids, dev_centroids, device):
    """Monitoring diagnostic: the model's own "gs"/"gs2x" negative-clue
    residual, built from the TRAINED model's predicted centroids (hat_c_mix
    from the mixture branch, hat_c_s1 from the enroll branch). For each dev
    mixture (pair_key "spkA_spkB", see dataset.pair_key), spkA is the
    "enrolled" half and spkB's oracle centroid (not a model prediction) is
    the comparison target -- matching gs's real inference-time use case (no
    oracle access to the interference speaker's own audio).

    Reports both:
      - "gs" (hat_c_mix - hat_c_s1): the actual clue formula used at
        inference (compute_pseudo_interference_centroid).
      - "gs2x" (2*hat_c_mix - hat_c_s1): diagnostic only, a ceiling valid if
        hat_c_mix ~= 0.5*(hat_c_s1 + hat_c_s2) held cleanly (paper Sec. 3.1)
        -- not a better formula.

    Returns (cos_sim_gs, mse_gs, cos_sim_gs2x, mse_gs2x), each averaged over
    every dev mixture in mixture_val_loader."""
    total_n = 0
    cos_gs_sum = mse_gs_sum = cos_gs2x_sum = mse_gs2x_sum = 0.0

    for mix_fbank, _, pair_keys in mixture_val_loader:
        mix_fbank = mix_fbank.to(device)
        hat_c_mix = model(mix_fbank, lengths=None)["centroid"]

        hat_c_s1 = torch.stack([dev_enroll_centroids[pk.split("_")[0]] for pk in pair_keys])
        oracle_s2 = torch.stack([dev_centroids[pk.split("_")[1]] for pk in pair_keys]).to(device)

        pseudo_gs = hat_c_mix - hat_c_s1
        pseudo_gs2x = 2 * hat_c_mix - hat_c_s1

        batch_n = mix_fbank.size(0)
        cos_gs_sum += F.cosine_similarity(pseudo_gs, oracle_s2, dim=-1).sum().item()
        mse_gs_sum += F.mse_loss(pseudo_gs, oracle_s2, reduction="none").mean(dim=-1).sum().item()
        cos_gs2x_sum += F.cosine_similarity(pseudo_gs2x, oracle_s2, dim=-1).sum().item()
        mse_gs2x_sum += F.mse_loss(pseudo_gs2x, oracle_s2, reduction="none").mean(dim=-1).sum().item()
        total_n += batch_n

    return cos_gs_sum / total_n, mse_gs_sum / total_n, cos_gs2x_sum / total_n, mse_gs2x_sum / total_n


@torch.no_grad()
def _evaluate_branch(val_loader, model, reference_fn, device):
    """Shared per-branch pass used by evaluate() for both the enroll branch
    and the mixture branch. Returns (val_loss, cos_sim)."""
    total_n = 0
    cos_sim_sum = 0.0
    loss_sum = 0.0

    for fbank, lengths, keys in val_loader:
        fbank = fbank.to(device)
        lengths = lengths.to(device) if lengths is not None else None
        reference = reference_fn(keys)

        pred = model(fbank, lengths=lengths)["centroid"]
        batch_n = fbank.size(0)

        cos_sim_sum += F.cosine_similarity(pred, reference, dim=-1).sum().item()
        loss_sum += F.mse_loss(pred, reference).item() * batch_n
        total_n += batch_n

    return loss_sum / total_n, cos_sim_sum / total_n


@torch.no_grad()
def evaluate(val_loader, mixture_val_loader, model, train_reference_fn, mixture_reference_fn,
             device, logger, epoch, mix_loss_weight, compute_dev_enroll_centroids_fn, dev_centroids):
    """val_loader/mixture_val_loader yield (fbank, lengths, keys) batches;
    *_reference_fn(keys) -> the reference centroids (speaker centroid for
    enroll, C1+C2 sum for mixture). Computes MSE loss for each branch plus
    cos_sim (monitoring only). The returned val_loss is
    val_loss_enroll + mix_loss_weight * val_loss_mixture, matching what
    training optimizes -- used for best-checkpoint selection.

    Also runs evaluate_gs_residual (logged under val_mixture/gs_*,
    val_mixture/gs2x_*) -- a monitoring diagnostic only, not used for
    best-checkpoint selection.

    Returns (val_loss, cos_sim) -- cos_sim is the enroll branch's only."""
    model.eval()
    val_loss, cos_sim = _evaluate_branch(val_loader, model, train_reference_fn, device)
    logger.info(
        f"[Epoch {epoch}] val[enroll] loss={val_loss:.4e}  "
        f"cos(predicted_centroid, true_centroid)={cos_sim:.4f}"
    )
    wandb.log({"val/cos_sim": cos_sim, "val/loss": val_loss, "epoch": epoch})

    mix_val_loss, mix_cos_sim = _evaluate_branch(mixture_val_loader, model, mixture_reference_fn, device)
    logger.info(
        f"[Epoch {epoch}] val[mixture] loss={mix_val_loss:.4e}  "
        f"cos(predicted_centroid, true_centroid)={mix_cos_sim:.4f}"
    )
    wandb.log({"val_mixture/cos_sim": mix_cos_sim, "val_mixture/loss": mix_val_loss, "epoch": epoch})
    val_loss = val_loss + mix_loss_weight * mix_val_loss

    dev_enroll_centroids = compute_dev_enroll_centroids_fn(model)
    gs_cos, gs_mse, gs2x_cos, gs2x_mse = evaluate_gs_residual(
        mixture_val_loader, model, dev_enroll_centroids, dev_centroids, device,
    )
    logger.info(
        f"[Epoch {epoch}] val[mixture] gs: cos(hat_c_mix - hat_c_s1, oracle_s2)={gs_cos:.4f} "
        f"mse={gs_mse:.4e}  |  gs2x: cos(2*hat_c_mix - hat_c_s1, oracle_s2)={gs2x_cos:.4f} "
        f"mse={gs2x_mse:.4e}"
    )
    wandb.log({
        "val_mixture/gs_cos_sim": gs_cos,
        "val_mixture/gs_mse": gs_mse,
        "val_mixture/gs2x_cos_sim": gs2x_cos,
        "val_mixture/gs2x_mse": gs2x_mse,
        "epoch": epoch,
    })

    return val_loss, cos_sim


if __name__ == "__main__":
    main()
