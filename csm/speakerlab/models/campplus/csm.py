"""Centroid-Space Mapper (CSM) built on a frozen CAM++ backbone: conv +
self-attention + attentive statistics pooling (paper Sec. 3.2), plus the
speaker/mixture centroid utilities (paper Sec. 3.1) and losses used to
train and evaluate it."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive statistics pooling (ASP): attentive mean + std pooling.

    Input: x [B, T, D], padding_mask [B, T] (True = padded position).
    Output: pooled [B, 2D], attention [B, T].
    """

    def __init__(
        self,
        input_dim: int,
        attention_hidden_dim: int = 128,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1),
        )
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"x must be [B, T, D], but got {x.shape}")

        logits = self.attention(x).squeeze(-1)  # [B, T]

        if padding_mask is not None:
            if padding_mask.shape != logits.shape:
                raise ValueError(
                    f"padding_mask must be {logits.shape}, "
                    f"but got {padding_mask.shape}"
                )

            logits = logits.masked_fill(
                padding_mask,
                torch.finfo(logits.dtype).min,
            )

        weights = torch.softmax(logits, dim=1)  # [B, T]
        weights_expanded = weights.unsqueeze(-1)

        mean = torch.sum(weights_expanded * x, dim=1)

        second_moment = torch.sum(
            weights_expanded * x.square(),
            dim=1,
        )
        variance = (second_moment - mean.square()).clamp_min(self.eps)
        std = torch.sqrt(variance)

        pooled = torch.cat([mean, std], dim=-1)

        return pooled, weights


class DepthwiseConvModule(nn.Module):
    """
    Conformer-style local module (paper's depthwise conv layer), inserted
    into the residual stream between input_norm and the transformer stack:
    LayerNorm -> pointwise conv -> GLU -> depthwise conv -> GroupNorm(1
    group) -> SiLU -> pointwise conv -> dropout, added back onto the input.

    Uses GroupNorm(num_groups=1) rather than the original Conformer's
    BatchNorm1d: it normalizes per-sample, so behavior doesn't depend on
    batch composition or size (needed since eval batches can be as small
    as 1). Operates on [B, T, D] (channels-last).
    """

    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd for symmetric 'same' padding, got {kernel_size}")

        self.norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim,
        )
        self.conv_norm = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [B, T, D]. padding_mask: [B, T], True = padded -- zeroed out
        before the conv so padding can't leak into valid frames through the
        depthwise conv's receptive field."""
        residual = x
        x = self.norm(x)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        x = x.transpose(1, 2)  # [B, D, T]
        x = self.pointwise_conv1(x)  # [B, 2D, T]
        x = F.glu(x, dim=1)  # [B, D, T]
        x = self.depthwise_conv(x)
        x = self.conv_norm(x)
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)  # [B, T, D]

        return residual + self.dropout(x)


class SelfAttentionOnlyEncoderLayer(nn.Module):
    """Bare self-attention block, pre-norm residual: only the token-mixing
    half of nn.TransformerEncoderLayer (x = x + MultiheadAttention(LayerNorm(x))),
    with the per-frame feed-forward sublayer dropped. See CAMPlusCSM's
    use_transformer='attn_only' option."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=src_key_padding_mask, need_weights=False)
        return residual + self.dropout(attn_out)


class SelfAttentionOnlyEncoder(nn.Module):
    """Stack of SelfAttentionOnlyEncoderLayer + a final LayerNorm. Mirrors
    nn.TransformerEncoder's call signature so CAMPlusCSM.forward can use
    either interchangeably."""

    def __init__(self, d_model: int, nhead: int, num_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            SelfAttentionOnlyEncoderLayer(d_model, nhead, dropout=dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return self.norm(x)


class CAMPlusCSM(nn.Module):
    """
    CSM (Centroid-Space Mapper): a depthwise conv layer + self-attention
    encoder + attentive statistics pooling over a frozen CAM++ backbone's
    frame-level features, projected to a centroid estimate (paper Sec. 3.2).
    The centroid is the branch's own projection output, with no residual
    connection to CAM++'s own StatsPool+Dense embedding.

    use_transformer selects the encoder between conv_module and ASP:
      - True (default): full nn.TransformerEncoderLayer stack (self-
        attention + per-frame feed-forward each layer).
      - 'attn_only': SelfAttentionOnlyEncoder -- same self-attention,
        without the feed-forward sublayer.
      - False: no attention encoder; requires use_conv=True so some
        temporal-mixing encoder remains.

    Expected CAM++ structure: campp.head, campp.xvector (tdnn, block*,
    transit*, out_nonlinear, stats, dense).

    Input: fbank [B, T, 80], lengths [B] (valid frame counts).
    Output: centroid [B, embedding_dim].
    """

    def __init__(
        self,
        campp: nn.Module,
        campp_frame_dim: int = 512,
        transformer_dim: int = 256,
        embedding_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        asp_hidden_dim: int = 128,
        freeze_campp: bool = True,
        campp_time_stride: int = 2,
        use_conv: bool = False,
        conv_kernel_size: int = 3,
        use_transformer: bool | str = True,
    ) -> None:
        super().__init__()

        if use_transformer not in (True, False, "attn_only"):
            raise ValueError(f"use_transformer must be True, False, or 'attn_only', got {use_transformer!r}")
        if use_transformer is False and not use_conv:
            raise ValueError(
                "use_transformer=False needs use_conv=True -- with both off there is no "
                "temporal-mixing encoder left (input_projection alone is frame-independent)."
            )

        if not hasattr(campp, "head"):
            raise ValueError("CAM++ module must have a `head` attribute.")

        if not hasattr(campp, "xvector"):
            raise ValueError("CAM++ module must have an `xvector` attribute.")

        self.campp = campp
        self.freeze_campp = freeze_campp
        self.campp_time_stride = campp_time_stride
        self.use_transformer = use_transformer

        if freeze_campp:
            for parameter in self.campp.parameters():
                parameter.requires_grad = False
            self.campp.eval()

        if campp_frame_dim == transformer_dim:
            self.input_projection = nn.Identity()
        else:
            self.input_projection = nn.Linear(
                campp_frame_dim,
                transformer_dim,
            )

        self.input_norm = nn.LayerNorm(transformer_dim)

        self.conv_module = (
            DepthwiseConvModule(transformer_dim, kernel_size=conv_kernel_size, dropout=dropout)
            if use_conv else None
        )

        if use_transformer is True:
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=transformer_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )

            self.transformer = nn.TransformerEncoder(
                transformer_layer,
                num_layers=num_layers,
                norm=nn.LayerNorm(transformer_dim),
            )
        elif use_transformer == "attn_only":
            self.transformer = SelfAttentionOnlyEncoder(
                d_model=transformer_dim, nhead=num_heads, num_layers=num_layers, dropout=dropout,
            )
        else:
            self.transformer = None

        self.asp = AttentiveStatisticsPooling(
            input_dim=transformer_dim,
            attention_hidden_dim=asp_hidden_dim,
        )

        # Default-initialized (not zero-init): with no residual base, a
        # zero-init would make the centroid identically zero.
        self.output_projection = nn.Linear(transformer_dim * 2, embedding_dim)

    def train(self, mode: bool = True) -> "CAMPlusCSM":
        """
        Prevent frozen CAM++ BatchNorm statistics from being updated when
        the whole estimator is switched to train mode.
        """
        super().train(mode)

        if self.freeze_campp:
            self.campp.eval()

        return self

    def extract_campp_frame_features(
        self,
        fbank: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract CAM++ features immediately before StatsPool.

        Args:
            fbank: [B, T, F]

        Returns:
            frame_features: [B, C, T']
        """
        if fbank.ndim != 3:
            raise ValueError(
                f"fbank must be [B, T, F], but got {fbank.shape}"
            )

        context = torch.no_grad() if self.freeze_campp else nullcontext()

        with context:
            x = fbank.transpose(1, 2)  # [B, T, F] -> [B, F, T]
            x = self.campp.head(x)

            # Stop before CAM++'s own pooling layer.
            for name, module in self.campp.xvector.named_children():
                if name in {"stats", "dense"}:
                    break
                x = module(x)

        return x

    def _make_padding_mask(
        self,
        input_lengths: torch.Tensor,
        output_time: int,
    ) -> torch.Tensor:
        """
        Official CAM++ uses temporal stride 2 in the initial TDNN layer.
        """
        output_lengths = torch.div(
            input_lengths + self.campp_time_stride - 1,
            self.campp_time_stride,
            rounding_mode="floor",
        )

        output_lengths = output_lengths.clamp(
            min=1,
            max=output_time,
        )

        time_index = torch.arange(
            output_time,
            device=input_lengths.device,
        ).unsqueeze(0)

        return time_index >= output_lengths.unsqueeze(1)

    def forward(
        self,
        fbank: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            fbank:
                [B, T, 80]
            lengths:
                Valid fbank-frame lengths [B]. Optional.

        Returns:
            Dictionary containing:
                centroid: [B, embedding_dim]
                frame_features: [B, T', transformer_dim]
                attention: [B, T']
                padding_mask: [B, T']
        """
        cam_features = self.extract_campp_frame_features(fbank)
        # [B, C, T'] -> [B, T', C]
        cam_features = cam_features.transpose(1, 2)

        x = self.input_projection(cam_features)
        x = self.input_norm(x)

        padding_mask = None
        if lengths is not None:
            padding_mask = self._make_padding_mask(
                input_lengths=lengths,
                output_time=x.size(1),
            )

        if self.conv_module is not None:
            x = self.conv_module(x, padding_mask=padding_mask)

        if self.transformer is not None:
            x = self.transformer(
                x,
                src_key_padding_mask=padding_mask,
            )

        # Self-attention (or conv) -> frame-wise L2 normalization -> ASP.
        x = F.normalize(x, p=2, dim=-1)

        pooled, attention = self.asp(
            x,
            padding_mask=padding_mask,
        )

        centroid_raw = self.output_projection(pooled)
        # No final L2 normalize: keeps magnitude for the MSE loss against
        # non-normalized reference centroids. centroid == centroid_raw here;
        # both keys are kept for downstream compatibility.
        centroid = centroid_raw

        output: Dict[str, torch.Tensor] = {
            "centroid": centroid,
            "centroid_raw": centroid_raw,
            "frame_features": x,
            "attention": attention,
        }

        if padding_mask is not None:
            output["padding_mask"] = padding_mask

        return output
    
    

class ModelCheckpointProxy:
    """
    Wraps a CAMPlusCSM for speakerlab.utils.checkpoint.Checkpointer, saving/
    loading everything except the frozen `campp.*` backbone -- it never
    changes during training and is always reloaded fresh from
    config.campplus["ckpt_path"] at model-construction time.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def save(self, path) -> None:
        state = {k: v for k, v in self.model.state_dict().items() if not k.startswith("campp.")}
        torch.save(state, path)

    def load(self, path, device=None) -> None:
        state = torch.load(path, map_location=device)
        result = self.model.load_state_dict(state, strict=False)
        unexpected_missing = [k for k in result.missing_keys if not k.startswith("campp.")]
        if unexpected_missing or result.unexpected_keys:
            raise RuntimeError(
                "Checkpoint/model mismatch beyond the expected excluded "
                f"campp.* keys -- missing={unexpected_missing}, "
                f"unexpected={result.unexpected_keys}"
            )


def load_pretrained_campplus(
    ckpt_path: str,
    embedding_size: int,
    device: torch.device,
) -> nn.Module:
    """
    Load a frozen, eval-mode CAM++ backbone from a checkpoint (e.g. the
    ModelScope iic/speech_campplus_sv_zh_en_16k-common_advanced weights,
    which use embedding_size=192).
    """
    from speakerlab.models.campplus.DTDNN import CAMPPlus

    model = CAMPPlus(feat_dim=80, embedding_size=embedding_size)
    state = torch.load(os.path.expanduser(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device)


@torch.no_grad()
def compute_speaker_centroid(
    campp: nn.Module,
    utterance_fbanks: list[torch.Tensor],
    device: torch.device,
    final_normalize: bool = False,
) -> torch.Tensor:
    """
    Speaker centroid: mean of per-utterance L2-normalized CAM++ embeddings
    (paper Sec. 3.1). Each embedding is unit-norm, so the mean's magnitude
    in [0, 1] reflects intra-speaker consistency.

    final_normalize=False (default) keeps that magnitude, so the centroid
    can serve as a magnitude-aware MSE target. Set True to re-normalize the
    mean back to unit norm.

    utterance_fbanks: list of [T_n, 80] tensors.
    Returns: centroid [embedding_dim].
    """
    campp.eval()
    embeddings = []

    for fbank in utterance_fbanks:
        fbank = fbank.unsqueeze(0).to(device)  # [1, T, 80]

        embedding = campp(fbank)               # [1, embedding_dim]
        embedding = F.normalize(
            embedding,
            p=2,
            dim=-1,
        )

        embeddings.append(embedding.squeeze(0).cpu())

    embeddings = torch.stack(embeddings, dim=0)
    centroid = embeddings.mean(dim=0)
    if final_normalize:
        centroid = F.normalize(centroid, p=2, dim=0)

    return centroid



def compute_pseudo_interference_centroid(
    mix_centroid: torch.Tensor,
    enroll_centroid: torch.Tensor,
) -> torch.Tensor:
    """
    Interference-speaker centroid via superposition subtraction: assuming
    mix_centroid ~= target_centroid + interference_centroid (paper Sec. 3.2),
    the interference estimate is mix_centroid - enroll_centroid. Used since
    no oracle interference enrollment is available at inference time.

    Magnitude-preserving (no normalization), so pass the non-normalized
    `centroid_raw` (== `centroid`) outputs for both arguments. Mirrors
    BSRNN/modules/spkadapt.py::gs_invert_batch, kept as a local copy so this
    module has no cross-project dependency.

    Args:
        mix_centroid: [B, D] centroid estimate from the mixture.
        enroll_centroid: [B, D] centroid estimate from the target enrollment.

    Returns:
        pseudo_interference_centroid: [B, D]
    """
    return mix_centroid - enroll_centroid


class InBatchCentroidLoss(nn.Module):
    def __init__(self, initial_scale: float = 20.0) -> None:
        super().__init__()

        self.log_scale = nn.Parameter(
            torch.tensor(initial_scale).log()
        )
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        prediction: torch.Tensor,
        reference_centroids: torch.Tensor,
    ) -> torch.Tensor:
        """
        prediction[i] and reference_centroids[i] must belong
        to the same speaker.

        Each batch item should ideally have a different speaker.
        """
        prediction = F.normalize(prediction, dim=-1)
        reference_centroids = F.normalize(
            reference_centroids,
            dim=-1,
        )

        scale = self.log_scale.exp().clamp(max=100.0)

        logits = (
            scale * prediction @ reference_centroids.transpose(0, 1)
            + self.bias
        )

        labels = torch.arange(
            prediction.size(0),
            device=prediction.device,
        )

        return F.cross_entropy(logits, labels)