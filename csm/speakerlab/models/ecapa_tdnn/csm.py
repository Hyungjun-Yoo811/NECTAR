"""ECAPA-TDNN analog of speakerlab.models.campplus.csm: the same
Centroid-Space Mapper (conv + self-attention + attentive statistics
pooling, paper Sec. 3.2) built on a frozen ECAPA-TDNN backbone instead of
CAM++."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from speakerlab.models.campplus.csm import (
    AttentiveStatisticsPooling,
    DepthwiseConvModule,
    SelfAttentionOnlyEncoder,
)

# ECAPA-TDNN's blocks never downsample time (stride=1 'same' padding
# throughout, unlike CAM++'s FCM head which halves T via stride=2), so
# extract_ecapa_frame_features's output T' always equals the input fbank T.
ECAPA_TIME_STRIDE = 1


class ECAPACSM(nn.Module):
    """
    ECAPA-TDNN analog of CAMPlusCSM: the frozen backbone's frame-level
    features (extracted immediately before its own pooling layer) feed a
    Transformer + ASP + Linear head trained from scratch, with no residual
    connection to ECAPA's own embedding.

    Expected ECAPA-TDNN structure (speakerlab.models.ecapa_tdnn.ECAPA_TDNN):
    ecapa.blocks (initial TDNNBlock + SERes2NetBlock * 3), ecapa.mfa
    (multi-layer feature aggregation). extract_ecapa_frame_features stops
    right after `mfa`, before ECAPA's own asp/asp_bn/fc.

    use_transformer: same three modes as CAMPlusCSM (see its docstring).

    Input: fbank [B, T, 80], lengths [B] (valid frame counts).
    Output: centroid [B, embedding_dim].
    """

    def __init__(
        self,
        ecapa: nn.Module,
        ecapa_frame_dim: int = 3072,
        transformer_dim: int = 256,
        embedding_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        asp_hidden_dim: int = 128,
        freeze_ecapa: bool = True,
        ecapa_time_stride: int = ECAPA_TIME_STRIDE,
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

        if not hasattr(ecapa, "blocks"):
            raise ValueError("ECAPA-TDNN module must have a `blocks` attribute.")

        if not hasattr(ecapa, "mfa"):
            raise ValueError("ECAPA-TDNN module must have an `mfa` attribute.")

        self.ecapa = ecapa
        self.freeze_ecapa = freeze_ecapa
        self.ecapa_time_stride = ecapa_time_stride
        self.use_transformer = use_transformer

        if freeze_ecapa:
            for parameter in self.ecapa.parameters():
                parameter.requires_grad = False
            self.ecapa.eval()

        if ecapa_frame_dim == transformer_dim:
            self.input_projection = nn.Identity()
        else:
            self.input_projection = nn.Linear(
                ecapa_frame_dim,
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

        # Default-initialized (not zero-init): no residual base to fall back on.
        self.output_projection = nn.Linear(transformer_dim * 2, embedding_dim)

    def train(self, mode: bool = True) -> "ECAPACSM":
        """Prevent frozen ECAPA-TDNN BatchNorm statistics from being updated
        when the whole estimator is switched to train mode."""
        super().train(mode)

        if self.freeze_ecapa:
            self.ecapa.eval()

        return self

    def extract_ecapa_frame_features(
        self,
        fbank: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract ECAPA-TDNN features immediately after multi-layer feature
        aggregation (`mfa`), right before its own Attentive Statistics
        Pooling. Replays ECAPA_TDNN.forward()'s blocks/mfa loop manually
        since ECAPA_TDNN has no public method that stops there.

        Args:
            fbank: [B, T, F]

        Returns:
            frame_features: [B, C, T'] (T' == T -- see ECAPA_TIME_STRIDE)
        """
        if fbank.ndim != 3:
            raise ValueError(
                f"fbank must be [B, T, F], but got {fbank.shape}"
            )

        context = torch.no_grad() if self.freeze_ecapa else nullcontext()

        with context:
            # Official ECAPA_TDNN.forward: [B, T, F] -> [B, F, T]
            x = fbank.transpose(1, 2)

            xl = []
            for layer in self.ecapa.blocks:
                try:
                    x = layer(x, lengths=None)
                except TypeError:
                    x = layer(x)
                xl.append(x)

            x = torch.cat(xl[1:], dim=1)
            x = self.ecapa.mfa(x)

        return x

    def _make_padding_mask(
        self,
        input_lengths: torch.Tensor,
        output_time: int,
    ) -> torch.Tensor:
        output_lengths = torch.div(
            input_lengths + self.ecapa_time_stride - 1,
            self.ecapa_time_stride,
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
            fbank: [B, T, 80]
            lengths: Valid fbank-frame lengths [B]. Optional.

        Returns:
            Dictionary containing:
                centroid: [B, embedding_dim]
                frame_features: [B, T', transformer_dim]
                attention: [B, T']
                padding_mask: [B, T']
        """
        ecapa_features = self.extract_ecapa_frame_features(fbank)
        # [B, C, T'] -> [B, T', C]
        ecapa_features = ecapa_features.transpose(1, 2)

        x = self.input_projection(ecapa_features)
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

        # Same ordering as CAMPlusCSM: MHA -> frame-wise L2
        # normalization -> ASP.
        x = F.normalize(x, p=2, dim=-1)

        pooled, attention = self.asp(
            x,
            padding_mask=padding_mask,
        )

        centroid_raw = self.output_projection(pooled)
        # No final L2 normalize -- see CAMPlusCSM.forward.
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
    """Same pattern as campplus.csm.ModelCheckpointProxy, but
    excludes `ecapa.*` (the frozen backbone) instead of `campp.*`."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def save(self, path) -> None:
        state = {k: v for k, v in self.model.state_dict().items() if not k.startswith("ecapa.")}
        torch.save(state, path)

    def load(self, path, device=None) -> None:
        state = torch.load(path, map_location=device)
        result = self.model.load_state_dict(state, strict=False)
        unexpected_missing = [k for k in result.missing_keys if not k.startswith("ecapa.")]
        if unexpected_missing or result.unexpected_keys:
            raise RuntimeError(
                "Checkpoint/model mismatch beyond the expected excluded "
                f"ecapa.* keys -- missing={unexpected_missing}, "
                f"unexpected={result.unexpected_keys}"
            )


def load_pretrained_ecapa(
    ckpt_path: str,
    embedding_size: int,
    device: torch.device,
) -> nn.Module:
    """
    Load a frozen, eval-mode ECAPA-TDNN backbone from a checkpoint (e.g. the
    ModelScope iic/speech_ecapa-tdnn_sv_en_voxceleb_16k weights, which use
    embedding_size=192, channels=[1024,1024,1024,1024,3072] rather than the
    speakerlab default channels=[512]*4+[1536]). Mirrors load_pretrained_campplus.
    """
    from speakerlab.models.ecapa_tdnn.ECAPA_TDNN import ECAPA_TDNN

    model = ECAPA_TDNN(
        input_size=80,
        lin_neurons=embedding_size,
        channels=[1024, 1024, 1024, 1024, 3072],
    )
    state = torch.load(os.path.expanduser(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device)
