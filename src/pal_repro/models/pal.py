"""Projection-Free Anchor Learning (PAL) core module.

This implements the paper's strict anchor-only path:

1. frozen image/text token embeddings are L2-normalized;
2. modality-specific learnable anchors are L2-normalized;
3. token-to-anchor cosine similarities form token-level relative representations;
4. Cross-Attention Pooling (CAP) applies an anchor-wise softmax over tokens;
5. pooled image/text profiles are L2-normalized for contrastive learning.

No projection heads, routers, fixed anchor banks, or auxiliary trainable modules
are introduced. The trainable parameter boundary is exactly two anchor matrices.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PALOutput:
    """Container returned by :class:`ProjectionFreeAnchorLearning`.

    Attributes:
        image: L2-normalized image relative profiles shaped ``(B, K)``.
        text: L2-normalized text relative profiles shaped ``(B, K)``.
        image_token_sims: Optional image token-to-anchor similarities shaped
            ``(B, T_v, K)``.
        text_token_sims: Optional text token-to-anchor similarities shaped
            ``(B, T_l, K)``.
    """

    image: torch.Tensor
    text: torch.Tensor
    image_token_sims: torch.Tensor | None = None
    text_token_sims: torch.Tensor | None = None


class ProjectionFreeAnchorLearning(nn.Module):
    """Strict PAL module with anchor-only trainable parameters.

    Args:
        dim_img: Last dimension of frozen image token embeddings.
        dim_txt: Last dimension of frozen text token embeddings.
        num_anchors: Number of modality-specific anchors ``K``.
        pool_temperature: CAP temperature ``tau_p`` used in
            ``softmax(similarity / tau_p, dim=tokens)``.
        init_std: Standard deviation for Gaussian anchor initialization.
    """

    def __init__(
        self,
        dim_img: int,
        dim_txt: int,
        num_anchors: int = 512,
        pool_temperature: float = 0.03,
        pooling_mode: str = "cap",
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if dim_img <= 0 or dim_txt <= 0:
            raise ValueError("Encoder dimensions must be positive.")
        if num_anchors <= 0:
            raise ValueError("num_anchors must be positive.")
        if pool_temperature <= 0:
            raise ValueError("pool_temperature must be positive.")
        if init_std <= 0:
            raise ValueError("init_std must be positive.")
        if pooling_mode not in {"cap", "mean", "global"}:
            raise ValueError("pooling_mode must be one of: cap, mean, global.")

        self.dim_img = int(dim_img)
        self.dim_txt = int(dim_txt)
        self.num_anchors = int(num_anchors)
        self.pool_temperature = float(pool_temperature)
        self.pooling_mode = pooling_mode
        self.init_std = float(init_std)

        self.anchors_img = nn.Parameter(torch.empty(self.num_anchors, self.dim_img))
        self.anchors_txt = nn.Parameter(torch.empty(self.num_anchors, self.dim_txt))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize anchor banks with small Gaussian noise."""

        nn.init.normal_(self.anchors_img, mean=0.0, std=self.init_std)
        nn.init.normal_(self.anchors_txt, mean=0.0, std=self.init_std)

    @staticmethod
    def _validate_tokens(name: str, tokens: torch.Tensor, dim: int) -> None:
        if tokens.dim() != 3:
            raise ValueError(f"{name} must be a 3D tensor shaped (B, T, D).")
        if tokens.shape[0] <= 0 or tokens.shape[1] <= 0:
            raise ValueError(f"{name} must have non-empty batch and token dimensions.")
        if tokens.shape[-1] != dim:
            raise ValueError(
                f"{name} last dimension must be {dim}, got {tokens.shape[-1]}."
            )

    @staticmethod
    def _validate_mask(mask: torch.Tensor, token_shape: torch.Size) -> torch.Tensor:
        if mask.shape != token_shape[:2]:
            raise ValueError(
                f"txt_mask must be shaped {tuple(token_shape[:2])}, got {tuple(mask.shape)}."
            )
        valid = mask.bool()
        if not valid.any(dim=1).all():
            raise ValueError("Each sample must have at least one valid token.")
        return valid

    def _cap_profile(
        self,
        tokens: torch.Tensor,
        anchors: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute CAP-pooled relative profile for one modality.

        Args:
            tokens: Frozen token embeddings shaped ``(B, T, D)``.
            anchors: Learnable anchor matrix shaped ``(K, D)``.
            mask: Optional valid-token mask shaped ``(B, T)``.

        Returns:
            ``(profile, token_sims)`` where profile is L2-normalized ``(B,K)``
            and token_sims is raw cosine similarity ``(B,T,K)``.
        """

        normalized_tokens = F.normalize(tokens, dim=-1)
        normalized_anchors = F.normalize(anchors, dim=-1)
        token_sims = normalized_tokens @ normalized_anchors.T

        logits = token_sims / self.pool_temperature
        if mask is not None:
            valid = self._validate_mask(mask, tokens.shape)
            logits = logits.masked_fill(~valid.unsqueeze(-1), float("-inf"))

        if self.pooling_mode == "cap":
            # Anchor-wise softmax over token positions: alpha_{t,k}.
            token_attention = F.softmax(logits, dim=1)
            raw_profile = (token_attention * token_sims).sum(dim=1)
        elif self.pooling_mode == "mean":
            if mask is None:
                raw_profile = token_sims.mean(dim=1)
            else:
                valid = self._validate_mask(mask, tokens.shape)
                weights = valid.to(dtype=token_sims.dtype, device=token_sims.device).unsqueeze(-1)
                raw_profile = (token_sims * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        elif self.pooling_mode == "global":
            if mask is not None:
                valid = self._validate_mask(mask, tokens.shape)
                if not valid[:, 0].all():
                    raise ValueError("global pooling requires the first token to be valid for every sample.")
            raw_profile = token_sims[:, 0, :]
        else:  # defensive; __init__ validates this branch is unreachable.
            raise RuntimeError(f"Unknown pooling_mode: {self.pooling_mode}")
        profile = F.normalize(raw_profile, dim=-1)
        return profile, token_sims

    def encode_image(
        self,
        img_tokens: torch.Tensor,
        return_token_sims: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode image tokens into PAL anchor-relative profiles.

        This modality-specific entry point is required for paper evaluations
        where the number of images differs from the number of class prompts.
        """

        self._validate_tokens("img_tokens", img_tokens, self.dim_img)
        profile, token_sims = self._cap_profile(img_tokens, self.anchors_img)
        if return_token_sims:
            return profile, token_sims
        return profile

    def encode_text(
        self,
        txt_tokens: torch.Tensor,
        txt_mask: torch.Tensor | None = None,
        return_token_sims: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode text tokens into PAL anchor-relative profiles."""

        self._validate_tokens("txt_tokens", txt_tokens, self.dim_txt)
        profile, token_sims = self._cap_profile(txt_tokens, self.anchors_txt, mask=txt_mask)
        if return_token_sims:
            return profile, token_sims
        return profile

    def forward(
        self,
        img_tokens: torch.Tensor,
        txt_tokens: torch.Tensor,
        txt_mask: torch.Tensor | None = None,
        return_token_sims: bool = False,
    ) -> PALOutput:
        """Return image/text PAL profiles for a paired batch.

        Args:
            img_tokens: Image patch/CLS token tensor shaped ``(B,Tv,Dv)``.
            txt_tokens: Text token tensor shaped ``(B,Tl,Dl)``.
            txt_mask: Optional text attention mask shaped ``(B,Tl)``. Passing
                the tokenizer attention mask is recommended for padded batches.
            return_token_sims: Whether to include raw token-to-anchor matrices.

        Returns:
            :class:`PALOutput` with normalized ``(B,K)`` image/text profiles.
        """

        if img_tokens.shape[0] != txt_tokens.shape[0]:
            raise ValueError(
                "img_tokens and txt_tokens must have the same batch size, "
                f"got {img_tokens.shape[0]} and {txt_tokens.shape[0]}."
            )

        if return_token_sims:
            image, image_token_sims = self.encode_image(img_tokens, return_token_sims=True)
            text, text_token_sims = self.encode_text(txt_tokens, txt_mask, return_token_sims=True)
            return PALOutput(
                image=image,
                text=text,
                image_token_sims=image_token_sims,
                text_token_sims=text_token_sims,
            )

        image = self.encode_image(img_tokens)
        text = self.encode_text(txt_tokens, txt_mask)
        return PALOutput(image=image, text=text)


def pal_trainable_parameter_names(model: nn.Module) -> list[str]:
    """Return names of trainable parameters for PAL boundary checks."""

    return [name for name, param in model.named_parameters() if param.requires_grad]
