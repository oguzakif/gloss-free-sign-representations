# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from xformers.components.attention import (
    Attention,
    AttentionConfig,
    AttentionMask,
    maybe_sparsify,
    register_attention,
    sparsify,
)
from xformers.components.attention.attention_patterns import (
    causal_1d_pattern,
    local_1d_pattern,
)
from xformers.components.attention.core import scaled_dot_product_attention
import xformers.ops as xops


@dataclass
class LocalMaskAttentionConfig(AttentionConfig):
    causal: Optional[bool] = None
    window_size: Optional[int] = None
    force_sparsity: Optional[bool] = None


@register_attention("local_mask", LocalMaskAttentionConfig)
class LocalMaskAttention(Attention):
    def __init__(
        self,
        dropout: float = 0.0,
        causal: bool = False,
        window_size: int = 5,
        force_sparsity: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.attn_drop = dropout
        self.causal = causal
        self.force_sparsity = force_sparsity

        if not self.causal:
            assert (
                window_size % 2 == 1
            ), "The window size is assumed to be odd (counts self-attention + 2 wings)"

        self.window_size = window_size
        self.attention_mask: Optional[torch.Tensor] = None
        self.requires_same_k_q_dimensions = True

        self.supports_attention_mask = True
        self.supports_key_padding_mask = False

    def _get_local_mask(self, shape: torch.Size) -> torch.Tensor:
        window_size = self.window_size * 2 + 1 if self.causal else self.window_size
        mask = local_1d_pattern(shape[1], window_size)

        if self.causal:
            mask &= causal_1d_pattern(shape[1])

        return mask

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Optional[Union[torch.Tensor, AttentionMask]] = None,
        *args,
        **kwargs,
    ):
        if self.attention_mask is None or self.attention_mask.shape[1] != q.shape[1]:
            self.attention_mask = self._get_local_mask(q.shape).to(q.device)

        if att_mask is None:
            mask = self.attention_mask
        else:
            if isinstance(att_mask, AttentionMask):
                att_mask = att_mask.to_bool()
            heads = v.shape[0] // att_mask.shape[0]
            am = (
                att_mask.unsqueeze(1)
                .expand(att_mask.shape[0], heads, att_mask.shape[-1])
                .reshape(-1, att_mask.shape[1])
            )
            mask = self.attention_mask.unsqueeze(0) & am.unsqueeze(
                1
            )
            mask = mask | (
                torch.eye(mask.shape[1], device=mask.device).unsqueeze(0) == 1
            )

            mask = (~mask).type(q.dtype).masked_fill(~mask, torch.finfo(q.dtype).min)

        # Prefer xFormers fast attention, but fall back to PyTorch SDPA if unavailable
        p = self.attn_drop if self.training else 0.0
        try:
            return xops.memory_efficient_attention(q, k, v, attn_bias=mask, p=p)
        except Exception:
            # Fallback: use torch.nn.functional.scaled_dot_product_attention
            # Accept both [BH, L, D] and [B, L, H, D] formats
            attn_bias = mask  # already additive 0 / -inf

            if q.dim() == 3:
                # [BH, L, D]
                q_bh, k_bh, v_bh = q, k, v
                attn_mask_bh = attn_bias.contiguous() if attn_bias is not None else None
                out_bh = F.scaled_dot_product_attention(
                    q_bh, k_bh, v_bh, attn_mask=attn_mask_bh, dropout_p=p, is_causal=False
                )
                return out_bh

            elif q.dim() == 4:
                # [B, L, H, D] -> [BH, L, D]
                B, Lq, H, D = q.shape
                Lk = k.shape[1]
                q_bh = q.permute(0, 2, 1, 3).reshape(B * H, Lq, D)
                k_bh = k.permute(0, 2, 1, 3).reshape(B * H, Lk, D)
                v_bh = v.permute(0, 2, 1, 3).reshape(B * H, Lk, D)

                if attn_bias is None:
                    attn_mask_bh = None
                else:
                    # Ensure broadcast to [BH, Lq, Lk]
                    if attn_bias.dim() == 2:
                        attn_mask_bh = attn_bias.unsqueeze(0).expand(B * H, -1, -1).contiguous()
                    elif attn_bias.dim() == 3:
                        attn_mask_bh = (
                            attn_bias.expand(B * H, -1, -1).contiguous()
                            if attn_bias.shape[0] == 1
                            else attn_bias.contiguous()
                        )
                    else:
                        attn_mask_bh = None

                out_bh = F.scaled_dot_product_attention(
                    q_bh, k_bh, v_bh, attn_mask=attn_mask_bh, dropout_p=p, is_causal=False
                )
                out = out_bh.reshape(B, H, Lq, D).permute(0, 2, 1, 3).contiguous()
                return out

            else:
                raise ValueError(f"Unsupported q.dim()={q.dim()} in LocalMaskAttention fallback")
