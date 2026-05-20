# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def fast_rope_apply(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply RoPE using the same public behavior as the fast-copy tree."""
    if torch.is_complex(freqs):
        freqs = torch.angle(freqs)

    squeeze = False
    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze = True

    batch_size, seq_len, n_heads, head_dim = x.shape
    _ = batch_size

    freqs = freqs.view(seq_len, head_dim // 2)
    cos = torch.cos(freqs).to(torch.float32)
    sin = torch.sin(freqs).to(torch.float32)
    cos = rearrange(cos, "n d -> 1 1 n d")
    sin = rearrange(sin, "n d -> 1 1 n d")

    rotated = (x.to(torch.float32) * cos) + (rotate_half(x.to(torch.float32)) * sin)
    rotated = rotated.to(x.dtype)

    if squeeze:
        rotated = rotated.squeeze(0)
    return rotated


def apply_rotary_complex(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    seq_len, num_heads, head_dim = x.shape
    x_complex = torch.view_as_complex(
        x.to(torch.float64).reshape(seq_len, num_heads, -1, 2)
    )
    rotated = torch.view_as_real(x_complex * freqs).flatten(2)
    return rotated.float()


class RotaryPositionalEmbedding1D(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.base = 10000

    @lru_cache(maxsize=32)
    def precompute_freqs_cis_1d(self, pos_indices: torch.Tensor) -> torch.Tensor:
        freqs = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.head_dim, 2)[: (self.head_dim // 2)].float()
                / self.head_dim
            )
        )
        freqs = freqs.to(pos_indices.device)
        freqs = torch.einsum("..., f -> ... f", pos_indices.float(), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        return freqs

    def forward(self, x: torch.Tensor, pos_indices: torch.Tensor) -> torch.Tensor:
        """1D RoPE.

        Args:
            x: [B, head, seq, head_dim]
            pos_indices: [seq,]
        Returns:
            query with the same shape as input.
        """
        freqs_cis = self.precompute_freqs_cis_1d(pos_indices)
        x_ = x.float()
        freqs_cis = freqs_cis.float().to(x.device)
        cos, sin = freqs_cis.cos(), freqs_cis.sin()
        cos, sin = rearrange(cos, "n d -> 1 1 n d"), rearrange(sin, "n d -> 1 1 n d")
        x_ = (x_ * cos) + (rotate_half(x_) * sin)
        return x_.type_as(x)


class VideoRopePosition3DEmb(nn.Module):
    def __init__(
        self,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
    ):
        super().__init__()
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, (
            f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"
        )
        self._dim_h = dim_h
        self._dim_t = dim_t

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))

        self._is_initialized = False

    def cache_parameters(self) -> None:
        if self._is_initialized:
            return

        dim_h = self._dim_h
        dim_t = self._dim_t

        self.seq = torch.arange(max(self.max_h, self.max_w, self.max_t)).float().cuda()
        self.dim_spatial_range = (
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float().cuda() / dim_h
        )
        self.dim_temporal_range = (
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float().cuda() / dim_t
        )
        self._is_initialized = True

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ):
        self.cache_parameters()

        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta**self.dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta**self.dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta**self.dim_temporal_range)

        B, T, H, W, _ = B_T_H_W_C
        assert H <= self.max_h and W <= self.max_w, (
            f"Input dimensions (H={H}, W={W}) exceed the maximum dimensions "
            f"(max_h={self.max_h}, max_w={self.max_w})"
        )
        freqs_h = torch.outer(self.seq[:H], h_spatial_freqs)
        freqs_w = torch.outer(self.seq[:W], w_spatial_freqs)
        freqs_t = torch.outer(self.seq[:T], temporal_freqs)

        freqs_T_H_W_D = torch.cat(
            [
                repeat(freqs_t, "t d -> t h w d", h=H, w=W),
                repeat(freqs_h, "h d -> t h w d", t=T, w=W),
                repeat(freqs_w, "w d -> t h w d", t=T, h=H),
            ],
            dim=-1,
        )

        return rearrange(freqs_T_H_W_D, "t h w d -> (t h w) d").float()

    @property
    def seq_dim(self):
        return 0
