# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from functools import lru_cache, partial
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
from einops import rearrange, repeat

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ModuleNotFoundError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def fast_rope_apply(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(freqs):
        freqs = torch.angle(freqs)
    batch_size, seq_len, n_heads, head_dim = x.shape
    _ = batch_size

    freqs = freqs.view(seq_len, head_dim // 2)
    cos = torch.cos(freqs).to(torch.float32)
    sin = torch.sin(freqs).to(torch.float32)

    rotated = apply_rotary_emb(
        x.to(torch.float32), cos, sin, interleaved=True, inplace=False
    )

    return rotated.to(x.dtype)


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


if TRITON_AVAILABLE:

    def apply_rotary_emb(
        x,
        cos,
        sin,
        interleaved=False,
        inplace=False,
        seqlen_offsets: Union[int, Tensor] = 0,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        return ApplyRotaryEmb.apply(
            x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen
        )

    apply_rotary_emb_func = apply_rotary_emb

    def _apply_rotary_emb_qkv(
        qkv,
        cos,
        sin,
        cos_k=None,
        sin_k=None,
        interleaved=False,
        inplace=False,
        conjugate=False,
        seqlen_offsets: Union[int, Tensor] = 0,
        num_heads_q: Optional[int] = None,
    ):
        apply_rotary_fn = partial(
            apply_rotary,
            interleaved=interleaved,
            inplace=inplace,
            conjugate=conjugate,
            seqlen_offsets=seqlen_offsets,
        )
        if cos_k is None and sin_k is None and qkv.is_contiguous():
            if qkv.dim() == 5:
                batch, seqlen, three, nheads, headdim = qkv.shape
                assert three == 3
                qk = qkv[:, :, :2].reshape(batch, seqlen, -1, headdim)
                qk = apply_rotary_fn(qk, cos, sin)
            else:
                assert qkv.dim() == 4
                assert num_heads_q is not None
                num_heads_k = (qkv.shape[2] - num_heads_q) // 2
                assert qkv.shape[2] == num_heads_q + 2 * num_heads_k
                qk = qkv[:, :, : num_heads_q + num_heads_k]
                qk = apply_rotary_fn(qk, cos, sin)
            if not inplace:
                if qkv.dim() == 5:
                    qkv = torch.cat(
                        [rearrange(qk, "b s (t h) d -> b s t h d", t=2), qkv[:, :, 2:]],
                        dim=2,
                    )
                else:
                    qkv = torch.cat([qk, qkv[:, :, num_heads_q + num_heads_k :]], dim=2)
        else:
            cos_k = cos if cos_k is None else cos_k
            sin_k = sin if sin_k is None else sin_k
            if qkv.dim() == 5:
                batch, seqlen, three, nheads, headdim = qkv.shape
                assert three == 3
                q, k = qkv[:, :, 0], qkv[:, :, 1]
            else:
                assert qkv.dim() == 4
                assert num_heads_q is not None
                num_heads_k = (qkv.shape[2] - num_heads_q) // 2
                assert qkv.shape[2] == num_heads_q + 2 * num_heads_k
                q, k = (
                    qkv[:, :, :num_heads_q],
                    qkv[:, :, num_heads_q : num_heads_q + num_heads_k],
                )
            q = apply_rotary_fn(q, cos, sin)
            k = apply_rotary_fn(k, cos_k, sin_k)
            if not inplace:
                if qkv.dim() == 5:
                    qkv = torch.stack([q, k, qkv[:, :, 2]], dim=2)
                else:
                    qkv = torch.cat([q, k, qkv[:, :, num_heads_q + num_heads_k :]], dim=2)
        return qkv

    class ApplyRotaryEmb(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            x,
            cos,
            sin,
            interleaved=False,
            inplace=False,
            seqlen_offsets: Union[int, Tensor] = 0,
            cu_seqlens: Optional[Tensor] = None,
            max_seqlen: Optional[int] = None,
        ):
            out = apply_rotary(
                x,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                interleaved=interleaved,
                inplace=inplace,
            )
            if isinstance(seqlen_offsets, int):
                ctx.save_for_backward(cos, sin, cu_seqlens)
                ctx.seqlen_offsets = seqlen_offsets
            else:
                ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
                ctx.seqlen_offsets = None
            ctx.interleaved = interleaved
            ctx.inplace = inplace
            ctx.max_seqlen = max_seqlen
            return out if not inplace else x

        @staticmethod
        def backward(ctx, do):
            seqlen_offsets = ctx.seqlen_offsets
            if seqlen_offsets is None:
                cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
            else:
                cos, sin, cu_seqlens = ctx.saved_tensors
            dx = apply_rotary(
                do,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                cu_seqlens=cu_seqlens,
                max_seqlen=ctx.max_seqlen,
                interleaved=ctx.interleaved,
                inplace=ctx.inplace,
                conjugate=True,
            )
            return dx, None, None, None, None, None, None, None

    def apply_rotary_emb_qkv_(
        qkv,
        cos,
        sin,
        cos_k=None,
        sin_k=None,
        interleaved=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        num_heads_q: Optional[int] = None,
    ):
        return ApplyRotaryEmbQKV_.apply(
            qkv, cos, sin, cos_k, sin_k, interleaved, seqlen_offsets, num_heads_q
        )

    class ApplyRotaryEmbQKV_(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            qkv,
            cos,
            sin,
            cos_k=None,
            sin_k=None,
            interleaved=False,
            seqlen_offsets: Union[int, torch.Tensor] = 0,
            num_heads_q: Optional[int] = None,
        ):
            qkv = _apply_rotary_emb_qkv(
                qkv,
                cos,
                sin,
                cos_k,
                sin_k,
                interleaved=interleaved,
                inplace=True,
                seqlen_offsets=seqlen_offsets,
                num_heads_q=num_heads_q,
            )
            if isinstance(seqlen_offsets, int):
                ctx.save_for_backward(cos, sin, cos_k, sin_k)
                ctx.seqlen_offsets = seqlen_offsets
            else:
                ctx.save_for_backward(cos, sin, cos_k, sin_k, seqlen_offsets)
                ctx.seqlen_offsets = None
            ctx.interleaved = interleaved
            ctx.num_heads_q = num_heads_q
            return qkv

        @staticmethod
        def backward(ctx, dqkv):
            seqlen_offsets = ctx.seqlen_offsets
            if seqlen_offsets is None:
                cos, sin, cos_k, sin_k, seqlen_offsets = ctx.saved_tensors
            else:
                cos, sin, cos_k, sin_k = ctx.saved_tensors
            dqkv = _apply_rotary_emb_qkv(
                dqkv,
                cos,
                sin,
                cos_k,
                sin_k,
                interleaved=ctx.interleaved,
                inplace=True,
                seqlen_offsets=seqlen_offsets,
                num_heads_q=ctx.num_heads_q,
                conjugate=True,
            )
            return dqkv, None, None, None, None, None, None, None

    def apply_rotary_emb_kv_(
        kv,
        cos,
        sin,
        interleaved=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
    ):
        return ApplyRotaryEmbKV_.apply(kv, cos, sin, interleaved, seqlen_offsets)

    class ApplyRotaryEmbKV_(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            kv,
            cos,
            sin,
            interleaved=False,
            seqlen_offsets: Union[int, torch.Tensor] = 0,
        ):
            batch, seqlen, two, nheads, headdim = kv.shape
            assert two == 2
            k = kv[:, :, 0]
            apply_rotary(
                k,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                interleaved=interleaved,
                inplace=True,
            )
            if isinstance(seqlen_offsets, int):
                ctx.save_for_backward(cos, sin)
                ctx.seqlen_offsets = seqlen_offsets
            else:
                ctx.save_for_backward(cos, sin, seqlen_offsets)
                ctx.seqlen_offsets = None
            ctx.interleaved = interleaved
            return kv

        @staticmethod
        def backward(ctx, dkv):
            seqlen_offsets = ctx.seqlen_offsets
            if seqlen_offsets is None:
                cos, sin, seqlen_offsets = ctx.saved_tensors
            else:
                cos, sin = ctx.saved_tensors
            apply_rotary(
                dkv[:, :, 0],
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                interleaved=ctx.interleaved,
                inplace=True,
                conjugate=True,
            )
            return dkv, None, None, None, None

    apply_rotary_emb_kv_ = ApplyRotaryEmbKV_.apply

    class RotaryEmbedding(torch.nn.Module):
        def __init__(
            self,
            dim: int,
            base=10000.0,
            interleaved=False,
            scale_base=None,
            device=None,
        ):
            super().__init__()
            self.dim = dim
            self.base = float(base)
            inv_freq = self._compute_inv_freq(device)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.interleaved = interleaved
            self.scale_base = scale_base
            scale = (
                (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim)
                / (1.4 * dim)
                if scale_base is not None
                else None
            )
            self.register_buffer("scale", scale, persistent=False)
            self._seq_len_cached = 0
            self._cos_cached = None
            self._sin_cached = None
            self._cos_k_cached = None
            self._sin_k_cached = None

        def _compute_inv_freq(self, device=None):
            return 1.0 / (
                self.base
                ** (
                    torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                    / self.dim
                )
            )

        def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
            if (
                seqlen > self._seq_len_cached
                or self._cos_cached is None
                or self._cos_cached.device != device
                or self._cos_cached.dtype != dtype
                or (self.training and self._cos_cached.is_inference())
            ):
                self._seq_len_cached = seqlen
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq
                freqs = torch.outer(t, inv_freq)
                if self.scale is None:
                    self._cos_cached = torch.cos(freqs).to(dtype)
                    self._sin_cached = torch.sin(freqs).to(dtype)
                else:
                    power = (
                        torch.arange(
                            seqlen, dtype=self.scale.dtype, device=self.scale.device
                        )
                        - seqlen // 2
                    ) / self.scale_base
                    scale = self.scale.to(device=power.device) ** rearrange(
                        power, "s -> s 1"
                    )
                    self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                    self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                    self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                    self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

        def forward(
            self,
            qkv: torch.Tensor,
            kv: Optional[torch.Tensor] = None,
            seqlen_offset: Union[int, torch.Tensor] = 0,
            max_seqlen: Optional[int] = None,
            num_heads_q: Optional[int] = None,
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            seqlen = qkv.shape[1]
            if max_seqlen is not None:
                self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
            elif isinstance(seqlen_offset, int):
                self._update_cos_sin_cache(
                    seqlen + seqlen_offset, device=qkv.device, dtype=qkv.dtype
                )
            if kv is None:
                return apply_rotary_emb_qkv_(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    self._cos_k_cached if self.scale is not None else None,
                    self._sin_k_cached if self.scale is not None else None,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                    num_heads_q=num_heads_q,
                )
            q = qkv
            q = apply_rotary_emb_func(
                q,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                inplace=True,
                seqlen_offsets=seqlen_offset,
            )
            kv = apply_rotary_emb_kv_(
                kv,
                self._cos_cached if self.scale is None else self._cos_k_cached,
                self._sin_cached if self.scale is None else self._sin_k_cached,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
            )
            return q, kv

    @triton.jit
    def rotary_kernel(
        OUT,
        X,
        COS,
        SIN,
        CU_SEQLENS,
        SEQLEN_OFFSETS,
        seqlen,
        nheads,
        seqlen_ro,
        stride_out_batch,
        stride_out_seqlen,
        stride_out_nheads,
        stride_out_headdim,
        stride_x_batch,
        stride_x_seqlen,
        stride_x_nheads,
        stride_x_headdim,
        ROTARY_DIM: tl.constexpr,
        IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        INTERLEAVED: tl.constexpr,
        CONJUGATE: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        BLOCK_K: tl.constexpr = triton.next_power_of_2(ROTARY_DIM)
        ROTARY_DIM_HALF = ROTARY_DIM // 2
        pid_head = tl.program_id(axis=0)
        pid_m = tl.program_id(axis=1)
        pid_batch = tl.program_id(axis=2)

        if not IS_VARLEN:
            X = X + pid_batch * stride_x_batch
            OUT = OUT + pid_batch * stride_out_batch
        else:
            start_idx = tl.load(CU_SEQLENS + pid_batch)
            seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
            X = X + start_idx * stride_x_seqlen
            OUT = OUT + start_idx * stride_out_seqlen

        if pid_m * BLOCK_M >= seqlen:
            return

        rh = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        if not IS_SEQLEN_OFFSETS_TENSOR:
            rm_cs = rm + SEQLEN_OFFSETS
        else:
            rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)

        rk_half = tl.arange(0, BLOCK_K // 2)
        COS = COS + (rm_cs[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
        SIN = SIN + (rm_cs[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
        mask_cs = (rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < ROTARY_DIM_HALF)
        cos = tl.load(COS, mask=mask_cs, other=1.0).to(tl.float32)
        sin = tl.load(SIN, mask=mask_cs, other=0.0).to(tl.float32)
        if CONJUGATE:
            sin = -sin

        if not INTERLEAVED:
            X = X + (
                rh[:, None, None] * stride_x_nheads
                + rm[None, :, None] * stride_x_seqlen
                + rk_half[None, None, :] * stride_x_headdim
            )
            OUT = OUT + (
                rh[:, None, None] * stride_out_nheads
                + rm[None, :, None] * stride_out_seqlen
                + rk_half[None, None, :] * stride_out_headdim
            )
            mask = (
                (rh[:, None, None] < nheads)
                & (rm[None, :, None] < seqlen)
                & (rk_half[None, None, :] < ROTARY_DIM_HALF)
            )
            x0 = tl.load(X, mask=mask, other=0.0).to(tl.float32)
            x1 = tl.load(
                X + ROTARY_DIM_HALF * stride_x_headdim,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            o0 = x0 * cos - x1 * sin
            o1 = x0 * sin + x1 * cos
            tl.store(OUT, o0, mask=mask)
            tl.store(OUT + ROTARY_DIM_HALF * stride_out_headdim, o1, mask=mask)
        else:
            rk = tl.arange(0, BLOCK_K)
            X = X + (
                rh[:, None, None] * stride_x_nheads
                + rm[None, :, None] * stride_x_seqlen
                + rk[None, None, :] * stride_x_headdim
            )
            OUT = OUT + (
                rh[:, None, None] * stride_out_nheads
                + rm[None, :, None] * stride_out_seqlen
                + rk[None, None, :] * stride_out_headdim
            )
            mask = (
                (rh[:, None, None] < nheads)
                & (rm[None, :, None] < seqlen)
                & (rk[None, None, :] < ROTARY_DIM)
            )
            x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
            x0, x1 = tl.split(tl.reshape(x, [BLOCK_H, BLOCK_M, BLOCK_K // 2, 2]))
            o0 = x0 * cos - x1 * sin
            o1 = x0 * sin + x1 * cos
            o = tl.reshape(tl.join(o0, o1), [BLOCK_H, BLOCK_M, BLOCK_K])
            tl.store(OUT, o, mask=mask)

    def apply_rotary(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        interleaved=False,
        inplace=False,
        conjugate=False,
    ) -> torch.Tensor:
        is_varlen = cu_seqlens is not None
        if not is_varlen:
            batch, seqlen, nheads, headdim = x.shape
        else:
            assert max_seqlen is not None, (
                "If cu_seqlens is passed in, then max_seqlen must be passed"
            )
            total_seqlen, nheads, headdim = x.shape
            batch_p_1 = cu_seqlens.shape[0]
            batch = batch_p_1 - 1
            seqlen = max_seqlen
        seqlen_ro, rotary_dim = cos.shape
        assert sin.shape == cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
        assert headdim <= 256, "Only support headdim <= 256"
        assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"

        cos, sin = cos.contiguous(), sin.contiguous()
        if isinstance(seqlen_offsets, torch.Tensor):
            assert seqlen_offsets.shape == (batch,)
            assert seqlen_offsets.dtype in [torch.int32, torch.int64]
            seqlen_offsets = seqlen_offsets.contiguous()
        else:
            assert seqlen_offsets + seqlen <= seqlen_ro

        output = torch.empty_like(x) if not inplace else x
        if rotary_dim < headdim and not inplace:
            output[..., rotary_dim:].copy_(x[..., rotary_dim:])

        grid = lambda META: (
            triton.cdiv(nheads, META["BLOCK_H"]),
            triton.cdiv(seqlen, META["BLOCK_M"]),
            batch,
        )
        BLOCK_M = 8 if rotary_dim <= 128 else 4

        with torch.cuda.device(x.device.index):
            torch.library.wrap_triton(rotary_kernel)[grid](
                output,
                x,
                cos,
                sin,
                cu_seqlens,
                seqlen_offsets,
                seqlen,
                nheads,
                seqlen_ro,
                output.stride(0) if not is_varlen else 0,
                output.stride(-3),
                output.stride(-2),
                output.stride(-1),
                x.stride(0) if not is_varlen else 0,
                x.stride(-3),
                x.stride(-2),
                x.stride(-1),
                rotary_dim,
                isinstance(seqlen_offsets, torch.Tensor),
                is_varlen,
                interleaved,
                conjugate,
                BLOCK_M=BLOCK_M,
                BLOCK_H=2,
            )
        return output

else:

    def apply_rotary_emb(
        x,
        cos,
        sin,
        interleaved=False,
        inplace=False,
        seqlen_offsets: Union[int, Tensor] = 0,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        if cu_seqlens is not None:
            raise RuntimeError("Triton is required for varlen rotary embedding.")
        out = x if inplace else x.clone()
        if out.size(-1) > cos.shape[-1] * 2:
            out[..., cos.shape[-1] * 2 :].copy_(x[..., cos.shape[-1] * 2 :])
        if isinstance(seqlen_offsets, int):
            cos = cos[seqlen_offsets : seqlen_offsets + x.shape[1]]
            sin = sin[seqlen_offsets : seqlen_offsets + x.shape[1]]
        else:
            raise RuntimeError("Tensor seqlen offsets require Triton in this port.")
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        rotary_dim = cos.shape[-1] * 2
        out[..., :rotary_dim] = (
            out[..., :rotary_dim].float() * cos
            + rotate_half(out[..., :rotary_dim].float(), interleaved=interleaved) * sin
        ).type_as(out)
        return out

    apply_rotary_emb_func = apply_rotary_emb

    def apply_rotary_emb_qkv_(
        qkv,
        cos,
        sin,
        cos_k=None,
        sin_k=None,
        interleaved=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        num_heads_q: Optional[int] = None,
    ):
        if qkv.dim() == 5:
            q = apply_rotary_emb(
                qkv[:, :, 0], cos, sin, interleaved=interleaved, seqlen_offsets=seqlen_offsets
            )
            k = apply_rotary_emb(
                qkv[:, :, 1],
                cos if cos_k is None else cos_k,
                sin if sin_k is None else sin_k,
                interleaved=interleaved,
                seqlen_offsets=seqlen_offsets,
            )
            return torch.stack([q, k, qkv[:, :, 2]], dim=2)
        raise RuntimeError("MQA/GQA rotary qkv fallback is not implemented without Triton.")

    def apply_rotary_emb_kv_(
        kv,
        cos,
        sin,
        interleaved=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
    ):
        kv = kv.clone()
        kv[:, :, 0] = apply_rotary_emb(
            kv[:, :, 0], cos, sin, interleaved=interleaved, seqlen_offsets=seqlen_offsets
        )
        return kv

    class RotaryEmbedding(torch.nn.Module):
        def __init__(
            self,
            dim: int,
            base=10000.0,
            interleaved=False,
            scale_base=None,
            device=None,
        ):
            super().__init__()
            self.dim = dim
            self.base = float(base)
            inv_freq = self._compute_inv_freq(device)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.interleaved = interleaved
            self.scale_base = scale_base
            scale = (
                (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim)
                / (1.4 * dim)
                if scale_base is not None
                else None
            )
            self.register_buffer("scale", scale, persistent=False)
            self._seq_len_cached = 0
            self._cos_cached = None
            self._sin_cached = None
            self._cos_k_cached = None
            self._sin_k_cached = None

        def _compute_inv_freq(self, device=None):
            return 1.0 / (
                self.base
                ** (
                    torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                    / self.dim
                )
            )

        def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
            if (
                seqlen > self._seq_len_cached
                or self._cos_cached is None
                or self._cos_cached.device != device
                or self._cos_cached.dtype != dtype
            ):
                self._seq_len_cached = seqlen
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                inv_freq = self._compute_inv_freq(device=device)
                freqs = torch.outer(t, inv_freq)
                if self.scale is None:
                    self._cos_cached = torch.cos(freqs).to(dtype)
                    self._sin_cached = torch.sin(freqs).to(dtype)
                else:
                    power = (
                        torch.arange(
                            seqlen, dtype=self.scale.dtype, device=self.scale.device
                        )
                        - seqlen // 2
                    ) / self.scale_base
                    scale = self.scale.to(device=power.device) ** rearrange(
                        power, "s -> s 1"
                    )
                    self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                    self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                    self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                    self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

        def forward(
            self,
            qkv: torch.Tensor,
            kv: Optional[torch.Tensor] = None,
            seqlen_offset: Union[int, torch.Tensor] = 0,
            max_seqlen: Optional[int] = None,
            num_heads_q: Optional[int] = None,
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            seqlen = qkv.shape[1]
            self._update_cos_sin_cache(seqlen if max_seqlen is None else max_seqlen, device=qkv.device, dtype=qkv.dtype)
            if kv is None:
                return apply_rotary_emb_qkv_(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    self._cos_k_cached if self.scale is not None else None,
                    self._sin_k_cached if self.scale is not None else None,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                    num_heads_q=num_heads_q,
                )
            q = apply_rotary_emb(
                qkv,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                inplace=False,
                seqlen_offsets=seqlen_offset,
            )
            kv = apply_rotary_emb_kv_(
                kv,
                self._cos_cached if self.scale is None else self._cos_k_cached,
                self._sin_cached if self.scale is None else self._sin_k_cached,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
            )
            return q, kv

    def apply_rotary(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        interleaved=False,
        inplace=False,
        conjugate=False,
    ) -> torch.Tensor:
        return apply_rotary_emb(
            x,
            cos,
            -sin if conjugate else sin,
            interleaved=interleaved,
            inplace=inplace,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
