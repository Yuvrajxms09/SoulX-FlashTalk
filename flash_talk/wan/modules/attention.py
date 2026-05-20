# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import warnings

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

__all__ = ["flash_attention", "attention"]


def _fast_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    _b, l_k, _n_k, _d = k.shape

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    q = half(q)
    k = half(k)
    v = half(v)

    q = q.to(v.dtype)
    k = k.to(v.dtype)
    if l_k < 512:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return scaled_dot_product_attention(
                q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
            ).permute(0, 2, 1, 3)
    return sageattn(q=q, k=k, v=v, tensor_layout="NHD", output_dtype=dtype)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    Legacy wrapper around the fast-copy attention kernel.

    The fast path matches `fast_flashtalk/kernels/attn.py`.
    The varlen path is kept for the legacy multi-human / CP call sites.
    """
    if q_lens is None and k_lens is None:
        if not causal and dropout_p == 0.0 and version is None:
            return _fast_attention(q, k, v, dtype=dtype)

        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes
        assert q.device.type == "cuda" and q.size(-1) <= 256
        out_dtype = q.dtype

        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)

        q = half(q)
        k = half(k)
        v = half(v)
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        x = scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            attn_mask=None,
            is_causal=causal,
            dropout_p=dropout_p,
        ).permute(0, 2, 1, 3)
        return x.type(out_dtype)

    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn("Flash attention 3 is not available, use flash attention 2 instead.")

    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32
            ).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32
            ).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )
        try:
            x = x.unflatten(0, (b, lq))
        except Exception:
            x = x[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32
            ).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32
            ).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))

    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    dtype=torch.bfloat16,
):
    return _fast_attention(q, k, v, dtype=dtype)
