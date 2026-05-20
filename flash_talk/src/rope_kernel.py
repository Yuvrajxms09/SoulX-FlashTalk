# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch


def apply_rotary_complex(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply complex rotary embeddings to a single padded sequence tensor.

    Args:
        x: Tensor shaped [seq_len, num_heads, head_dim].
        freqs: Complex tensor shaped [seq_len, 1, head_dim // 2].
    """
    seq_len, num_heads, head_dim = x.shape
    x_complex = torch.view_as_complex(
        x.to(torch.float64).reshape(seq_len, num_heads, -1, 2)
    )
    rotated = torch.view_as_real(x_complex * freqs).flatten(2)
    return rotated.float()
