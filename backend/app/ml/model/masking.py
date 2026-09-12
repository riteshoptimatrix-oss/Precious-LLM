"""
Precious Edu LLM — Causal Attention Masking

Implements lower-triangular causal masking to prevent future token leakage during
self-attention computation.
"""

import torch


def create_causal_mask(
    seq_len: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate a lower-triangular causal mask tensor.

    Tokens at position i can only attend to positions j <= i.
    Positions where j > i are set to -inf (or -1e9), which results in 0.0 attention
    weight after softmax.

    Args:
        seq_len: Current sequence length (T).
        device: PyTorch device (CPU/CUDA).
        dtype: PyTorch float dtype.

    Returns:
        Tensor of shape [1, 1, seq_len, seq_len] containing 0.0 for visible positions
        and float('-inf') for masked future positions.
    """
    # Create upper triangular mask of ones above main diagonal
    # tril has 1s on and below diagonal, 0s above
    tril = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
    
    # Mask with 0.0 where tril is True (visible), -inf where False (future)
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    mask = torch.masked_fill(mask, tril, 0.0)
    
    # Reshape for broadcasting over batch and head dimensions: [1, 1, T, T]
    return mask.unsqueeze(0).unsqueeze(0)
