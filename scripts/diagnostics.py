"""
Model diagnostics for debugging during model porting.

This module provides utilities to check model state and catch common
misconfigurations that can cause silent training failures.
"""

import logging
import torch
import torch.nn as nn
from typing import Any, Dict
from collections import defaultdict

# Module-level logger
logger = logging.getLogger(__name__)

# Prefix constants for log messages
class Prefix:
    """Log message prefixes for filtering and grepping."""
    INIT = "INIT"   # Weight sums, initialization checks
    FWD = "FWD"     # Forward pass hooks
    BWD = "BWD"     # Backward pass hooks
    STATE = "STATE" # Model state diagnostics (training mode, frozen params, device, dtype)


def get_rank() -> int:
    """Get the current distributed rank, or 0 if not in distributed mode."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _format_size(num_params: int) -> str:
    """Format parameter count with human-readable suffix."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


def print_training_mode(model: nn.Module) -> Dict[str, bool]:
    """
    Print training/eval mode of all modules (recursively).

    Args:
        model: PyTorch model

    Returns:
        Dict mapping module names to their training mode
    """
    rank = get_rank()

    by_mode = defaultdict(list)
    for name, module in model.named_modules():
        mode = "training" if module.training else "eval"
        by_mode[mode].append(name or "root")

    logger.info(f"[Rank {rank}][{Prefix.STATE}] Module training/eval mode: training={len(by_mode['training'])} modules, eval={len(by_mode['eval'])} modules")

    if by_mode["eval"] and by_mode["training"]:
        logger.warning(f"[Rank {rank}][{Prefix.STATE}] Mixed modes! Modules in eval: {by_mode['eval']}")

    return {name: module.training for name, module in model.named_modules()}


def print_frozen_params(model: nn.Module) -> Dict[str, bool]:
    """
    Print frozen parameters (requires_grad=False) recursively.

    Args:
        model: PyTorch model

    Returns:
        Dict mapping param names to requires_grad status
    """
    rank = get_rank()

    frozen = []
    trainable = []
    frozen_size = 0
    trainable_size = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.append(name)
            trainable_size += param.numel()
        else:
            frozen.append(name)
            frozen_size += param.numel()

    total_size = frozen_size + trainable_size

    logger.info(
        f"[Rank {rank}][{Prefix.STATE}] Parameter requires_grad: "
        f"trainable={len(trainable)} ({_format_size(trainable_size)}), "
        f"frozen={len(frozen)} ({_format_size(frozen_size)}), "
        f"total={len(trainable) + len(frozen)} ({_format_size(total_size)})"
    )

    if frozen:
        logger.info(f"[Rank {rank}][{Prefix.STATE}] Frozen parameters: {frozen}")

    return {name: param.requires_grad for name, param in model.named_parameters()}


def print_device(model: nn.Module) -> Dict[str, torch.device]:
    """
    Print device of all parameters (recursively).

    Args:
        model: PyTorch model

    Returns:
        Dict mapping param names to their device
    """
    rank = get_rank()

    by_device = defaultdict(list)
    devices = {}

    for name, param in model.named_parameters():
        devices[name] = param.device
        by_device[str(param.device)].append(name)

    device_summary = ", ".join(f"{device}={len(names)}" for device, names in by_device.items())
    logger.info(f"[Rank {rank}][{Prefix.STATE}] Parameter devices: {device_summary}")

    return devices


def print_dtype(model: nn.Module) -> Dict[str, torch.dtype]:
    """
    Print dtype of all parameters (recursively).

    Args:
        model: PyTorch model

    Returns:
        Dict mapping param names to their dtype
    """
    rank = get_rank()

    by_dtype = defaultdict(list)
    dtypes = {}

    for name, param in model.named_parameters():
        dtypes[name] = param.dtype
        by_dtype[str(param.dtype)].append(name)

    dtype_summary = ", ".join(f"{dtype}={len(names)}" for dtype, names in by_dtype.items())
    logger.info(f"[Rank {rank}][{Prefix.STATE}] Parameter dtypes: {dtype_summary}")

    return dtypes


def print_weight_sums(model: nn.Module) -> Dict[str, float]:
    """
    Print the sum of weights for each module parameter.

    This is useful for verifying weight initialization correctness when porting
    a model between frameworks. By comparing the sum of weights, you can quickly
    identify if weights were loaded/initialized correctly.

    Args:
        model: PyTorch model

    Returns:
        Dict mapping param names to their weight sums (as float32)
    """
    rank = get_rank()

    weight_sums = {}
    for name, param in model.named_parameters():
        # Compute sum in float32 to avoid overflow
        weight_sum = torch.sum(param.detach().to(torch.float32)).item()
        weight_sums[name] = weight_sum
        logger.info(f"[Rank {rank}][{Prefix.INIT}] weight_sum {name}: {weight_sum}")

    logger.info(f"[Rank {rank}][{Prefix.INIT}] Total parameters: {len(weight_sums)}")

    return weight_sums


def visualize_attn_mask(
    mask: torch.Tensor,
    mode: str = "color",
    title: str = "Attention Mask",
    save_path: str = None,
    figsize: tuple = (10, 10),
    cmap: str = "viridis",
) -> None:
    """
    Visualize an attention mask.

    Args:
        mask: Attention mask tensor (2D or higher, will use last 2 dims)
        mode: "color" for heatmap visualization, "number" for text-based display
        title: Title for the plot (only used in color mode)
        save_path: Path to save the figure (only used in color mode). If None, displays interactively.
        figsize: Figure size as (width, height) tuple (only used in color mode)
        cmap: Colormap for heatmap (only used in color mode)

    Example:
        >>> mask = torch.triu(torch.ones(8, 8), diagonal=1)  # causal mask
        >>> visualize_attn_mask(mask, mode="color")
        >>> visualize_attn_mask(mask, mode="number")
    """
    rank = get_rank()

    # Handle multi-dimensional masks - use last 2 dims
    if mask.dim() > 2:
        # Take first element from batch/head dimensions
        while mask.dim() > 2:
            mask = mask[0]

    mask_np = mask.detach().cpu().float().numpy()
    h, w = mask_np.shape

    if mode == "number":
        logger.info(f"[Rank {rank}][{Prefix.STATE}] Attention Mask ({h}x{w}):")
        logger.info("-" * (w * 6 + 1))

        for i in range(h):
            row_str = "|"
            for j in range(w):
                val = mask_np[i, j]
                if val == 0:
                    row_str += "  0  |"
                elif val == 1:
                    row_str += "  1  |"
                elif val == float("-inf") or val < -1e9:
                    row_str += " -inf|"
                else:
                    row_str += f"{val:5.2f}|"
            logger.info(row_str)

        logger.info("-" * (w * 6 + 1))

    elif mode == "color":
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.warning(f"[Rank {rank}][{Prefix.STATE}] matplotlib not available, falling back to number mode")
            visualize_attn_mask(mask, mode="number", title=title)
            return

        fig, ax = plt.subplots(figsize=figsize)

        # Handle -inf values for visualization
        mask_vis = np.where(np.isinf(mask_np), np.nan, mask_np)

        im = ax.imshow(mask_vis, cmap=cmap, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Mask Value")

        # Add grid
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"[Rank {rank}][{Prefix.STATE}] Saved attention mask to: {save_path}")
            plt.close()
        else:
            plt.show()

    else:
        raise ValueError(f"mode must be 'color' or 'number', got '{mode}'")


def diagnose_model(model: nn.Module) -> Dict:
    """
    Run all diagnostics on a model (recursively traverses all nested modules).

    Args:
        model: PyTorch model

    Returns:
        Dict with all diagnostic results
    """
    rank = get_rank()

    logger.info(f"[Rank {rank}][{Prefix.STATE}] " + "=" * 50)
    logger.info(f"[Rank {rank}][{Prefix.STATE}] Model Diagnostics")
    logger.info(f"[Rank {rank}][{Prefix.STATE}] " + "=" * 50)

    results = {
        "training_mode": print_training_mode(model),
        "requires_grad": print_frozen_params(model),
        "device": print_device(model),
        "dtype": print_dtype(model),
    }

    logger.info(f"[Rank {rank}][{Prefix.STATE}] " + "=" * 50)

    return results


def print_rng_state() -> Dict[str, Any]:
    """
    Print current RNG states for torch and cuda.

    Prints seed values and state checksums that can be used to verify
    RNG state consistency across different runs or frameworks.

    Returns:
        Dict containing RNG state information:
            - torch_seed: Initial torch seed
            - torch_sum: Checksum of torch RNG state
            - torch_head: First 8 bytes of torch RNG state
            - cuda_seed: Initial CUDA seed (if available)
            - cuda_sum: Checksum of CUDA RNG state (if available)
            - cuda_head: First 8 bytes of CUDA RNG state (if available)
    """
    rank = get_rank()

    result: Dict[str, Any] = {}

    # Torch RNG state
    torch_seed = torch.initial_seed()
    torch_state = torch.get_rng_state()
    torch_sum = torch_state.to(torch.float32).sum().item()
    torch_head = torch_state[:8].tolist()

    result["torch_seed"] = torch_seed
    result["torch_sum"] = torch_sum
    result["torch_head"] = torch_head

    # CUDA RNG state (if available)
    cuda_seed = None
    cuda_sum = None
    cuda_head = None

    if torch.cuda.is_available():
        cuda_seed = torch.cuda.initial_seed()
        cuda_state = torch.cuda.get_rng_state()
        cuda_sum = cuda_state.to(torch.float32).sum().item()
        cuda_head = cuda_state[:8].tolist()

        result["cuda_seed"] = cuda_seed
        result["cuda_sum"] = cuda_sum
        result["cuda_head"] = cuda_head

    # Format as single line for grepping
    parts = [
        f"torch_seed={torch_seed}",
        f"torch_sum={int(torch_sum)}",
        f"torch_head={torch_head}",
    ]

    if cuda_seed is not None:
        parts.extend([
            f"cuda_seed={cuda_seed}",
            f"cuda_sum={int(cuda_sum)}",
            f"cuda_head={cuda_head}",
        ])

    logger.info(f"[Rank {rank}][{Prefix.INIT}] RNG: {' '.join(parts)}")

    return result
