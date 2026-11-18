"""Deterministic FP32 execution for the v2 CUDA stages."""

import os

import torch
from torch.nn import functional as F


def configure_cuda():
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting CUDA")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def bilinear_zeros(features, grid, *, mode, padding_mode, align_corners):
    if (mode, padding_mode, align_corners) != ("bilinear", "zeros", True):
        raise ValueError("unexpected CTF sampling geometry")
    if torch.is_grad_enabled() and torch.are_deterministic_algorithms_enabled():
        # PyTorch's decomposed bilinear sampler has differentiable deterministic gathers.
        # Native CUDA grid_sampler_2d_backward uses atomic floating point addition.
        from torch._decomp.decompositions import grid_sampler_2d

        return grid_sampler_2d(features, grid, 0, 0, True)
    return F.grid_sample(features, grid, mode=mode, padding_mode=padding_mode, align_corners=True)


def pool_map(features, size):
    height, width = features.shape[-2:]
    if torch.are_deterministic_algorithms_enabled() and height % size[0] == width % size[1] == 0:
        # For a divisible map these non-overlapping bins are exactly the adaptive windows.
        kernel = (height // size[0], width // size[1])
        return F.avg_pool2d(features, kernel, stride=kernel)
    return F.adaptive_avg_pool2d(features, size)
