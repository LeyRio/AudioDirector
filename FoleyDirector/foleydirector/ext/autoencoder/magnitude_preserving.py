# Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn.functional as F


def normalize(x: torch.Tensor, dim=None, eps: float = 1e-4) -> torch.Tensor:
    if dim is None:
        dim = tuple(range(1, x.ndim))

    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    scale = math.sqrt(norm.numel() / x.numel())
    denominator = torch.add(eps, norm, alpha=scale)
    return x / denominator.to(dtype=x.dtype)


def mp_silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x) / 0.596


def mp_sum(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    scale = math.sqrt((1.0 - t) ** 2 + t**2)
    return a.lerp(b, t) / scale


class MPConv1D(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        self.weight_norm_removed = False

    def forward(self, x: torch.Tensor, gain: float = 1) -> torch.Tensor:
        assert self.weight_norm_removed, "call remove_weight_norm() before inference"

        weight = self.weight * gain
        if weight.ndim == 2:
            return x @ weight.t()
        assert weight.ndim == 3
        return F.conv1d(x, weight, padding=weight.shape[-1] // 2)

    def remove_weight_norm(self):
        weight = normalize(self.weight.to(torch.float32))
        weight = weight / math.sqrt(weight[0].numel())
        self.weight.data.copy_(weight.to(self.weight.dtype))
        self.weight_norm_removed = True
        return self
