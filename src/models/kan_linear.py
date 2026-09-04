"""Small GPU-friendly KAN classifier for bounded 4-D quantum latents."""
import torch
from torch import nn


class KANLinear(nn.Module):
    """Edge-wise piecewise-linear spline map with no dense FC base path.

    Inputs from the quantum layer are expectation values in [-1, 1].  A
    two-cell linear spline therefore has three knots (-1, 0, 1) per edge.  It
    remains deliberately small: ``out_features * (3 * in_features + 1)``
    trainable values, including bias.
    """

    def __init__(self, in_features, out_features, grid_size=2):
        super().__init__()
        if grid_size != 2:
            raise ValueError("the compact classifier intentionally fixes grid_size=2")
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, 3))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.bias)

    def basis(self, x):
        """Triangular degree-1 B-spline bases centered at -1, 0, and 1."""
        x = x.clamp(-1.0, 1.0).unsqueeze(-1)
        centers = x.new_tensor((-1.0, 0.0, 1.0))
        return (1.0 - (x - centers).abs()).clamp_min(0.0)

    def forward(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected final dimension {self.in_features}, got {x.shape[-1]}")
        return torch.einsum("...ig,oig->...o", self.basis(x), self.spline_weight) + self.bias

    def smoothness_penalty(self):
        """Penalize spline curvature, not the learned affine trend."""
        curvature = self.spline_weight[..., 0] - 2.0 * self.spline_weight[..., 1] + self.spline_weight[..., 2]
        return curvature.square().mean()
