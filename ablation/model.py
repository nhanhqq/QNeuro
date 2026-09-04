"""HybridNode11 ablation models; the full variant is architecture-identical."""
from __future__ import annotations

import torch
from torch import nn

from src.models.kan_linear import KANLinear
from src.models.quantum_latent import ClassicalLatent16, QuantumLatent16


VARIANTS = (
    "full", "classical_latent", "no_entanglement", "no_reupload",
    "uniform_channel_pool", "no_bilstm", "linear_classifier",
    "no_spectral", "no_hjorth", "no_connectivity",
)
FEATURE_MASKS = {
    "full": (), "classical_latent": (), "no_entanglement": (),
    "no_reupload": (), "uniform_channel_pool": (), "no_bilstm": (),
    "linear_classifier": (), "no_spectral": tuple(range(7)),
    "no_hjorth": (7, 8), "no_connectivity": (9, 10),
}


class HybridNode11Ablation(nn.Module):
    """Component-isolated variants with no target-data-dependent branches."""
    def __init__(self, num_classes: int, num_channels: int, variant: str, rz_noise_std: float):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant}")
        self.variant, self.num_channels = variant, int(num_channels)
        self.register_buffer("feature_mask", torch.ones(11), persistent=True)
        if FEATURE_MASKS[variant]:
            self.feature_mask[list(FEATURE_MASKS[variant])] = 0.0
        self.uniform_channel_pool = variant == "uniform_channel_pool"
        if not self.uniform_channel_pool:
            self.channel_attention = nn.Parameter(torch.zeros(self.num_channels))
        self.no_bilstm = variant == "no_bilstm"
        if self.no_bilstm:
            self.temporal_norm = nn.LayerNorm(11)
            self.to_quantum = nn.Linear(11, 4)
        else:
            self.bilstm = nn.LSTM(11, 5, num_layers=1, batch_first=True, bidirectional=True)
            self.temporal_fusion = nn.Parameter(torch.zeros(2))
            self.temporal_norm = nn.LayerNorm(10)
            self.to_quantum = nn.Linear(10, 4)
        if variant == "classical_latent":
            self.quantum = ClassicalLatent16()
        else:
            self.quantum = QuantumLatent16(
                depth=2,
                entanglement="none" if variant == "no_entanglement" else "ring",
                reupload=variant != "no_reupload",
                rz_noise_std=rz_noise_std,
            )
        self.classifier = nn.Linear(4, num_classes) if variant == "linear_classifier" else KANLinear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[2] != self.num_channels or x.shape[3] != 11:
            raise ValueError(f"expected [B,L,{self.num_channels},11], got {tuple(x.shape)}")
        x = x * self.feature_mask.view(1, 1, 1, -1)
        if self.uniform_channel_pool:
            sequence = x.mean(dim=2)
        else:
            attention = torch.softmax(self.channel_attention, dim=0)
            sequence = (x * attention.view(1, 1, -1, 1)).sum(dim=2)
        if self.no_bilstm:
            temporal = self.temporal_norm(sequence.mean(dim=1))
        else:
            output, (hidden, _) = self.bilstm(sequence)
            endpoint = torch.cat((hidden[-2], hidden[-1]), dim=-1)
            fusion = torch.softmax(self.temporal_fusion, dim=0)
            temporal = self.temporal_norm(fusion[0] * endpoint + fusion[1] * output.mean(dim=1))
        angles = torch.pi * torch.tanh(self.to_quantum(temporal))
        return self.classifier(self.quantum(angles))


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
