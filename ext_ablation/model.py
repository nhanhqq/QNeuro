"""Independent, strict-LOSO extended HybridNode11 component replacements."""
from __future__ import annotations

from collections import OrderedDict
import torch
from torch import nn
from src.models.kan_linear import KANLinear
from src.models.quantum_latent import ClassicalLatent16, QuantumLatent16

# Each family renders one paper table.  ``full`` is re-run in this campaign.
FAMILIES = OrderedDict((
    ("quantum_latent", ("full", "classical_latent", "quantum_depth_1", "quantum_depth_3")),
    ("entanglement", ("full", "no_entanglement", "linear_entanglement")),
    ("reuploading", ("full", "no_reupload", "frozen_reupload_scale")),
    ("channel_pooling", ("full", "uniform_channel_pool", "max_channel_pool")),
    ("temporal_encoder", ("full", "no_bilstm", "endpoint_bilstm", "mean_bilstm")),
    ("classifier", ("full", "linear_classifier", "mlp_classifier")),
    ("spectral_features", ("full", "no_spectral", "no_bandpowers", "no_entropy")),
    ("hjorth_features", ("full", "no_hjorth", "no_hjorth_mobility", "no_hjorth_complexity")),
    ("connectivity_features", ("full", "no_connectivity", "no_mean_connectivity", "no_derivative_connectivity")),
    ("rz_augmentation", ("full", "no_rz_augmentation", "high_rz_augmentation")),
))
VARIANTS = tuple(dict.fromkeys(v for members in FAMILIES.values() for v in members))
VARIANT_FAMILY = {v: f for f, members in FAMILIES.items() for v in members if v != "full"}
FEATURE_MASKS = {
    "full": (), "classical_latent": (), "quantum_depth_1": (), "quantum_depth_3": (),
    "no_entanglement": (), "linear_entanglement": (), "no_reupload": (), "frozen_reupload_scale": (),
    "uniform_channel_pool": (), "max_channel_pool": (), "no_bilstm": (), "endpoint_bilstm": (), "mean_bilstm": (),
    "linear_classifier": (), "mlp_classifier": (),
    "no_spectral": tuple(range(7)), "no_bandpowers": tuple(range(5)), "no_entropy": (5, 6),
    "no_hjorth": (7, 8), "no_hjorth_mobility": (7,), "no_hjorth_complexity": (8,),
    "no_connectivity": (9, 10), "no_mean_connectivity": (9,), "no_derivative_connectivity": (10,),
    "no_rz_augmentation": (), "high_rz_augmentation": (),
}


class HybridNode11ExtendedAblation(nn.Module):
    """Exactly one named intervention per non-reference variant."""
    def __init__(self, num_classes: int, num_channels: int, variant: str, rz_noise_std: float):
        super().__init__()
        if variant not in VARIANTS: raise ValueError(f"unknown variant {variant}")
        self.variant, self.num_channels = variant, int(num_channels)
        self.register_buffer("feature_mask", torch.ones(11), persistent=True)
        if FEATURE_MASKS[variant]: self.feature_mask[list(FEATURE_MASKS[variant])] = 0.0
        self.pooling = "max" if variant == "max_channel_pool" else ("uniform" if variant == "uniform_channel_pool" else "learned")
        if self.pooling == "learned": self.channel_attention = nn.Parameter(torch.zeros(self.num_channels))
        self.no_bilstm = variant == "no_bilstm"
        if self.no_bilstm:
            self.temporal_norm = nn.LayerNorm(11); self.to_quantum = nn.Linear(11, 4)
        else:
            self.bilstm = nn.LSTM(11, 5, num_layers=1, batch_first=True, bidirectional=True)
            self.temporal_fusion = nn.Parameter(torch.zeros(2)); self.temporal_norm = nn.LayerNorm(10); self.to_quantum = nn.Linear(10, 4)
        if variant == "classical_latent": self.quantum = ClassicalLatent16()
        else:
            depth = 1 if variant == "quantum_depth_1" else (3 if variant == "quantum_depth_3" else 2)
            entanglement = "none" if variant == "no_entanglement" else ("linear" if variant == "linear_entanglement" else "ring")
            self.quantum = QuantumLatent16(depth=depth, entanglement=entanglement, reupload=variant != "no_reupload", rz_noise_std=rz_noise_std)
            if variant == "frozen_reupload_scale": self.quantum.encoding_scale.requires_grad_(False)
        if variant == "linear_classifier": self.classifier = nn.Linear(4, num_classes)
        elif variant == "mlp_classifier": self.classifier = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, num_classes))
        else: self.classifier = KANLinear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[2] != self.num_channels or x.shape[3] != 11: raise ValueError(f"expected [B,L,{self.num_channels},11], got {tuple(x.shape)}")
        x = x * self.feature_mask.view(1, 1, 1, -1)
        if self.pooling == "uniform": sequence = x.mean(dim=2)
        elif self.pooling == "max": sequence = x.max(dim=2).values
        else:
            attention = torch.softmax(self.channel_attention, dim=0); sequence = (x * attention.view(1, 1, -1, 1)).sum(dim=2)
        if self.no_bilstm: temporal = self.temporal_norm(sequence.mean(dim=1))
        else:
            output, (hidden, _) = self.bilstm(sequence); endpoint = torch.cat((hidden[-2], hidden[-1]), dim=-1); mean = output.mean(dim=1)
            if self.variant == "endpoint_bilstm": temporal = self.temporal_norm(endpoint)
            elif self.variant == "mean_bilstm": temporal = self.temporal_norm(mean)
            else:
                fusion = torch.softmax(self.temporal_fusion, dim=0); temporal = self.temporal_norm(fusion[0] * endpoint + fusion[1] * mean)
        return self.classifier(self.quantum(torch.pi * torch.tanh(self.to_quantum(temporal))))


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
