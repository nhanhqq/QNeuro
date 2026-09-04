"""Sub-1000 parameter QNeuro variant for HybridNode11 sequences."""

from __future__ import annotations

import torch
from torch import nn

from src.models.kan_linear import KANLinear
from src.models.quantum_latent import QuantumLatent16


class PaperLiteQNeuro(nn.Module):
    """Hybrid node features -> channel attention -> BiLSTM -> VQC -> KAN.

    Endpoint and sequence-mean temporal summaries use a learned two-scalar
    fusion, a cheap analogue of the paper's layer-wise fusion.  Node identity
    is retained until a learned C-scalar attention pool.  This avoids both PCA
    and a CxC graph while allowing source-only training to learn which
    electrodes matter.  There is no input FC.
    """

    def __init__(self, num_classes: int, num_channels: int, rz_noise_std: float = 0.10):
        super().__init__()
        self.num_channels = int(num_channels)
        self.channel_attention = nn.Parameter(torch.zeros(self.num_channels))
        self.bilstm = nn.LSTM(
            input_size=11,
            hidden_size=5,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.temporal_fusion = nn.Parameter(torch.zeros(2))
        self.temporal_norm = nn.LayerNorm(10)
        self.to_quantum = nn.Linear(10, 4)
        self.quantum = QuantumLatent16(
            depth=2,
            entanglement="ring",
            reupload=True,
            rz_noise_std=rz_noise_std,
        )
        self.classifier = KANLinear(4, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        if x.ndim != 4 or x.shape[2] != self.num_channels or x.shape[3] != 11:
            raise ValueError(
                f"expected [B,L,{self.num_channels},11], got {tuple(x.shape)}"
            )
        attention = torch.softmax(self.channel_attention, dim=0)
        sequence = (x * attention.view(1, 1, -1, 1)).sum(dim=2)
        output, (hidden, _) = self.bilstm(sequence)
        endpoint = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        fusion = torch.softmax(self.temporal_fusion, dim=0)
        temporal = self.temporal_norm(
            fusion[0] * endpoint + fusion[1] * output.mean(dim=1)
        )
        angles = torch.pi * torch.tanh(self.to_quantum(temporal))
        quantum_latent = self.quantum(angles)
        logits = self.classifier(quantum_latent)
        if return_features:
            return {
                "channel_attention": attention,
                "sequence": sequence,
                "temporal": temporal,
                "angles": angles,
                "quantum_latent": quantum_latent,
                "logits": logits,
            }
        return logits


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
