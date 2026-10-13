#!/usr/bin/env python3
"""Standalone QNeuro full-model training program.

This file has no imports from the QNeuro repository and creates its own
feature cache, LOSO splits, checkpoints, CSV logs, and final metrics.  It uses
the 12 per-channel features described in report.md: five relative band powers,
Shannon band entropy, spectral entropy, log Hjorth activity, Hjorth mobility,
Hjorth complexity, mean channel correlation, and mean derivative correlation.

The hybrid classifier is channel attention -> BiLSTM -> temporal fusion ->
4-qubit PennyLane VQC (two re-upload blocks, trainable encoding scales, cyclic
CNOT entanglement) -> residual MLP classification head.

Expected input directory:
  X.npy       float EEG windows with shape [samples, channels, time_points]
  labels.npy  one class label per EEG window
  meta.csv    one row per window, with a subject/participant ID column

Install runtime packages with:
  python -m pip install numpy torch pennylane tqdm

Example:
  python camera_ready.py --data ./CHSZ --output ./runs/chsz --epochs 100

PennyLane's default.qubit analytic state-vector simulator is used on CPU.  This
keeps the script portable and avoids requiring a PennyLane GPU plugin.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pennylane as qml
import torch
from torch import nn
from tqdm.auto import tqdm


FEATURE_NAMES = (
    "log_relative_delta",
    "log_relative_theta",
    "log_relative_alpha",
    "log_relative_beta",
    "log_relative_gamma",
    "shannon_band_entropy",
    "spectral_entropy",
    "log_hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    "mean_channel_correlation",
    "mean_derivative_correlation",
)
BANDS = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))
DATASET_DEFAULTS = {
    "2014001": (250.0, 1.0, 0.5),
    "chb-mit": (256.0, 1.0, 0.5),
    "chsz": (500.0, 1.0, 0.5),
    "cog-bci": (500.0, 0.5, 0.25),
    "faced": (250.0, 1.0, 0.5),
    "isruc-s3": (200.0, 1.0, 0.5),
    "physionet-mi": (160.0, 1.0, 0.5),
    "seed": (200.0, 1.0, 0.5),
    "sleep-edf-20": (100.0, 1.0, 0.5),
    "stew": (128.0, 0.5, 0.25),
}
SUBJECT_COLUMN_CANDIDATES = (
    "subject",
    "subject_id",
    "participant",
    "participant_id",
    "group",
    "patient",
    "patient_id",
)
EPS = 1e-12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def stable_label_encoding(raw_labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    values = [str(value) for value in np.asarray(raw_labels).reshape(-1).tolist()]
    classes = sorted(set(values))
    lookup = {name: index for index, name in enumerate(classes)}
    encoded = np.fromiter((lookup[value] for value in values), dtype=np.int64)
    return encoded, classes


def _find_subject_column(columns: list[str], requested: str | None) -> str:
    by_lower = {column.strip().lower(): column for column in columns}
    if requested:
        key = requested.strip().lower()
        if key not in by_lower:
            raise ValueError(
                f"subject column {requested!r} not found in metadata columns {columns}"
            )
        return by_lower[key]
    for candidate in SUBJECT_COLUMN_CANDIDATES:
        if candidate in by_lower:
            return by_lower[candidate]
    raise ValueError(
        "could not find a subject ID column in meta.csv; expected one of "
        + ", ".join(SUBJECT_COLUMN_CANDIDATES)
        + ". Set --subject-column to its actual name."
    )


def load_dataset(
    data_path: Path, subject_column: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], str]:
    """Load standalone NPY/CSV data or a single NPZ archive."""
    if data_path.is_file() and data_path.suffix.lower() == ".npz":
        archive = np.load(data_path, allow_pickle=True)
        keys = set(archive.files)
        x_key = next((name for name in ("X", "x", "signals", "eeg") if name in keys), None)
        y_key = next((name for name in ("labels", "y", "targets") if name in keys), None)
        group_key = next((name for name in ("subjects", "subject", "groups", "group") if name in keys), None)
        if x_key is None or y_key is None or group_key is None:
            raise ValueError(
                "NPZ input must contain signal, label, and subject arrays; accepted "
                "keys are X/x/signals/eeg, labels/y/targets, and subjects/subject/groups/group"
            )
        signals = archive[x_key]
        raw_labels = archive[y_key]
        raw_groups = archive[group_key]
        source_name = data_path.name
    else:
        if not data_path.is_dir():
            raise FileNotFoundError(f"data path does not exist: {data_path}")
        x_path = data_path / "X.npy"
        y_path = data_path / "labels.npy"
        meta_path = data_path / "meta.csv"
        groups_path = data_path / "subjects.npy"
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"{data_path} must contain X.npy and labels.npy (or pass an NPZ archive)"
            )
        signals = np.load(x_path, mmap_mode="r", allow_pickle=False)
        raw_labels = np.load(y_path, allow_pickle=True)
        if meta_path.exists():
            with meta_path.open("r", newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                columns = list(reader.fieldnames or [])
                selected_column = _find_subject_column(columns, subject_column)
                raw_groups = [row[selected_column] for row in reader]
        elif groups_path.exists():
            raw_groups = np.load(groups_path, allow_pickle=True)
        else:
            raise FileNotFoundError(
                f"{data_path} must contain meta.csv with subject IDs or subjects.npy"
            )
        source_name = data_path.name

    signals = np.asarray(signals)
    raw_labels = np.asarray(raw_labels).reshape(-1)
    raw_groups = np.asarray(raw_groups).reshape(-1)
    if signals.ndim != 3:
        raise ValueError(
            f"expected EEG windows shaped [N,C,T], got {tuple(signals.shape)}"
        )
    if len(signals) != len(raw_labels) or len(signals) != len(raw_groups):
        raise ValueError(
            "sample counts differ: "
            f"X={len(signals)}, labels={len(raw_labels)}, subjects={len(raw_groups)}"
        )
    if len(signals) == 0:
        raise ValueError("the dataset contains no EEG samples")
    if not np.issubdtype(signals.dtype, np.number):
        raise TypeError("X must contain numeric EEG values")
    if not np.isfinite(np.asarray(signals[: min(len(signals), 8)])).all():
        raise ValueError("X contains NaN or infinite values")
    labels, classes = stable_label_encoding(raw_labels)
    groups = np.asarray([str(value) for value in raw_groups], dtype=str)
    if np.any(groups == ""):
        raise ValueError("subject IDs must be non-empty")
    if len(np.unique(groups)) < 2:
        raise ValueError("LOSO training requires at least two distinct subjects")
    return signals, labels, groups, classes, source_name


def mean_off_diagonal_correlation(x: np.ndarray) -> np.ndarray:
    """Compute exact mean pairwise correlation without building a C x C matrix."""
    n_samples, channels, _ = x.shape
    if channels < 2:
        return np.zeros(n_samples, dtype=np.float32)
    centered = x - x.mean(axis=-1, keepdims=True)
    rms = np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True))
    standardized = centered / np.maximum(rms, 1e-12)
    channel_sum = standardized.sum(axis=1)
    average_all_pairs = np.mean(channel_sum * channel_sum, axis=-1)
    return ((average_all_pairs - channels) / (channels * (channels - 1))).astype(
        np.float32
    )


def extract_feature_batch(
    raw_batch: np.ndarray,
    sampling_rate: float,
    frame_seconds: float,
    hop_seconds: float,
) -> np.ndarray:
    """Convert [N,C,T] raw windows to [N,L,C,12] EEG node features."""
    x = np.asarray(raw_batch, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"expected [N,C,T], got {tuple(x.shape)}")
    frame_length = int(round(sampling_rate * frame_seconds))
    hop_length = int(round(sampling_rate * hop_seconds))
    if frame_length < 4 or hop_length < 1 or x.shape[-1] < frame_length:
        raise ValueError(
            f"invalid framing: samples={x.shape[-1]}, frame={frame_length}, hop={hop_length}"
        )
    if sampling_rate <= 90.0:
        raise ValueError(
            "the configured 30-45 Hz gamma band needs a sampling rate above 90 Hz"
        )

    starts = tuple(range(0, x.shape[-1] - frame_length + 1, hop_length))
    frequencies = np.fft.rfftfreq(frame_length, d=1.0 / sampling_rate)
    analysis_mask = (frequencies >= 1.0) & (frequencies < 45.0)
    band_masks = tuple(
        (frequencies >= low) & (frequencies < high) for low, high in BANDS
    )
    if not analysis_mask.any() or any(not mask.any() for mask in band_masks):
        raise ValueError(
            "frame duration and sampling rate do not represent every 1-45 Hz EEG band; "
            "increase --frame-seconds or adjust --sampling-rate"
        )
    window = np.hanning(frame_length).astype(np.float32)
    n_samples, channels, _ = x.shape
    output = np.empty(
        (n_samples, len(starts), channels, len(FEATURE_NAMES)), dtype=np.float32
    )

    for frame_index, start in enumerate(starts):
        frame = x[:, :, start : start + frame_length]
        centered = frame - frame.mean(axis=-1, keepdims=True)
        variance = np.mean(centered * centered, axis=-1)
        power = np.abs(np.fft.rfft(centered * window, axis=-1)) ** 2
        spectrum = power[:, :, analysis_mask]
        total_power = spectrum.sum(axis=-1)
        band_power = np.stack(
            [power[:, :, mask].sum(axis=-1) for mask in band_masks], axis=-1
        )
        log_relative_power = np.log(
            (band_power + EPS) / (total_power[..., None] + EPS)
        )
        band_probability = band_power / np.maximum(
            band_power.sum(axis=-1, keepdims=True), EPS
        )
        shannon_band_entropy = -(
            band_probability * np.log(band_probability + EPS)
        ).sum(axis=-1)
        spectral_probability = spectrum / np.maximum(
            spectrum.sum(axis=-1, keepdims=True), EPS
        )
        spectral_entropy = -(
            spectral_probability * np.log(spectral_probability + EPS)
        ).sum(axis=-1)

        first_derivative = np.diff(centered, axis=-1)
        second_derivative = np.diff(first_derivative, axis=-1)
        std_signal = np.sqrt(np.maximum(variance, 1e-24))
        std_first = np.sqrt(
            np.maximum(np.mean(first_derivative * first_derivative, axis=-1), 1e-24)
        )
        std_second = np.sqrt(
            np.maximum(np.mean(second_derivative * second_derivative, axis=-1), 1e-24)
        )
        hjorth_mobility = std_first / std_signal
        hjorth_complexity = (std_second / std_first) / np.maximum(
            hjorth_mobility, 1e-12
        )
        log_activity = np.log(variance + 1e-24)
        correlation = mean_off_diagonal_correlation(centered)[:, None, None]
        derivative_correlation = mean_off_diagonal_correlation(first_derivative)[
            :, None, None
        ]
        values = np.concatenate(
            (
                log_relative_power,
                shannon_band_entropy[..., None],
                spectral_entropy[..., None],
                log_activity[..., None],
                hjorth_mobility[..., None],
                hjorth_complexity[..., None],
                np.broadcast_to(correlation, (n_samples, channels, 1)),
                np.broadcast_to(derivative_correlation, (n_samples, channels, 1)),
            ),
            axis=-1,
        )
        output[:, frame_index] = np.nan_to_num(
            values, nan=0.0, posinf=30.0, neginf=-30.0
        ).astype(np.float32, copy=False)
    return output


def _cache_configuration(
    x_path: Path,
    signals: np.ndarray,
    sampling_rate: float,
    frame_seconds: float,
    hop_seconds: float,
) -> dict[str, Any]:
    source_stat = x_path.stat() if x_path.exists() else None
    return {
        "format": "qneuro_standalone_features_v1",
        "feature_names": list(FEATURE_NAMES),
        "sampling_rate": float(sampling_rate),
        "frame_seconds": float(frame_seconds),
        "hop_seconds": float(hop_seconds),
        "source_path": str(x_path.resolve()),
        "source_shape": list(signals.shape),
        "source_size": source_stat.st_size if source_stat else None,
        "source_mtime_ns": source_stat.st_mtime_ns if source_stat else None,
    }


def build_feature_cache(
    signals: np.ndarray,
    data_path: Path,
    output_dir: Path,
    sampling_rate: float,
    frame_seconds: float,
    hop_seconds: float,
    chunk_size: int,
) -> np.ndarray:
    """Extract every frame once and keep features in a reproducible mmap cache."""
    cache_dir = output_dir / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "features.npy"
    config_path = cache_dir / "features.json"
    x_path = data_path if data_path.is_file() else data_path / "X.npy"
    expected = _cache_configuration(
        x_path, signals, sampling_rate, frame_seconds, hop_seconds
    )
    if cache_path.exists() and config_path.exists():
        try:
            current = json.loads(config_path.read_text(encoding="utf-8"))
            cached = np.load(cache_path, mmap_mode="r", allow_pickle=False)
            if current == expected and cached.shape[0] == len(signals):
                if cached.shape[-1] == len(FEATURE_NAMES):
                    print(f"Using feature cache: {cache_path}")
                    return cached
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    preview = extract_feature_batch(
        np.asarray(signals[:1]), sampling_rate, frame_seconds, hop_seconds
    )
    shape = (len(signals),) + tuple(preview.shape[1:])
    cache = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.float32, shape=shape
    )
    starts = range(0, len(signals), chunk_size)
    for start in tqdm(starts, desc="Extracting EEG features", unit="chunk"):
        stop = min(start + chunk_size, len(signals))
        batch = np.asarray(signals[start:stop], dtype=np.float32)
        cache[start:stop] = extract_feature_batch(
            batch, sampling_rate, frame_seconds, hop_seconds
        )
    cache.flush()
    del cache
    config_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return np.load(cache_path, mmap_mode="r", allow_pickle=False)


def make_loso_splits(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, str]]:
    splits = []
    for subject in np.unique(groups):
        test_idx = np.flatnonzero(groups == subject).astype(np.int64)
        train_idx = np.flatnonzero(groups != subject).astype(np.int64)
        if not len(train_idx) or not len(test_idx):
            continue
        splits.append((train_idx, test_idx, str(subject)))
    return splits


def split_inner_validation(
    source_indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole source subjects, preserving every source class in fit data."""
    source_groups = np.unique(groups[source_indices])
    if len(source_groups) < 2 or validation_fraction <= 0.0:
        return source_indices.copy(), np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    order = source_groups.copy()
    rng.shuffle(order)
    n_validation_groups = max(1, int(round(len(order) * validation_fraction)))
    n_validation_groups = min(n_validation_groups, len(order) - 1)
    source_classes = set(np.unique(labels[source_indices]).tolist())

    for offset in range(len(order)):
        rotated = np.concatenate((order[offset:], order[:offset]))
        validation_groups = set(rotated[:n_validation_groups].tolist())
        validation_mask = np.isin(groups[source_indices], list(validation_groups))
        fit_indices = source_indices[~validation_mask]
        validation_indices = source_indices[validation_mask]
        if len(fit_indices) and len(validation_indices):
            fit_classes = set(np.unique(labels[fit_indices]).tolist())
            if fit_classes == source_classes:
                return fit_indices, validation_indices
    return source_indices.copy(), np.empty(0, dtype=np.int64)


def fit_feature_scaler(
    features: np.ndarray, train_indices: np.ndarray, chunk_size: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """Fit scalar feature normalization using only inner-training subjects."""
    total = np.zeros(features.shape[-1], dtype=np.float64)
    total_sq = np.zeros(features.shape[-1], dtype=np.float64)
    count = 0
    for start in range(0, len(train_indices), chunk_size):
        ids = train_indices[start : start + chunk_size]
        block = np.asarray(features[ids], dtype=np.float64)
        total += block.sum(axis=(0, 1, 2))
        total_sq += np.square(block).sum(axis=(0, 1, 2))
        count += int(np.prod(block.shape[:-1]))
    mean = total / max(count, 1)
    variance = np.maximum(total_sq / max(count, 1) - mean * mean, 1e-12)
    scale = np.sqrt(variance)
    return mean.astype(np.float32), scale.astype(np.float32)


def load_feature_batch(
    features: np.ndarray,
    indices: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    batch = np.array(features[indices], dtype=np.float32, copy=True)
    batch -= mean.reshape(1, 1, 1, -1)
    batch /= scale.reshape(1, 1, 1, -1)
    return torch.from_numpy(batch).to(device)


def make_quantum_layer(depth: int = 2, n_qubits: int = 4) -> nn.Module:
    """Create a differentiable PennyLane TorchLayer with cyclic CNOT entanglement."""
    quantum_device = qml.device("default.qubit", wires=n_qubits, shots=None)

    @qml.qnode(quantum_device, interface="torch", diff_method="backprop")
    def circuit(inputs: torch.Tensor, weights: torch.Tensor, encoding_scale: torch.Tensor):
        for layer in range(depth):
            for wire in range(n_qubits):
                qml.RY(inputs[..., wire] * encoding_scale[layer, wire], wires=wire)
                qml.RZ(inputs[..., n_qubits + layer * n_qubits + wire], wires=wire)
                qml.RY(weights[layer, wire, 0], wires=wire)
                qml.RZ(weights[layer, wire, 1], wires=wire)
            # The report calls this all-to-all circular entanglement: each
            # neighboring pair is connected and the last wire closes the ring.
            for wire in range(n_qubits):
                qml.CNOT(wires=(wire, (wire + 1) % n_qubits))
        return tuple(qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits))

    weight_shapes = {
        "weights": (depth, n_qubits, 2),
        "encoding_scale": (depth, n_qubits),
    }
    init_method = {
        "weights": lambda tensor: nn.init.normal_(tensor, mean=0.0, std=0.05),
        "encoding_scale": lambda tensor: nn.init.ones_(tensor),
    }
    return qml.qnn.TorchLayer(circuit, weight_shapes, init_method=init_method)


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(x + self.layers(x)))


class QNeuroFullModel(nn.Module):
    """Channel attention, temporal BiLSTM, PennyLane VQC, and residual MLP."""

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        feature_count: int = len(FEATURE_NAMES),
        lstm_hidden: int = 12,
        head_width: int = 32,
        dropout: float = 0.20,
        quantum_depth: int = 2,
        rz_noise_std: float = 0.10,
    ):
        super().__init__()
        if num_channels < 1 or num_classes < 2:
            raise ValueError("model needs at least one channel and two classes")
        self.num_channels = int(num_channels)
        self.feature_count = int(feature_count)
        self.rz_noise_std = float(rz_noise_std)
        self.channel_attention = nn.Parameter(torch.zeros(num_channels))
        self.bilstm = nn.LSTM(
            input_size=feature_count,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        temporal_width = 2 * lstm_hidden
        self.temporal_fusion = nn.Parameter(torch.zeros(2))
        self.temporal_norm = nn.LayerNorm(temporal_width)
        self.to_quantum = nn.Linear(temporal_width, 4)
        self.quantum = make_quantum_layer(depth=quantum_depth, n_qubits=4)
        self.head_input = nn.Sequential(
            nn.Linear(4, head_width),
            nn.LayerNorm(head_width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_blocks = nn.Sequential(
            ResidualMLPBlock(head_width, dropout),
            ResidualMLPBlock(head_width, dropout),
        )
        self.classifier = nn.Linear(head_width, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        if x.ndim != 4:
            raise ValueError(f"expected [B,L,C,F] features, got {tuple(x.shape)}")
        if x.shape[2] != self.num_channels or x.shape[3] != self.feature_count:
            raise ValueError(
                f"expected [B,L,{self.num_channels},{self.feature_count}], "
                f"got {tuple(x.shape)}"
            )
        attention = torch.softmax(self.channel_attention, dim=0)
        sequence = (x * attention.view(1, 1, -1, 1)).sum(dim=2)
        outputs, (hidden, _) = self.bilstm(sequence)
        endpoint = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        temporal_mix = torch.softmax(self.temporal_fusion, dim=0)
        temporal = self.temporal_norm(
            temporal_mix[0] * endpoint + temporal_mix[1] * outputs.mean(dim=1)
        )
        angles = math.pi * torch.tanh(self.to_quantum(temporal))
        if self.training and self.rz_noise_std > 0.0:
            rz_noise = torch.randn(
                angles.shape[0], 8, dtype=angles.dtype, device=angles.device
            ) * self.rz_noise_std
        else:
            rz_noise = torch.zeros(
                angles.shape[0], 8, dtype=angles.dtype, device=angles.device
            )
        quantum_input = torch.cat((angles, rz_noise), dim=-1)
        quantum_latent = self.quantum(quantum_input)
        if quantum_latent.ndim == 1:
            quantum_latent = quantum_latent.unsqueeze(0)
        hidden_head = self.head_input(quantum_latent)
        hidden_head = self.residual_blocks(hidden_head)
        logits = self.classifier(hidden_head)
        if return_features:
            return {
                "channel_attention": attention,
                "temporal": temporal,
                "quantum_angles": angles,
                "quantum_latent": quantum_latent,
                "logits": logits,
            }
        return logits


def classification_metrics(
    targets: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, n_classes: int
) -> dict[str, float]:
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if len(targets) == 0:
        return {"accuracy": 0.0, "balanced_accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    true_positive = np.diag(confusion).astype(np.float64)
    recall = np.divide(true_positive, support, out=np.zeros(n_classes), where=support > 0)
    precision = np.divide(
        true_positive, predicted_count, out=np.zeros(n_classes), where=predicted_count > 0
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(n_classes),
        where=(precision + recall) > 0,
    )
    present = support > 0
    return {
        "accuracy": float(true_positive.sum() / len(targets)),
        "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)) if support.sum() else 0.0,
    }


@torch.no_grad()
def predict_indices(
    model: nn.Module,
    features: np.ndarray,
    indices: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_out = []
    for start in range(0, len(indices), batch_size):
        ids = indices[start : start + batch_size]
        x = load_feature_batch(features, ids, mean, scale, device)
        logits_out.append(model(x).cpu())
    logits = torch.cat(logits_out, dim=0)
    return logits.argmax(dim=1).numpy(), logits.softmax(dim=1).numpy()


def class_weights_for(labels: np.ndarray, indices: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels[indices], minlength=n_classes).astype(np.float64)
    weights = np.zeros(n_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = len(indices) / (present.sum() * counts[present])
    return torch.from_numpy(weights)


def train_one_fold(
    fold: int,
    subject: str,
    source_indices: np.ndarray,
    test_indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    classes: list[str],
    features: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    fit_indices, validation_indices = split_inner_validation(
        source_indices,
        labels,
        groups,
        args.validation_fraction,
        args.seed + fold,
    )
    fit_classes = set(np.unique(labels[fit_indices]).tolist())
    source_classes = set(np.unique(labels[source_indices]).tolist())
    if fit_classes != source_classes:
        raise RuntimeError(
            f"fold {fold}: source training split lost classes; "
            f"fit={sorted(fit_classes)}, source={sorted(source_classes)}"
        )
    mean, scale = fit_feature_scaler(features, fit_indices)
    model = QNeuroFullModel(
        num_channels=features.shape[2],
        num_classes=len(classes),
        feature_count=features.shape[3],
        lstm_hidden=args.lstm_hidden,
        head_width=args.head_width,
        dropout=args.dropout,
        quantum_depth=args.quantum_depth,
        rz_noise_std=args.rz_noise_std,
    ).to(device)
    weights = class_weights_for(labels, fit_indices, len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.minimum_learning_rate
    )
    fold_dir = output_dir / f"fold_{fold:03d}_subject_{safe_name(subject)}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_path = fold_dir / "best_validation.pt"
    history_path = fold_dir / "epochs.csv"
    history_rows: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = -1
    stale_epochs = 0
    train_rng = np.random.default_rng(args.seed + 1009 * (fold + 1))
    has_validation = len(validation_indices) > 0
    fixed_epoch_count = args.epochs

    epoch_bar = tqdm(range(fixed_epoch_count), desc=f"LOSO {fold + 1}: train", unit="epoch")
    for epoch in epoch_bar:
        epoch_started = time.time()
        model.train()
        train_rng.shuffle(fit_indices)
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        batch_starts = range(0, len(fit_indices), args.batch_size)
        for start in tqdm(
            batch_starts,
            desc=f"epoch {epoch + 1} batches",
            unit="batch",
            leave=False,
            disable=len(fit_indices) <= args.batch_size,
        ):
            batch_indices = fit_indices[start : start + args.batch_size]
            x_batch = load_feature_batch(features, batch_indices, mean, scale, device)
            y_batch = torch.as_tensor(labels[batch_indices], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_indices)
            total_correct += int((logits.argmax(dim=1) == y_batch).sum().detach())
            total_seen += len(batch_indices)

        train_loss = total_loss / max(total_seen, 1)
        train_accuracy = total_correct / max(total_seen, 1)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        row: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "learning_rate": learning_rate,
            "validation_accuracy": "",
            "validation_balanced_accuracy": "",
            "validation_macro_f1": "",
            "validation_weighted_f1": "",
            "epoch_seconds": 0.0,
        }
        if has_validation:
            val_prediction, val_probability = predict_indices(
                model,
                features,
                validation_indices,
                mean,
                scale,
                device,
                args.batch_size,
            )
            val_target = labels[validation_indices]
            val_metrics = classification_metrics(
                val_target, val_prediction, val_probability, len(classes)
            )
            row["validation_accuracy"] = val_metrics["accuracy"]
            row["validation_balanced_accuracy"] = val_metrics["balanced_accuracy"]
            row["validation_macro_f1"] = val_metrics["macro_f1"]
            row["validation_weighted_f1"] = val_metrics["weighted_f1"]
            score = val_metrics[args.selection_metric]
        else:
            # With only one source subject, no subject-disjoint validation is
            # possible; use the final epoch and evaluate the test subject once.
            score = float(epoch)
        row["epoch_seconds"] = time.time() - epoch_started
        history_rows.append(row)
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mean": mean.tolist(),
                    "scale": scale.tolist(),
                    "epoch": best_epoch,
                    "selection_metric": args.selection_metric if has_validation else "last_epoch_fallback",
                    "validation_score": best_score,
                    "train_subjects": np.unique(groups[fit_indices]).tolist(),
                    "validation_subjects": np.unique(groups[validation_indices]).tolist(),
                    "target_subject": subject,
                    "classes": classes,
                    "feature_names": list(FEATURE_NAMES),
                    "config": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                },
                best_path,
            )
        else:
            stale_epochs += 1
        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            train_acc=f"{train_accuracy:.3f}",
            val=f"{best_score:.3f}" if has_validation else "n/a",
        )
        if has_validation and stale_epochs >= args.patience:
            epoch_bar.write(
                f"Early stop at epoch {epoch + 1}; best validation epoch={best_epoch}"
            )
            break

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(history_rows[0]) if history_rows else ["epoch"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    # This is the only evaluation of the outer held-out subject in the fold.
    test_prediction, test_probability = predict_indices(
        model,
        features,
        test_indices,
        mean,
        scale,
        device,
        args.batch_size,
    )
    test_metrics = classification_metrics(
        labels[test_indices], test_prediction, test_probability, len(classes)
    )
    result = {
        "fold": fold,
        "target_subject": subject,
        "train_subjects": sorted(np.unique(groups[fit_indices]).tolist()),
        "validation_subjects": sorted(np.unique(groups[validation_indices]).tolist()),
        "test_subjects": [subject],
        "n_train_windows": int(len(fit_indices)),
        "n_validation_windows": int(len(validation_indices)),
        "n_test_windows": int(len(test_indices)),
        "selected_epoch": int(best_epoch),
        "selection_metric": args.selection_metric if has_validation else "last_epoch_fallback",
        "test_metrics": test_metrics,
        "test_evaluations": 1,
    }
    (fold_dir / "final_test_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def safe_name(value: str) -> str:
    name = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return name[:80] or "subject"


def resolve_signal_settings(args: argparse.Namespace, data_path: Path) -> tuple[float, float, float]:
    dataset_name = args.dataset or (data_path.stem if data_path.is_file() else data_path.name)
    key = dataset_name.strip().lower()
    defaults = DATASET_DEFAULTS.get(key, (128.0, 1.0, 0.5))
    sampling_rate = args.sampling_rate or defaults[0]
    frame_seconds = args.frame_seconds or defaults[1]
    hop_seconds = args.hop_seconds or defaults[2]
    return float(sampling_rate), float(frame_seconds), float(hop_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train standalone PennyLane QNeuro with subject-independent LOSO evaluation."
    )
    parser.add_argument("--data", type=Path, required=True, help="folder with X.npy/labels.npy/meta.csv, or an NPZ")
    parser.add_argument("--output", type=Path, default=Path("camera_ready_runs"))
    parser.add_argument("--dataset", default=None, help="dataset name used to select built-in sampling defaults")
    parser.add_argument("--subject-column", default=None, help="subject ID column name in meta.csv")
    parser.add_argument("--sampling-rate", type=float, default=None, help="sampling frequency in Hz")
    parser.add_argument("--frame-seconds", type=float, default=None)
    parser.add_argument("--hop-seconds", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-chunk-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--lstm-hidden", type=int, default=12)
    parser.add_argument("--head-width", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--quantum-depth", type=int, default=2)
    parser.add_argument("--rz-noise-std", type=float, default=0.10)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--selection-metric", choices=("macro_f1", "accuracy"), default="macro_f1")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-folds", type=int, default=None, help="optional fold limit for a partial run")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.feature_chunk_size < 1:
        parser.error("epochs, batch-size, and feature-chunk-size must be positive")
    if args.quantum_depth < 1 or args.lstm_hidden < 1 or args.head_width < 1:
        parser.error("quantum-depth, lstm-hidden, and head-width must be positive")
    if not 0.0 <= args.validation_fraction < 1.0:
        parser.error("validation-fraction must be in [0,1)")
    if args.patience < 1 or args.rz_noise_std < 0:
        parser.error("patience must be positive and rz-noise-std non-negative")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.data = args.data.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    sampling_rate, frame_seconds, hop_seconds = resolve_signal_settings(args, args.data)
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading data")
    signals, labels, groups, classes, source_name = load_dataset(
        args.data, subject_column=args.subject_column
    )
    print(
        f"Dataset={source_name} samples={len(signals)} channels={signals.shape[1]} "
        f"points={signals.shape[2]} subjects={len(np.unique(groups))} classes={len(classes)}"
    )
    print(
        f"Features: fs={sampling_rate:g} Hz, frame={frame_seconds:g}s, "
        f"hop={hop_seconds:g}s, node_features={len(FEATURE_NAMES)}"
    )
    features = build_feature_cache(
        signals,
        args.data,
        args.output,
        sampling_rate,
        frame_seconds,
        hop_seconds,
        args.feature_chunk_size,
    )
    if features.ndim != 4 or features.shape[-1] != len(FEATURE_NAMES):
        raise RuntimeError(f"invalid cached feature shape {features.shape}")

    device = torch.device("cpu")
    print("PennyLane simulator=default.qubit | training device=CPU")
    splits = make_loso_splits(groups)
    if args.max_folds is not None:
        if args.max_folds < 1:
            raise ValueError("--max-folds must be at least 1")
        splits = splits[: args.max_folds]
    if not splits:
        raise RuntimeError("no LOSO folds could be created")

    config = {
        "source": source_name,
        "input_shape": list(signals.shape),
        "feature_shape": list(features.shape),
        "feature_names": list(FEATURE_NAMES),
        "class_mapping": {str(index): name for index, name in enumerate(classes)},
        "sampling_rate": sampling_rate,
        "frame_seconds": frame_seconds,
        "hop_seconds": hop_seconds,
        "model": "channel attention -> BiLSTM temporal fusion -> 4-qubit PennyLane VQC -> residual MLP",
        "quantum_backend": "PennyLane default.qubit, analytic shots=None, Torch backprop",
        "quantum_depth": args.quantum_depth,
        "quantum_entanglement": "cyclic CNOT ring",
        "data_reuploading": True,
        "training_protocol": "outer LOSO; source-subject inner validation; test subject evaluated once after checkpoint selection",
        "selection_metric": args.selection_metric,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "pennylane_version": qml.__version__,
        "numpy_version": np.__version__,
    }
    (args.output / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    results = []
    for fold, (source_indices, test_indices, subject) in enumerate(
        tqdm(splits, desc="LOSO folds", unit="fold")
    ):
        set_seed(args.seed + fold)
        result = train_one_fold(
            fold,
            subject,
            source_indices,
            test_indices,
            labels,
            groups,
            classes,
            features,
            args.output,
            args,
            device,
        )
        results.append(result)
        summary = ", ".join(
            f"{name}={value:.4f}" for name, value in result["test_metrics"].items()
        )
        print(f"Fold {fold} subject={subject}: {summary}")
        (args.output / "fold_results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )

    metric_names = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
    summary = {}
    for name in metric_names:
        values = np.asarray([row["test_metrics"][name] for row in results], dtype=float)
        summary[name] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    final = {
        "fold_count": len(results),
        "protocol": config["training_protocol"],
        "test_evaluations_per_fold": 1,
        "metrics": summary,
        "folds": results,
    }
    (args.output / "final_results.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )
    csv_path = args.output / "final_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        columns = [
            "fold",
            "target_subject",
            "n_train_windows",
            "n_validation_windows",
            "n_test_windows",
            "selected_epoch",
            *metric_names,
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "fold": row["fold"],
                    "target_subject": row["target_subject"],
                    "n_train_windows": row["n_train_windows"],
                    "n_validation_windows": row["n_validation_windows"],
                    "n_test_windows": row["n_test_windows"],
                    "selected_epoch": row["selected_epoch"],
                    **row["test_metrics"],
                }
            )
    print(f"Completed {len(results)} LOSO folds. Results: {args.output / 'final_results.json'}")
    for name in metric_names:
        item = summary[name]
        print(f"{name}: {item['mean']:.4f} +/- {item['std']:.4f}")


if __name__ == "__main__":
    main()
