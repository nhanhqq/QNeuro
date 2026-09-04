"""Fast graph-free EEG features inspired by the Rashomon-GNN paper.

The paper explores a costly cross product of node features and connectivity
graphs.  PaperLite keeps the complementary spectral/Hjorth idea, but replaces
all CxC graph builders with two exact mean-correlation sketches.  The sketches
are O(C*T), never materialize an adjacency matrix, and are deterministic.
"""

from __future__ import annotations

import json
import mmap as mmap_module
import os
from pathlib import Path

import numpy as np
import torch


FEATURE_VERSION = "hybridnode11_v2"
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
FEATURE_NAMES = tuple(f"log_relative_{name}" for name in BANDS) + (
        "spectral_entropy",
        "differential_entropy",
        "hjorth_mobility",
        "hjorth_complexity",
        "mean_channel_correlation",
        "mean_derivative_correlation",
)


def _standardize_channels(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=-1, keepdims=True)
    rms = np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True))
    return centered / np.maximum(rms, 1e-12)


def _mean_off_diagonal_correlation(x: np.ndarray) -> np.ndarray:
    """Exact mean Pearson correlation without constructing [N,C,C]."""
    channels = x.shape[1]
    if channels < 2:
        return np.zeros(x.shape[0], dtype=np.float32)
    z = _standardize_channels(x)
    summed = z.sum(axis=1)
    all_pair_energy = np.mean(summed * summed, axis=-1)
    return ((all_pair_energy - channels) / (channels * (channels - 1))).astype(np.float32)


def paperlite_features(
    x: np.ndarray,
    fs: float,
    frame_seconds: float = 1.0,
    hop_seconds: float = 0.5,
) -> np.ndarray:
    """CPU reference: [N,C,T] EEG to node-preserving [N,L,C,11]."""
    one_sample = x.ndim == 2
    if one_sample:
        x = x[None]
    if x.ndim != 3:
        raise ValueError(f"expected [N,C,T] time-domain EEG, got shape {x.shape}")
    n_samples, channels, time_points = x.shape
    frame_length = int(round(float(fs) * frame_seconds))
    hop_length = int(round(float(fs) * hop_seconds))
    if frame_length < 4 or hop_length < 1 or time_points < frame_length:
        raise ValueError(
            f"invalid framing: T={time_points}, frame={frame_length}, hop={hop_length}"
        )

    starts = range(0, time_points - frame_length + 1, hop_length)
    starts = tuple(starts)
    window = np.hanning(frame_length).astype(np.float32)
    frequencies = np.fft.rfftfreq(frame_length, d=1.0 / float(fs))
    analysis_mask = (frequencies >= 1.0) & (frequencies < 45.0)
    band_masks = tuple(
        (frequencies >= low) & (frequencies < high) for low, high in BANDS.values()
    )
    if not analysis_mask.any() or any(not mask.any() for mask in band_masks):
        raise ValueError(f"sampling rate {fs} cannot represent the configured 1-45 Hz bands")

    output = np.empty(
        (n_samples, len(starts), channels, len(FEATURE_NAMES)), dtype=np.float32
    )
    for frame_index, start in enumerate(starts):
        frame = np.asarray(x[:, :, start : start + frame_length], dtype=np.float32)
        centered = frame - frame.mean(axis=-1, keepdims=True)
        variance = np.mean(centered * centered, axis=-1)

        power = np.abs(np.fft.rfft(centered * window, axis=-1)) ** 2
        total_power = power[:, :, analysis_mask].sum(axis=-1)
        band_power = np.stack(
            [power[:, :, mask].sum(axis=-1) for mask in band_masks], axis=-1
        )
        log_relative = np.log((band_power + 1e-12) / (total_power[..., None] + 1e-12))

        spectrum = power[:, :, analysis_mask]
        probability = spectrum / np.maximum(spectrum.sum(axis=-1, keepdims=True), 1e-12)
        spectral_entropy = -(probability * np.log(probability + 1e-12)).sum(axis=-1)

        first = np.diff(centered, axis=-1)
        second = np.diff(first, axis=-1)
        std_x = np.sqrt(np.maximum(variance, 1e-24))
        std_first = np.sqrt(np.maximum(np.mean(first * first, axis=-1), 1e-24))
        std_second = np.sqrt(np.maximum(np.mean(second * second, axis=-1), 1e-24))
        mobility = std_first / std_x
        complexity = (std_second / std_first) / np.maximum(mobility, 1e-12)
        differential_entropy = 0.5 * np.log(2.0 * np.pi * np.e * variance + 1e-24)

        correlation = _mean_off_diagonal_correlation(centered)[:, None, None]
        derivative_correlation = _mean_off_diagonal_correlation(first)[:, None, None]
        values = np.concatenate(
            (
                log_relative,
                spectral_entropy[..., None],
                differential_entropy[..., None],
                mobility[..., None],
                complexity[..., None],
                np.broadcast_to(correlation, (n_samples, channels, 1)),
                np.broadcast_to(derivative_correlation, (n_samples, channels, 1)),
            ),
            axis=-1,
        )
        output[:, frame_index] = np.nan_to_num(
            values, nan=0.0, posinf=30.0, neginf=-30.0
        ).astype(np.float32)

    return output[0] if one_sample else output


def _torch_mean_off_diagonal_correlation(x: torch.Tensor) -> torch.Tensor:
    channels = x.shape[1]
    if channels < 2:
        return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    centered = x - x.mean(dim=-1, keepdim=True)
    rms = centered.square().mean(dim=-1, keepdim=True).clamp_min(1e-24).sqrt()
    summed = (centered / rms).sum(dim=1)
    return (summed.square().mean(dim=-1) - channels) / (channels * (channels - 1))


def paperlite_features_cuda(
    x: torch.Tensor,
    fs: float,
    frame_seconds: float = 1.0,
    hop_seconds: float = 0.5,
) -> torch.Tensor:
    """CUDA implementation used by every production cache build."""
    if not x.is_cuda:
        raise RuntimeError("PaperLite production preprocessing is CUDA-only")
    if x.ndim != 3:
        raise ValueError(f"expected [N,C,T], got {tuple(x.shape)}")
    frame_length = int(round(float(fs) * frame_seconds))
    hop_length = int(round(float(fs) * hop_seconds))
    if frame_length < 4 or hop_length < 1 or x.shape[-1] < frame_length:
        raise ValueError("invalid temporal framing")
    starts = tuple(range(0, x.shape[-1] - frame_length + 1, hop_length))
    window = torch.hann_window(
        frame_length, periodic=False, dtype=torch.float32, device=x.device
    )
    frequencies = torch.fft.rfftfreq(frame_length, d=1.0 / float(fs), device=x.device)
    analysis_mask = (frequencies >= 1.0) & (frequencies < 45.0)
    band_masks = tuple(
        (frequencies >= low) & (frequencies < high) for low, high in BANDS.values()
    )
    output = torch.empty(
        (x.shape[0], len(starts), x.shape[1], len(FEATURE_NAMES)),
        dtype=torch.float32,
        device=x.device,
    )
    for frame_index, start in enumerate(starts):
        frame = x[:, :, start : start + frame_length].float()
        centered = frame - frame.mean(dim=-1, keepdim=True)
        variance = centered.square().mean(dim=-1)
        power = torch.fft.rfft(centered * window, dim=-1).abs().square()
        total_power = power[:, :, analysis_mask].sum(dim=-1)
        band_power = torch.stack(
            [power[:, :, mask].sum(dim=-1) for mask in band_masks], dim=-1
        )
        log_relative = torch.log(
            (band_power + 1e-12) / (total_power.unsqueeze(-1) + 1e-12)
        )
        spectrum = power[:, :, analysis_mask]
        probability = spectrum / spectrum.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        spectral_entropy = -(probability * torch.log(probability + 1e-12)).sum(dim=-1)
        first = torch.diff(centered, dim=-1)
        second = torch.diff(first, dim=-1)
        std_x = variance.clamp_min(1e-24).sqrt()
        std_first = first.square().mean(dim=-1).clamp_min(1e-24).sqrt()
        std_second = second.square().mean(dim=-1).clamp_min(1e-24).sqrt()
        mobility = std_first / std_x
        complexity = (std_second / std_first) / mobility.clamp_min(1e-12)
        differential_entropy = 0.5 * torch.log(2.0 * torch.pi * torch.e * variance + 1e-24)
        correlation = _torch_mean_off_diagonal_correlation(centered)[:, None, None]
        derivative_correlation = _torch_mean_off_diagonal_correlation(first)[:, None, None]
        values = torch.cat(
            (
                log_relative,
                spectral_entropy.unsqueeze(-1),
                differential_entropy.unsqueeze(-1),
                mobility.unsqueeze(-1),
                complexity.unsqueeze(-1),
                correlation.expand(-1, x.shape[1], -1),
                derivative_correlation.expand(-1, x.shape[1], -1),
            ),
            dim=-1,
        )
        output[:, frame_index] = torch.nan_to_num(
            values, nan=0.0, posinf=30.0, neginf=-30.0
        )
    return output


def build_paperlite_cache(
    x_path: str | Path,
    cache_dir: str | Path,
    fs: float,
    frame_seconds: float,
    hop_seconds: float,
    chunk_size: int = 128,
) -> Path:
    """Build or validate a deterministic mmap cache entirely with CUDA math."""
    if not torch.cuda.is_available():
        raise RuntimeError("PaperLite cache generation requires CUDA")
    x_path = Path(x_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / f"{FEATURE_VERSION}.npy"
    config_path = cache_dir / f"{FEATURE_VERSION}.json"
    x = np.load(x_path, mmap_mode="r")
    if x.ndim != 3:
        raise RuntimeError("PaperLite requires actual [N,C,T] time-domain EEG")
    source = x_path.stat()
    expected = {
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "sampling_rate": float(fs),
        "frame_seconds": float(frame_seconds),
        "hop_seconds": float(hop_seconds),
        "source_shape": list(x.shape),
        "source_size": source.st_size,
        "source_mtime_ns": source.st_mtime_ns,
        "compute_backend": "torch_cuda",
        "compute_device": torch.cuda.get_device_name(0),
    }
    if output_path.exists() and config_path.exists():
        try:
            current = json.loads(config_path.read_text())
            cached = np.load(output_path, mmap_mode="r")
            if (
                current == expected
                and cached.shape[0] == x.shape[0]
                and cached.shape[2] == x.shape[1]
                and cached.shape[-1] == len(FEATURE_NAMES)
            ):
                return output_path
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    device = torch.device("cuda")
    first = paperlite_features_cuda(
        torch.from_numpy(np.array(x[:1], copy=True)).to(device), fs, frame_seconds, hop_seconds
    )
    cache_mmap = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(x), first.shape[1], first.shape[2], first.shape[3]),
    )
    for start in range(0, len(x), chunk_size):
        stop = min(start + chunk_size, len(x))
        raw = torch.from_numpy(np.array(x[start:stop], copy=True)).to(device, non_blocking=False)
        values = paperlite_features_cuda(raw, fs, frame_seconds, hop_seconds)
        cache_mmap[start:stop] = values.cpu().numpy()
        del raw, values
    cache_mmap.flush()
    del cache_mmap, first
    torch.cuda.empty_cache()
    config_path.write_text(json.dumps(expected, indent=2))
    # mmap page cache is reclaimable, but explicitly release it between large
    # datasets so the 32 GiB cgroup cannot be killed by accumulated file cache.
    if hasattr(x, "_mmap") and hasattr(x._mmap, "madvise"):
        x._mmap.madvise(mmap_module.MADV_DONTNEED)
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        for path in (x_path, output_path):
            with path.open("rb") as handle:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return output_path
