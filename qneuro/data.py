"""Dataset, masks and source-only normalization for QNeuro-v2."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

QUANKAN = Path('/home/namphuongtran9196/intel_project/QuanKAN')
if str(QUANKAN) not in sys.path:
    sys.path.insert(0, str(QUANKAN))
from data.dataset import (load_seed_all, load_seediv_all, load_seedv_all,
                          load_seedvii_all, read_locs_to_3d)

DATASETS = ('seed', 'seediv', 'seedv', 'seedvii')
ROOTS = {'seed': '/home/namphuongtran9196/intel_project/datasets/SEED',
         'seediv': '/home/namphuongtran9196/intel_project/eeg_feature_smooth',
         'seedv': '/home/namphuongtran9196/intel_project/datasets/SEED_V',
         'seedvii': '/home/namphuongtran9196/intel_project/datasets/SEED_VII'}
CLASSES = {'seed': 3, 'seediv': 4, 'seedv': 5, 'seedvii': 7}
CHANNEL_NAMES, CHANNEL_POSITIONS = read_locs_to_3d(str(QUANKAN / 'channel_62_pos.locs'))
CHANNEL_POSITIONS = np.asarray(CHANNEL_POSITIONS, dtype=np.float32)
CHANNEL_POSITIONS = (CHANNEL_POSITIONS - CHANNEL_POSITIONS.mean(0)) / np.maximum(CHANNEL_POSITIONS.std(0), 1e-6)
CHANNEL_TO_INDEX = {name: index for index, name in enumerate(CHANNEL_NAMES)}


def lengths_from_x(x):
    valid = np.any(np.abs(x) > 0, axis=(2, 3))
    lengths = valid.sum(axis=1).astype(np.int64)
    if np.any(lengths <= 0):
        raise ValueError('found an empty EEG sequence')
    expected = np.arange(x.shape[1])[None, :] < lengths[:, None]
    if not np.array_equal(valid, expected):
        raise ValueError('padding is not trailing/contiguous')
    return lengths


class EEG(Dataset):
    def __init__(self, x, y, lengths, subject_ids):
        self.x = x.astype('float32', copy=False)
        self.y = y.astype('int64', copy=False)
        self.lengths = lengths.astype('int64', copy=False)
        self.subject_ids = subject_ids.astype('int64', copy=False)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return self.x[i], self.y[i], self.lengths[i], self.subject_ids[i]


def load(dataset, root=None):
    root = root or ROOTS[dataset]
    if dataset == 'seed': x, y, p, _ = load_seed_all(root, [1])
    elif dataset == 'seediv': x, y, p, _ = load_seediv_all(root, [1, 2, 3])
    elif dataset == 'seedv': x, y, p, _ = load_seedv_all(root)
    else: x, y, p, _ = load_seedvii_all(root)
    p = np.asarray(p)
    lengths = lengths_from_x(x)
    names = sorted(set(p), key=lambda s: int(str(s)[1:]))
    subject_to_id = {s: i for i, s in enumerate(names)}
    subject_ids = np.asarray([subject_to_id[s] for s in p], dtype=np.int64)
    return x, np.asarray(y), p, subject_ids, CLASSES[dataset]


def source_normalize(x, source_indices, lengths=None, mode='source_global'):
    """Fit band statistics on valid source values only; target never participates."""
    lengths = lengths_from_x(x) if lengths is None else np.asarray(lengths)
    valid = np.arange(x.shape[1])[None, :] < lengths[:, None]
    out = x.copy().astype(np.float32, copy=False)
    stats = []
    for band in range(x.shape[-1]):
        values = x[source_indices, :, :, band][valid[source_indices]]
        mean, std = float(values.mean()), float(max(values.std(), 1e-4))
        stats.append({'band': band, 'mean': mean, 'std': std, 'fit_samples': int(values.size)})
        z = (out[:, :, :, band] - mean) / std
        out[:, :, :, band] = np.where(valid[:, :, None], z, 0.0)
    return out, stats


def tensor_mask(lengths, t, device=None):
    lengths = torch.as_tensor(lengths, device=device)
    return torch.arange(t, device=lengths.device)[None, :] < lengths[:, None]
