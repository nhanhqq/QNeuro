"""Source-only LOSO loading shared with the verified QuanKAN SEED loaders."""
from pathlib import Path
import sys
import numpy as np
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
CHANNEL_TO_INDEX = {name: index for index, name in enumerate(CHANNEL_NAMES)}

class EEG(Dataset):
    def __init__(self, x, y): self.x, self.y = x.astype('float32', copy=False), y.astype('int64', copy=False)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

def load(dataset, root=None):
    root = root or ROOTS[dataset]
    if dataset == 'seed': x,y,p,_ = load_seed_all(root, [1])
    elif dataset == 'seediv': x,y,p,_ = load_seediv_all(root, [1,2,3])
    elif dataset == 'seedv': x,y,p,_ = load_seedv_all(root)
    else: x,y,p,_ = load_seedvii_all(root)
    return x, y, np.asarray(p), CLASSES[dataset]

def source_normalize(x, source_indices):
    """Fit each DE band only on N-1 source subjects; target never participates."""
    valid = np.any(np.abs(x) > 0, axis=(2,3)); out = x.copy()
    stats=[]
    for band in range(x.shape[-1]):
        values=x[source_indices,:,:,band][valid[source_indices]]
        mean, std=float(values.mean()), float(max(values.std(),1e-4)); stats.append((mean,std))
        z=(out[:,:,:,band]-mean)/std; out[:,:,:,band]=np.where(valid[:,:,None],z,0.)
    return out, stats
