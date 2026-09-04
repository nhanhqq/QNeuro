import json
from pathlib import Path
import numpy as np
from .spectral import spectral_features,BANDS,FEATURE_NAMES
def build_cache(x_path, cache_dir, fs=128., chunk=128, frame_seconds=1., hop_seconds=.5):
 x=np.load(x_path,mmap_mode='r')
 if x.ndim!=3: raise RuntimeError('CHSZ package contains precomputed features rather than time-domain EEG. FFT/BiLSTM temporal pipeline requires original EEG.')
 p=Path(cache_dir);p.mkdir(parents=True,exist_ok=True); out=p/'chsz_spectral.npy'
 first=spectral_features(x[:1],fs,frame_seconds,hop_seconds);mm=np.lib.format.open_memmap(out,mode='w+',dtype='float32',shape=(len(x),)+first.shape[1:])
 for i in range(0,len(x),chunk): mm[i:i+chunk]=spectral_features(x[i:i+chunk],fs,frame_seconds,hop_seconds)
 del mm
 (p/'chsz_spectral_config.json').write_text(json.dumps({'sampling_rate':fs,'frame_seconds':frame_seconds,'hop_seconds':hop_seconds,'frame_length':int(round(fs*frame_seconds)),'hop_length':int(round(fs*hop_seconds)),'frequency_bands':BANDS,'number_channels':int(x.shape[1]),'spectral_feature_names':FEATURE_NAMES,'source_X_shape':list(x.shape)},indent=2))
 return out
