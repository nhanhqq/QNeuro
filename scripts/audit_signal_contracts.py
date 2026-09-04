#!/usr/bin/env python3
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
 contracts=json.loads((ROOT/'configs/dataset_signal_contracts.json').read_text());rows=[]
 for name,c in contracts.items():
  d=ROOT/name;x=np.load(d/'X.npy',mmap_mode='r');y=np.load(d/'labels.npy');m=pd.read_csv(d/'meta.csv')
  expected=(c['channels'],c['samples']);ok=x.ndim==3 and tuple(x.shape[1:])==expected and len(x)==len(y)==len(m) and 'subject' in m
  actual_duration=x.shape[2]/c['sampling_rate'];expected_steps=1+(x.shape[2]-round(c['sampling_rate']*c['frame_seconds']))//round(c['sampling_rate']*c['hop_seconds'])
  rows.append({'dataset':name,'valid':bool(ok),'X_shape':list(x.shape),'dtype':str(x.dtype),'classes':int(len(np.unique(y))),'subjects':int(m.subject.nunique()),'sampling_rate':c['sampling_rate'],'duration_seconds':actual_duration,'frame_seconds':c['frame_seconds'],'hop_seconds':c['hop_seconds'],'sequence_steps':int(expected_steps),'preprocessing':c['preprocessing'],'evidence':c['evidence']})
  if not ok:raise RuntimeError(f'contract mismatch: {name}')
 out=ROOT/'results'/'dataset_signal_audit.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
