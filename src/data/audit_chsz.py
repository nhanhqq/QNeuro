"""Non-destructive CHSZ package audit.  This is deliberately run before FFT."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def audit(data_dir: str, results_dir: str, seed: int = 7):
    d, out = Path(data_dir), Path(results_dir); out.mkdir(parents=True, exist_ok=True)
    X=np.load(d/'X.npy', mmap_mode='r'); y=np.load(d/'labels.npy'); meta=pd.read_csv(d/'meta.csv')
    subject_col=next((c for c in ('subject','subject_id','patient','patient_id') if c in meta),None)
    if subject_col is None: raise ValueError('meta.csv has no supported subject identifier')
    classes, counts=np.unique(y,return_counts=True)
    rng=np.random.default_rng(seed); examples=[]
    for i in rng.choice(len(y),size=min(10,len(y)),replace=False):
        a=X[i]; examples.append({'index':int(i),'shape':list(a.shape),'mean':float(a.mean()),'std':float(a.std()),'min':float(a.min()),'max':float(a.max()),'label':str(y[i]),'subject':str(meta.iloc[i][subject_col])})
    subject=pd.crosstab(meta[subject_col],y).reset_index(); subject['samples']=meta.groupby(subject_col).size().values
    subject.to_csv(out/'subject_distribution.csv',index=False)
    pd.DataFrame({'label':classes,'samples':counts}).to_csv(out/'class_distribution.csv',index=False)
    report={'source':str(d),'X_shape':list(X.shape),'X_dtype':str(X.dtype),'y_shape':list(y.shape),'y_dtype':str(y.dtype),'unique_labels':[str(v) for v in classes],'samples_per_label':{str(k):int(v) for k,v in zip(classes,counts)},'meta_columns':meta.columns.tolist(),'meta_shape':list(meta.shape),'subject_column':subject_col,'unique_subjects':int(meta[subject_col].nunique()),'X_ndim':int(X.ndim),'X_min':float(X.min()),'X_max':float(X.max()),'X_mean':float(X.mean()),'X_std':float(X.std()),'nan_count':int(np.isnan(X).sum()),'inf_count':int(np.isinf(X).sum()),'examples':examples,'sample_representation':('windowed time-domain EEG [channels,time]; FFT temporal pipeline is applicable' if X.ndim==3 else 'precomputed 2-D features; FFT/BiLSTM temporal pipeline MUST NOT run')}
    (out/'data_audit.json').write_text(json.dumps(report,indent=2))
    return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',default='CHSZ');p.add_argument('--results',default='results');a=p.parse_args();print(json.dumps(audit(a.data,a.results),indent=2))
