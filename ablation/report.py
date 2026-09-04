#!/usr/bin/env python3
"""Completion-only strict-LOSO statistics; partial folds are never pooled."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ablation.model import VARIANTS


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output-root',default='ablation/results_strict_loso_seed7'); parser.add_argument('--datasets',default='CHSZ Sleep-EDF-20'); parser.add_argument('--data-root',default='.'); args=parser.parse_args()
    root=Path(args.output_root); rows=[]
    for data in sorted(x for x in root.iterdir() if x.is_dir() and (x/'runs').exists()):
        if data.name not in set(args.datasets.split()): continue
        # Expected LOSO folds come from authoritative input metadata, never
        # from dynamically-created output split files.
        expected=pd.read_csv(Path(args.data_root)/data.name/'meta.csv')['subject'].nunique()
        for variant in VARIANTS:
            values=[]; variant_dir=data/'runs'/variant
            for run in variant_dir.glob('fold_*_seed_*') if variant_dir.exists() else []:
                _, fold, _, seed=run.name.split('_')
                metric=run/'strict_test_once.json'; status=data/'runtime'/'job_status'/f'{variant}_{fold}_{seed}.code'
                if metric.exists() and status.exists() and status.read_text().strip()=='0': values.append(json.loads(metric.read_text()))
            row={'dataset':data.name,'variant':variant,'complete_folds':len(values),'expected_folds':expected}
            for key in ('accuracy','balanced_accuracy','macro_f1','weighted_f1'):
                x=np.asarray([value[key]*100 for value in values],float)
                row[key+'_mean']=x.mean() if len(x)==expected and expected else np.nan
                row[key+'_std']=x.std() if len(x)==expected and expected else np.nan
            rows.append(row)
    frame=pd.DataFrame(rows); print(frame.to_string(index=False,float_format=lambda x:f'{x:.2f}')); frame.to_csv(root/'strict_loso_summary.csv',index=False)


if __name__=='__main__': main()
