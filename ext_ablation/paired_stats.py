#!/usr/bin/env python3
"""Fold-paired, multiplicity-controlled comparisons for complete ablations."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from ext_ablation.model import VARIANTS


def holm(values):
    """Holm-adjust p-values without an additional dependency."""
    order=np.argsort(values); adjusted=np.empty(len(values),float); running=0.
    for rank,index in enumerate(order):
        running=max(running,(len(values)-rank)*values[index]); adjusted[index]=min(1.,running)
    return adjusted


def fold_values(root, dataset, variant, metric, expected):
    out={}
    for fold in range(expected):
        run=root/dataset/'runs'/variant/f'fold_{fold}_seed_7'
        path=run/'strict_test_once.json'; status=root/dataset/'runtime'/'job_status'/f'{variant}_{fold}_7.code'
        if path.exists() and status.exists() and status.read_text().strip()=='0': out[fold]=json.loads(path.read_text())[metric]*100
    return out


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output-root',default='ext_ablation/results_strict_loso_seed7'); p.add_argument('--datasets',default='CHSZ Sleep-EDF-20'); p.add_argument('--data-root',default='.'); a=p.parse_args()
    root=Path(a.output_root); rows=[]
    for dataset in a.datasets.split():
        expected=pd.read_csv(Path(a.data_root)/dataset/'meta.csv')['subject'].nunique()
        for metric in ('accuracy','macro_f1'):
            base=fold_values(root,dataset,'full',metric,expected)
            candidates=[]
            for variant in VARIANTS[1:]:
                other=fold_values(root,dataset,variant,metric,expected)
                if len(base)==expected and len(other)==expected:
                    delta=np.asarray([base[i]-other[i] for i in range(expected)])
                    try: p_raw=float(wilcoxon(delta,alternative='two-sided',zero_method='wilcox',method='auto').pvalue)
                    except ValueError: p_raw=1.0
                    dz=float(delta.mean()/delta.std(ddof=1)) if delta.std(ddof=1)>0 else 0.0
                    candidates.append((variant,delta,p_raw,dz))
            if candidates:
                corrected=holm(np.asarray([x[2] for x in candidates]))
                for (variant,delta,p_raw,dz),p_holm in zip(candidates,corrected):
                    rows.append({'dataset':dataset,'metric':metric,'reference':'full','ablation':variant,
                                 'folds':expected,'full_minus_ablation_mean':delta.mean(),
                                 'full_minus_ablation_std':delta.std(ddof=0),'cohen_dz':dz,
                                 'wilcoxon_p_raw':p_raw,'wilcoxon_p_holm':p_holm})
    frame=pd.DataFrame(rows); frame.to_csv(root/'paired_strict_statistics.csv',index=False)
    lines=['# Paired strict-LOSO component comparisons','',
           'Each comparison pairs the same held-out subject fold. Delta is `full − ablation` in percentage points. '
           'Two-sided Wilcoxon signed-rank tests are Holm-adjusted within each dataset × metric family. '
           'A single seed is used, so these tests quantify held-out-subject consistency, not seed-level uncertainty.','']
    if len(frame):
        lines += ['| Dataset | Metric | Ablation | Folds | Delta (pp) | Cohen dz | Raw p | Holm p |','|---|---|---|---:|---:|---:|---:|---:|']
        for row in frame.itertuples(): lines.append(f'| {row.dataset} | {row.metric} | `{row.ablation}` | {row.folds} | {row.full_minus_ablation_mean:.2f} ± {row.full_minus_ablation_std:.2f} | {row.cohen_dz:.3f} | {row.wilcoxon_p_raw:.4g} | {row.wilcoxon_p_holm:.4g} |')
    else: lines.append('No comparison has both complete variants yet.')
    (root/'PAIRED_STRICT_STATISTICS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__': main()
