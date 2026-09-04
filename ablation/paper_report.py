#!/usr/bin/env python3
"""Render a conservative paper-ready strict-LOSO ablation report."""
import argparse
from pathlib import Path

import pandas as pd

from ablation.model import VARIANTS

DESCRIPTIONS={
    'full':'HybridNode11 + learned channel attention + BiLSTM + 4Q ring VQC + KAN.',
    'classical_latent':'Replace the 4Q VQC by a bias-free classical 4-to-4 latent control.',
    'no_entanglement':'Keep VQC depth/re-uploading, remove all CNOT entanglement.',
    'no_reupload':'Encode data in the first VQC layer only; later layers have no data re-uploading.',
    'uniform_channel_pool':'Replace learned channel attention with fixed uniform averaging.',
    'no_bilstm':'Replace BiLSTM endpoint/mean fusion with temporal mean pooling.',
    'linear_classifier':'Replace spline KAN classifier with a linear classifier.',
    'no_spectral':'Mask five band powers plus spectral and differential entropy.',
    'no_hjorth':'Mask Hjorth mobility and complexity.',
    'no_connectivity':'Mask mean channel and derivative-correlation sketches.',
}

def cell(row, metric):
    if row is None or int(row.complete_folds) != int(row.expected_folds): return '--'
    return f'{getattr(row, metric+"_mean"):.2f} ± {getattr(row, metric+"_std"):.2f}'

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output-root',default='ablation/results_strict_loso_seed7'); p.add_argument('--datasets',default='CHSZ Sleep-EDF-20'); a=p.parse_args()
    root=Path(a.output_root); summary=root/'strict_loso_summary.csv'
    if not summary.exists(): raise SystemExit('run ablation.report first')
    frame=pd.read_csv(summary); selected=a.datasets.split(); lines=[
        '# Strict-LOSO HybridNode11 Ablation Results', '',
        '## Protocol', '',
        'One deterministic seed (7) is used for every fold. The outer split holds out exactly one subject. '
        'Within the remaining N-1 source samples, a deterministic stratified 80/20 source-pool split selects the '
        'epoch with highest source-validation macro-F1. The held-out target subject is evaluated exactly once after '
        'that selection. No epoch log contains a target metric; no target-selected checkpoint is used. Feature scaling '
        'and class weights fit source-training samples only. `num_workers=0`; at most four folds run concurrently.', '',
        'All values are percent mean ± population standard deviation across held-out subjects. `--` means incomplete '
        'and is intentionally not interpreted.', '',
        '## Interventions', ''
    ]
    for i,v in enumerate(VARIANTS,1): lines.append(f'{i}. `{v}` — {DESCRIPTIONS[v]}')
    for dataset in selected:
        lines += ['', f'## {dataset}', '', '| Variant | Folds | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |', '|---|---:|---:|---:|---:|']
        data=frame[frame.dataset==dataset]
        for v in VARIANTS:
            hit=data[data.variant==v]
            row=hit.iloc[0] if len(hit) else None
            folds=f'{int(row.complete_folds)}/{int(row.expected_folds)}' if row is not None else '--'
            lines.append(f'| `{v}` | {folds} | {cell(row,"accuracy")} | {cell(row,"balanced_accuracy")} | {cell(row,"macro_f1")} |')
    lines += ['', '## Interpretation boundary', '',
        'These controlled one-factor interventions support component-level evidence only after both datasets and all '
        'folds are complete. This report does not pool datasets, does not treat a single seed as seed-level uncertainty, '
        'and does not claim a causal improvement from incomplete cells.']
    (root/'PAPER_ABLATION_RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__': main()
