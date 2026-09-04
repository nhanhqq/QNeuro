#!/usr/bin/env python3
"""Render ten completion-only strict-LOSO result-table files."""
import argparse
from pathlib import Path
import pandas as pd
from ext_ablation.model import FAMILIES

DESCRIPTIONS = {
 "full":"Independent matched reference.", "classical_latent":"Bias-free classical 4-to-4 control.", "quantum_depth_1":"One VQC layer.", "quantum_depth_3":"Three VQC layers.", "no_entanglement":"Remove CNOT gates.", "linear_entanglement":"Linear CNOT chain.", "no_reupload":"Encode only in layer 1.", "frozen_reupload_scale":"Freeze encoding scales at one.", "uniform_channel_pool":"Uniform channel mean.", "max_channel_pool":"Maximum channel pool.", "no_bilstm":"Temporal mean.", "endpoint_bilstm":"BiLSTM endpoint only.", "mean_bilstm":"BiLSTM mean only.", "linear_classifier":"Linear head.", "mlp_classifier":"Two-layer tanh MLP head.", "no_spectral":"Remove spectral + entropy.", "no_bandpowers":"Remove five band powers.", "no_entropy":"Remove two entropy features.", "no_hjorth":"Remove both Hjorth features.", "no_hjorth_mobility":"Remove Hjorth mobility.", "no_hjorth_complexity":"Remove Hjorth complexity.", "no_connectivity":"Remove both correlation sketches.", "no_mean_connectivity":"Remove mean correlation.", "no_derivative_connectivity":"Remove derivative correlation.", "no_rz_augmentation":"RZ noise = 0.", "high_rz_augmentation":"RZ noise = 0.20 rad."}

def cell(row, metric):
    if row is None or int(row.complete_folds) != int(row.expected_folds): return '--'
    return f'{getattr(row,metric+"_mean"):.2f} +/- {getattr(row,metric+"_std"):.2f}'

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output-root',default='ext_ablation/results_strict_loso_seed7'); p.add_argument('--datasets',default='CHSZ Sleep-EDF-20'); a=p.parse_args()
    root=Path(a.output_root); source=root/'strict_loso_summary.csv'
    if not source.exists(): raise SystemExit('run ext_ablation.report first')
    frame=pd.read_csv(source); tables=root/'tables'; tables.mkdir(exist_ok=True)
    index=['# Extended HybridNode11 strict-LOSO results','','Seed 7; strict LOSO; deterministic source-pool 80/20 validation; source-validation macro-F1 selection; one held-out test evaluation; train-only scaling/class weights; `num_workers=0`. Results are percent mean +/- population standard deviation. `--` is incomplete and excluded.','','## Ten result tables','']
    for number,(family,variants) in enumerate(FAMILIES.items(),1):
        filename=f'{number:02d}_{family}.md'; index.append(f'{number}. [{family}](tables/{filename})')
        lines=[f'# Table {number}: {family}','','Same strict-LOSO protocol as `../PROTOCOL.md`.','']
        for dataset in a.datasets.split():
            lines += [f'## {dataset}','','| Variant | Intervention | Folds | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |','|---|---|---:|---:|---:|---:|']
            data=frame[frame.dataset==dataset]
            for variant in variants:
                hit=data[data.variant==variant]; row=hit.iloc[0] if len(hit) else None
                folds=f'{int(row.complete_folds)}/{int(row.expected_folds)}' if row is not None else '--'
                lines.append(f'| `{variant}` | {DESCRIPTIONS[variant]} | {folds} | {cell(row,"accuracy")} | {cell(row,"balanced_accuracy")} | {cell(row,"macro_f1")} |')
            lines.append('')
        (tables/filename).write_text('\n'.join(lines)+'\n')
    (root/'EXTENDED_ABLATION_RESULTS.md').write_text('\n'.join(index)+'\n')

if __name__=='__main__': main()
