#!/usr/bin/env python3
"""Reject results that do not prove the strict source-validation contract."""
import argparse
import json
from pathlib import Path

from ablation.model import VARIANTS


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output-root',default='ablation/results_strict_loso_seed7'); p.add_argument('--datasets',default='CHSZ Sleep-EDF-20'); p.add_argument('--data-root',default='.'); a=p.parse_args(); root=Path(a.output_root)
    selected=set(a.datasets.split()); invalid=[]; complete=0; expected_configurations=len(selected)*len(VARIANTS)
    for data in sorted(x for x in root.iterdir() if x.is_dir() and (x/'runs').exists()):
        if data.name not in selected: continue
        expected=__import__('pandas').read_csv(Path(a.data_root)/data.name/'meta.csv')['subject'].nunique()
        for variant in VARIANTS:
            runs=list((data/'runs'/variant).glob('fold_*_seed_*')) if (data/'runs'/variant).exists() else []
            valid=0
            for run in runs:
                # A worker creates its run directory before terminal metadata.
                # That is an in-progress fold, not a protocol violation.
                if not (run/'run.json').exists():
                    continue
                try:
                    info=json.loads((run/'run.json').read_text()); split=info['split']
                    if info['target_selection'] or info['test_evaluations'] != 1 or info['num_workers'] != 0: raise ValueError('selection/worker contract')
                    if split['validation_scheme'] != 'stratified source-pool holdout' or split['test_evaluations'] != 1: raise ValueError('split contract')
                    if set(split['source_subjects']) & set(split['target_subjects']): raise ValueError('subject leakage')
                    if not (run/'strict_test_once.json').exists() or not (run/'final_selected_on_source_validation.pt').exists(): raise ValueError('missing terminal artifacts')
                    valid += 1
                except Exception as exc: invalid.append({'dataset':data.name,'variant':variant,'run':run.name,'error':str(exc)})
            if valid == expected and expected: complete += 1
    result={'complete_configurations':complete,'expected_configurations':expected_configurations,
            'invalid':invalid,'complete':not invalid and complete == expected_configurations}
    print(json.dumps(result,indent=2)); (root/'audit.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__': main()
