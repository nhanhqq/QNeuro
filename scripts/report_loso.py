#!/usr/bin/env python3
"""Report the actual dynamic LOSO campaign; no hard-coded seeds or folds."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', default='results/all_quantum')
    args = parser.parse_args()
    root = Path(args.output_root)
    rows = []
    for data_dir in sorted(p for p in root.iterdir() if p.is_dir() and (p / 'runs').is_dir()):
        complete = []
        active = []
        for run in sorted((data_dir / 'runs').glob('quantum_fold*_seed*')):
            target = run / 'target_selected_test.json'
            suffix = run.name[len('quantum_fold'):]
            fold, seed_part = suffix.split('_seed', 1)
            code = data_dir / 'runtime' / 'job_status' / f'{fold}_{seed_part}.code'
            finished = code.exists() and code.read_text().strip() == '0'
            if finished and target.exists():
                metric = json.loads(target.read_text())
                complete.append(metric)
            elif (run / 'epochs.csv').exists():
                try:
                    active.append(len(pd.read_csv(run / 'epochs.csv')))
                except pd.errors.EmptyDataError:
                    # A worker creates the file before writing its first row.
                    # Treat that short interval as active rather than failing
                    # the whole live report.
                    active.append(0)
        expected = len(list((data_dir / 'splits').glob('fold_*.json')))
        def stat(key):
            values = [m[key] * 100.0 for m in complete]
            return (float(np.mean(values)), float(np.std(values))) if values else (float('nan'), float('nan'))
        acc, acc_std = stat('accuracy')
        bacc, bacc_std = stat('balanced_accuracy')
        f1, f1_std = stat('macro_f1')
        rows.append({'dataset': data_dir.name, 'complete_folds': len(complete), 'expected_loso_folds': expected,
                     'active_folds': len(active), 'active_epoch_min': min(active) if active else None,
                     'active_epoch_max': max(active) if active else None,
                     'accuracy_mean': acc, 'accuracy_std': acc_std,
                     'balanced_accuracy_mean': bacc, 'balanced_accuracy_std': bacc_std,
                     'macro_f1_mean': f1, 'macro_f1_std': f1_std})
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    frame.to_csv(root / 'live_loso_summary.csv', index=False)


if __name__ == '__main__':
    main()
