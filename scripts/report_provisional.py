#!/usr/bin/env python3
"""Write a live, explicitly provisional LOSO summary for HybridNode runs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summary(values):
    if not values:
        return "—"
    return f"{np.mean(values):.2f} ± {np.std(values):.2f}"


def as_markdown(rows):
    columns = [
        "Dataset",
        "Completed Folds",
        "Final Accuracy (%)",
        "Active Folds",
        "Best-so-far Accuracy (%)",
        "Active Epoch Range",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    lines.extend("| " + " | ".join(str(row[column]) for column in columns) + " |" for row in rows)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="results/hybridnode_quantum")
    args = parser.parse_args()
    root = Path(args.output_root)
    rows = []

    for dataset in sorted(path for path in root.iterdir() if (path / "runs").is_dir()):
        complete, active, active_epochs = [], [], []
        for run in sorted((dataset / "runs").glob("quantum_fold*_seed*")):
            fold = run.name.split("_seed", 1)[0].replace("quantum_fold", "", 1)
            status = dataset / "runtime" / "job_status" / f"{fold}_7.code"
            selected = run / "target_selected_test.json"
            if status.exists() and status.read_text().strip() == "0" and selected.exists():
                complete.append(json.loads(selected.read_text())["accuracy"] * 100.0)
                continue
            epochs = run / "epochs.csv"
            if not epochs.exists() or not epochs.stat().st_size:
                continue
            try:
                frame = pd.read_csv(epochs)
            except pd.errors.EmptyDataError:
                continue
            if len(frame):
                active.append(float(frame["test_accuracy"].max() * 100.0))
                active_epochs.append(int(frame["epoch"].max() + 1))

        rows.append(
            {
                "Dataset": dataset.name,
                "Completed Folds": f"{len(complete)}/{len(list((dataset / 'splits').glob('fold_*.json')))}",
                "Final Accuracy (%)": summary(complete),
                "Active Folds": len(active),
                "Best-so-far Accuracy (%)": summary(active),
                "Active Epoch Range": "—" if not active_epochs else f"{min(active_epochs)}–{max(active_epochs)}",
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(root / "live_provisional_summary.csv", index=False)
    markdown = as_markdown(rows)
    (root / "live_provisional_summary.md").write_text(
        "# Live provisional target-selected results\n\n"
        "Completed rows are final for this campaign. Active rows are the maximum "
        "held-out target-subject test accuracy observed so far and will change.\n\n"
        + markdown
        + "\n"
    )
    print(markdown)


if __name__ == "__main__":
    main()
