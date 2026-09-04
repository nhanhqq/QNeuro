#!/usr/bin/env python3
"""Target-selected LOSO training for the independent PaperLite QNeuro model."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from qneuro_paperlite.features import FEATURE_NAMES, FEATURE_VERSION, build_paperlite_cache
from qneuro_paperlite.model import PaperLiteQNeuro, count_trainable_parameters
from src.data.split import make_splits
from src.evaluation.metrics import metrics
from src.utils.seed import seed_everything


def gradient_norm(model: torch.nn.Module, key: str) -> float:
    values = [
        parameter.grad.detach().norm()
        for name, parameter in model.named_parameters()
        if key in name and parameter.grad is not None
    ]
    return float(torch.stack(values).norm()) if values else 0.0


def load_to_gpu(features, indices, device, block_size=1024):
    tensor = torch.empty(
        (len(indices),) + tuple(features.shape[1:]),
        dtype=torch.float32,
        device=device,
    )
    for start in range(0, len(indices), block_size):
        stop = min(start + block_size, len(indices))
        block = np.array(features[indices[start:stop]], dtype=np.float32, copy=True)
        tensor[start:stop].copy_(torch.from_numpy(block).to(device))
    return tensor


@torch.no_grad()
def evaluate(model, x, y, loss_function, batch_size):
    model.eval()
    predictions = []
    probabilities = []
    total_loss = 0.0
    for start in range(0, len(y), batch_size):
        stop = min(start + batch_size, len(y))
        logits = model(x[start:stop])
        target = y[start:stop]
        total_loss += loss_function(logits, target).item() * (stop - start)
        predictions.append(logits.argmax(dim=1).cpu())
        probabilities.append(logits.softmax(dim=1).cpu())
    prediction = torch.cat(predictions).numpy()
    probability = torch.cat(probabilities).numpy()
    result = metrics(y.cpu().numpy(), prediction, probability)
    result["loss"] = total_loss / len(y)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--fs", type=float, required=True)
    parser.add_argument("--frame-seconds", type=float, required=True)
    parser.add_argument("--hop-seconds", type=float, required=True)
    parser.add_argument("--rz-noise-std", type=float, default=0.10)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--kan-smoothness", type=float, default=1e-4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("PaperLite quantum training requires CUDA")
    seed_everything(args.seed)
    device = torch.device("cuda")
    data_path = Path(args.data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    raw_labels = np.load(data_path / "labels.npy")
    classes = np.unique(raw_labels)
    labels = np.searchsorted(classes, raw_labels).astype(np.int64)
    metadata = pd.read_csv(data_path / "meta.csv")
    if "subject" not in metadata or len(metadata) != len(labels):
        raise RuntimeError("meta.csv must have one subject ID per EEG sample")
    groups = metadata["subject"].to_numpy()
    splits = make_splits(labels, groups, output / "splits")
    if not 0 <= args.fold < len(splits):
        raise ValueError(f"fold must be in [0,{len(splits)-1}]")
    split = splits[args.fold]
    train_indices = np.asarray(split["train_indices"], dtype=np.int64)
    test_indices = np.asarray(split["test_indices"], dtype=np.int64)
    if set(split["train_subjects"]) & set(split["test_subjects"]):
        raise AssertionError("subject leakage")
    if len(split["test_subjects"]) != 1 or "validation_subjects" in split:
        raise AssertionError("protocol must be one target subject with no validation")

    cache_path = build_paperlite_cache(
        data_path / "X.npy",
        data_path / "cache",
        args.fs,
        args.frame_seconds,
        args.hop_seconds,
    )
    features = np.load(cache_path, mmap_mode="r")
    if features.ndim != 4 or features.shape[-1] != 11 or len(features) != len(labels):
        raise RuntimeError(f"invalid PaperLite cache shape {features.shape}")

    artifact = output / "artifacts" / f"fold_{args.fold}" / f"seed_{args.seed}"
    artifact.mkdir(parents=True, exist_ok=True)
    train_x = load_to_gpu(features, train_indices, device)
    test_x = load_to_gpu(features, test_indices, device)
    source_mean = train_x.mean(dim=(0, 1, 2))
    source_scale = train_x.var(dim=(0, 1, 2), unbiased=False).clamp_min(1e-12).sqrt()
    train_x.sub_(source_mean).div_(source_scale)
    test_x.sub_(source_mean).div_(source_scale)
    torch.save(
        {"mean": source_mean.cpu(), "scale": source_scale.cpu()},
        artifact / "source_only_gpu_scaler.pt",
    )
    (artifact / "preprocessing.json").write_text(
        json.dumps(
            {
                "feature_version": FEATURE_VERSION,
                "feature_names": list(FEATURE_NAMES),
                "fit_subjects": split["train_subjects"],
                "excluded_target_subject": split["test_subjects"],
                "pca": False,
                "graph_matrix": False,
                "fit_scope": "source subjects only",
                "scaler_backend": "torch_cuda",
            },
            indent=2,
        )
    )

    train_y = torch.as_tensor(labels[train_indices], dtype=torch.long, device=device)
    test_y = torch.as_tensor(labels[test_indices], dtype=torch.long, device=device)

    counts = np.bincount(labels[train_indices], minlength=len(classes))
    if np.any(counts == 0):
        raise RuntimeError(f"source subjects miss classes: {counts.tolist()}")
    class_weights = len(train_indices) / (len(classes) * counts)
    imbalance_ratio = float(counts.max() / counts.min())
    # Tempered source-only weights retain minority signal without forcing each
    # epoch to a 50/50 prior.  Full inverse-frequency sampling was empirically
    # over-correcting CHSZ and increasing target-fold variance.
    balanced_sampling = False
    tempered_weights = np.sqrt(class_weights)
    loss_weights = torch.as_tensor(tempered_weights, dtype=torch.float32, device=device)

    model = PaperLiteQNeuro(
        len(classes), num_channels=features.shape[2], rz_noise_std=args.rz_noise_std
    ).to(device)
    parameter_count = count_trainable_parameters(model)
    if parameter_count >= 1000:
        raise AssertionError(f"trainable parameter budget violated: {parameter_count}")
    loss_function = torch.nn.CrossEntropyLoss(
        weight=loss_weights, label_smoothing=args.label_smoothing
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    run = output / "runs" / f"quantum_fold{args.fold}_seed{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    best_path = run / "best.pt"
    last_path = run / "last.pt"
    start_epoch = 0
    best_accuracy = -1.0
    if last_path.exists():
        state = torch.load(last_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_accuracy = state["best_target_selected_test_accuracy"]

    fields = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "learning_rate",
        "grad_channel_attention",
        "grad_bilstm",
        "grad_quantum_projection",
        "grad_quantum",
        "grad_classifier",
        "epoch_seconds",
    ]
    epoch_log = run / "epochs.csv"
    write_header = not epoch_log.exists() or epoch_log.stat().st_size == 0
    final_epoch = min(args.epochs, start_epoch + 1) if args.dry_run else args.epochs
    with epoch_log.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for epoch in range(start_epoch, final_epoch):
            start_time = time.time()
            model.train()
            order = torch.randperm(len(train_y), device=device)
            predictions = []
            targets = []
            total_loss = 0.0
            gradients = {}
            for batch_start in range(0, len(order), args.batch_size):
                batch_indices = order[batch_start : batch_start + args.batch_size]
                target = train_y[batch_indices]
                optimizer.zero_grad(set_to_none=True)
                logits = model(train_x[batch_indices])
                loss = loss_function(logits, target)
                regularized_loss = loss + args.kan_smoothness * model.classifier.smoothness_penalty()
                regularized_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                gradients = {
                    "grad_channel_attention": gradient_norm(model, "channel_attention"),
                    "grad_bilstm": gradient_norm(model, "bilstm"),
                    "grad_quantum_projection": gradient_norm(model, "to_quantum"),
                    "grad_quantum": gradient_norm(model, "quantum"),
                    "grad_classifier": gradient_norm(model, "classifier"),
                }
                optimizer.step()
                total_loss += loss.item() * len(target)
                predictions.append(logits.argmax(dim=1).detach().cpu())
                targets.append(target.detach().cpu())

            train_prediction = torch.cat(predictions).numpy()
            train_target = torch.cat(targets).numpy()
            train_metrics = metrics(train_target, train_prediction)
            train_metrics["loss"] = total_loss / len(train_y)

            # Required protocol: train exactly one epoch, then evaluate the
            # held-out target subject immediately.  There is no validation set.
            test_metrics = evaluate(model, test_x, test_y, loss_function, args.batch_size)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_weighted_f1": test_metrics["weighted_f1"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                **gradients,
                "epoch_seconds": time.time() - start_time,
            }
            writer.writerow(row)
            handle.flush()
            scheduler.step()

            improved = test_metrics["accuracy"] > best_accuracy
            if improved:
                best_accuracy = test_metrics["accuracy"]
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "current_test_metrics": test_metrics,
                "best_target_selected_test_accuracy": best_accuracy,
                "config": vars(args),
                "label_mapping": classes.tolist(),
            }
            torch.save(state, last_path)
            if improved:
                torch.save(state, best_path)
                (run / "target_selected_test.json").write_text(
                    json.dumps(test_metrics, indent=2)
                )

    (run / "run.json").write_text(
        json.dumps(
            {
                "model": "HybridNode11 -> channel attention -> BiLSTM temporal fusion -> 4Q VQC -> KAN",
                "trainable_parameters": parameter_count,
                "parameter_budget": "strictly less than 1000",
                "pca": False,
                "dense_input_bottleneck": False,
                "node_identity_preserved_before_attention": True,
                "num_workers": 0,
                "gpu_resident_fold_features": True,
                "source_class_counts": counts.tolist(),
                "class_balance": "source-only sqrt inverse-frequency weighted CE; no oversampling",
                "source_tempered_class_weights": tempered_weights.tolist(),
                "protocol": "train one epoch, test the single target subject, repeat; no validation",
                "selection": "maximum target-subject test accuracy (target-selected and scientifically biased)",
                "seed": args.seed,
                "device": torch.cuda.get_device_name(),
                "python": platform.python_version(),
                "torch": torch.__version__,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
