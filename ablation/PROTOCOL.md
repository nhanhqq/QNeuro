# Strict HybridNode11 ablation protocol

This directory is independent of every existing trainer, output root, cache, and result.

The campaign scope is intentionally restricted to `CHSZ` and `Sleep-EDF-20`, the two
datasets whose prior completed accuracy summaries exceeded 90%. Any earlier partial
output for another dataset is retained but excluded from this campaign and its report.

For every `(dataset, variant, held-out subject)`, the deterministic single seed is `7`.
The outer split holds out exactly one subject. The remaining N-1 source samples are split
stratified 80/20 into source training and source validation. Feature scaling and class
weights are fit only on source training. Epoch selection uses maximum source-validation
macro-F1 with patience 12. Only after selection, the held-out subject is evaluated once.
There is no target-selected checkpoint and no target metric in epoch logs.

The primary model is exactly `HybridNode11 -> learned channel attention -> BiLSTM
endpoint/mean fusion -> 4-qubit, 2-layer ring VQC with re-uploading -> spline KAN`.
The ten preregistered component tests are: full; bias-free classical latent control;
no quantum entanglement; no quantum data re-uploading; uniform channel pooling; no
BiLSTM; linear classifier replacing KAN; no spectral features; no Hjorth features;
and no correlation sketches. All use identical data, outer folds, seed, optimizer,
epochs, read-only feature cache, and source-only selection rule except for the named intervention.

`AB_MAX_PARALLEL=8` is the requested parallelism limit. Every subprocess sets one BLAS thread;
the Python training code creates no DataLoader, hence `num_workers=0`. The trainer never
creates or rebuilds caches. Scheduler admission
uses cgroup RAM plus a 2 GiB VRAM reserve. Results are complete only with exit code zero,
source-selected checkpoint, and `strict_test_once.json`; `report.py` excludes partial runs.
