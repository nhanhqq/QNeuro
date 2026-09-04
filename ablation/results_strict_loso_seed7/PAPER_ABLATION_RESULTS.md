# Strict-LOSO HybridNode11 Ablation Results

## Protocol

One deterministic seed (7) is used for every fold. The outer split holds out exactly one subject. Within the remaining N-1 source samples, a deterministic stratified 80/20 source-pool split selects the epoch with highest source-validation macro-F1. The held-out target subject is evaluated exactly once after that selection. No epoch log contains a target metric; no target-selected checkpoint is used. Feature scaling and class weights fit source-training samples only. `num_workers=0`; at most four folds run concurrently.

All values are percent mean ± population standard deviation across held-out subjects. `--` means incomplete and is intentionally not interpreted.

## Interventions

1. `full` — HybridNode11 + learned channel attention + BiLSTM + 4Q ring VQC + KAN.
2. `classical_latent` — Replace the 4Q VQC by a bias-free classical 4-to-4 latent control.
3. `no_entanglement` — Keep VQC depth/re-uploading, remove all CNOT entanglement.
4. `no_reupload` — Encode data in the first VQC layer only; later layers have no data re-uploading.
5. `uniform_channel_pool` — Replace learned channel attention with fixed uniform averaging.
6. `no_bilstm` — Replace BiLSTM endpoint/mean fusion with temporal mean pooling.
7. `linear_classifier` — Replace spline KAN classifier with a linear classifier.
8. `no_spectral` — Mask five band powers plus spectral and differential entropy.
9. `no_hjorth` — Mask Hjorth mobility and complexity.
10. `no_connectivity` — Mask mean channel and derivative-correlation sketches.

## CHSZ

| Variant | Folds | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |
|---|---:|---:|---:|---:|
| `full` | 27/27 | 90.88 ± 7.36 | 80.54 ± 17.46 | 75.42 ± 17.68 |
| `classical_latent` | 27/27 | 89.81 ± 10.05 | 79.12 ± 17.38 | 73.72 ± 19.03 |
| `no_entanglement` | 27/27 | 89.85 ± 10.14 | 80.51 ± 17.35 | 74.94 ± 18.65 |
| `no_reupload` | 27/27 | 90.61 ± 10.28 | 81.23 ± 18.48 | 75.79 ± 19.12 |
| `uniform_channel_pool` | 27/27 | 90.22 ± 8.01 | 80.06 ± 17.53 | 74.84 ± 17.22 |
| `no_bilstm` | 27/27 | 87.80 ± 10.80 | 72.78 ± 16.83 | 69.59 ± 16.87 |
| `linear_classifier` | 27/27 | 91.24 ± 8.71 | 81.17 ± 17.45 | 76.55 ± 18.48 |
| `no_spectral` | 27/27 | 81.17 ± 20.48 | 55.19 ± 12.34 | 49.94 ± 16.50 |
| `no_hjorth` | 27/27 | 90.74 ± 8.87 | 79.15 ± 17.28 | 75.39 ± 18.09 |
| `no_connectivity` | 27/27 | 92.36 ± 7.25 | 81.67 ± 17.95 | 77.37 ± 18.50 |

## Sleep-EDF-20

| Variant | Folds | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |
|---|---:|---:|---:|---:|
| `full` | 20/20 | 91.08 ± 3.21 | 76.58 ± 5.04 | 73.06 ± 5.43 |
| `classical_latent` | 20/20 | 90.67 ± 3.87 | 76.69 ± 5.77 | 73.04 ± 6.59 |
| `no_entanglement` | 20/20 | 90.44 ± 3.94 | 75.41 ± 6.36 | 72.15 ± 6.17 |
| `no_reupload` | 20/20 | 90.28 ± 3.99 | 75.34 ± 5.95 | 71.81 ± 6.32 |
| `uniform_channel_pool` | 20/20 | 91.27 ± 3.20 | 76.43 ± 4.91 | 73.45 ± 4.63 |
| `no_bilstm` | 20/20 | 89.37 ± 4.10 | 72.95 ± 5.27 | 69.65 ± 5.63 |
| `linear_classifier` | 20/20 | 90.54 ± 3.61 | 74.08 ± 5.89 | 71.04 ± 6.52 |
| `no_spectral` | 20/20 | 86.39 ± 4.92 | 65.46 ± 6.10 | 62.16 ± 6.61 |
| `no_hjorth` | 20/20 | 90.55 ± 4.18 | 75.62 ± 5.76 | 72.35 ± 6.04 |
| `no_connectivity` | 20/20 | 89.82 ± 4.93 | 74.89 ± 6.53 | 71.68 ± 6.42 |

## Interpretation boundary

These controlled one-factor interventions support component-level evidence only after both datasets and all folds are complete. This report does not pool datasets, does not treat a single seed as seed-level uncertainty, and does not claim a causal improvement from incomplete cells.
