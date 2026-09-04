# Paired strict-LOSO component comparisons

Each comparison pairs the same held-out subject fold. Delta is `full − ablation` in percentage points. Two-sided Wilcoxon signed-rank tests are Holm-adjusted within each dataset × metric family. A single seed is used, so these tests quantify held-out-subject consistency, not seed-level uncertainty.

| Dataset | Metric | Ablation | Folds | Delta (pp) | Cohen dz | Raw p | Holm p |
|---|---|---|---:|---:|---:|---:|---:|
| CHSZ | accuracy | `classical_latent` | 27 | 1.07 ± 7.47 | 0.140 | 0.6849 | 1 |
| CHSZ | accuracy | `no_entanglement` | 27 | 1.03 ± 8.01 | 0.126 | 0.6012 | 1 |
| CHSZ | accuracy | `no_reupload` | 27 | 0.27 ± 7.64 | 0.035 | 0.6373 | 1 |
| CHSZ | accuracy | `uniform_channel_pool` | 27 | 0.66 ± 2.84 | 0.228 | 0.1701 | 1 |
| CHSZ | accuracy | `no_bilstm` | 27 | 3.08 ± 6.90 | 0.438 | 0.1161 | 0.8126 |
| CHSZ | accuracy | `linear_classifier` | 27 | -0.36 ± 6.10 | -0.057 | 0.7751 | 1 |
| CHSZ | accuracy | `no_spectral` | 27 | 9.71 ± 17.28 | 0.551 | 0.009233 | 0.0831 |
| CHSZ | accuracy | `no_hjorth` | 27 | 0.14 ± 5.82 | 0.024 | 0.9772 | 1 |
| CHSZ | accuracy | `no_connectivity` | 27 | -1.48 ± 5.63 | -0.257 | 0.09434 | 0.7547 |
| CHSZ | macro_f1 | `classical_latent` | 27 | 1.70 ± 8.28 | 0.202 | 0.5034 | 1 |
| CHSZ | macro_f1 | `no_entanglement` | 27 | 0.48 ± 7.42 | 0.063 | 0.372 | 1 |
| CHSZ | macro_f1 | `no_reupload` | 27 | -0.37 ± 7.54 | -0.048 | 0.523 | 1 |
| CHSZ | macro_f1 | `uniform_channel_pool` | 27 | 0.58 ± 3.55 | 0.159 | 0.2311 | 1 |
| CHSZ | macro_f1 | `no_bilstm` | 27 | 5.83 ± 10.23 | 0.560 | 0.01506 | 0.1205 |
| CHSZ | macro_f1 | `linear_classifier` | 27 | -1.13 ± 6.41 | -0.173 | 0.5296 | 1 |
| CHSZ | macro_f1 | `no_spectral` | 27 | 25.48 ± 22.46 | 1.113 | 4.1e-05 | 0.000369 |
| CHSZ | macro_f1 | `no_hjorth` | 27 | 0.03 ± 6.28 | 0.005 | 0.8612 | 1 |
| CHSZ | macro_f1 | `no_connectivity` | 27 | -1.95 ± 6.46 | -0.296 | 0.09185 | 0.643 |
| Sleep-EDF-20 | accuracy | `classical_latent` | 20 | 0.40 ± 1.37 | 0.285 | 0.114 | 0.4216 |
| Sleep-EDF-20 | accuracy | `no_entanglement` | 20 | 0.63 ± 1.78 | 0.348 | 0.1054 | 0.4216 |
| Sleep-EDF-20 | accuracy | `no_reupload` | 20 | 0.79 ± 1.56 | 0.495 | 0.02664 | 0.1599 |
| Sleep-EDF-20 | accuracy | `uniform_channel_pool` | 20 | -0.19 ± 1.40 | -0.133 | 0.6215 | 0.6215 |
| Sleep-EDF-20 | accuracy | `no_bilstm` | 20 | 1.70 ± 2.04 | 0.814 | 0.002712 | 0.0217 |
| Sleep-EDF-20 | accuracy | `linear_classifier` | 20 | 0.53 ± 1.21 | 0.428 | 0.01362 | 0.09532 |
| Sleep-EDF-20 | accuracy | `no_spectral` | 20 | 4.68 ± 4.31 | 1.058 | 3.624e-05 | 0.0003262 |
| Sleep-EDF-20 | accuracy | `no_hjorth` | 20 | 0.52 ± 2.22 | 0.230 | 0.1769 | 0.4216 |
| Sleep-EDF-20 | accuracy | `no_connectivity` | 20 | 1.25 ± 2.72 | 0.448 | 0.03277 | 0.1638 |
| Sleep-EDF-20 | macro_f1 | `classical_latent` | 20 | 0.03 ± 2.46 | 0.010 | 0.7285 | 1 |
| Sleep-EDF-20 | macro_f1 | `no_entanglement` | 20 | 0.91 ± 2.06 | 0.429 | 0.07585 | 0.3793 |
| Sleep-EDF-20 | macro_f1 | `no_reupload` | 20 | 1.25 ± 2.78 | 0.438 | 0.07585 | 0.3793 |
| Sleep-EDF-20 | macro_f1 | `uniform_channel_pool` | 20 | -0.39 ± 2.41 | -0.156 | 0.6477 | 1 |
| Sleep-EDF-20 | macro_f1 | `no_bilstm` | 20 | 3.42 ± 2.72 | 1.222 | 8.202e-05 | 0.0006561 |
| Sleep-EDF-20 | macro_f1 | `linear_classifier` | 20 | 2.02 ± 3.04 | 0.648 | 0.00639 | 0.04473 |
| Sleep-EDF-20 | macro_f1 | `no_spectral` | 20 | 10.91 ± 6.73 | 1.579 | 3.815e-06 | 3.433e-05 |
| Sleep-EDF-20 | macro_f1 | `no_hjorth` | 20 | 0.71 ± 3.17 | 0.218 | 0.498 | 1 |
| Sleep-EDF-20 | macro_f1 | `no_connectivity` | 20 | 1.38 ± 2.33 | 0.577 | 0.01718 | 0.1031 |
