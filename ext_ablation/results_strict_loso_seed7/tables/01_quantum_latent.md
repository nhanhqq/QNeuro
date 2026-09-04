# Table 1: quantum_latent

Same strict-LOSO protocol as `../PROTOCOL.md`.

## CHSZ

| Variant | Intervention | Folds | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|---:|---:|
| `full` | Independent matched reference. | 27/27 | 90.88 +/- 7.36 | 80.54 +/- 17.46 | 75.42 +/- 17.68 |
| `classical_latent` | Bias-free classical 4-to-4 control. | 27/27 | 89.81 +/- 10.05 | 79.12 +/- 17.38 | 73.72 +/- 19.03 |
| `quantum_depth_1` | One VQC layer. | 27/27 | 90.59 +/- 8.63 | 80.90 +/- 17.57 | 75.64 +/- 18.26 |
| `quantum_depth_3` | Three VQC layers. | 27/27 | 92.26 +/- 7.25 | 81.36 +/- 17.91 | 77.04 +/- 18.50 |

## Sleep-EDF-20

| Variant | Intervention | Folds | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|---:|---:|
| `full` | Independent matched reference. | 20/20 | 91.08 +/- 3.21 | 76.58 +/- 5.04 | 73.06 +/- 5.43 |
| `classical_latent` | Bias-free classical 4-to-4 control. | 20/20 | 90.67 +/- 3.87 | 76.69 +/- 5.77 | 73.04 +/- 6.59 |
| `quantum_depth_1` | One VQC layer. | 20/20 | 90.34 +/- 3.60 | 74.04 +/- 5.99 | 70.97 +/- 6.12 |
| `quantum_depth_3` | Three VQC layers. | 20/20 | 90.19 +/- 3.99 | 75.26 +/- 6.42 | 71.89 +/- 6.07 |

