# Table 7: spectral_features

Same strict-LOSO protocol as `../PROTOCOL.md`.

## CHSZ

| Variant | Intervention | Folds | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|---:|---:|
| `full` | Independent matched reference. | 27/27 | 90.88 +/- 7.36 | 80.54 +/- 17.46 | 75.42 +/- 17.68 |
| `no_spectral` | Remove spectral + entropy. | 27/27 | 81.17 +/- 20.48 | 55.19 +/- 12.34 | 49.94 +/- 16.50 |
| `no_bandpowers` | Remove five band powers. | 27/27 | 90.44 +/- 10.57 | 80.46 +/- 17.49 | 75.31 +/- 18.86 |
| `no_entropy` | Remove two entropy features. | 27/27 | 86.46 +/- 10.66 | 74.95 +/- 15.73 | 70.27 +/- 16.33 |

## Sleep-EDF-20

| Variant | Intervention | Folds | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|---:|---:|
| `full` | Independent matched reference. | 20/20 | 91.08 +/- 3.21 | 76.58 +/- 5.04 | 73.06 +/- 5.43 |
| `no_spectral` | Remove spectral + entropy. | 20/20 | 86.39 +/- 4.92 | 65.46 +/- 6.10 | 62.16 +/- 6.61 |
| `no_bandpowers` | Remove five band powers. | 20/20 | 88.14 +/- 3.77 | 72.85 +/- 5.27 | 68.81 +/- 5.44 |
| `no_entropy` | Remove two entropy features. | 20/20 | 90.00 +/- 2.96 | 74.24 +/- 5.23 | 70.58 +/- 5.58 |

