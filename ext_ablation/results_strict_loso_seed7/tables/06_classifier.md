# Table 6: classifier

Same strict-LOSO protocol as `../PROTOCOL.md`.

## CHSZ

| Variant | Intervention | Folds | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|---:|---:|
| `full` | Independent matched reference. | 27/27 | 90.88 +/- 7.36 | 80.54 +/- 17.46 | 75.42 +/- 17.68 |
| `linear_classifier` | Linear head. | 27/27 | 91.24 +/- 8.71 | 81.17 +/- 17.45 | 76.55 +/- 18.48 |
| `mlp_classifier` | Two-layer tanh MLP head. | 27/27 | 91.15 +/- 8.41 | 81.52 +/- 17.59 | 76.34 +/- 18.35 |

## Sleep-EDF-20

| Variant | Intervention | Folds | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|---:|---:|
| `full` | Independent matched reference. | 20/20 | 91.08 +/- 3.21 | 76.58 +/- 5.04 | 73.06 +/- 5.43 |
| `linear_classifier` | Linear head. | 20/20 | 90.54 +/- 3.61 | 74.08 +/- 5.89 | 71.04 +/- 6.52 |
| `mlp_classifier` | Two-layer tanh MLP head. | 20/20 | 90.73 +/- 3.54 | 75.49 +/- 6.24 | 72.31 +/- 5.90 |

