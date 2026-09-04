# Table 2: entanglement

Same strict-LOSO protocol as `../PROTOCOL.md`.

## CHSZ

| Variant | Intervention | Folds | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|---:|---:|
| `full` | Independent matched reference. | 27/27 | 90.88 +/- 7.36 | 80.54 +/- 17.46 | 75.42 +/- 17.68 |
| `no_entanglement` | Remove CNOT gates. | 27/27 | 89.85 +/- 10.14 | 80.51 +/- 17.35 | 74.94 +/- 18.65 |
| `linear_entanglement` | Linear CNOT chain. | 27/27 | 90.32 +/- 10.54 | 79.39 +/- 17.69 | 75.05 +/- 19.14 |

## Sleep-EDF-20

| Variant | Intervention | Folds | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|---:|---:|
| `full` | Independent matched reference. | 20/20 | 91.08 +/- 3.21 | 76.58 +/- 5.04 | 73.06 +/- 5.43 |
| `no_entanglement` | Remove CNOT gates. | 20/20 | 90.44 +/- 3.94 | 75.41 +/- 6.36 | 72.15 +/- 6.17 |
| `linear_entanglement` | Linear CNOT chain. | 20/20 | 90.53 +/- 3.60 | 74.24 +/- 6.56 | 71.25 +/- 6.55 |

