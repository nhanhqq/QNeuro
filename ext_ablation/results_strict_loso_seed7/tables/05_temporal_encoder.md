# Table 5: temporal_encoder

Same strict-LOSO protocol as `../PROTOCOL.md`.

## CHSZ

| Variant | Intervention | Folds | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|---:|---:|
| `full` | Independent matched reference. | 27/27 | 90.88 +/- 7.36 | 80.54 +/- 17.46 | 75.42 +/- 17.68 |
| `no_bilstm` | Temporal mean. | 27/27 | 87.80 +/- 10.80 | 72.78 +/- 16.83 | 69.59 +/- 16.87 |
| `endpoint_bilstm` | BiLSTM endpoint only. | 27/27 | 90.27 +/- 7.64 | 80.09 +/- 16.95 | 75.28 +/- 17.10 |
| `mean_bilstm` | BiLSTM mean only. | 27/27 | 90.39 +/- 6.44 | 80.23 +/- 16.82 | 74.48 +/- 16.13 |

## Sleep-EDF-20

| Variant | Intervention | Folds | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|---:|---:|
| `full` | Independent matched reference. | 20/20 | 91.08 +/- 3.21 | 76.58 +/- 5.04 | 73.06 +/- 5.43 |
| `no_bilstm` | Temporal mean. | 20/20 | 89.37 +/- 4.10 | 72.95 +/- 5.27 | 69.65 +/- 5.63 |
| `endpoint_bilstm` | BiLSTM endpoint only. | 20/20 | 90.99 +/- 3.03 | 75.91 +/- 5.50 | 73.00 +/- 5.26 |
| `mean_bilstm` | BiLSTM mean only. | 20/20 | 90.83 +/- 3.23 | 76.16 +/- 6.08 | 72.73 +/- 5.79 |

