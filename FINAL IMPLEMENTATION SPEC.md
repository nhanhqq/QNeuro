# FINAL IMPLEMENTATION SPEC
## LatentSet-QuanKAN + Progressive Global–Adjacent Distillation
### Full-Quantum-from-Scratch Training for 62→3→1 Edge EEG

---

# 0. MAIN RESEARCH OBJECTIVE

Implement a compact hybrid EEG model that:

1. trains from scratch with the quantum branch active from epoch 1;
2. achieves maximum possible cross-subject accuracy using N−1 training subjects;
3. supports a real variable number of EEG electrodes;
4. progressively transfers knowledge:

```text
62 → 61 → 60 → ... → 3 → 2 → 1
```

5. uses two frozen teachers at every transition:

```text
Global Teacher G62
+
Adjacent Teacher S(K+1)
```

6. performs knowledge distillation AND supervised fine-tuning simultaneously;
7. produces true 3-electrode and 1-electrode edge models;
8. records the COMPLETE experiment:
   - all losses;
   - all metrics;
   - all node scores;
   - quantum behaviour;
   - figures;
   - checkpoints;
   - ablations;
   - runtime/size metrics.

No:

```text
GRL
DANN
domain discriminator
target adaptation
pseudo-labeling
MMD
CORAL
```

---

# 1. CURRENT TARGET-SELECTED LOSO PROTOCOL

Temporarily use:

```text
Target subject = P_i

Training:
all subjects except P_i

Testing:
P_i
```

Example:

```text
P1:
Train P2...PN
Test P1

P2:
Train P1,P3...PN
Test P2
```

CLI:

```bash
python run_target.py --target P1
```

or:

```bash
python run_all_targets.py
```

The target subject may be evaluated after every epoch for monitoring.

However target data MUST NOT participate in:

```text
gradient computation
optimizer
normalization
electrode selection
knowledge distillation
teacher weighting
```

Save both:

```text
final_epoch.pt
```

and optionally:

```text
DEV_ONLY_best_target.pt
```

The latter is for development diagnostics only.

Main reported protocol should use the fixed final epoch unless explicitly running the development-selected experiment.

---

# 2. MODEL

Implement:

```python
class LatentSetQuanKAN(nn.Module):
```

Flow:

```text
EEG
[B,T,N,5]

  ↓

Adaptive Spectral Gate

  ↓

Shared Electrode Embedding
+
Electrode ID
+
3D Position

  ↓

6 Brain Latent Tokens

  ↓

2 × Electrode→Latent Cross Attention

  ↓

6×32 → 48

  ↓

Multi-Scale Temporal Mixer

  ↓

Temporal Attention Pool

  ↓

48-D EEG Embedding

  ↓

TinyKAN
48 → 16

  ↓
              ┌────────────────────┐
              │                    │
              ▼                    ▼
       Classical KAN       Quantum Residual
          16 → C             4-qubit QNN
              │                    │
              │                 12 meas.
              │                    │
              │              TinyKAN 12→16
              │                    │
              └──── Entropy Gate ──┘
                        │
                        ▼
               Hybrid 16-D Feature

                        ↓

                 TinyKAN 16→C

                        ↓

                     Emotion
```

Target:

```text
preferably < 100K parameters
```

Accuracy has priority over forcing an arbitrary 50K/60K limit.

---

# 3. ADAPTIVE SPECTRAL GATE

Input:

```text
[B,T,N,5]
```

Use:

```text
LayerNorm(5)
↓
Linear 5→12
↓
SiLU
↓
Linear 12→5
↓
tanh
```

Apply:

```math
X' = X \odot (1 + 0.5\tanh(g))
```

Therefore scaling range is:

```text
0.5× → 1.5×
```

allowing both:

```text
frequency suppression
frequency enhancement
```

---

# 4. ELECTRODE TOKENIZATION

Shared DE projection:

```text
5 → 32
```

Add:

```text
Electrode-ID embedding: 32
3-D coordinate embedding: 3→12→32
```

Then:

```text
LayerNorm(32)
```

Do not flatten the 62 electrodes.

Do not make network parameters dependent on N.

---

# 5. LATENT SPATIAL ENCODER

Default:

```text
num_latents = 6
latent_dim = 32
heads = 4
blocks = 2
```

Each block:

```text
Latent Queries
    ↓
Cross-Attention
Q = latent
K,V = electrodes
    ↓
Residual
LayerNorm
    ↓
FFN 32→64→32
GELU
    ↓
Residual
LayerNorm
```

Complexity:

```text
O(LN)
```

with:

```text
L=6
```

rather than O(N²).

Support preliminary alternatives:

```text
4 / 6 / 8 latent tokens
```

---

# 6. SPATIAL COMPRESSION

Output:

```text
[B*T,6,32]
```

Flatten only latent tokens:

```text
192
```

then:

```text
Linear 192→48
LayerNorm
SiLU
```

Result:

```text
[B,T,48]
```

This representation is invariant in dimensionality across:

```text
62ch
32ch
16ch
3ch
1ch
```

---

# 7. TEMPORAL MIXER

Three depthwise branches:

```text
k=3, dilation=1
k=5, dilation=2
k=7, dilation=4
```

Input:

```text
[B,48,T]
```

Each produces 48 channels.

Concatenate:

```text
144
```

then:

```text
Pointwise Conv
144→96
```

split:

```text
content = 48
gate = 48
```

GLU:

```math
H = content \odot sigmoid(gate)
```

Residual-add original feature.

Then LayerNorm.

Add second lightweight block:

```text
DWConv 48→48 k=3
Pointwise 48→96
GLU →48
Residual
LayerNorm
```

No LSTM.

No GRU.

No large temporal Transformer.

---

# 8. TEMPORAL ATTENTION POOLING

For each temporal token:

```text
48→1
```

Compute:

```math
a_t=softmax(score_t)
```

and:

```math
z=\sum_t a_th_t
```

Output:

```text
z ∈ R^48
```

Save:

```python
features["pooled"]
```

This is the primary feature used for KD.

---

# 9. TINY KAN PROJECTION

Use:

```text
TinyKANLinear
48 → 16
```

Default KAN:

```text
grid_size = 3
spline_order = 2
```

Output:

```text
h_clean ∈ R^16
```

---

# 10. CLASSICAL AUXILIARY CLASSIFIER

From:

```text
h_clean
```

use:

```text
TinyKAN 16→C
```

to produce:

```text
classical_logits
```

This branch is ALWAYS active.

It provides:

```text
stable supervised path
entropy for quantum gating
quantum residual target
```

---

# 11. QUANTUM TRAINING — IMPORTANT CHANGE

Quantum is active FROM SCRATCH.

Do NOT use:

```text
q_strength warm-up
classical-only initial epochs
```

For every epoch:

```python
q_strength = 1.0
```

including:

```text
epoch 1
```

and including every progressive distillation stage.

Quantum parameters are randomly initialized together with the whole model.

The full model therefore learns:

```text
classical
KAN
quantum
hybrid residual
```

jointly from the beginning.

---

# 12. QUANTUM STABILITY MECHANISMS

Although quantum is active immediately, retain stability safeguards.

## 12.1 Detached quantum input

```python
q_input = h_clean.detach()
```

Quantum gradients must not directly modify the upstream EEG feature extractor.

## 12.2 Bounded residual

```math
\alpha_q =
0.25 \sigma(s_q)
```

Therefore:

```text
0 < αq < 0.25
```

## 12.3 Entropy gate

Quantum correction is controlled by classical uncertainty.

## 12.4 Residual auxiliary objective

Quantum explicitly learns the classical prediction error.

These mechanisms replace the need for quantum warm-up.

---

# 13. QUANTUM CIRCUIT

Use QuanKAN-style circuit.

Default:

```text
4 qubits
4 layers
```

Reason:

4 layers add very few actual quantum parameters while providing stronger data re-uploading.

Ablation:

```text
2 layers
```

Each layer:

```text
RY(data)
RZ(data)

Rot(
    theta,
    phi,
    omega
)
```

for each qubit.

Alternating ring CNOT.

Even layer:

```text
0→1
1→2
2→3
3→0
```

Odd layer:

```text
1→0
2→1
3→2
0→3
```

---

# 14. QUANTUM DATA RE-UPLOADING

For four layers:

```text
4 layers ×
4 qubits ×
2 data angles
=
32 angles
```

Generate:

```text
h_clean [16]
↓
TinyKAN 16→32
↓
π × tanh
```

reshape:

```text
[B,4,4,2]
```

---

# 15. QUANTUM PARAMETERS

Trainable Rot parameters:

```text
4 layers
× 4 qubits
× 3
=
48 parameters
```

Initialize:

```python
0.05 * torch.randn(...)
```

---

# 16. QUANTUM MEASUREMENTS

Measure:

```text
Z0 Z1 Z2 Z3
X0 X1 X2 X3
Z0Z1
Z1Z2
Z2Z3
Z3Z0
```

Total:

```text
12
```

Then:

```text
TinyKAN 12→16
```

producing:

```text
q_residual ∈ R^16
```

---

# 17. FAST QUANTUM BACKEND

Default:

```text
TorchStatevectorQuantumBranch
```

NOT PennyLane.

Everything remains on CUDA:

```text
state
angles
Rot weights
CNOT
measurements
gradients
```

With 4 qubits:

```text
state vector size = 16
```

which is extremely small.

PennyLane remains only for:

```text
numerical reference
unit test
fallback
```

---

# 18. ENTROPY GATE

From:

```text
classical_logits
```

compute:

```math
p=softmax(z)
```

```math
H=
-\sum_cp_c\log(p_c+\epsilon)
```

normalize:

```math
H_n=H/\log(C)
```

then:

```math
g_q=H_n^{1.5}
```

Range:

```text
[0,1]
```

High uncertainty:

```text
higher quantum contribution
```

Low uncertainty:

```text
lower quantum contribution
```

Zero trainable gate parameters.

---

# 19. HYBRID REPRESENTATION

Define:

```math
h_{hybrid}
=
LayerNorm(
h_{clean}
+
g_q
\alpha_q
q_{residual}
)
```

Then:

```text
TinyKAN 16→C
```

produces:

```text
final_logits
```

Final prediction ALWAYS uses:

```text
final_logits
```

not classical logits.

---

# 20. QUANTUM RESIDUAL AUXILIARY HEAD

Add:

```text
q_feature → Linear → C
```

Initialize:

```text
weight = 0
bias = 0
```

Generate residual target:

```python
with torch.no_grad():

    p_classical = softmax(classical_logits)

    y_onehot = one_hot(labels)

    residual_target =
        y_onehot - p_classical
```

Quantum residual loss:

```math
L_Q =
SmoothL1(
tanh(q_aux),
residual_target
)
```

This encourages quantum to learn:

```text
what the classical branch is currently missing
```

rather than duplicate the classical classifier.

---

# 21. PHASE A — FULL 62-CHANNEL TRAINING FROM SCRATCH

For each target:

```text
Train:
N-1 subjects

Test:
1 target subject
```

Initialize the ENTIRE model randomly:

```text
spectral gate
latent encoder
temporal mixer
KAN layers
quantum angle encoder
quantum rotations
quantum readout
final hybrid classifier
```

Quantum is active immediately.

Train:

```text
100 epochs
```

Fixed.

No early stop.

---

# 22. FULL MODEL LOSS

For every training batch:

```math
L_{base}
=
L_{final}
+
0.25L_{classical}
+
0.15L_Q
+
\lambda_{KAN}R_{KAN}
```

where:

```math
L_{final}
=
CE(final\_logits,y)
```

```math
L_{classical}
=
CE(classical\_logits,y)
```

Recommended:

```text
label smoothing = 0.05
λKAN = 1e-5
```

---

# 23. FULL MODEL OPTIMIZER

Use parameter groups.

```text
Backbone:
3e-4

Spectral/latent/temporal modules:
3e-4

KAN projection:
5e-4

Classical KAN classifier:
5e-4

Quantum angle KAN:
5e-4

Quantum circuit Rot parameters:
5e-4

Quantum readout KAN:
5e-4

Final KAN:
5e-4

Quantum scale:
5e-4
```

Optimizer:

```text
AdamW
weight_decay = 1e-4
```

Scheduler:

```text
CosineAnnealingLR
T_max = 100
eta_min = 1e-6
```

Gradient clipping:

```text
1.0
```

---

# 24. GLOBAL TEACHER CREATION

After exactly 100 epochs:

```text
G62 = deepcopy(full_model)
```

Freeze:

```text
requires_grad = False
eval()
```

This becomes the permanent:

```text
GLOBAL TEACHER
```

for every subsequent transition.

It never changes.

---

# 25. START PROGRESSIVE REDUCTION

Initialize:

```text
Student S62
=
copy of G62
```

Active channels:

```text
C62
```

Then repeat:

```text
62→61
61→60
60→59
...
```

until:

```text
3
```

and finally:

```text
3→2→1
```

---

# 26. HARDWARE-CONSTRAINED PATH

Until K=3 protect:

```text
Fp1
Fpz
Fp2
```

When these three remain:

```text
SAVE 3-channel checkpoint
```

After 3-channel model is finalized:

change protected set:

```text
Fp1
```

Continue:

```text
3→2→1
```

---

# 27. ELECTRODE SELECTION

At current stage K:

For every removable electrode e:

```text
C_candidate =
C_K - {e}
```

Run current student with the candidate electrode set.

Use TRAINING subjects only.

Calculate:

```text
Cross Entropy
Accuracy
Macro F1
```

Baseline current model:

```text
CE_base
F1_base
```

Candidate:

```text
CE_e
F1_e
```

Then:

```math
ΔCE_e = CE_e-CE_{base}
```

```math
ΔF1_e = F1_{base}-F1_e
```

Normalize candidate values within stage.

Score:

```math
Score(e)
=
0.70\,norm(ΔCE_e)
+
0.30\,norm(ΔF1_e)
```

Select:

```math
e^* = argmin_e Score(e)
```

This means:

```text
remove the electrode whose loss causes
the smallest degradation.
```

---

# 28. RECOVERY-AWARE LOOKAHEAD

Do NOT waste recovery search on all 62 stages.

Use:

```text
K > 16:
standard greedy
```

Use:

```text
K <= 16:
top-3 recovery lookahead
```

For top three candidates:

```text
clone current student
remove candidate
train 3 mini epochs
using dual KD + CE
measure recovery
```

Choose candidate with:

```text
best recovered training Macro-F1
```

This spends extra compute where each remaining electrode matters most.

---

# 29. ADJACENT TEACHER

Immediately before removal:

```python
adjacent_teacher =
freeze(deepcopy(current_student))
```

Adjacent Teacher sees:

```text
K electrodes
```

Student sees:

```text
K-1 electrodes
```

Therefore:

```text
Global Teacher:
stable full-EEG semantic anchor

Adjacent Teacher:
locally attainable teacher
```

---

# 30. STUDENT INITIALIZATION

Do NOT train reduced students from scratch.

For:

```text
K → K-1
```

use:

```python
student_new.load_state_dict(
    student_previous.state_dict()
)
```

Transfer:

```text
EEG backbone
spectral gate
latent tokens
temporal mixer
KAN
quantum Rot parameters
quantum scale
quantum readout
classifier
```

Reset ONLY:

```text
optimizer
scheduler
```

---

# 31. DUAL-TEACHER FORWARD PASS

Each training sample generates three views.

## GLOBAL

```text
G62
input:
62 channels
```

outputs:

```text
global_logits
global_pooled_48
```

## ADJACENT

```text
A_K
input:
K channels
```

outputs:

```text
adjacent_logits
adjacent_pooled_48
```

## STUDENT

```text
S_(K-1)
input:
K-1 channels
```

outputs:

```text
student_logits
student_classical_logits
student_pooled_48
student_quantum
```

---

# 32. TEACHER WEIGHTING

Use stage-aware + disagreement-aware weighting.

Stage prior:

```math
r_K=
clip(
\sqrt{K/62},
0.15,
0.85
)
```

Calculate:

```math
d =
JS(p_G,p_A)
```

Then:

```math
r_d=
\exp(-d/0.5)
```

Global unnormalized reliability:

```math
u_G = r_Kr_d
```

Adjacent reliability:

```math
u_A=1-r_K
```

Normalize:

```math
w_G=
\frac{u_G}{u_G+u_A+\epsilon}
```

```math
w_A=1-w_G
```

Detach teacher weighting from gradients.

Ablation modes:

```text
fixed 0.5/0.5
linear K/62
sqrt K/62
adaptive proposed
```

---

# 33. LOGIT DISTILLATION

Temperature:

```text
T=4
```

Implementation:

```python
KD(
    student,
    teacher
) = F.kl_div(
    F.log_softmax(student/T),
    F.softmax(teacher/T),
    reduction="batchmean"
) * T**2
```

Global:

```text
KD_G
```

Adjacent:

```text
KD_A
```

Combined:

```math
L_{KD}
=
w_G KD_G
+
w_A KD_A
```

---

# 34. FEATURE DISTILLATION

Use:

```text
48-D pooled feature
```

Global:

```math
L_{FG}
=
1-\cos(f_S,f_G)
```

Adjacent:

```math
L_{FA}
=
1-\cos(f_S,f_A)
```

Combined:

```math
L_F
=
w_GL_{FG}
+
w_AL_{FA}
```

Do NOT distill raw quantum measurement by default.

---

# 35. DISTILLATION + FINETUNING LOSS

This is important:

The reduced student is NOT doing KD only.

It is simultaneously:

```text
distilling teacher knowledge
+
fine-tuning on ground-truth EEG labels
```

Loss:

```math
L_{stage}
=
L_{final}
+
0.15L_{classical}
+
0.10L_Q
+
1.00L_{KD}
+
0.25L_F
+
\lambda_{KAN}R_{KAN}
```

Therefore every stage retains supervised task learning.

---

# 36. INTERMEDIATE TRANSITION TRAINING BUDGET

Do NOT train all 61 transitions for 100 epochs.

That would unnecessarily multiply compute.

Because every student is warm-started, use:

```text
K > 32:
20 epochs

17 ≤ K ≤ 32:
30 epochs

9 ≤ K ≤ 16:
40 epochs

4 ≤ K ≤ 8:
50 epochs
```

Every epoch performs:

```text
CE fine-tuning
+
Global KD
+
Adjacent KD
+
feature KD
+
quantum residual learning
```

Quantum remains fully active from epoch 1 of every stage.

---

# 37. 3-CHANNEL FINAL CONSOLIDATION

When the active channels become:

```text
Fp1
Fpz
Fp2
```

do NOT immediately continue to 2 channels.

First save:

```text
3ch_preconsolidation.pt
```

Then perform a final:

```text
100 epochs
```

of:

```text
Dual Teacher Distillation
+
Ground Truth Fine-Tuning
```

using:

```text
Global Teacher = G62

Adjacent Teacher =
last frozen 4-channel teacher
```

Same loss:

```math
L_{stage}
```

After 100 epochs:

```text
SAVE final_3ch.pt
```

This is the final 3-channel deployment model.

---

# 38. CONTINUE 3→1

Use:

```text
final_3ch.pt
```

as current student.

Protect:

```text
Fp1
```

Perform:

```text
3→2
```

with dual teacher + fine-tune.

Then:

```text
2→1
```

with dual teacher + fine-tune.

---

# 39. 1-CHANNEL FINAL CONSOLIDATION

After Fp1 becomes the only channel:

save:

```text
1ch_preconsolidation.pt
```

Then train:

```text
100 epochs
```

with:

```text
Global Teacher = G62

Adjacent Teacher =
frozen final 2-channel model
```

Loss:

```math
L_{stage}
```

Final:

```text
final_1ch.pt
```

---

# 40. THREE MAIN FINAL MODELS

Every target must therefore produce:

```text
G62
final_3ch
final_1ch
```

These are the primary reported models.

---

# 41. BASE 100-EPOCH LOGGING

CSV:

```text
base_training.csv
```

Columns:

```text
epoch
lr_backbone
lr_head

train_total_loss
train_final_ce
train_classical_ce
train_quantum_residual_loss
train_kan_regularization

train_acc
train_macro_precision
train_macro_recall
train_macro_f1
train_weighted_f1

test_loss
test_acc
test_macro_precision
test_macro_recall
test_macro_f1
test_weighted_f1

quantum_gate_mean
quantum_gate_std
quantum_alpha
quantum_feature_rms
quantum_residual_rms

backbone_grad_norm
kan_grad_norm
quantum_grad_norm

epoch_seconds
peak_vram_mb
```

---

# 42. PROGRESSIVE HISTORY

CSV:

```text
progressive_history.csv
```

Columns:

```text
target
stage
K_before
K_after
removed_channel

epoch

w_global_mean
w_adjacent_mean
teacher_js_mean

loss_total

loss_ce
loss_classical
loss_q_residual

loss_kd_global
loss_kd_adjacent

loss_feature_global
loss_feature_adjacent

train_acc
train_macro_f1

test_acc
test_macro_f1

quantum_gate_mean
quantum_alpha
quantum_correction_ratio

grad_backbone
grad_kan
grad_quantum

epoch_seconds
peak_vram
```

---

# 43. ELECTRODE SEARCH LOG

Save:

```text
candidate_scores.csv
```

For every candidate:

```text
target
stage
K

candidate_channel

baseline_ce
candidate_ce

delta_ce

baseline_acc
candidate_acc
delta_acc

baseline_f1
candidate_f1
delta_f1

normalized_ce
normalized_f1

score

lookahead_used
recovered_f1

selected
```

---

# 44. QUANTUM CORRECTION RATIO

Compute diagnostic:

```math
QCR =
\frac{
||g_q \alpha_q q||
}{
||h_{clean}||+\epsilon
}
```

Log:

```text
QCR mean
QCR std
```

for every epoch and stage.

This gives:

```text
How much is the quantum residual actually
modifying the representation?
```

---

# 45. REQUIRED FIGURES — FULL MODEL

Save:

```text
figures/base/
```

## Fig 1

```text
Epoch vs Train/Test Accuracy
```

## Fig 2

```text
Epoch vs Train/Test Macro F1
```

## Fig 3

```text
Epoch vs Loss
```

with:

```text
total
final CE
classical CE
quantum residual
```

## Fig 4

```text
Quantum gate vs epoch
```

## Fig 5

```text
Quantum alpha vs epoch
```

## Fig 6

```text
QCR vs epoch
```

## Fig 7

```text
Gradient norms:
backbone / KAN / quantum
```

## Fig 8

```text
62ch confusion matrix
```

---

# 46. REQUIRED FIGURES — PROGRESSIVE REDUCTION

Save:

```text
figures/progressive/
```

## Fig 9

```text
Accuracy vs Remaining Electrodes
```

x-axis reversed:

```text
62 → 1
```

## Fig 10

```text
Macro F1 vs Remaining Electrodes
```

## Fig 11

```text
Performance Retention Ratio vs K
```

## Fig 12

```text
Removed electrode sequence
```

## Fig 13

```text
Global/Adjacent teacher weight vs K
```

## Fig 14

```text
Teacher JS disagreement vs K
```

## Fig 15

```text
KD Global / KD Adjacent loss vs K
```

## Fig 16

```text
Feature KD loss vs K
```

## Fig 17

```text
Quantum Gate vs K
```

## Fig 18

```text
Quantum Correction Ratio vs K
```

## Fig 19

```text
Stage training time vs K
```

---

# 47. FINAL CONFUSION MATRICES

Always generate:

```text
CM_62ch.png
CM_3ch.png
CM_1ch.png
```

same class ordering and layout.

---

# 48. FINAL METRICS

For:

```text
62ch
3ch
1ch
```

save:

```text
Accuracy

Macro Precision
Macro Recall
Macro F1

Weighted Precision
Weighted Recall
Weighted F1

Balanced Accuracy

Per-Class:
Precision
Recall
F1
Support

Cross Entropy

Confusion Matrix
```

If class probabilities are valid and all classes occur:

```text
Macro AUROC OVR
```

---

# 49. PERFORMANCE RETENTION

Compute:

```math
PRR(K)=
\frac{
Acc_K-Acc_{chance}
}{
Acc_{62}-Acc_{chance}
}
```

Also F1 retention:

```math
F1R(K)=
\frac{F1_K}{F1_{62}}
```

Report:

```text
PRR_3
PRR_1
F1R_3
F1R_1
```

---

# 50. EDGE METRICS

For:

```text
62ch
3ch
1ch
```

measure:

```text
trainable parameters

classical parameters
KAN parameters
quantum circuit parameters

FP32 checkpoint size

MACs/FLOPs classical portion

quantum state dimension

forward latency
batch=1

forward latency
batch=32

peak inference memory
```

Also export:

```text
ONNX/classical-compatible path
```

when practical.

Quantum runtime benchmark must be separate.

---

# 51. CORE MODEL ABLATIONS

These prove the Hybrid QuanKAN classifier.

## M0 — Pure MLP

```text
48
→ Linear
→ C
```

## M1 — TinyKAN

```text
48→16→C
```

No quantum.

## M2 — Quantum + Linear

```text
48→16
+
quantum residual
→ Linear C
```

No final KAN.

## M3 — Full Hybrid

```text
TinyKAN
+
Quantum
+
TinyKAN
```

PROPOSED.

---

# 52. QUANTUM ABLATIONS

## Q0 — No Quantum

```text
alpha_q = 0
```

## Q1 — No Entropy Gate

```text
gate = 1
```

## Q2 — Fixed Quantum Scale

Do not train:

```text
quantum_scale
```

## Q3 — No Quantum Residual Objective

Set:

```text
lambda_Q = 0
```

## Q4 — No Detach

Allow quantum gradient into backbone.

Diagnostic only.

## Q5 — No Entanglement

Remove CNOT.

## Q6 — Z-only

Only measure:

```text
Z0...Z3
```

## Q7 — 2 layers

versus proposed:

```text
4 layers
```

These quantify what the quantum component actually contributes.

---

# 53. BACKBONE ABLATIONS

Secondary.

## A0

No spectral gate.

## A1

4 latent tokens.

## A2

6 latent tokens.

PROPOSED.

## A3

8 latent tokens.

## A4

Mean electrode pooling instead of latent attention.

## A5

No multi-scale temporal mixer.

These may first be screened on one target.

Shortlisted variants should later run full target-selected LOSO.

---

# 54. DISTILLATION ABLATIONS — MANDATORY

These are more important than backbone ablations.

## D0 — Scratch 3ch

Train:

```text
Fp1/Fpz/Fp2
```

from random initialization.

## D1 — Scratch 1ch

Train:

```text
Fp1
```

from random initialization.

## D2 — Direct KD 62→3

No progressive path.

## D3 — Direct KD 62→1

No progressive path.

## D4 — Progressive CE-only

```text
62→61→...→1
```

warm-start + CE only.

No teacher KD.

This tests whether gains simply come from progressive fine-tuning.

## D5 — Global Teacher Only

No Adjacent Teacher.

## D6 — Adjacent Teacher Only

No G62.

## D7 — Dual Teacher Fixed 50/50

## D8 — Dual Teacher Adaptive

Proposed weighting.

## D9 — No Feature KD

Logit KD only.

## D10 — Random Progressive Order + Dual KD

Random removable node at every stage.

## D11 — Greedy Order without Lookahead

## D12 — Full Proposed

```text
Greedy
+
low-channel recovery lookahead
+
Global Teacher
+
Adjacent Teacher
+
adaptive weighting
+
feature KD
+
CE fine-tuning
```

---

# 55. MAIN PAPER ABLATION SET

Because full 61-step runs are expensive, the mandatory FULL-target ablations should be:

```text
1. Scratch 1ch/3ch
2. Direct KD
3. Progressive CE-only
4. Global-only
5. Adjacent-only
6. Dual fixed
7. Random path + Dual
8. Full Proposed
9. No Quantum
10. Full Quantum/KAN
```

Other ablations can be detailed supplementary experiments.

---

# 56. ABLATION OUTPUT

Generate:

```text
ablation_results.csv
```

Columns:

```text
variant
target
channels

accuracy
macro_f1
precision
recall

params
latency

delta_vs_full
```

Generate figures:

```text
Ablation Accuracy bar plot
Ablation Macro-F1 bar plot

3ch comparison

1ch comparison

Quantum ablation plot

Distillation ablation plot
```

---

# 57. LOSO SUMMARY

After all targets:

```text
loso_summary.csv
```

Report for:

```text
62ch
32ch
16ch
8ch
5ch
3ch
1ch
```

Each:

```text
Accuracy mean ± std
Macro-F1 mean ± std
Precision mean ± std
Recall mean ± std
PRR mean ± std
```

---

# 58. REQUIRED FINAL FIGURES ACROSS SUBJECTS

Generate:

## LOSO Accuracy Boxplot

```text
62 / 32 / 16 / 8 / 5 / 3 / 1
```

## LOSO Macro-F1 Boxplot

## Mean Accuracy Retention Curve

## Mean Macro-F1 Retention Curve

## Fold × Channel-count Heatmap

## Fold × Final 3ch Accuracy

## Fold × Final 1ch Accuracy

---

# 59. CHECKPOINT STRUCTURE

```text
results/
└── target_Pxx/

    ├── config.yaml

    ├── base/
    │   ├── final_epoch.pt
    │   ├── DEV_ONLY_best_target.pt
    │   └── base_training.csv

    ├── progressive/
    │   ├── electrode_path.json
    │   ├── candidate_scores.csv
    │   ├── progressive_history.csv
    │
    │   ├── 62ch.pt
    │   ├── 61ch.pt
    │   ├── ...
    │   ├── 4ch.pt
    │
    │   ├── 3ch_preconsolidation.pt
    │   ├── final_3ch.pt
    │
    │   ├── 2ch.pt
    │
    │   ├── 1ch_preconsolidation.pt
    │   └── final_1ch.pt

    ├── metrics/
    │   ├── 62ch.json
    │   ├── 3ch.json
    │   └── 1ch.json

    ├── figures/
    │   ├── base/
    │   ├── progressive/
    │   ├── confusion/
    │   └── quantum/

    └── runtime/
        ├── parameter_report.json
        ├── latency.json
        └── memory.json
```

---

# 60. RESUME SUPPORT

Save every epoch:

```text
model
optimizer
scheduler

epoch

active channels
removed history

global teacher checkpoint
adjacent teacher checkpoint

loss history
metrics history

random RNG state
NumPy RNG state
Torch RNG
CUDA RNG
```

Any interrupted stage must resume exactly where it stopped.

---

# 61. TRAINING COMMANDS

Base + progressive target:

```bash
python run_target.py \
    --target P1 \
    --epochs-base 100 \
    --quantum-backend torch \
    --quantum-layers 4 \
    --run-progressive
```

All targets:

```bash
python run_all_targets.py \
    --epochs-base 100
```

Ablation:

```bash
python run_ablation.py \
    --variant progressive_ce_only
```

---

# 62. EXECUTION ORDER FOR CODING AGENT

Implement exactly in this order.

## P0

Dataset loading.

## P1

Target N−1 / 1 split.

## P2

Variable-electrode classical backbone.

## P3

TinyKAN.

## P4

Torch state-vector quantum backend.

## P5

QuanKAN residual hybrid head.

## P6

100-epoch full 62-channel training.

Verify quantum receives gradients from epoch 1.

## P7

Full metric/logging system.

## P8

Variable N inference test.

## P9

Greedy electrode search.

## P10

Adjacent Teacher snapshot.

## P11

Global + Adjacent KD.

## P12

Progressive KD + fine-tuning.

## P13

62→3 pipeline.

## P14

100-epoch 3ch consolidation.

## P15

3→1 pipeline.

## P16

100-epoch 1ch consolidation.

## P17

Figures.

## P18

Efficiency benchmark.

## P19

Core ablations.

## P20

All-target experiment.

---

# 63. ABSOLUTE TRAINING RULE

The training process is:

```text
RANDOM INITIALIZATION

        ↓

Full hybrid quantum model

        ↓

100 epochs
Quantum active from Epoch 1

        ↓

Converged 62ch model

        ↓

Freeze as G62

        ↓

62 → 61
Dual KD + Fine-tune

        ↓

61 → 60
Dual KD + Fine-tune

        ↓

...

        ↓

4 → 3
Dual KD + Fine-tune

        ↓

100 epoch 3ch consolidation

        ↓

3 → 2
Dual KD + Fine-tune

        ↓

2 → 1
Dual KD + Fine-tune

        ↓

100 epoch 1ch consolidation

        ↓

FINAL EDGE MODEL
```

Quantum is NEVER disabled during normal training.

There is NO quantum warm-up.

---

# 64. CENTRAL SCIENTIFIC IDEA

The framework solves a sequence of increasingly difficult observability constraints.

Instead of:

```text
62 → 1
```

directly:

```text
62→61→60→...→3→2→1
```

At every step:

```text
Global Teacher
=
stable full-observability semantic anchor

Adjacent Teacher
=
closest attainable previous-observability target

Ground Truth CE
=
continuous task fine-tuning

Quantum–KAN
=
compact nonlinear specialization mechanism
```

Therefore the student is never asked to reproduce unavailable full-brain knowledge without a nearby teacher.

---

# 65. FINAL SUCCESS CRITERIA

The proposed method must demonstrate:

```text
Full hybrid 62ch
>
classical-only 62ch
or competitive with it

AND

Proposed 3ch
>
3ch scratch

Proposed 3ch
>
direct 62→3 KD

Proposed 3ch
>
progressive CE-only

AND

Proposed 1ch
>
1ch scratch

Proposed 1ch
>
direct 62→1 KD

Proposed 1ch
>
progressive CE-only

AND

Dual Teacher
>
Global-only

Dual Teacher
>
Adjacent-only

AND

Greedy trajectory
>
Random trajectory
```

The goal is not necessarily:

```text
1ch ≈ 62ch
```

because physical EEG information has been removed.

The goal is:

```text
maximize task-performance retention
under extreme electrode reduction
while producing a truly compact
1-channel / 3-channel edge EEG model.
```