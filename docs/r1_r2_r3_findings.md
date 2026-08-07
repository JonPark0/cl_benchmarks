# R1/R2/R3: A-GEM Defect Isolation and Protocol-Matched Replication

**Date:** 2026-07-28
**Environment:** WSL2 Ubuntu 24.04, `torch==2.6.0+cu124` (see `docs/r0_platform_gate.md` for platform equivalence).

## Background

`mammoth/models/agem.py:end_task` computes `samples_per_task = buffer_size // N_TASKS`
but never uses it, storing only one minibatch (64 samples) per task regardless of
`buffer_size` — verified by diffing `buffer_500` vs `buffer_1000` logs for `agem`
(bit-identical except the `buffer_size` field). `mammoth/models/agem_fixed.py` is a new
model (not an edit of the pinned submodule) whose `end_task` iterates the full task
loader, letting the `Buffer` class's own reservoir sampling honor `buffer_size` correctly.
Patch verified: `agem_fixed` buf=500 vs buf=1000 logs now genuinely differ (12 diff lines).

## R1 + R2: Buffer sweep, original protocol (mlp_hidden_size=100, ep=1, buf∈{100,200,500,1000})

Source: `raw_data/wsl/sweep_buffer/buffer_{100,200,500,1000}/{agem_fixed,gem}/.../logs.pyd`,
aggregated in `results/wsl/buffer_sweep_agem_fixed_gem.csv`.

| Model | buf=100 | buf=200 | buf=500 | buf=1000 |
|---|---|---|---|---|
| A-GEM (unpatched, Windows) | 23.86±2.80 | 23.77±1.17 | 24.04±2.07 | 24.04±2.07 |
| A-GEM (patched, WSL2) | 26.24±1.48 | 24.81±2.07 | 24.42±1.08 | 25.60±1.03 |
| GEM (WSL2) | 76.04±1.42 | 83.72±0.83 | 89.43±0.92 | 92.04±0.59 |

**Finding:** patching the buffer-fill bug does *not* meaningfully change A-GEM's
performance (24–26% either way). GEM — which uses the same full-buffer construction —
scales cleanly with buffer size and exceeds ER/DER++ at buf=1000. The flat A-GEM curve
is therefore predominantly algorithmic (single averaged gradient direction), not an
artifact of the implementation defect.

## R3: van de Ven et al. (2022) protocol-matched replication

Source: `raw_data/wsl/r3_vandeven/{model}/.../logs.pyd`, aggregated in
`results/wsl/r3_vandeven_protocol.csv`. Protocol: `--mlp_hidden_size 400
--fitting_mode iters --n_iters 2000 --batch_size 128 --permute_classes 1`, buffer=1000
where applicable. 7 models × 5 seeds = 35 runs, plus 5 more for `agem_fixed`.

| Model | Literature | Original (ep=1, mlp=100) | Protocol-matched |
|---|---|---|---|
| SGD | 19.89±0.02 | 19.50±0.07 | 19.91±0.06 |
| Joint | 98.17±0.04 | 93.91±0.14 | 93.07±0.50 |
| EWC Online | 20.64±0.52 | 19.51±0.06 | 19.91±0.06 |
| SI | 21.20±0.57 | 22.15±4.19 | 19.96±0.36 |
| LwF | 21.89±0.32 | 19.34±0.47 | 20.31±0.78 |
| ER | 88.79±0.20 | 81.38±0.70 | 91.06±1.27 |
| A-GEM | 65.10±3.64 | 23.77±1.17 | 45.33±3.26 |
| A-GEM (patched) | — | — | 45.57±7.23 |

**Finding:** the original 41%p A-GEM gap decomposes as ≈21%p from protocol mismatch
(backbone width, training budget, buffer size, class-order randomization) and a residual
≈20%p that protocol-matching does not close. `agem_fixed` under the matched protocol
(45.57±7.23) is statistically indistinguishable from unpatched A-GEM (45.33±3.26),
confirming the buffer-fill defect is not the source of the residual gap either. The exact
cause of the remaining ≈20%p (e.g. finer implementation differences between Mammoth's
A-GEM and the original paper's) was not identified further — left as an open question
(thesis §6.5).

## Guardrails maintained

- `results/*.csv|json` (Windows-era) and `raw_data/split_mnist`, `raw_data/sweep_epochs`,
  `raw_data/sweep_buffer` (Windows-era) are untouched — verified via `git status --porcelain`.
- All new data under `raw_data/wsl/` and `results/wsl/`.
- `mammoth/` submodule diff is exactly one new untracked file (`models/agem_fixed.py`);
  the pinned commit's tracked files are unmodified.
