# R0: WSL2 / Windows Cross-Platform Comparability Gate

**Date:** 2026-07-28
**Verdict: PASS**

## Setup
- WSL2 Ubuntu 24.04, Python 3.12.3, `torch==2.6.0+cu124` (matching Windows `docs/environment.md`), NVIDIA driver reports CUDA 13.1, RTX 3060 Laptop (6GB) via WSL2 GPU passthrough.
- 3 models (`sgd`, `er`, `agem`) x 5 seeds `[42, 123, 456, 789, 1024]` at `n_epochs=1, buffer_size=200`.
- Output written to `raw_data/wsl/split_mnist/{model}/...` — existing `raw_data/split_mnist/` (Windows) and `results/` trees untouched.

## Per-seed comparison (AA %, BWT %)

| Model | Seed | AA (Win) | AA (WSL) | ΔAA | BWT (Win) | BWT (WSL) | ΔBWT |
|---|---|---|---|---|---|---|---|
| sgd | 42 | 19.6067 | 19.5966 | -0.0101 | -99.2641 | -99.2641 | +0.0000 |
| sgd | 123 | 19.4453 | 19.4453 | +0.0000 | -99.1232 | -99.1232 | +0.0000 |
| sgd | 456 | 19.5461 | 19.5461 | +0.0000 | -99.1410 | -99.1410 | +0.0000 |
| sgd | 789 | 19.4453 | 19.4453 | +0.0000 | -99.1253 | -99.1253 | +0.0000 |
| sgd | 1024 | 19.4554 | 19.4554 | +0.0000 | -99.1999 | -99.1999 | +0.0000 |
| er | 42 | 82.1531 | 82.2023 | +0.0492 | -19.8595 | -19.8106 | +0.0489 |
| er | 123 | 81.2942 | 81.2942 | +0.0000 | -20.9664 | -20.9664 | +0.0000 |
| er | 456 | 81.3358 | 81.3162 | -0.0195 | -20.6239 | -20.6490 | -0.0252 |
| er | 789 | 81.7994 | 81.7994 | +0.0000 | -20.1856 | -20.1856 | +0.0000 |
| er | 1024 | 80.2997 | 80.2997 | +0.0000 | -21.9686 | -21.9686 | +0.0000 |
| agem | 42 | 25.5461 | 25.5461 | +0.0000 | -91.8435 | -91.8435 | +0.0000 |
| agem | 123 | 22.6944 | 22.6755 | -0.0189 | -95.2133 | -95.2370 | -0.0236 |
| agem | 456 | 22.7557 | 22.8597 | +0.1040 | -95.1175 | -94.9875 | +0.1300 |
| agem | 789 | 24.0917 | 24.2143 | +0.1226 | -93.1916 | -93.0257 | +0.1659 |
| agem | 1024 | 23.7525 | 23.4162 | -0.3363 | -93.8418 | -94.2496 | -0.4078 |

## Interpretation

Max |ΔAA| = 0.34%p (A-GEM, seed 1024). Several seeds are bit-identical across platforms (deterministic SGD path). All deviations are well inside each model's own reported cross-seed standard deviation (A-GEM: ±1.17 to ±2.80 across buffer sizes in the original Windows sweep), consistent with ordinary cuDNN/CUDA kernel-scheduling nondeterminism between platforms rather than any systematic Windows/WSL2 divergence.

**Conclusion:** new WSL2-produced results (R1 patched A-GEM sweep, R2 GEM) may be placed in the same tables as the existing Windows-produced results, with a footnote noting the platform difference.
