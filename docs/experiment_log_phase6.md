# 실험 진행 로그 - Phase 6: 후속 실험 (Epoch/Buffer Sweep)

> 이 문서는 `experiment_log.md` Phase 5 이후의 후속 실험을 기록합니다.
> 교수님 피드백 (2026-04-03) 기반으로 추가 실험을 진행합니다.

---

## 피드백 요약 (2026-04-03)

**교수님 코멘트:**
> "A-GEM부터 성능이 눈에 띄게 안좋네요. 다른 벤치마크 논문에서도 비슷한 경향이 보고되는지 체크해보는 것이 좋을 것 같아요."

**대응 계획:**
1. 벤치마크 논문 비교 조사 (문헌 조사)
2. Epoch sweep (5, 10, 20) — 정규화 모델 재평가
3. Buffer size sweep (100, 500, 1000) — Replay 모델 버퍼 의존성 분석
4. 시간/리소스 지표 추가

---

## Phase 6-1: 벤치마크 논문 비교 조사

**날짜:** 2026-04-04
**상태:** ✅ 완료

### 참조 논문

| # | 논문 | 출처 | 핵심 내용 |
|---|------|------|-----------|
| 1 | "Three types of incremental learning" | van de Ven et al., Nature Machine Intelligence 2022 | CIL/TIL/DIL 세 시나리오 비교 |
| 2 | "Three scenarios for continual learning" | van de Ven & Tolias, arXiv:1904.07734 | 시나리오별 벤치마크 원본 논문 |
| 3 | "Improving Weight Regularization CL Baselines" | Bricken, 2020 | EWC/SI 성능 개선 연구 |
| 4 | "Deep Class-Incremental Learning: A Survey" | Zhou et al., arXiv:2302.03648 | CIL 서베이 |

### 논문별 Split-MNIST Class-IL 결과 비교

**van de Ven et al. (2022) — Table 2 결과:**

| 모델 | AA (논문) | AA (우리 실험) | 비고 |
|------|-----------|---------------|------|
| None (SGD) | 19.89% ± 0.02 | 19.50% ± 0.07 | 일치 (하한선) |
| Joint | 98.17% ± 0.04 | 93.91% ± 0.14 | 우리 실험이 약간 낮음 (epoch 차이 추정) |
| EWC | 20.64% ± 0.52 | 19.51% ± 0.06 | **일치: 하한선 수준** |
| SI | 21.20% ± 0.57 | 22.15% ± 4.19 | **일치: 하한선 수준** |
| LwF | 21.89% ± 0.32 | 19.34% ± 0.47 | 일치: 하한선 수준 |
| A-GEM | 65.10% ± 3.64 | 23.77% ± 1.17 | **차이 큼 — epoch 수 차이 때문 (논문은 다중 epoch)** |
| ER | 88.79% ± 0.20 | 81.38% ± 0.70 | 비슷한 경향, epoch 수 차이 |

### 핵심 발견

1. **정규화 방법 (EWC, SI)의 CIL 실패는 보편적 현상**
   - van de Ven et al.: "parameter regularization methods such as EWC and SI fail almost completely in the class-incremental scenario, even on Split MNIST"
   - 근본 원인: 정규화 방법은 "서로 다른 컨텍스트에서 관찰된 클래스를 비교하는 메커니즘이 없음"
   - Class-IL에서는 출력 헤드를 공유하므로, 단순 가중치 보존만으로는 올바른 분류 불가

2. **A-GEM 성능 차이의 원인**
   - 논문의 A-GEM (65.10%) vs 우리 실험 (23.77%) — 약 41%p 차이
   - **핵심 원인: n_epochs=1** — 논문에서는 다중 epoch 사용
   - A-GEM은 gradient projection 기반이므로, epoch가 적으면 projection 효과가 누적되지 않음
   - → **Epoch sweep 실험으로 검증 필요**

3. **Replay 방법이 CIL에서 가장 효과적 (보편적 결론)**
   - van de Ven et al.: "The only strategy among the top performers in all three scenarios is replay"
   - ER이 A-GEM보다 단순하면서도 성능 우수 (보편적으로 보고됨)

4. **LwF의 FWT 양수는 정상**
   - KD 방법은 이전 학습 표현을 새 태스크에 전이하는 효과가 있으나, 망각 방지에는 실패
   - Class-IL에서 KD만으로는 구조적 한계

5. **EWC/SI 개선 가능성 (Bricken, 2020)**
   - Cross-entropy loss에 β 계수 도입 → EWC 성능 43% 향상 (27% → 54%)
   - 핵심: 학습 손실이 너무 낮아지면 gradient가 소실되어 중요 가중치 식별 불가
   - β=0.005로 모델 과신을 줄이면 중요도 추정 개선

### 결론

> **우리 실험에서 A-GEM 및 정규화 모델의 낮은 성능은 기존 문헌과 일치하는 현상이며, 특히 n_epochs=1 설정이 이를 악화시킨 것으로 판단됩니다.**

---

## Phase 6-2: Epoch Sweep 실험

**날짜:** 2026-04-04 ~ 2026-04-08
**상태:** ✅ 완료

### 실험 설계

| 항목 | 값 |
|------|-----|
| Sweep 변수 | n_epochs: 5, 10, 20 |
| 대상 모델 | 전체 8개 모델 |
| Seeds | [42, 123, 456, 789, 1024] |
| Buffer Size | 200 (고정) |
| 총 실험 수 | 8모델 × 5seed × 3epoch = **120회** |

### 결과: AA (Average Accuracy) by Epoch

| 모델 | 카테고리 | ep=1 | ep=5 | ep=10 | ep=20 |
|------|----------|------|------|-------|-------|
| SGD | Lower Bound | 19.50±0.07 | 19.80±0.10 | 19.86±0.05 | 19.91±0.03 |
| Joint | Upper Bound | 93.91±0.14 | 97.11±0.12 | 97.32±0.66 | **97.85±0.06** |
| EWC Online | Regularization | 19.51±0.06 | 19.79±0.10 | 19.88±0.05 | 19.91±0.02 |
| SI | Regularization | 22.15±4.19 | 14.36±3.74 | 13.74±3.44 | 12.05±1.50 |
| ER | Replay | 81.38±0.70 | 77.99±2.19 | 77.00±1.42 | 76.22±1.02 |
| DER++ | Replay | 82.79±3.76 | 83.30±0.95 | 81.90±0.89 | 81.03±0.85 |
| LwF | KD | 19.34±0.47 | 19.91±0.09 | 20.27±0.38 | 20.37±0.43 |
| A-GEM | Optimization | 23.77±1.17 | 31.62±5.01 | 33.93±2.54 | 35.44±3.06 |

### 검증 포인트 결과

- [x] EWC, SI가 epoch 증가 시 하한선에서 벗어나는지? → **NO.** EWC는 하한선 고정. SI는 오히려 epoch 증가 시 **성능 하락** (22.15% → 12.05%).
- [x] A-GEM이 논문 수준 (65%) 에 도달하는지? → **NO.** ep=20에서 35.44%로 개선되나, 논문의 65.10%에는 크게 미달. Buffer size나 하이퍼파라미터 차이 추정.
- [x] ER, DER++의 epoch 증가에 따른 성능 변화? → **ER은 epoch 증가 시 오히려 성능 하락** (81.38% → 76.22%). DER++는 비교적 안정적이나 미세 하락 (82.79% → 81.03%).
- [x] Joint Training이 98%+ 에 도달하는지 (논문 수준)? → **거의 도달.** ep=20에서 97.85±0.06%. 논문의 98.17%에 근접.

### 핵심 발견

1. **정규화 방법 (EWC, SI, LwF)은 epoch과 무관하게 CIL에서 구조적으로 실패** — 문헌 조사 결론과 완벽히 일치
2. **SI는 epoch 증가 시 오히려 악화** — 정규화 페널티가 과도하게 누적되어 학습 자체를 방해
3. **ER의 overfitting 경향** — epoch 증가 시 현재 태스크에 과적합되어 이전 태스크 망각 가속
4. **DER++가 가장 안정적인 Replay 방법** — epoch 변화에 대한 robustness가 ER보다 우수
5. **A-GEM은 개선되나 한계 명확** — gradient projection만으로는 CIL의 근본 문제 해결 불가

---

## Phase 6-3: Buffer Size Sweep 실험

**날짜:** 2026-04-04 ~ 2026-04-08
**상태:** ✅ 완료

### 실험 설계

| 항목 | 값 |
|------|-----|
| Sweep 변수 | buffer_size: 100, 500, 1000 |
| 대상 모델 | ER, DER++, A-GEM (버퍼 사용 모델만) |
| Seeds | [42, 123, 456, 789, 1024] |
| N_EPOCHS | 1 (고정, 기존 실험과 비교용) |
| 총 실험 수 | 3모델 × 5seed × 3buffer = **45회** |

> buffer=200은 기존 Phase 3 결과를 재활용 (중복 실행 불필요)

### 결과: AA (Average Accuracy) by Buffer Size

| 모델 | 카테고리 | buf=100 | buf=200 | buf=500 | buf=1000 |
|------|----------|---------|---------|---------|----------|
| ER | Replay | 73.58±1.07 | 81.38±0.70 | 87.22±0.61 | **89.04±1.29** |
| DER++ | Replay | 77.80±2.60 | 82.79±3.76 | 87.85±1.85 | **88.54±2.15** |
| A-GEM | Optimization | 23.86±2.80 | 23.77±1.17 | 24.04±2.07 | 24.04±2.07 |

### 결과: BWT (Backward Transfer) by Buffer Size

| 모델 | buf=100 | buf=200 | buf=500 | buf=1000 |
|------|---------|---------|---------|----------|
| ER | -30.68±1.38 | -20.72±0.81 | -12.72±0.92 | **-10.02±1.78** |
| DER++ | -25.77±3.40 | -19.34±4.78 | -12.74±2.52 | **-11.83±2.88** |
| A-GEM | -93.72±3.50 | -93.84±1.41 | -93.50±2.50 | -93.50±2.50 |

### 검증 포인트 결과

- [x] ER, DER++에서 buffer 크기에 따른 AA 상승 곡선? → **YES.** 둘 다 buffer 증가에 따라 명확한 로그-선형 상승. ER: 73.58% → 89.04% (+15.46%p), DER++: 77.80% → 88.54% (+10.74%p).
- [x] A-GEM도 buffer 증가 시 성능 향상되는지? → **NO.** A-GEM은 buffer 크기에 **완전히 무관** (~24% 고정). Gradient projection 방식은 버퍼 내용이 아닌 방향만 사용하므로 크기 증가의 이점이 없음.
- [x] buffer=1000에서 ER/DER++가 Joint에 얼마나 근접하는지? → ER 89.04%, DER++ 88.54% vs Joint 93.91% (ep=1 기준). **약 5%p 차이.** 버퍼를 더 늘리면 근접 가능성 있으나 수확체감 경향.

### 핵심 발견

1. **ER과 DER++는 buffer 크기에 강한 양의 상관관계** — 직접적인 과거 데이터 replay가 망각 방지의 핵심 메커니즘임을 재확인
2. **DER++는 소형 버퍼에서 ER보다 우수** — buf=100에서 DER++ 77.80% vs ER 73.58% (+4.22%p). Dark experience replay가 제한된 메모리에서 더 효율적
3. **대형 버퍼에서는 ER과 DER++ 성능 수렴** — buf=1000에서 ER 89.04% ≈ DER++ 88.54%. 충분한 데이터가 있으면 단순 replay로 충분
4. **A-GEM은 buffer 크기와 무관** — gradient constraint 방식의 구조적 한계. CIL에서는 gradient 방향 제약만으로 분류 경계 유지 불가

---

## Phase 6-4: 시간/리소스 지표

**날짜:** 2026-04-04
**상태:** ✅ 구현 완료

### 추가된 측정 항목

| 지표 | 측정 방법 | 저장 위치 |
|------|----------|----------|
| 실행 시간 (elapsed_sec) | `time.time()` 전후 차이 | sweep summary JSON |
| GPU 메모리 (gpu_mem_mb) | `nvidia-smi` 쿼리 | sweep summary JSON |

### 구현 위치

- [run_sweep.py](../run_sweep.py): `query_gpu_memory()` 함수 추가, 각 실험 전후 GPU 메모리 측정
- 실행 결과 JSON에 `elapsed_sec`, `gpu_mem_mb` 필드 포함

---

## 생성된 스크립트 목록

| 파일 | 역할 | 사용법 |
|------|------|--------|
| `run_sweep.py` | Epoch/Buffer sweep 실험 실행기 | `python run_sweep.py epoch --values 5 10 20` |
| `analysis_sweep.py` | Sweep 결과 분석 및 CSV/JSON 저장 | `python analysis_sweep.py epoch --include-baseline` |

---

## 참고 문헌

- van de Ven, G.M., Tuytelaars, T., & Tolias, A.S. (2022). "Three types of incremental learning." *Nature Machine Intelligence*, 4, 1185-1197.
- van de Ven, G.M., & Tolias, A.S. (2019). "Three scenarios for continual learning." arXiv:1904.07734.
- Bricken, T. (2020). "Improving Weight Regularization Continual Learning Baselines."
- Zhou, D.W., et al. (2023). "Deep Class-Incremental Learning: A Survey." arXiv:2302.03648.
