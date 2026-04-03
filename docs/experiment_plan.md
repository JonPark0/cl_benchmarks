# 실험 계획 (Experiment Plan)

> 작성일: 2026-03-31
> 상태: 계획 수립 완료 / 실험 준비 중

---

## 1. 실험 목표

Split-MNIST 벤치마크를 사용하여 Continual Learning 알고리즘 카테고리별 대표 모델의 성능을 정량적으로 비교한다. 각 모델을 5회 반복 실행하여 통계적으로 유의미한 평균값(mean ± std)을 산출한다.

---

## 2. 데이터셋

### Split-MNIST

| 항목 | 상세 |
|------|------|
| 기반 데이터셋 | MNIST (LeCun et al., 1998) |
| CL 분할 방식 | 10개 클래스 → 5개 태스크 (태스크당 2클래스) |
| CL 시나리오 | Class-Incremental Learning (CIL) |
| 태스크 구성 | Task 1: {0,1}, Task 2: {2,3}, Task 3: {4,5}, Task 4: {6,7}, Task 5: {8,9} |
| 학습 데이터 | 60,000장 |
| 테스트 데이터 | 10,000장 |
| 이미지 크기 | 28×28 grayscale |
| CL 벤치마크 기원 | Lopez-Paz & Ranzato, "GEM", NeurIPS 2017 |

---

## 3. 실험 환경

| 항목 | 설정 |
|------|------|
| OS | Windows 11 Pro |
| Python | 3.12.10 |
| 가상환경 | `.venv/` (continual_learning/ 내) |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 (Driver: 13.1) |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| VRAM | 6.0 GB |
| CL 프레임워크 | Mammoth (주) |

---

## 4. 실험 대상 모델

### 4.1 모델 목록

| # | 카테고리 | 모델명 | Mammoth 식별자 | 근거 논문 |
|---|----------|--------|----------------|-----------|
| 1 | **Lower Bound** | Naive Fine-tuning (SGD) | `sgd` | - |
| 2 | **Upper Bound** | Joint Training | `joint` | - |
| 3 | **Regularization** | EWC (Elastic Weight Consolidation) | `ewc_on` | Kirkpatrick et al., 2017 |
| 4 | **Regularization** | SI (Synaptic Intelligence) | `si` | Zenke et al., 2017 |
| 5 | **Replay** | ER (Experience Replay) | `er` | Chaudhry et al., 2019 |
| 6 | **Replay** | DER++ (Dark Experience Replay++) | `derpp` | Buzzega et al., NeurIPS 2020 |
| 7 | **Knowledge Distillation** | LwF (Learning without Forgetting) | `lwf` | Li & Hoiem, TPAMI 2018 |
| 8 | **Optimization** | A-GEM (Averaged GEM) | `agem` | Chaudhry et al., ICLR 2019 |

> 총 8개 모델 × 5 seed = **40회 실험**

### 4.2 모델 선정 근거

- **SGD, Joint**: 성능 범위(하한·상한)를 정의하는 기준선
- **EWC**: CL 연구의 가장 표준적인 정규화 베이스라인. 논문 1·2·3 전부 사용
- **SI**: EWC의 온라인 변형. 논문 2·3에서 EWC와 비교
- **ER**: 가장 단순한 리플레이 방법. 복잡한 방법의 기준점
- **DER++**: Mammoth 프레임워크의 대표 SOTA. NeurIPS 2020 발표
- **LwF**: 메모리 버퍼 없는 KD 기반 방법. 세 논문 모두 참조
- **A-GEM**: Averaged GEM (Chaudhry et al., ICLR 2019). GEM의 메모리 효율 버전. GEM은 `quadprog` (Linux 전용) 의존성 문제로 Windows에서 실행 불가 → **A-GEM**으로 대체 (class-il 지원, Windows 호환)

---

## 5. 평가 지표

| 지표 | 수식 | 의미 |
|------|------|------|
| **AA** (Average Accuracy) | `(1/T) Σ a_{T,i}` | 모든 태스크 학습 후 평균 정확도 |
| **BWT** (Backward Transfer) | `(1/(T-1)) Σ (a_{T,i} - a_{i,i})` | 이전 태스크 망각 정도 (음수 = 망각) |
| **FWT** (Forward Transfer) | `(1/(T-1)) Σ (a_{i-1,i} - b_i)` | 이전 학습의 새 태스크 전이 효과 |
| **FM** (Forgetting Measure) | `(1/(T-1)) Σ max(a_{t,i}) - a_{T,i}` | 최고 성능 대비 최종 하락 |

**주요 보고 지표:** AA (mean ± std), BWT (mean ± std)
**보조 보고 지표:** FWT, FM

---

## 6. 반복 실험 설계

```
seeds = [42, 123, 456, 789, 1024]   # 5개 고정 seed
모델 수 = 8
총 실험 수 = 8 × 5 = 40회
```

각 실험 결과는 `raw_data/split_mnist/{model_name}/seed_{seed}.json` 형식으로 저장.

---

## 7. 디렉토리 구조

```
continual_learning/
├── .venv/                          # Python 가상환경
├── docs/
│   ├── pre_analysis.md             # 사전 분석 결과
│   ├── experiment_plan.md          # 본 문서 (실험 계획)
│   ├── experiment_log.md           # 실험 진행 로그 (지속 업데이트)
│   └── environment.md              # 환경 설정 문서
├── raw_data/
│   └── split_mnist/
│       ├── sgd/                    # seed별 결과 JSON
│       ├── joint/
│       ├── ewc_on/
│       ├── si/
│       ├── er/
│       ├── derpp/
│       ├── lwf/
│       └── hat/
├── results/
│   └── split_mnist_summary.csv     # 집계된 mean ± std 결과
├── configs/                        # 모델별 하이퍼파라미터 설정
├── run_experiments.sh              # 일괄 실험 실행 스크립트
└── analysis.py                     # 결과 집계 및 시각화
```

---

## 8. 실험 단계별 계획

| 단계 | 작업 | 상태 |
|------|------|------|
| Phase 0 | 환경 셋업 (venv, PyTorch CUDA) | ✅ 완료 |
| Phase 1 | Mammoth 설치 및 Split-MNIST 단일 실행 테스트 | ✅ 완료 |
| Phase 2 | 실험 스크립트 작성 (run_experiments.py, analysis.py) | ✅ 완료 |
| Phase 3 | 8개 모델 × 5 seed 전체 실험 실행 | ✅ 완료 |
| Phase 4 | 결과 집계 및 분석 (analysis.py) | ✅ 완료 |
| Phase 5 | 결과 정리 및 보고 | ✅ 완료 |

---

## 9. 실제 실험 결과 (Split-MNIST, Class-IL, n_epochs=1, buffer=200)

```csv
model,category,n_runs,aa_mean,aa_std,bwt_mean,bwt_std,fwt_mean,fwt_std
sgd,Lower Bound,5,19.50,0.07,-99.17,0.06,-1.49,0.20
joint,Upper Bound,5,93.91,0.14,68.74,8.10,0.00,0.00
ewc-on,Regularization,5,19.51,0.06,-99.17,0.07,-1.62,0.18
si,Regularization,5,22.15,4.19,-92.21,5.33,-1.57,0.22
er,Replay,5,81.38,0.70,-20.72,0.81,-1.54,0.16
derpp,Replay,5,82.79,3.76,-19.34,4.78,-1.54,0.19
lwf,Knowledge Distillation,5,19.34,0.47,-99.23,0.15,6.96,2.05
agem,Optimization,5,23.77,1.17,-93.84,1.41,-1.57,0.22
```

**주요 발견:**
- Replay 기반 (ER, DER++) 만이 망각 억제에 효과적 (AA ~82%)
- Regularization (EWC, SI) 및 LwF는 SGD 수준으로 붕괴 → Class-IL에서 헤드 분리 없이 한계
- A-GEM은 메모리를 활용하나 제약이 너무 강해 실질적 개선 미미
- Joint Training (상한) 93.91% vs SGD (하한) 19.50% → 성능 범위 확인

---

## 10. 참고 사항

- Mammoth GitHub: https://github.com/aimagelab/mammoth
- Mammoth 재현성 가이드: https://github.com/aimagelab/mammoth/blob/master/REPRODUCIBILITY.md
- Split-MNIST 벤치마크 원본 논문: Lopez-Paz & Ranzato, NeurIPS 2017
