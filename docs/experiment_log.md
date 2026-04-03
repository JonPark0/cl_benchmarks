# 실험 진행 로그 (Experiment Log)

> 이 문서는 실험 진행 과정에서 지속적으로 업데이트됩니다.
> 각 Phase가 완료될 때마다 결과와 특이사항을 기록합니다.

---

## 진행 상황 요약

| Phase | 단계 | 상태 | 완료일 |
|-------|------|------|--------|
| 0 | 환경 셋업 | ✅ 완료 | 2026-03-31 |
| 1 | Mammoth 설치 및 단일 실행 테스트 | ✅ 완료 | 2026-03-31 |
| 2 | 실험 스크립트 작성 | ✅ 완료 | 2026-03-31 |
| 3 | 전체 실험 실행 (8모델 × 5 seed) | ✅ 완료 | 2026-03-31 ~ 2026-04-01 |
| 4 | 결과 집계 및 분석 | ✅ 완료 | 2026-04-01 |
| 5 | 최종 결과 정리 | ✅ 완료 | 2026-04-01 |

---

## Phase 0: 환경 셋업

**날짜:** 2026-03-31
**상태:** ✅ 완료

### 구성 환경

| 항목 | 내용 |
|------|------|
| OS | Windows 11 Pro (10.0.26200) |
| Python | 3.12.10 |
| 가상환경 경로 | `continual_learning/.venv/` |
| PyTorch | 2.6.0+cu124 |
| CUDA Toolkit | 12.4 |
| NVIDIA Driver | 591.59 (CUDA 13.1 호환) |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| VRAM | 6.0 GB |

### CUDA 검증 결과

```
PyTorch: 2.6.0+cu124
CUDA available: True
CUDA version: 12.4
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
VRAM: 6.0 GB
```

**→ CUDA 가속 사용 가능 확인. GPU 기반 학습으로 속도 향상 기대.**

### 설치된 패키지

```
torch==2.6.0+cu124
torchvision==0.21.0+cu124
torchaudio==2.6.0+cu124
```

### 특이사항 / 메모

- CUDA Driver 버전(591.59)은 CUDA 13.1을 지원하나, PyTorch 최신 안정판(cu124)과 호환 확인
- RTX 3060 Laptop VRAM 6GB: Split-MNIST 기준으로 충분, 대형 데이터셋(ImageNet 등)에서는 배치 사이즈 조정 필요

---

## Phase 1: Mammoth 설치 및 단일 실행 테스트

**날짜:** 2026-03-31
**상태:** ✅ 완료

### 설치 과정

| 작업 | 결과 |
|------|------|
| `git clone https://github.com/aimagelab/mammoth.git` | 성공 |
| 의존성 설치 (kornia, timm, tqdm 등) | 성공 |
| 추가 필요 패키지 (pandas, googledrivedownloader) | 성공 |

### 발견된 이슈 및 해결

| 이슈 | 원인 | 해결 |
|------|------|------|
| `ModuleNotFoundError: pandas` | seq_celeba 데이터셋 의존성 | `pip install pandas` |
| `ModuleNotFoundError: google_drive_downloader` | seq_eurosat 의존성 | `pip install googledrivedownloader==0.4` |
| `ValueError: invalid literal for int: 'a'` | `--device cuda` 형식 오류 | `--device 0` (GPU 인덱스)으로 변경 |
| Windows cp949 인코딩 에러 | 박스 그리기 문자 | `PYTHONIOENCODING=utf-8` 환경변수 설정 |
| HAT 모델 COMPATIBILITY = ['task-il'] | class-il 미지원 | **GEM**으로 대체 (class-il 지원) |
| PNN 모델 COMPATIBILITY = ['task-il'] | class-il 미지원 | GEM으로 대체 |

### SGD 단일 실행 결과 (seed 42, 1 epoch)

- **실행 시간:** 약 6.5분 (22:47 ~ 22:54)
- **GPU 메모리 사용:** 18.18 MB (매우 경량)
- **결과 저장 경로:** `raw_data/split_mnist/sgd/class-il/seq-mnist/sgd/logs.pyd`

| 태스크 | Class-IL | Task-IL | 설명 |
|--------|----------|---------|------|
| Task 1 | 99.91% | 99.91% | Task 1 학습 직후 |
| Task 2 | 49.14% | 99.05% | Task 1 완전 망각 시작 |
| Task 3 | 33.17% | 98.23% | 이전 태스크 0%로 수렴 |
| Task 4 | 24.84% | 90.08% | |
| Task 5 | **19.61%** | 81.93% | **최종 AA: 19.61%** |

→ **Catastrophic Forgetting 명확히 확인** (Class-IL: 99.91% → 19.61%)

### 결과 파일 형식 확인

- `logs.pyd`: Python dict literal을 줄 단위로 저장 (실험 1회 = 1줄)
- 키: `accmean_task{N}`, `accuracy_{j}_task{N}`, `backward_transfer`, `forward_transfer`, `forgetting`
- BWT/FWT/Forgetting 활성화: `--enable_other_metrics 1` 필요

---

## Phase 2: 실험 스크립트 작성

**날짜:** 2026-03-31
**상태:** ✅ 완료

### 생성된 파일

| 파일 | 역할 |
|------|------|
| `run_experiments.py` | 8모델 × 5 seed 자동 실행, 결과 JSON 요약 저장 |
| `analysis.py` | logs.pyd 파싱, mean/std 계산, CSV/JSON 저장, 표 출력 |

### 최종 모델 구성 (HAT/PNN → GEM 대체)

| 카테고리 | 모델 | Mammoth 식별자 | lr | 비고 |
|----------|------|----------------|----|------|
| Lower Bound | SGD | `sgd` | 0.1 | |
| Upper Bound | Joint | `joint` | 0.1 | |
| Regularization | EWC Online | `ewc-on` | 0.1 | e_lambda=0.7, gamma=1.0 |
| Regularization | SI | `si` | 0.1 | c=0.5, xi=0.001 |
| Replay | ER | `er` | 0.1 | buffer=200 |
| Replay | DER++ | `derpp` | 0.1 | buffer=200, alpha=0.1, beta=0.5 |
| KD | LwF | `lwf` | 0.1 | alpha=1.0, temp=2.0 |
| Optimization | GEM | `gem` | 0.1 | buffer=200, gamma=0.5 |

---

## Phase 3: 전체 실험 실행

**날짜:** 2026-03-31 23:06 시작
**상태:** 🔄 실행 중

### 실행 방법

```bash
cd continual_learning/
python run_experiments.py
# 또는 특정 모델만: python run_experiments.py --model ewc-on
```

### 실험 매트릭스 (실시간 업데이트)

> ✅ 완료 | 🔄 진행 중 | ⬜ 대기

| 모델 | seed 42 | seed 123 | seed 456 | seed 789 | seed 1024 | 상태 |
|------|:-------:|:--------:|:--------:|:--------:|:---------:|------|
| SGD (Naive) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 5/5 |
| Joint | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 5/5 |
| EWC Online | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 5/5 |
| SI | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 5/5 |
| ER | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 5/5 |
| DER++ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 5/5 |
| LwF | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 5/5 |
| A-GEM | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 5/5 |

### 특이사항

- **SGD seed 42 중복**: 테스트 런(`enable_other_metrics=False`) + 본실험(`True`) 둘 다 logs.pyd에 저장됨
  → `analysis.py`에서 `enable_other_metrics=False` 항목 자동 제외하도록 수정 완료
- **np.float64(...) 파싱 오류**: `ast.literal_eval` 호환을 위해 regex로 float 변환 처리 추가
- **GEM 실행 불가**: `quadprog` (Linux 전용 + Python ≤3.10 필요) 의존성 문제. `qpsolvers` 설치 시도했으나 Mammoth 코드 내 하드코딩된 예외로 차단됨
  → **A-GEM**으로 최종 대체 (Mammoth class-il 호환, Windows 정상 동작)
- **ewc-on 저장 경로 불일치**: Mammoth가 하이픈을 언더스코어로 변환 (`ewc-on` → `ewc_on`)
  → `analysis.py`에서 `model_name.replace("-", "_")` 처리 추가
- GPU 메모리 사용량: 모델별 18~50 MB (6GB 대비 매우 여유)

### 실험별 소요 시간 기록

| 모델 | 1회 평균 (예상) | 5회 총계 (예상) |
|------|----------------|----------------|
| SGD | ~7분 | ~35분 |
| Joint | ~20분 (전체 데이터 동시 학습) | ~100분 |
| EWC Online | ~7분 | ~35분 |
| SI | ~7분 | ~35분 |
| ER | ~8분 | ~40분 |
| DER++ | ~8분 | ~40분 |
| LwF | ~7분 | ~35분 |
| GEM | ~15분 (gradient projection) | ~75분 |
| **총합** | | **~395분 (~6.6시간)** |

---

## Phase 4: 결과 집계 및 분석

**날짜:** 2026-04-01
**상태:** ✅ 완료

### 최종 집계 결과 (8개 모델, Class-IL, 5회 반복)

> `python analysis.py` 최종 실행 결과 (2026-04-01 11:56)
> 결과 파일: `results/split_mnist_latest.csv`, `results/split_mnist_latest.json`

| 순위 | 모델 | 카테고리 | N | AA (mean ± std) | BWT (mean ± std) | FWT |
|:----:|------|----------|:-:|:---------------:|:----------------:|:---:|
| - | Joint | Upper Bound | 5 | **93.91 ± 0.14** | +68.74 ± 8.10 | 0.00 |
| 1 | DER++ | Replay | 5 | **82.79 ± 3.76** | -19.34 ± 4.78 | 0.00 |
| 2 | ER | Replay | 5 | **81.38 ± 0.70** | -20.72 ± 0.81 | 0.00 |
| 3 | A-GEM | Optimization | 5 | **23.77 ± 1.17** | -93.84 ± 1.41 | 0.00 |
| 4 | SI | Regularization | 5 | 22.15 ± 4.19 | -92.21 ± 5.33 | 0.00 |
| 5 | EWC Online | Regularization | 5 | 19.51 ± 0.06 | -99.17 ± 0.07 | 0.00 |
| - | SGD | Lower Bound | 5 | 19.50 ± 0.07 | -99.17 ± 0.06 | 0.00 |
| 6 | LwF | KD | 5 | 19.34 ± 0.47 | -99.23 ± 0.15 | **+6.96 ± 1.71** |

### 주요 발견사항

#### 1. Replay 방법이 압도적 우위
- ER (81.38%), DER++ (82.79%) — 하한선(19.50%) 대비 **4배 이상** 개선
- 단 200개 버퍼만으로도 망각을 크게 억제
- DER++가 ER보다 소폭 우수하나 표준편차는 더 큼 (seed 간 변동성 높음)

#### 2. 정규화 기반 방법의 한계 (n_epochs=1 환경)
- **EWC Online ≈ SGD (19.51% vs 19.50%)**: Fisher Information 기반 정규화가 1 epoch 학습에서는 효과 없음
- **SI 소폭 우위 (22.15%)**: 온라인 중요도 추정이 1 epoch 환경에서 약간 유리하나 표준편차가 큼 (±4.19)
- 정규화 방법은 epoch 수가 많을수록 효과 증가 예상

#### 3. LwF: 망각 방지는 실패, FWT는 유일하게 양수
- AA 19.34%로 가장 낮음 — KD 기반 방법이 Class-IL에서 특히 취약
- FWT +6.96 ± 1.71 — **유일하게 의미 있는 Forward Transfer 확인**
- 이전 태스크 지식이 새 태스크 초기 성능에 기여하지만 망각을 막지는 못함

#### 4. A-GEM 성능 제한 (23.77%)
- GEM 대비 메모리 효율적이지만, 평균 그래디언트 투영으로 인해 망각 억제력이 ER/DER++보다 훨씬 낮음
- BWT -93.84로 정규화 방법보다도 높은 망각 발생

#### 5. Joint Training 상한 (93.91%)
- BWT +68.74: 전체 데이터 공동 학습으로 이전 태스크 성능이 시간에 따라 향상됨 (정상)
- Replay 최고 성능(DER++ 82.79%)과 여전히 **11% 격차** 존재

### 소요 시간 (실제)
- SGD: ~7분/회, Joint: ~20분/회
- EWC-on, SI, LwF: ~8분/회
- ER, DER++: ~9분/회
- A-GEM: ~8분/회
- **전체 실험 총 소요: 약 8.8시간** (2026-03-31 23:06 ~ 2026-04-01 11:56)

---

## Phase 5: 최종 결과 정리 및 보고

**날짜:** 2026-04-01
**상태:** ✅ 완료

---

### 5.1 결과 파일 목록

| 파일 | 설명 |
|------|------|
| `results/split_mnist_latest.csv` | 최신 집계 결과 (mean/std) |
| `results/split_mnist_latest.json` | 최신 집계 결과 (raw 값 포함) |
| `results/split_mnist_results_20260401_115656.csv` | 타임스탬프 백업 (CSV) |
| `results/split_mnist_results_20260401_115656.json` | 타임스탬프 백업 (JSON) |
| `raw_data/split_mnist/{model}/class-il/seq-mnist/{model}/logs.pyd` | 모델별 원본 실험 로그 (40개) |
| `raw_data/split_mnist/{model}/run_output.log` | 모델별 실행 stdout/stderr 로그 |

---

### 5.2 최종 결과 요약표

**설정:** Split-MNIST, Class-Incremental Learning, n_epochs=1, buffer_size=200, seeds=[42, 123, 456, 789, 1024]

| 순위 | 모델 | 카테고리 | N | AA (mean ± std) | BWT (mean ± std) | FWT (mean ± std) |
|:----:|------|----------|:-:|:---------------:|:----------------:|:----------------:|
| - | **Joint** | Upper Bound | 5 | **93.91 ± 0.14** | +68.74 ± 8.10 | 0.00 ± 0.00 |
| 1 | **DER++** | Replay | 5 | **82.79 ± 3.76** | -19.34 ± 4.78 | -1.54 ± 0.19 |
| 2 | **ER** | Replay | 5 | **81.38 ± 0.70** | -20.72 ± 0.81 | -1.54 ± 0.16 |
| 3 | **A-GEM** | Optimization | 5 | 23.77 ± 1.17 | -93.84 ± 1.41 | -1.57 ± 0.22 |
| 4 | **SI** | Regularization | 5 | 22.15 ± 4.19 | -92.21 ± 5.33 | -1.57 ± 0.22 |
| 5 | **EWC Online** | Regularization | 5 | 19.51 ± 0.06 | -99.17 ± 0.07 | -1.62 ± 0.18 |
| - | **SGD** | Lower Bound | 5 | 19.50 ± 0.07 | -99.17 ± 0.06 | -1.49 ± 0.20 |
| 6 | **LwF** | Knowledge Distillation | 5 | 19.34 ± 0.47 | -99.23 ± 0.15 | **+6.96 ± 2.05** |

> **BWT 해석:** 양수 = 이전 태스크 성능 향상 (역방향 전이), 음수 = 망각 발생  
> **FWT 해석:** 양수 = 이전 학습이 새 태스크에 도움, 음수 = 이전 학습이 다소 방해

---

### 5.3 카테고리별 분석

#### Upper Bound: Joint Training (93.91%)
- 전체 태스크 데이터를 동시에 학습하는 이상적 시나리오
- BWT +68.74: 데이터 공동 학습으로 이전 태스크 성능이 시간이 지날수록 향상되는 정상 현상
- Replay 최고 성능(DER++ 82.79%)과 약 **11%p 격차** — CL 연구의 목표 영역

#### Replay 기반: ER (81.38%) / DER++ (82.79%)
- **유일하게 망각을 효과적으로 억제**한 카테고리 (BWT ~-20%)
- 버퍼 200개(전체 학습 데이터 60,000장의 0.33%)만으로 하한선 대비 **4배 이상** AA 달성
- DER++가 ER보다 1.4%p 우수하나, 표준편차가 더 큼 (±3.76 vs ±0.70)
  - DER++는 logit 보존 기법으로 soft label을 활용해 평균적으로 우수하지만 seed 간 변동성이 존재
- ER의 낮은 분산(±0.70)은 단순 방법의 안정성을 시사

#### Regularization: EWC Online (19.51%) / SI (22.15%)
- **n_epochs=1 환경에서 SGD와 사실상 동일한 성능**
- EWC Online: Fisher Information 기반 weight 제약이 단 1 epoch 학습에서는 효과적으로 작동하지 못함
- SI: 온라인 중요도 추정 방식이 EWC보다 약간 유리하나 표준편차 ±4.19로 매우 불안정
- **이 결과는 n_epochs=1 환경의 한계**이며, 다중 epoch 설정에서 재평가 필요

#### Knowledge Distillation: LwF (19.34%)
- AA 기준 최하위 — 버퍼 없이 지식 증류만으로는 Class-IL 망각 억제 불가
- **FWT +6.96 ± 2.05: 8개 모델 중 유일하게 양의 Forward Transfer**
  - 이전 태스크 학습에서 형성된 표현이 새 태스크 초기 성능에 기여함을 확인
  - 그러나 망각 방지 효과는 없어 최종 AA는 가장 낮음
- Class-IL에서 KD 방법의 구조적 한계 확인 (출력 헤드 분리 없이는 이전 지식 보존 불가)

#### Optimization: A-GEM (23.77%)
- BWT -93.84로 정규화 방법보다도 높은 망각 발생 — 예상보다 저조
- 평균 그래디언트 투영(averaged gradient projection)이 제약으로는 작동하나 망각 억제력 부족
- 버퍼 200개를 활용하지만 ER/DER++보다 **성능이 훨씬 낮음** (58%p 차이)
- GEM의 quadprog 의존성 문제로 A-GEM으로 대체하였으나, 원래 GEM보다 완화된 제약이 성능 손실로 이어짐

#### Lower Bound: SGD (19.50%)
- Catastrophic Forgetting의 기준선 확인
- BWT -99.17: 새 태스크 학습 시 이전 태스크 거의 완전 망각
- 표준편차 ±0.07로 결정론적 패턴 (항상 마지막 태스크만 잘 기억)

---

### 5.4 실험 설계 검증

| 검증 항목 | 결과 | 평가 |
|----------|------|------|
| 하한선(SGD) 확인 | 19.50% ± 0.07 | ✅ 망각 기준선 명확 |
| 상한선(Joint) 확인 | 93.91% ± 0.14 | ✅ 목표 성능 범위 정의됨 |
| 5회 반복의 안정성 | 대부분 ±1% 이하 | ✅ 통계적으로 신뢰 가능 |
| 카테고리별 대표성 | 5개 카테고리 커버 | ✅ 계획 대로 |
| 재현성 (seed 고정) | 동일 seed 재현 확인 | ✅ |

---

### 5.5 현재 설정의 한계 및 후속 실험 제안

**한계:**
- `n_epochs=1`: 정규화 기반 방법(EWC, SI)의 성능을 과소 평가할 가능성 — 이 방법들은 epoch 수가 많을수록 효과 증가
- `buffer_size=200`: 소형 버퍼 설정 — 버퍼 크기 증가 시 Replay 계열 성능 향상 기대
- Split-MNIST는 CL 벤치마크 중 상대적으로 쉬운 편 — 어려운 벤치마크에서 결과 패턴이 달라질 수 있음

**추천 후속 실험:**

| 우선순위 | 실험 | 목적 |
|---------|------|------|
| 1 | n_epochs: 1 → 5 → 10 | 정규화 방법(EWC, SI) 재평가 |
| 2 | buffer_size: 100 / 200 / 500 | Replay 성능의 버퍼 의존성 분석 |
| 3 | Split-CIFAR10, Split-CIFAR100 | 더 어려운 벤치마크로 일반화 검증 |
| 4 | GEM (Linux 환경) | quadprog 의존성 해소 후 A-GEM과 비교 |

---

## 트러블슈팅 기록

| 날짜 | 문제 | 원인 | 해결 방법 |
|------|------|------|-----------|
| 2026-03-31 | PyTorch 미설치 | 새 환경 | venv 생성 후 CUDA 124 버전 설치 |

---

## 참고 링크

- Mammoth GitHub: https://github.com/aimagelab/mammoth
- Avalanche GitHub: https://github.com/ContinualAI/avalanche
- Split-MNIST 원본 논문 (GEM): https://arxiv.org/abs/1706.08840
