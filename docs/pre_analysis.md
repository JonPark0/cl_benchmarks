# 사전 분석 결과 (Pre-Analysis)

> 작성일: 2026-03-31
> 참고 자료: `journal/research_continual_learning.md`, 논문 1·2·3 (journal/1, journal/2, journal/3)

---

## 1. 참고 논문 개요

### 논문 1 — Class-Incremental Learning: A Survey
- **저자:** Da-Wei Zhou, Qi-Wei Wang et al. (Nanjing University + NTU)
- **게재:** IEEE TPAMI, 2024
- **링크:** https://doi.org/10.1109/TPAMI.2024.3429383
- **코드:** https://github.com/zhoudw-zdw/CIL_Survey/
- **핵심 내용:**
  - CIL(Class-Incremental Learning) 방법론 **7개 카테고리** 분류 및 17개 방법론 통합 평가
  - 벤치마크: **CIFAR-100**, **ImageNet-100/1000**
  - Memory budget 공정 비교 프레임워크 제안
  - ViT 기반 최신 방법론까지 포함
- **프레임워크:** 자체 커스텀 코드 (CIL_Survey GitHub 레포지토리), Mammoth/Avalanche 미사용

| 카테고리 | 대표 알고리즘 |
|----------|--------------|
| Data Replay (Direct) | iCaRL, ER, EEIL |
| Data Replay (Generative) | DGR, CCGAN |
| Data Regularization | GDumb, Mnemonics |
| Dynamic Networks (Neuron/Backbone/Prompt Expansion) | DER, L2P, CODA-Prompt |
| Parameter Regularization | EWC, SI, MAS, LwF |
| Knowledge Distillation (Logit/Feature/Relational) | LwF, PODNet, DER++ |
| Model Rectify | WA, BiC, FOSTER |
| Template-Based Classification | MEMO, SLCA |

---

### 논문 2 — A Comprehensive Survey of Continual Learning: Theory, Method and Application
- **저자:** Liyuan Wang, Xingxing Zhang, Hang Su, Jun Zhu (Tsinghua University)
- **게재:** IEEE TPAMI, 2024
- **핵심 내용:**
  - CL의 이론·방법론·응용을 포괄하는 **가장 넓은 범위**의 서베이
  - **5개 카테고리** 분류 체계 (기존 3개에서 확장)
  - 7가지 학습 시나리오 정의 (IIL, DIL, TIL, CIL, TFCL, OCL, CPT)
  - 평가 지표: AA, AIA, BWT, FWT, FM, IM 전체 다룸
- **프레임워크:** 자체 커스텀 실험 코드, 별도 CL 라이브러리 미사용

| 카테고리 | 설명 | 대표 알고리즘 |
|----------|------|--------------|
| Regularization-based | 중요 파라미터 보호 | EWC, SI, MAS, VCL |
| Replay-based | 이전 데이터 재현 | ER, GEM, A-GEM, iCaRL |
| Optimization-based | 그래디언트 방향 제어 | OGD, PCGrad |
| Representation-based | 강건한 표현 학습 | PNN, PackNet |
| Architecture-based | 태스크별 구조 분리 | HAT, SupSup |

---

### 논문 3 — A Continual Learning Survey: Defying Forgetting in Classification Tasks
- **저자:** Matthias De Lange, Rahaf Aljundi et al. (KU Leuven)
- **게재:** IEEE TPAMI, 2022
- **핵심 내용:**
  - **Task-Incremental Learning** 중심 서베이 (Multi-head 설정)
  - **11개 방법론** 종합 실험 평가
  - 자체 **Continual Hyperparameter Framework** 제안 (task 간 데이터 없이 하이퍼파라미터 선정)
  - 벤치마크: **Tiny ImageNet**, **iNaturalist** (대규모, 불균형), **RecogSeq** (이질적 태스크 시퀀스)
- **프레임워크:** 자체 커스텀 실험 코드, Mammoth/Avalanche 미사용

| 카테고리 | 대표 알고리즘 |
|----------|--------------|
| Replay Methods | iCaRL, GEM, A-GEM, CoPE |
| Regularization-based | EWC, SI, MAS, LwF, VCL |
| Parameter Isolation | PackNet, HAT, PNN |

---

## 2. 프레임워크 3종 비교 분석

### 2.1 개요

| 항목 | Avalanche | Mammoth | 논문 커스텀 코드 |
|------|-----------|---------|-----------------|
| **출처** | ContinualAI (비영리 조직) | AImageLab (Univ. of Modena) | 각 논문 저자 |
| **목적** | 종합 연구·실용 라이브러리 | 재현성 중심 연구 프레임워크 | 특정 논문 실험 재현 |
| **기반** | PyTorch | PyTorch | PyTorch |
| **GitHub Stars** | ~2,000 | ~793 | - |
| **알고리즘 수** | ~20개 | **70개 이상** | 11~17개 |
| **데이터셋 수** | 다수 + 사용자 정의 | 23개 | 3~5개 |
| **논문** | JMLR 2023, CVPR-W 2021 | NeurIPS 2020, TPAMI 2022 | 각 서베이 논문 |
| **최신 버전** | v0.6.0 (2024.10, Beta) | 지속 업데이트 중 | 논문 출판 시점 |

### 2.2 모듈 구조 비교

**Avalanche** — 5개 독립 모듈
```
Benchmarks  →  데이터 처리 및 CL 벤치마크 생성
Training    →  학습 루프 및 사전 구현 Strategy
Evaluation  →  평가 지표 (TensorBoard 연동)
Models      →  모델 유틸리티 및 사전 학습 아키텍처
Logging     →  stdout / 파일 / TensorBoard 로깅
```

**Mammoth** — 모델 중심 플랫 구조
```
models/     →  70+ CL 알고리즘 (각각 독립 파일)
datasets/   →  23개 데이터셋 로더
backbone/   →  ResNet, ViT 등 백본
utils/      →  공통 유틸리티
```

**논문 커스텀** — 스크립트 형태
```
특정 알고리즘 세트만 구현
논문 재현에 최적화, 확장성 낮음
```

### 2.3 알고리즘 카테고리 커버리지

| 카테고리 | Avalanche | Mammoth | 논문1 | 논문2 | 논문3 |
|----------|:---------:|:-------:|:-----:|:-----:|:-----:|
| Regularization | ✅ EWC, SI, MAS, LwF | ✅ EWC, SI, MAS, LwF + 다수 | ✅ | ✅ | ✅ |
| Replay/Rehearsal | ✅ ER, GEM, A-GEM, iCaRL | ✅ ER, DER, DER++, GEM + 다수 | ✅ | ✅ | ✅ |
| Architecture/Isolation | ✅ PNN, Multi-head | ✅ PackNet, HAT + 다수 | ✅ | ✅ | ✅ |
| Knowledge Distillation | ✅ LwF | ✅ DER, DER++, PODNet + 다수 | ✅ | 제한적 | 제한적 |
| Optimization-based | ⚠️ 제한적 | ✅ OGD, 다수 | 제한적 | ✅ | 제한적 |

> **결론:** 5개 카테고리 완전 커버 — **Mammoth (70+) > Avalanche (~20) > 논문 커스텀 (11~17)**

---

## 3. 성능 평가 지표 비교

| 지표 | Avalanche | Mammoth | 논문 커스텀 |
|------|:---------:|:-------:|:-----------:|
| Average Accuracy (AA) | ✅ 내장 | ✅ 내장 | ✅ |
| Average Incremental Accuracy (AIA) | ✅ 내장 | ✅ 내장 | ✅ (논문1·2) |
| Backward Transfer (BWT) | ✅ 내장 | ✅ 내장 | ✅ |
| Forward Transfer (FWT) | ✅ 내장 | ✅ 내장 | 논문1·2만 |
| Forgetting Measure (FM) | ✅ 내장 | ⚠️ 수동 계산 | 논문3 사용 |
| Intransigence Measure (IM) | ⚠️ 제한적 | ⚠️ 수동 계산 | 논문2 사용 |

---

## 4. Split-MNIST 구동 용이성 비교

| 프레임워크 | 지원 여부 | 구동 방법 | 난이도 |
|-----------|:---------:|-----------|:------:|
| **Avalanche** | ✅ `SplitMNIST` 내장 | `benchmark = SplitMNIST(n_experiences=5)` | ⭐ 매우 쉬움 |
| **Mammoth** | ✅ `seq-mnist` 내장 | `python main.py --model ewc --dataset seq-mnist` | ⭐ 매우 쉬움 |
| 논문1 (CIL_Survey) | ❌ CIFAR100·ImageNet 전용 | 직접 구현 필요 | ⭐⭐⭐⭐ 어려움 |
| 논문3 (De Lange) | ⚠️ 부분 지원 | TinyImageNet 기준 수정 필요 | ⭐⭐⭐ 보통 |

> **결론:** Avalanche와 Mammoth 모두 Split-MNIST를 즉시 사용 가능. 알고리즘 수·재현성 관점에서 **Mammoth**를 주 실험 프레임워크로 선정.

---

## 5. 실험 프레임워크 선정 결론

**선정: Mammoth** (주) + **Avalanche** (보조 검증용)

| 선정 이유 | 내용 |
|----------|------|
| 알고리즘 다양성 | 70개 이상 구현 → 카테고리당 1~2개 선별 용이 |
| 재현성 | `REPRODUCIBILITY.md`로 하이퍼파라미터 문서화 |
| Split-MNIST 지원 | CLI 한 줄로 즉시 실행 |
| 확장성 | 새 모델 추가 용이한 플랫 구조 |
| 벤치마크 기반 논문 | NeurIPS 2020, TPAMI 2022 (피어리뷰 완료) |

---

## 6. 실험 대상 모델 (Split-MNIST 기준)

| 카테고리 | 선별 모델 | 선정 근거 |
|----------|----------|-----------|
| **Lower Bound** | SGD (Naive Fine-tuning) | 망각 하한선 |
| **Upper Bound** | Joint Training | 성능 상한선 |
| **Regularization** | EWC | CL 표준 베이스라인, 논문 1·2·3 전부 사용 |
| **Regularization** | SI (Synaptic Intelligence) | EWC의 온라인 버전, 논문 2·3 사용 |
| **Replay** | ER (Experience Replay) | 가장 기초적인 리플레이 방법 |
| **Replay** | DER++ | Mammoth 대표 SOTA, NeurIPS 2020 |
| **Knowledge Distillation** | LwF | 메모리 없는 KD 기반, 세 논문 모두 참조 |
| **Architecture** | PackNet 또는 HAT | 파라미터 격리 방식, 논문3 사용 |

**총 8개 모델 × 5회 반복 = 40회 실험**

---

## 7. 평가 프로토콜 요약

| 항목 | 설정 |
|------|------|
| 데이터셋 | Split-MNIST |
| CL 시나리오 | Class-Incremental Learning (CIL) |
| 태스크 수 | 5개 (각 2 클래스) |
| 반복 횟수 | 각 모델 × 5회 (다른 random seed) |
| 보고 지표 | AA (mean ± std), BWT (mean ± std), FWT |
| 디바이스 | NVIDIA RTX 3060 Laptop (CUDA 12.4) |
| 프레임워크 | Mammoth (주), PyTorch 2.6.0 |
