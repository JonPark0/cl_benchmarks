# 3차 실험 계획: Split-CIFAR10 벤치마크 확장

## 배경 및 동기

1·2차 실험은 Split-MNIST (Class-IL 5-task) 벤치마크에 집중되었습니다. 이 과정에서
다음의 구조적 결과가 확인되었습니다.

- 정규화 계열(EWC-on, SI, LwF)은 epoch 수와 무관하게 하한선(≈20%) 근방에 머무름.
- A-GEM은 gradient projection 기법의 한계로 buffer 크기와 무관하게 ≈24%에 수렴.
- ER/DER++만 buffer·epoch 증가에 유의미하게 반응하며, buffer=1000에서 AA 89% 선에 도달.

그러나 Split-MNIST는 상대적으로 난이도가 낮아, 다음과 같은 의문이 남아 있습니다.

- 더 복잡한 입력 분포(자연 이미지)에서도 Replay 방법이 동일하게 우월한가?
- Joint 상한선과 실제 Replay 방법 간 격차가 어느 정도까지 벌어지는가?
- Task-IL 시나리오에서는 Class-IL에서 실패한 정규화 방법이 회복되는가?

이러한 질문에 답하기 위해 3차 실험은 **Split-CIFAR10** 벤치마크로 확장합니다.

---

## 실험 목표

1. Split-CIFAR10 Class-IL 환경에서 1차 실험과 동일한 8개 모델의 AA/BWT/FWT를 측정합니다.
2. Task-IL 시나리오를 병행 실행하여 시나리오별 난이도 차이를 정량화합니다.
3. Replay 방법(ER, DER++)의 buffer size 의존성이 CIFAR10에서도 일관되게 나타나는지 검증합니다.
4. 입력 복잡도가 증가했을 때 Joint 상한선과 Replay 방법의 격차를 측정합니다.

---

## 실험 설계

### 공통 설정

| 항목 | 값 |
|------|-----|
| 데이터셋 | CIFAR-10 (5 task, 클래스 2개씩) |
| 백본 | ResNet-18 (Mammoth 기본) |
| Scenario | Class-IL (기본), Task-IL (병행) |
| Seeds | [42, 123, 456, 789, 1024] |
| Optimizer | SGD, lr=0.03 (Mammoth 기본값 기준) |
| Batch size | 32 |

### Phase 7-1: 기본 벤치마크 (Class-IL)

| 항목 | 값 |
|------|-----|
| 대상 모델 | 전체 8개 (sgd, joint, ewc-on, si, lwf, agem, er, derpp) |
| n_epochs | 50 (CIFAR10 표준 설정) |
| buffer_size | 200 (ER, DER++, A-GEM) |
| 총 실험 수 | 8 × 5 = 40회 |

### Phase 7-2: Task-IL 시나리오 비교

| 항목 | 값 |
|------|-----|
| 대상 모델 | ewc-on, si, lwf, agem, er, derpp (정규화/KD/버퍼 방법) |
| scenario | task-il |
| n_epochs | 50 |
| buffer_size | 200 |
| 총 실험 수 | 6 × 5 = 30회 |

> **가설:** Task-IL에서는 출력 헤드가 태스크별로 분리되어 있으므로, Class-IL에서 하한선에
> 머물렀던 EWC/SI/LwF가 실질적 성능(>60%)을 회복할 것으로 기대됩니다.

### Phase 7-3: Buffer Sweep (Replay 방법)

| 항목 | 값 |
|------|-----|
| 대상 모델 | er, derpp (A-GEM은 1차 결과상 buffer 무관하므로 제외) |
| buffer_size | 200, 500, 2000, 5120 |
| n_epochs | 50 |
| 총 실험 수 | 2 × 5 × 4 = 40회 |

> buffer_size=5120은 CIFAR10 CL 논문에서 자주 쓰이는 상한 설정(전체 훈련셋의 약 10%).

---

## 검증 포인트

- [ ] Split-MNIST에서 확인된 "정규화 방법의 Class-IL 실패" 현상이 CIFAR10에서도 반복되는가?
- [ ] Task-IL 시나리오에서 EWC/SI/LwF가 유의미한 성능 회복을 보이는가?
- [ ] ER/DER++의 buffer 의존성이 CIFAR10에서 더 가파른 곡선을 그리는가?
- [ ] Joint 상한선과 최고 Replay 방법 간 격차가 MNIST보다 큰가 (약 5%p → ?%p)?
- [ ] DER++가 CIFAR10에서도 소형 버퍼에서 ER보다 우월한가?

---

## 예상 리소스

| 항목 | 추정치 | 근거 |
|------|--------|------|
| GPU 메모리 | 3~5 GB | ResNet-18 + batch=32 |
| 실험당 소요 시간 | 약 15~25분 | CIFAR10 / 5task / 50 epoch / ResNet-18 |
| 전체 소요 시간 (Phase 7-1+2+3) | 약 30~45시간 | 110회 × 평균 20분 |

> 2차 실험(Phase 6) 총 소요 시간(약 30시간)과 비슷한 규모. 부분 실행 후 중간 점검 권장.

---

## 기대 산출물

1. `results/cifar10_class_il.csv` — Phase 7-1 집계 결과
2. `results/cifar10_task_il.csv` — Phase 7-2 집계 결과
3. `results/cifar10_buffer_sweep.csv` — Phase 7-3 집계 결과
4. `results/figures/figure_cifar10_*.png` — 대응 그래프 (카테고리 바, 시나리오 비교, buffer sweep)
5. `docs/experiment_log_phase7.md` — 실행 로그 및 분석
6. Notion: "3차 실험 상세 일지" 페이지 신규 생성

---

## 실행 순서 제안

1. **사전 점검**: Mammoth의 `seq-cifar10` 데이터셋 설정 확인, ResNet-18 기본값 확인
2. **파일럿 실행**: 각 모델당 seed 1개만으로 Phase 7-1 수행 → 시간·메모리 실측 확인
3. **본 실행 (Phase 7-1)**: 5 seed 전체 → Class-IL 기본 결과 확보
4. **본 실행 (Phase 7-2)**: Task-IL 시나리오
5. **본 실행 (Phase 7-3)**: Buffer sweep
6. **분석 및 시각화**: analysis_sweep.py/analysis_plots.py에 CIFAR10 지원 추가
7. **문서화**: experiment_log_phase7.md 작성 및 Notion 업데이트

---

## 리스크 및 대응

| 리스크 | 대응 |
|--------|------|
| 학습 시간이 예상보다 길어짐 | Phase 7-3의 buffer 값 축소 또는 seed 수 3개로 감소 |
| Mammoth의 CIFAR10 하이퍼파라미터 기본값과 논문 값 불일치 | 파일럿 단계에서 Joint 결과로 정합성 검증 |
| GPU OOM | batch_size 감소 또는 ResNet-18을 최소 구성으로 변경 |
| 재현 불가능한 결과 | deterministic 플래그 재확인, seed 로깅 강화 |

---

## 후속 실험 아이디어 (Phase 7 이후 선택지)

- **Split-CIFAR100**: 10 task, 더 긴 시퀀스로 망각 누적 효과 관찰
- **EWC/SI 개선 실험**: Bricken (2020) β 계수 도입 버전 구현 및 효과 측정
- **GEM vs A-GEM 재실행**: 1차에서 GEM 컴파일 실패 이슈 해결 후 비교
- **Hybrid 방법 (ER + LwF)**: Replay와 KD를 결합한 단순 하이브리드의 효과 검증
