# docs

실험을 진행하면서 쓴 기록 문서이다. 논문의 최종 결과는 2026-10-03 지도교수 면담(교정 1) 이후 Linux 단일 환경에서 다시 실행한 실험(`run_linux.py`, `raw_data/linux/`)에서 나왔다. 아래 문서는 모두 그 이전 단계의 기록이며, 최종 실험 조건은 저장소 [README](../README.md)와 `run_linux.py` 머리말을 기준으로 한다.

## 최종 실험과 달라진 점

| 항목 | 교정 전 기록 | 최종 실험 |
|---|---|---|
| 실행 환경 | Windows 11과 WSL2를 함께 사용하고 플랫폼 간 일치를 검증 | WSL2 Ubuntu 24.04 하나 |
| 옵티마이저 | SGD, 학습률 0.1 | Adam, 학습률 0.001로 통일 |
| 최적화 기반 모델 | A-GEM(수정본 `agem_fixed`) | GEM |
| 시드 | 5~10개 | 기본 실험 20개 |
| 지표 | AA, BWT, FWT | AA, BWT(Task-IL AA는 보조) |
| 결과 위치 | `raw_data/`, `results/` | `raw_data/linux/` |

## 문서 목록

| 문서 | 작성 시기 | 내용 |
|---|---|---|
| [pre_analysis.md](pre_analysis.md) | 2026-03 | 실험 전 문헌 조사와 알고리즘 선정 |
| [experiment_plan.md](experiment_plan.md) | 2026-03 | 1차 실험 계획 |
| [environment.md](environment.md) | 2026-03 | Windows 환경 설정. 최종 환경은 `raw_data/linux/environment.json` |
| [experiment_log.md](experiment_log.md) | 2026-03~04 | 1차 실험 진행 기록(Phase 1~5) |
| [experiment_log_phase6.md](experiment_log_phase6.md) | 2026-04 | 에폭, 버퍼 민감도 실험 기록 |
| [figure_insertion_guide.md](figure_insertion_guide.md) | 2026-04 | 교정 전 그림을 Notion에 붙이는 안내 |
| [next_experiment_plan.md](next_experiment_plan.md) | 2026-04 | Split-CIFAR10 확장 계획. 진행하지 않았다 |
| [r0_platform_gate.md](r0_platform_gate.md) | 2026-07 | Windows와 WSL2 결과 일치 검증 |
| [r1_r2_r3_findings.md](r1_r2_r3_findings.md) | 2026-07 | Mammoth A-GEM의 버퍼 크기 결함 분석과 수정본 검증 |
