# 그래프 삽입 가이드 (Notion 수동 첨부용)

> Notion MCP는 이미지 직접 업로드를 지원하지 않아, 생성된 PNG 파일을 수동으로 첨부해야 합니다.
> 이 문서는 각 그래프 파일을 어느 Notion 페이지의 어느 위치에 삽입해야 하는지 안내합니다.

---

## 생성된 그래프 파일

모든 PNG 파일은 `results/figures/` 디렉토리에 저장되어 있습니다 (200 DPI).

| # | 파일명 | 설명 |
|---|--------|------|
| 1 | `figure_category_bar.png` | 1차 실험 모델별 AA 바 차트 (카테고리별 색상) |
| 2 | `figure_paper_comparison.png` | 논문 vs 본 실험 AA 비교 바 차트 |
| 3 | `figure_epoch_sweep.png` | Epoch sweep 라인 차트 (8개 모델) |
| 4 | `figure_buffer_sweep.png` | Buffer sweep 라인 차트 (ER/DER++/A-GEM) |
| 5 | `figure_bwt_aa_scatter.png` | BWT-AA 산점도 (Trade-off) |

---

## 페이지별 삽입 위치

### 1. 메인 보고서 — "Split-MNIST Continual Learning 실험 보고서"

| 삽입 파일 | 삽입 위치 (앞 섹션 바로 아래) |
|-----------|------------------------------|
| `figure_category_bar.png` | `## 1차 실험 → ### 결과` 섹션의 AA/BWT 표 아래 |
| `figure_bwt_aa_scatter.png` | `## 1차 실험 → ### 분석` 섹션 상단 (Trade-off 분석 설명 앞) |
| `figure_paper_comparison.png` | `## 2차 실험 → ### 벤치마크 논문 비교 조사` 표 아래 |
| `figure_epoch_sweep.png` | `## 2차 실험 → ### Epoch Sweep 실험` 표 아래 |
| `figure_buffer_sweep.png` | `## 2차 실험 → ### Buffer Size Sweep 실험` 표 아래 |

### 2. 1차 실험 상세 일지

| 삽입 파일 | 삽입 위치 |
|-----------|-----------|
| `figure_category_bar.png` | `Phase 4 → 결과 집계` 섹션의 최종 AA 표 아래 |
| `figure_bwt_aa_scatter.png` | `Phase 4 → 결과 집계` 섹션 말미 (BWT 해석 문단 위) |

### 3. 2차 실험 상세 일지

| 삽입 파일 | 삽입 위치 |
|-----------|-----------|
| `figure_paper_comparison.png` | `벤치마크 논문 비교 조사` 섹션 표 아래 |
| `figure_epoch_sweep.png` | `Epoch Sweep 실험 → 결과` 표 아래 |
| `figure_buffer_sweep.png` | `Buffer Size Sweep 실험 → 결과` 표 아래 |

### 4. CL 학습 시나리오 및 알고리즘 카테고리

> 이 페이지는 개념 설명 중심이므로 별도 그래프 첨부 없음.

---

## Notion에서 이미지 첨부 방법

1. Notion 페이지에서 삽입 위치로 이동
2. 빈 줄에서 `/image` 입력 → "Image" 블록 선택
3. "Upload" 탭에서 해당 PNG 파일 선택
4. 업로드 후 캡션 추가 (아래 권장 캡션 참조)

## 권장 캡션

| 파일 | 캡션 |
|------|------|
| `figure_category_bar.png` | Figure 1. Split-MNIST Class-IL 평균 정확도 (n_epochs=1, buffer=200, 5 seeds). 오차 막대는 표준편차. |
| `figure_bwt_aa_scatter.png` | Figure 2. Backward Transfer (BWT)와 Average Accuracy (AA)의 관계. 좌측 하단은 심각한 망각, 우측 상단은 이상적 영역. |
| `figure_paper_comparison.png` | Figure 3. 문헌(van de Ven et al., 2022)과 본 실험의 Split-MNIST Class-IL 결과 비교. |
| `figure_epoch_sweep.png` | Figure 4. 태스크당 epoch 수에 따른 AA 변화 (buffer=200, 5 seeds). |
| `figure_buffer_sweep.png` | Figure 5. 메모리 버퍼 크기에 따른 AA 변화 (n_epochs=1, 5 seeds, 로그 스케일). 점선은 Joint 상한선. |

---

## 재생성 방법

그래프를 수정하거나 재생성하려면:

```bash
.venv/Scripts/python.exe analysis_plots.py
```

CSV 데이터는 `results/split_mnist_latest.csv`, `results/epoch_sweep_latest.csv`, `results/buffer_sweep_latest.csv`를 참조합니다.
