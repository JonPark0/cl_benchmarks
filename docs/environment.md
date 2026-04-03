# 환경 설정 (Environment Setup)

> 작성일: 2026-03-31
> 상태: 초기 셋업 완료

---

## 1. 시스템 환경

| 항목 | 내용 |
|------|------|
| OS | Windows 11 Pro (Build 10.0.26200) |
| Shell | bash (Git Bash / VSCode 통합 터미널) |
| Python | 3.12.10 |
| pip | 25.0.1 |

## 2. GPU / CUDA 환경

| 항목 | 내용 |
|------|------|
| GPU 모델 | NVIDIA GeForce RTX 3060 Laptop GPU |
| VRAM | 6.0 GB |
| NVIDIA Driver | 591.59 |
| 지원 CUDA 버전 | 13.1 (드라이버 지원 최대) |
| 설치된 CUDA Toolkit | 12.4 (PyTorch cu124) |
| CUDA 가속 가능 | ✅ 확인 완료 |

### CUDA 호환성 메모

- NVIDIA Driver 591.59는 CUDA 13.1까지 지원 (상위 호환)
- PyTorch 안정 최신 버전은 cu124 (CUDA 12.4) 기준으로 설치
- cu124는 드라이버 591.59와 완전 호환 확인됨

---

## 3. 가상환경 설정

### 경로

```
c:\Users\yeoch\Documents\vscode\ai_project_2\continual_learning\.venv\
```

### 생성 명령어

```bash
cd continual_learning/
python -m venv .venv
```

### 활성화 방법

```bash
# Windows (bash)
source .venv/Scripts/activate

# Windows (cmd)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 비활성화

```bash
deactivate
```

---

## 4. 설치 패키지

### 핵심 패키지

```bash
# PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

| 패키지 | 버전 | 용도 |
|--------|------|------|
| torch | 2.6.0+cu124 | 딥러닝 프레임워크 |
| torchvision | 0.21.0+cu124 | 이미지 데이터셋·변환 |
| torchaudio | 2.6.0+cu124 | (PyTorch 의존성) |

### 추가 설치 예정 (Phase 1에서 설치)

```bash
# Mammoth 프레임워크
git clone https://github.com/aimagelab/mammoth.git
cd mammoth
pip install -r requirements.txt
```

| 패키지 | 예상 버전 | 용도 |
|--------|----------|------|
| numpy | ~2.x | 수치 계산 |
| Pillow | ~10.x | 이미지 처리 |
| tqdm | ~4.x | 진행 표시 |
| wandb (선택) | ~0.x | 실험 추적 |
| matplotlib | ~3.x | 시각화 |

---

## 5. CUDA 검증 스크립트

```python
# .venv/Scripts/python 으로 실행
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_memory
    print(f"VRAM: {mem / 1024**3:.1f} GB")
```

**실행 결과 (2026-03-31):**

```
PyTorch: 2.6.0+cu124
CUDA available: True
CUDA version: 12.4
GPU count: 1
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
VRAM: 6.0 GB
```

---

## 6. 프로젝트 디렉토리 구조

```
continual_learning/
├── .venv/                          # Python 가상환경 (git 제외)
├── docs/
│   ├── pre_analysis.md             # 사전 분석 결과
│   ├── experiment_plan.md          # 실험 계획
│   ├── experiment_log.md           # 실험 진행 로그 (지속 업데이트)
│   └── environment.md              # 본 문서 (환경 설정)
├── raw_data/
│   └── split_mnist/                # 원본 실험 결과 JSON 저장
│       ├── sgd/
│       ├── joint/
│       ├── ewc_on/
│       ├── si/
│       ├── er/
│       ├── derpp/
│       ├── lwf/
│       └── hat/
├── results/                        # 집계된 결과 (CSV, 표)
├── configs/                        # 모델별 하이퍼파라미터 YAML
├── run_experiments.sh              # 일괄 실험 실행 스크립트
└── analysis.py                     # 결과 분석 스크립트
```

---

## 7. 재현성 메모

- 모든 실험은 고정된 5개 seed 사용: `[42, 123, 456, 789, 1024]`
- CUDA 연산의 결정론적 실행을 위해 실험 시 `torch.backends.cudnn.deterministic = True` 설정 권장
- 패키지 버전 고정을 위해 실험 완료 후 `pip freeze > requirements.txt` 실행 예정

---

## 8. 변경 이력

| 날짜 | 변경 내용 |
|------|-----------|
| 2026-03-31 | 초기 환경 셋업 완료: venv 생성, PyTorch 2.6.0+cu124 설치, CUDA 가속 확인 |
