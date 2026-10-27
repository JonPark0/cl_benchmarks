# cl_benchmarks

학사학위논문 「Split-MNIST 클래스 증분 학습에서 접근 방식별 지속적 학습 알고리즘의 비교 분석」(인천대학교 메카트로닉스공학과)의 실험 코드와 원시 기록이다. [Mammoth](https://github.com/aimagelab/mammoth) 0.2.2 위에서 지속적 학습 알고리즘 8개를 같은 조건으로 학습하고, 클래스 증분 학습(Class-IL)에서의 평균 정확도와 망각을 비교한다.

## 실험 조건

| 항목 | 값 |
|---|---|
| 벤치마크 | Split-MNIST(Mammoth `seq-mnist`), 숫자 2개씩 5개 작업, Class-IL |
| 백본 | MLP(은닉층 100-100, 출력 노드 10개 고정) |
| 옵티마이저 | Adam, 학습률 0.001 |
| 배치 / 에폭 | 64 / 작업당 1 |
| 메모리 버퍼 | 200(ER, DER++, GEM), 버퍼 미니 배치 64 |
| 시드 | 기본 실험 20개(42, 123, 456, 789, 1024, 2001~2015), 민감도 실험은 앞의 10개 |
| 환경 | WSL2 Ubuntu 24.04, Python 3.12.3, PyTorch 2.6.0+cu124, RTX 3060 Laptop GPU |

모델별 인자는 Mammoth 기본값이 있으면 그 값을 쓰고, 기본값이 없는 필수 인자는 Buzzega et al.(2020) 표 9의 Seq-MNIST 값을 쓴다.

| 모델 | 분류 | 인자 |
|---|---|---|
| SGD | 하한선 | 없음 |
| Joint | 상한선 | 없음 |
| Online EWC | 정규화 | `e_lambda=90`, `gamma=1.0` |
| SI | 정규화 | `c=1.0`, `xi=0.9` |
| ER | 리플레이 | 없음 |
| DER++ | 리플레이 | `alpha=0.2`, `beta=1.0` |
| LwF | 지식 증류 | Mammoth 기본값(`alpha=0.5`, `softmax_temp=2`) |
| GEM | 최적화 | Mammoth 기본값(`gamma=0.5`) |

## 주요 결과

기본 실험(20개 시드)의 Class-IL 평균 정확도(AA)와 역방향 전이(BWT)이다. 값은 평균 ± 표준편차(%)이다.

| 모델 | AA | BWT |
|---|---|---|
| Joint | 95.61 ± 0.34 | — |
| DER++ | 86.49 ± 1.36 | −15.05 ± 1.76 |
| GEM | 82.79 ± 0.73 | −19.67 ± 0.97 |
| ER | 78.45 ± 1.50 | −25.29 ± 1.86 |
| LwF | 19.67 ± 0.10 | −99.39 ± 0.10 |
| SI | 19.63 ± 0.05 | −99.26 ± 0.07 |
| Online EWC | 19.61 ± 0.06 | −99.25 ± 0.09 |
| SGD | 19.58 ± 0.03 | −99.28 ± 0.07 |

메모리 버퍼를 쓰는 세 모델만 하한선을 크게 넘었고, 정규화(Online EWC, SI)와 지식 증류(LwF)는 하한선 수준에 머물렀다. 논문은 이 차이의 원인을 출력 노드별 Fisher 측정, 페널티 강도 탐색, LwF 수치 안정성 분석으로 살펴본다.

## 설치

```bash
git clone --recurse-submodules https://github.com/JonPark0/cl_benchmarks.git
cd cl_benchmarks
python3.12 -m venv .venv
.venv/bin/pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -r mammoth/requirements.txt scipy matplotlib
```

`mammoth`는 `aimagelab/mammoth`의 커밋 `ea967c3`(2025-11-22)을 가리키는 서브모듈이다. 실행기는 `.venv/bin/python`을 직접 호출하므로 가상환경은 저장소 루트의 `.venv`에 만들어야 한다.

## 실행

`run_linux.py`가 모든 실험을 정의한다. 결과는 `raw_data/linux/<실험>/<조건>/s<시드>/`에 저장되고, 이미 `logs.pyd`가 있는 실행은 건너뛴다.

```bash
.venv/bin/python run_linux.py base --seeds 42 --workers 1   # 시운전
.venv/bin/python run_linux.py base --workers 4              # 기본 실험: 8개 모델 x 20개 시드
```

| 실험 | 내용 | 논문 |
|---|---|---|
| `base` | 8개 모델 기본 비교 | 4.1절 |
| `epoch` | 작업당 에폭 5, 10, 20 | 4.2절 |
| `buffer` | ER, DER++, GEM의 버퍼 100, 500, 1000 | 4.3절 |
| `lwf`, `lwf_stable` | LwF의 `alpha` x 온도 x 학습률 격자와 van de Ven(2022) 프로토콜, 원본 계산과 로그 소프트맥스 계산 | 4.4절 |
| `lwfep`, `lwfep_stable` | 에폭 탐색에서 발산한 LwF 조건(10, 20 에폭) 재실행 | 4.4절 |
| `penalty` | Online EWC `e_lambda`, SI `c`를 기준값의 0.01~1000배로 탐색 | 5.2절 |
| `fisher` | Online EWC의 출력 노드별 Fisher 기록(5개 시드) | 5.2절 |

`run_all_linux.sh`는 위 실험을 차례로 실행한다. `--models`, `--tags`로 일부 조건만 실행할 수 있다.

### 모델 패치

서브모듈에는 직접 커밋할 수 없으므로 추가 모델은 `models_patch/`에 두고, `run_linux.py`가 실행 전에 `mammoth/models/`로 복사한다.

- `ewc_on_fisher.py`: Online EWC를 상속해 작업마다 분류 층의 Fisher를 출력 노드별로 저장한다. 학습 로직은 그대로다.
- `lwf_probe.py`: LwF를 상속해 매 갱신 단계의 이전 클래스 최소 확률, 로짓 간격, 경사의 유한성을 기록한다. `--lwf_stable 0`이면 원본과 같은 계산을, `1`이면 증류 항을 로그 소프트맥스로 계산한다.
- `agem_fixed.py`: 교정 전 실험에서 쓴 A-GEM 수정본이다. 최종 실험에서는 A-GEM을 GEM으로 바꿔 쓰지 않는다.

## 집계와 그림

```bash
.venv/bin/python analysis_linux.py              # 실험별 조건별 AA, BWT, Task-IL AA 요약
.venv/bin/python report_linux.py base           # 표 4.1 행, 인접 모델 대응 표본 t 검정(Holm 보정), 그림 4.1·4.2
.venv/bin/python report_linux.py epoch buffer penalty fisher lwf lwf_stable
```

`report_linux.py`는 인자로 준 실험마다 표와 검정 결과를 출력하고 해당 그림(`*_linux.png`)을 저장소 밖의 `../figures/`에 쓴다. 논문 원고 폴더 안에서 실행하는 것을 전제로 한 경로다.

## 디렉터리

| 경로 | 내용 |
|---|---|
| `run_linux.py`, `run_*_linux.sh`, `run_lwfep.sh` | 최종 실험 실행기 |
| `analysis_linux.py`, `report_linux.py` | 최종 실험 집계와 표, 검정, 그림 |
| `models_patch/` | Mammoth 추가 모델 |
| `raw_data/linux/` | 최종 실험 원시 기록(`logs.pyd`, `run_output.log`, `probe.json`, Fisher JSON) |
| `raw_data/linux/environment.json` | 실행 환경 기록 |
| 그 밖의 `run_*.py`, `analysis*.py`, `raw_data/`, `results/` | 교정 전 실험(Windows, WSL2, SGD 옵티마이저) 기록 |
| `docs/` | 실험 기록 문서. [docs/README.md](docs/README.md)에서 문서별 상태를 정리한다 |

## 인용

이 저장소의 결과를 인용할 때는 위 논문을 인용한다. Mammoth는 Buzzega et al., "Dark Experience for General Continual Learning: a Strong, Simple Baseline", NeurIPS 2020과 Boschini et al., "Class-Incremental Continual Learning into the Extended DER-verse", TPAMI 2022를 인용한다.

## 라이선스

이 저장소의 코드와 실험 기록은 [MIT 라이선스](LICENSE)를 따른다. `mammoth` 서브모듈과 `models_patch/`가 상속하는 Mammoth 코드는 Mammoth 저작자의 MIT 라이선스(`mammoth/LICENSE`)를 따른다. MNIST 데이터는 저장소에 포함하지 않으며, 처음 실행할 때 Mammoth가 내려받는다.
