"""
Split-MNIST Continual Learning Experiment Runner
=================================================
8개 모델 × 5 seed = 40회 실험 자동 실행
결과는 raw_data/split_mnist/{model}/class-il/seq-mnist/{model}/logs.pyd 에 저장

사용법:
    python run_experiments.py                  # 전체 실험 실행
    python run_experiments.py --model sgd      # 특정 모델만 실행
    python run_experiments.py --dry-run        # 실행 명령만 출력 (실제 실행 X)
"""

import subprocess
import sys
import os
import argparse
import time
import json
from pathlib import Path
from datetime import datetime

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
MAMMOTH_DIR = BASE_DIR / "mammoth"
VENV_PYTHON = BASE_DIR / ".venv" / "Scripts" / "python.exe"
RAW_DATA_DIR = BASE_DIR / "raw_data"
DOCS_DIR = BASE_DIR / "docs"

# ─── 실험 설정 ────────────────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1024]
N_EPOCHS = 1  # Split-MNIST 표준 설정
BUFFER_SIZE = 200  # Replay 기반 모델 메모리 버퍼 크기

# ─── 모델별 하이퍼파라미터 ────────────────────────────────────────────────────
# 기준: Mammoth REPRODUCIBILITY.md + CL 논문 표준값 (Split-MNIST)
MODELS = {
    "sgd": {
        "category": "Lower Bound",
        "args": {
            "--lr": "0.1",
        },
        "description": "Naive Fine-tuning (하한선)"
    },
    "joint": {
        "category": "Upper Bound",
        "args": {
            "--lr": "0.1",
        },
        "description": "Joint Training (상한선)"
    },
    "ewc-on": {
        "category": "Regularization",
        "args": {
            "--lr": "0.1",
            "--e_lambda": "0.7",
            "--gamma": "1.0",
        },
        "description": "EWC Online (Kirkpatrick et al., 2017)"
    },
    "si": {
        "category": "Regularization",
        "args": {
            "--lr": "0.1",
            "--c": "0.5",
            "--xi": "0.001",
        },
        "description": "Synaptic Intelligence (Zenke et al., 2017)"
    },
    "er": {
        "category": "Replay",
        "args": {
            "--lr": "0.1",
            "--buffer_size": str(BUFFER_SIZE),
            "--minibatch_size": "64",
        },
        "description": "Experience Replay (Chaudhry et al., 2019)"
    },
    "derpp": {
        "category": "Replay",
        "args": {
            "--lr": "0.1",
            "--buffer_size": str(BUFFER_SIZE),
            "--alpha": "0.1",
            "--beta": "0.5",
        },
        "description": "DER++ (Buzzega et al., NeurIPS 2020)"
    },
    "lwf": {
        "category": "Knowledge Distillation",
        "args": {
            "--lr": "0.1",
            "--alpha": "1.0",
            "--softmax_temp": "2.0",
        },
        "description": "Learning without Forgetting (Li & Hoiem, TPAMI 2018)"
    },
    "agem": {
        "category": "Optimization",
        "args": {
            "--lr": "0.1",
            "--buffer_size": str(BUFFER_SIZE),
            "--minibatch_size": "64",
        },
        "description": "A-GEM (Chaudhry et al., ICLR 2019) - GEM의 평균화 버전, Windows 호환"
    },
}


def build_command(model_name: str, seed: int, dry_run: bool = False) -> list:
    """주어진 모델과 seed에 대한 Mammoth 실행 명령어를 생성"""
    model_cfg = MODELS[model_name]
    results_subpath = f"split_mnist/{model_name}"

    cmd = [
        str(VENV_PYTHON),
        str(MAMMOTH_DIR / "main.py"),
        "--model", model_name,
        "--dataset", "seq-mnist",
        "--seed", str(seed),
        "--device", "0",
        "--n_epochs", str(N_EPOCHS),
        "--base_path", str(RAW_DATA_DIR) + "/",
        "--results_path", results_subpath,
        "--enable_other_metrics", "1",  # BWT, FWT, Forgetting 활성화
        "--non_verbose", "1",
    ]

    # 모델별 추가 인수
    for key, val in model_cfg["args"].items():
        cmd.extend([key, val])

    return cmd


def run_experiment(model_name: str, seed: int, log_file: Path, dry_run: bool = False) -> dict:
    """단일 실험 실행 및 결과 반환"""
    cmd = build_command(model_name, seed, dry_run)

    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 모델: {model_name} | Seed: {seed}")
    print(f"  카테고리: {MODELS[model_name]['category']}")
    if dry_run:
        print(f"  [DRY-RUN] 명령어: {' '.join(cmd)}")
        return {"model": model_name, "seed": seed, "status": "dry_run"}

    start_time = time.time()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        cmd,
        cwd=str(MAMMOTH_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    elapsed = time.time() - start_time

    # 로그 저장
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Model: {model_name} | Seed: {seed} | Time: {elapsed:.1f}s\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write("STDOUT:\n")
        f.write(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
        if result.returncode != 0:
            f.write("\nSTDERR:\n")
            f.write(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)

    status = "success" if result.returncode == 0 else "failed"
    print(f"  상태: {status} | 소요시간: {elapsed:.1f}s")
    if result.returncode != 0:
        print(f"  [ERROR] returncode={result.returncode}")
        # 핵심 오류만 출력
        stderr_lines = result.stderr.strip().split('\n') if result.stderr else []
        for line in stderr_lines[-5:]:
            if line.strip():
                print(f"    {line}")

    return {
        "model": model_name,
        "seed": seed,
        "status": status,
        "elapsed_sec": round(elapsed, 1),
        "returncode": result.returncode,
    }


def run_all(target_models: list = None, dry_run: bool = False):
    """전체 실험 실행"""
    models_to_run = target_models if target_models else list(MODELS.keys())
    total = len(models_to_run) * len(SEEDS)

    print(f"\n{'='*60}")
    print(f"Split-MNIST Continual Learning 실험")
    print(f"  실험 모델: {models_to_run}")
    print(f"  Seeds: {SEEDS}")
    print(f"  총 실험 수: {total}회")
    print(f"  결과 저장: {RAW_DATA_DIR}")
    print(f"{'='*60}")

    # 런 로그 파일
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_path = RAW_DATA_DIR / f"run_log_{timestamp}.txt"

    results_summary = []
    completed = 0
    failed = 0

    for model_name in models_to_run:
        if model_name not in MODELS:
            print(f"[WARNING] 알 수 없는 모델: {model_name}")
            continue

        model_log = RAW_DATA_DIR / f"split_mnist" / model_name / "run_output.log"
        model_log.parent.mkdir(parents=True, exist_ok=True)

        for seed in SEEDS:
            result = run_experiment(model_name, seed, model_log, dry_run=dry_run)
            results_summary.append(result)

            if result["status"] == "success":
                completed += 1
            elif result["status"] == "failed":
                failed += 1

    # 요약 출력
    print(f"\n{'='*60}")
    print(f"실험 완료 요약")
    print(f"  총: {total}회 | 성공: {completed} | 실패: {failed}")
    print(f"{'='*60}")

    # 요약 JSON 저장
    summary_path = RAW_DATA_DIR / f"experiment_summary_{timestamp}.json"
    if not dry_run:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": timestamp,
                "models": models_to_run,
                "seeds": SEEDS,
                "total": total,
                "completed": completed,
                "failed": failed,
                "results": results_summary,
            }, f, indent=2, ensure_ascii=False)
        print(f"  요약 저장: {summary_path}")

    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split-MNIST CL 실험 실행기")
    parser.add_argument("--model", nargs="+", help="실행할 모델 이름 (기본: 전체)")
    parser.add_argument("--dry-run", action="store_true", help="명령어만 출력, 실제 실행 안 함")
    args = parser.parse_args()

    run_all(target_models=args.model, dry_run=args.dry_run)
