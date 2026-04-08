"""
Split-MNIST Continual Learning Sweep Runner
=============================================
Epoch sweep 및 Buffer size sweep 실험을 자동 실행.

사용법:
    # Epoch sweep: 모든 모델에 대해 여러 epoch 값으로 실험
    python run_sweep.py epoch --values 5 10 20

    # Buffer sweep: Replay 계열 모델에 대해 여러 buffer 크기로 실험
    python run_sweep.py buffer --values 100 500 1000

    # 특정 모델만 지정
    python run_sweep.py epoch --values 5 10 --model ewc-on si

    # Dry-run (명령만 출력)
    python run_sweep.py epoch --values 5 --dry-run
"""

import subprocess
import sys
import os
import argparse
import time
import json
from pathlib import Path
from datetime import datetime


# ─── GPU 메모리 측정 ──────────────────────────────────────────────────────────

def query_gpu_memory() -> int | None:
    """nvidia-smi로 현재 GPU 메모리 사용량 (MB) 조회. 실�� 시 None."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split('\n')[0])
    except Exception:
        pass
    return None


# ─── 경로 설정 ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
MAMMOTH_DIR = BASE_DIR / "mammoth"
VENV_PYTHON = BASE_DIR / ".venv" / "Scripts" / "python.exe"
RAW_DATA_DIR = BASE_DIR / "raw_data"

# ─── 기본 설정 ────────────────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1024]
DEFAULT_EPOCHS = 1
DEFAULT_BUFFER = 200

# ─── 모델별 하이퍼파라미터 ────────────────────────────────────────────────────
MODELS = {
    "sgd": {
        "category": "Lower Bound",
        "uses_buffer": False,
        "args": {"--lr": "0.1"},
    },
    "joint": {
        "category": "Upper Bound",
        "uses_buffer": False,
        "args": {"--lr": "0.1"},
    },
    "ewc-on": {
        "category": "Regularization",
        "uses_buffer": False,
        "args": {"--lr": "0.1", "--e_lambda": "0.7", "--gamma": "1.0"},
    },
    "si": {
        "category": "Regularization",
        "uses_buffer": False,
        "args": {"--lr": "0.1", "--c": "0.5", "--xi": "0.001"},
    },
    "er": {
        "category": "Replay",
        "uses_buffer": True,
        "args": {"--lr": "0.1", "--minibatch_size": "64"},
    },
    "derpp": {
        "category": "Replay",
        "uses_buffer": True,
        "args": {"--lr": "0.1", "--alpha": "0.1", "--beta": "0.5"},
    },
    "lwf": {
        "category": "Knowledge Distillation",
        "uses_buffer": False,
        "args": {"--lr": "0.1", "--alpha": "1.0", "--softmax_temp": "2.0"},
    },
    "agem": {
        "category": "Optimization",
        "uses_buffer": True,
        "args": {"--lr": "0.1", "--minibatch_size": "64"},
    },
}


def build_command(model_name: str, seed: int, n_epochs: int, buffer_size: int,
                  sweep_type: str, sweep_value: int) -> list:
    """실험 명령어 생성"""
    model_cfg = MODELS[model_name]

    # sweep 결과를 구분된 디렉토리에 저장
    if sweep_type == "epoch":
        results_subpath = f"sweep_epochs/epoch_{sweep_value}/{model_name}"
    else:
        results_subpath = f"sweep_buffer/buffer_{sweep_value}/{model_name}"

    cmd = [
        str(VENV_PYTHON),
        str(MAMMOTH_DIR / "main.py"),
        "--model", model_name,
        "--dataset", "seq-mnist",
        "--seed", str(seed),
        "--device", "0",
        "--n_epochs", str(n_epochs),
        "--base_path", str(RAW_DATA_DIR) + "/",
        "--results_path", results_subpath,
        "--enable_other_metrics", "1",
        "--non_verbose", "1",
    ]

    # 모델별 추가 인수
    for key, val in model_cfg["args"].items():
        cmd.extend([key, val])

    # 버퍼 크기 (버퍼 사용 모델만)
    if model_cfg["uses_buffer"]:
        cmd.extend(["--buffer_size", str(buffer_size)])

    return cmd


def run_single(model_name: str, seed: int, n_epochs: int, buffer_size: int,
               sweep_type: str, sweep_value: int, log_file: Path,
               dry_run: bool = False) -> dict:
    """단일 실험 실행"""
    cmd = build_command(model_name, seed, n_epochs, buffer_size, sweep_type, sweep_value)

    tag = f"[{sweep_type}={sweep_value}] {model_name} seed={seed}"
    print(f"  {tag}", end=" ... ", flush=True)

    if dry_run:
        print("DRY-RUN")
        print(f"    CMD: {' '.join(cmd)}")
        return {"model": model_name, "seed": seed, "status": "dry_run",
                "sweep_type": sweep_type, "sweep_value": sweep_value}

    # GPU 메모리 측정 (실험 전)
    gpu_mem_before = query_gpu_memory()

    start_time = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        cmd, cwd=str(MAMMOTH_DIR),
        capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )
    elapsed = time.time() - start_time
    status = "success" if result.returncode == 0 else "failed"

    # GPU 메모리 측정 (실험 후)
    gpu_mem_after = query_gpu_memory()
    gpu_mem_peak = max(gpu_mem_before or 0, gpu_mem_after or 0)

    # 로그 저장
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"{tag} | {status} | {elapsed:.1f}s | GPU: {gpu_mem_peak}MB\n")
        f.write(f"CMD: {' '.join(cmd)}\n")
        if result.returncode != 0:
            f.write("STDERR:\n")
            f.write(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)

    if status == "success":
        print(f"OK ({elapsed:.1f}s, GPU:{gpu_mem_peak}MB)")
    else:
        print(f"FAIL (rc={result.returncode})")
        stderr_lines = result.stderr.strip().split('\n') if result.stderr else []
        for line in stderr_lines[-3:]:
            if line.strip():
                print(f"    {line.strip()}")

    return {
        "model": model_name, "seed": seed, "status": status,
        "elapsed_sec": round(elapsed, 1), "returncode": result.returncode,
        "sweep_type": sweep_type, "sweep_value": sweep_value,
        "gpu_mem_mb": gpu_mem_peak,
    }


def run_epoch_sweep(epoch_values: list, target_models: list = None, dry_run: bool = False):
    """Epoch sweep 실행: 지정된 epoch 값들에 대해 모든(또는 지정) 모델 실행"""
    models_to_run = target_models if target_models else list(MODELS.keys())
    total = len(models_to_run) * len(SEEDS) * len(epoch_values)

    print(f"\n{'='*70}")
    print(f"  Epoch Sweep 실험")
    print(f"  Epoch 값: {epoch_values}")
    print(f"  모델: {models_to_run}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Buffer: {DEFAULT_BUFFER} (고정)")
    print(f"  총 실험 수: {total}회")
    print(f"{'='*70}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    success, failed = 0, 0

    for n_epochs in epoch_values:
        print(f"\n--- Epoch = {n_epochs} ---")
        log_dir = RAW_DATA_DIR / "sweep_epochs" / f"epoch_{n_epochs}"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "run_output.log"

        for model_name in models_to_run:
            if model_name not in MODELS:
                print(f"  [WARNING] 알 수 없는 모델: {model_name}")
                continue
            for seed in SEEDS:
                r = run_single(model_name, seed, n_epochs, DEFAULT_BUFFER,
                               "epoch", n_epochs, log_file, dry_run)
                all_results.append(r)
                if r["status"] == "success":
                    success += 1
                elif r["status"] == "failed":
                    failed += 1

    # 요약
    print(f"\n{'='*70}")
    print(f"  Epoch Sweep 완료: 총 {total} | 성공 {success} | 실패 {failed}")
    print(f"{'='*70}")

    if not dry_run:
        summary_path = RAW_DATA_DIR / f"sweep_epoch_summary_{timestamp}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "type": "epoch_sweep", "timestamp": timestamp,
                "epoch_values": epoch_values, "models": models_to_run,
                "seeds": SEEDS, "buffer_size": DEFAULT_BUFFER,
                "total": total, "success": success, "failed": failed,
                "results": all_results,
            }, f, indent=2, ensure_ascii=False)
        print(f"  요약 저장: {summary_path}")

    return all_results


def run_buffer_sweep(buffer_values: list, target_models: list = None,
                     n_epochs: int = DEFAULT_EPOCHS, dry_run: bool = False):
    """Buffer sweep 실행: 버퍼 사용 모델에 대해 다양한 버퍼 크기로 실행"""
    # 버퍼 사용 모델만 기본 선택
    buffer_models = [m for m in MODELS if MODELS[m]["uses_buffer"]]
    if target_models:
        models_to_run = [m for m in target_models if m in buffer_models]
        skipped = [m for m in target_models if m not in buffer_models]
        if skipped:
            print(f"  [INFO] 버퍼 미사용 모델 제외: {skipped}")
    else:
        models_to_run = buffer_models

    total = len(models_to_run) * len(SEEDS) * len(buffer_values)

    print(f"\n{'='*70}")
    print(f"  Buffer Size Sweep 실험")
    print(f"  Buffer 값: {buffer_values}")
    print(f"  모델: {models_to_run}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Epochs: {n_epochs} (고정)")
    print(f"  총 실험 수: {total}회")
    print(f"{'='*70}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    success, failed = 0, 0

    for buf_size in buffer_values:
        print(f"\n--- Buffer = {buf_size} ---")
        log_dir = RAW_DATA_DIR / "sweep_buffer" / f"buffer_{buf_size}"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "run_output.log"

        for model_name in models_to_run:
            for seed in SEEDS:
                r = run_single(model_name, seed, n_epochs, buf_size,
                               "buffer", buf_size, log_file, dry_run)
                all_results.append(r)
                if r["status"] == "success":
                    success += 1
                elif r["status"] == "failed":
                    failed += 1

    print(f"\n{'='*70}")
    print(f"  Buffer Sweep 완료: 총 {total} | 성공 {success} | 실패 {failed}")
    print(f"{'='*70}")

    if not dry_run:
        summary_path = RAW_DATA_DIR / f"sweep_buffer_summary_{timestamp}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "type": "buffer_sweep", "timestamp": timestamp,
                "buffer_values": buffer_values, "models": models_to_run,
                "seeds": SEEDS, "n_epochs": n_epochs,
                "total": total, "success": success, "failed": failed,
                "results": all_results,
            }, f, indent=2, ensure_ascii=False)
        print(f"  요약 저장: {summary_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split-MNIST Sweep Runner")
    subparsers = parser.add_subparsers(dest="sweep_type", required=True)

    # Epoch sweep
    ep = subparsers.add_parser("epoch", help="Epoch sweep 실행")
    ep.add_argument("--values", nargs="+", type=int, required=True,
                    help="테스트할 epoch 값 (예: 5 10 20)")
    ep.add_argument("--model", nargs="+", help="실행할 모델 (기본: 전체)")
    ep.add_argument("--dry-run", action="store_true")

    # Buffer sweep
    buf = subparsers.add_parser("buffer", help="Buffer size sweep 실행")
    buf.add_argument("--values", nargs="+", type=int, required=True,
                     help="테스트할 buffer 크기 (예: 100 500 1000)")
    buf.add_argument("--model", nargs="+", help="실행할 모델 (기본: 버퍼 사용 모델)")
    buf.add_argument("--n_epochs", type=int, default=DEFAULT_EPOCHS,
                     help=f"고정 epoch 수 (기본: {DEFAULT_EPOCHS})")
    buf.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.sweep_type == "epoch":
        run_epoch_sweep(args.values, target_models=args.model, dry_run=args.dry_run)
    elif args.sweep_type == "buffer":
        run_buffer_sweep(args.values, target_models=args.model,
                         n_epochs=args.n_epochs, dry_run=args.dry_run)
