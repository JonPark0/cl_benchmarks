"""
Split-MNIST Regularization Strength Sweep Runner
=================================================
Online EWC의 e_lambda, SI의 c 파라미터에 대한 로그스케일 민감도 분석.
기존 run_sweep.py(epoch/buffer)와 동일한 패턴을 따르되, 정규화 강도 자체를
변화시킨다. 기준 실험(e_lambda=0.7, c=0.5)은 raw_data/split_mnist/{model}에
이미 존재하므로 재사용하고, 그 값을 중심으로 10배씩 위아래로 2단계씩 이동한
4개 값만 추가로 실행한다.

사용법:
    python run_reg_sweep.py ewc --values 0.07 7 70 700
    python run_reg_sweep.py si --values 0.05 5 50 500
    python run_reg_sweep.py ewc --values 0.07 --dry-run
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
import sys as _sys
VENV_PYTHON = BASE_DIR / ".venv" / ("Scripts" if _sys.platform == "win32" else "bin") / ("python.exe" if _sys.platform == "win32" else "python")
RAW_DATA_DIR = BASE_DIR / "raw_data"

# ─── 기본 설정 ────────────────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1024]
N_EPOCHS = 1  # 기본 실험과 동일한 프로토콜(ep=1)에서 정규화 강도만 변화

# ewc: e_lambda 외 나머지는 기본 실험과 동일하게 고정
# si:  c 외 나머지는 기본 실험과 동일하게 고정
MODEL_CFG = {
    "ewc": {
        "model_name": "ewc-on",
        "sweep_arg": "--e_lambda",
        "fixed_args": {"--lr": "0.1", "--gamma": "1.0"},
        "baseline_value": 0.7,
        "sweep_dir": "sweep_reg_lambda",
        "value_prefix": "lambda_",
    },
    "si": {
        "model_name": "si",
        "sweep_arg": "--c",
        "fixed_args": {"--lr": "0.1", "--xi": "0.001"},
        "baseline_value": 0.5,
        "sweep_dir": "sweep_reg_c",
        "value_prefix": "c_",
    },
}


def fmt_value(v: float) -> str:
    """0.07 -> '0.07', 700.0 -> '700' 처럼 디렉토리 이름에 쓸 문자열 생성"""
    if v == int(v):
        return str(int(v))
    return str(v)


def build_command(cfg: dict, seed: int, value: float, results_subpath: str) -> list:
    cmd = [
        str(VENV_PYTHON),
        str(MAMMOTH_DIR / "main.py"),
        "--model", cfg["model_name"],
        "--dataset", "seq-mnist",
        "--seed", str(seed),
        "--device", "0",
        "--n_epochs", str(N_EPOCHS),
        "--base_path", str(RAW_DATA_DIR) + "/",
        "--results_path", results_subpath,
        "--enable_other_metrics", "1",
        "--non_verbose", "1",
    ]
    for key, val in cfg["fixed_args"].items():
        cmd.extend([key, val])
    cmd.extend([cfg["sweep_arg"], str(value)])
    return cmd


def run_single(cfg: dict, seed: int, value: float, log_file: Path, dry_run: bool = False) -> dict:
    value_str = fmt_value(value)
    results_subpath = f"{cfg['sweep_dir']}/{cfg['value_prefix']}{value_str}/{cfg['model_name']}"
    cmd = build_command(cfg, seed, value, results_subpath)

    tag = f"[{cfg['sweep_arg']}={value_str}] {cfg['model_name']} seed={seed}"
    print(f"  {tag}", end=" ... ", flush=True)

    if dry_run:
        print("DRY-RUN")
        print(f"    CMD: {' '.join(cmd)}")
        return {"model": cfg["model_name"], "seed": seed, "value": value, "status": "dry_run"}

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

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"{tag} | {status} | {elapsed:.1f}s\n")
        f.write(f"CMD: {' '.join(cmd)}\n")
        if result.returncode != 0:
            f.write("STDERR:\n")
            f.write(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)

    if status == "success":
        print(f"OK ({elapsed:.1f}s)")
    else:
        print(f"FAIL (rc={result.returncode})")
        stderr_lines = result.stderr.strip().split('\n') if result.stderr else []
        for line in stderr_lines[-3:]:
            if line.strip():
                print(f"    {line.strip()}")

    return {
        "model": cfg["model_name"], "seed": seed, "value": value, "status": status,
        "elapsed_sec": round(elapsed, 1), "returncode": result.returncode,
    }


def run_reg_sweep(target: str, values: list, dry_run: bool = False):
    cfg = MODEL_CFG[target]
    total = len(values) * len(SEEDS)

    print(f"\n{'='*70}")
    print(f"  정규화 강도 Sweep: {cfg['model_name']} ({cfg['sweep_arg']})")
    print(f"  값: {values} (기준값 {cfg['baseline_value']}은 기존 결과 재사용)")
    print(f"  Seeds: {SEEDS}")
    print(f"  Epochs: {N_EPOCHS} (고정)")
    print(f"  총 실험 수: {total}회")
    print(f"{'='*70}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    success, failed = 0, 0

    for value in values:
        value_str = fmt_value(value)
        print(f"\n--- {cfg['sweep_arg']} = {value_str} ---")
        log_dir = RAW_DATA_DIR / cfg["sweep_dir"] / f"{cfg['value_prefix']}{value_str}"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "run_output.log"

        for seed in SEEDS:
            r = run_single(cfg, seed, value, log_file, dry_run)
            all_results.append(r)
            if r["status"] == "success":
                success += 1
            elif r["status"] == "failed":
                failed += 1

    print(f"\n{'='*70}")
    print(f"  정규화 강도 Sweep 완료: 총 {total} | 성공 {success} | 실패 {failed}")
    print(f"{'='*70}")

    if not dry_run:
        summary_path = RAW_DATA_DIR / f"sweep_reg_{target}_summary_{timestamp}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "type": f"reg_sweep_{target}", "timestamp": timestamp,
                "sweep_arg": cfg["sweep_arg"], "values": values,
                "baseline_value": cfg["baseline_value"],
                "model": cfg["model_name"], "seeds": SEEDS, "n_epochs": N_EPOCHS,
                "total": total, "success": success, "failed": failed,
                "results": all_results,
            }, f, indent=2, ensure_ascii=False)
        print(f"  요약 저장: {summary_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split-MNIST 정규화 강도 Sweep Runner")
    parser.add_argument("target", choices=["ewc", "si"], help="ewc(e_lambda) 또는 si(c)")
    parser.add_argument("--values", nargs="+", type=float, required=True,
                        help="테스트할 값 목록 (예: 0.07 7 70 700)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_reg_sweep(args.target, args.values, dry_run=args.dry_run)
