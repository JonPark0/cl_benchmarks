"""
Split-MNIST Sweep 결과 분석 스크립트
=====================================
sweep_epochs/ 및 sweep_buffer/ 내 실험 결과를 파싱하여
파라미터 변화에 따른 성능 추이를 분석하고 CSV로 저장.

사용법:
    python analysis_sweep.py epoch                    # Epoch sweep 분석
    python analysis_sweep.py buffer                   # Buffer sweep 분석
    python analysis_sweep.py epoch --include-baseline  # 기존 epoch=1 결과도 포함
    python analysis_sweep.py all                      # 전체 분석
"""

import ast
import re
import json
import argparse
import csv
import numpy as np
from pathlib import Path
from datetime import datetime

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
RAW_DATA_DIR = BASE_DIR / "raw_data"
RESULTS_DIR = BASE_DIR / "results"

MODELS = ["sgd", "joint", "ewc-on", "si", "er", "derpp", "lwf", "agem"]
BUFFER_MODELS = ["er", "derpp", "agem"]
CATEGORIES = {
    "sgd": "Lower Bound", "joint": "Upper Bound",
    "ewc-on": "Regularization", "si": "Regularization",
    "er": "Replay", "derpp": "Replay",
    "lwf": "Knowledge Distillation", "agem": "Optimization",
}
N_TASKS = 5


def parse_logs_pyd(filepath: Path) -> list[dict]:
    """logs.pyd 파일 파싱 (analysis.py와 동일 로직)"""
    records = []
    if not filepath.exists():
        return records

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    content = re.sub(r"np\.float64\(([^)]+)\)", r"\1", content)
    content = re.sub(r"np\.float32\(([^)]+)\)", r"\1", content)
    content = re.sub(r"np\.int64\(([^)]+)\)", r"\1", content)
    content = re.sub(r"device\(type='cuda',\s*index=\d+\)", "'cuda:0'", content)
    content = re.sub(r"device\(type='cpu'\)", "'cpu'", content)
    content = re.sub(r"device\('cuda:\d+'\)", "'cuda:0'", content)

    seen_seeds = {}
    for line_num, line in enumerate(content.strip().split('\n'), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = ast.literal_eval(line)
            if not record.get("enable_other_metrics", True):
                continue
            seed = record.get("seed", line_num)
            seen_seeds[seed] = record
        except Exception as e:
            pass  # 파싱 실패는 조용히 무시

    return list(seen_seeds.values())


def extract_metrics(record: dict) -> dict:
    """핵심 지표 추출"""
    metrics = {"seed": record.get("seed"), "aa": None, "bwt": None,
               "fwt": None, "forgetting": None}

    for t in range(N_TASKS, 0, -1):
        key = f"accmean_task{t}"
        if key in record and record[key] is not None:
            metrics["aa"] = record[key]
            break

    metrics["bwt"] = record.get("backward_transfer")
    metrics["fwt"] = record.get("forward_transfer")
    metrics["forgetting"] = record.get("forgetting")
    return metrics


def compute_stats(values: list) -> dict:
    """mean ± std 계산"""
    valid = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not valid:
        return {"mean": None, "std": None, "n": 0}
    arr = np.array(valid, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "n": len(arr),
    }


def analyze_single(model_name: str, base_path: Path) -> dict | None:
    """특정 경로의 단일 모델 결과 분석"""
    model_dir_name = model_name.replace("-", "_")
    logs_path = base_path / model_name / "class-il" / "seq-mnist" / model_dir_name / "logs.pyd"

    records = parse_logs_pyd(logs_path)
    if not records:
        return None

    aa_list, bwt_list, fwt_list, fm_list = [], [], [], []
    for rec in records:
        m = extract_metrics(rec)
        if m["aa"] is not None: aa_list.append(m["aa"])
        if m["bwt"] is not None: bwt_list.append(m["bwt"])
        if m["fwt"] is not None: fwt_list.append(m["fwt"])
        if m["forgetting"] is not None: fm_list.append(m["forgetting"])

    return {
        "model": model_name,
        "category": CATEGORIES.get(model_name, "Unknown"),
        "n_runs": len(records),
        "aa": compute_stats(aa_list),
        "bwt": compute_stats(bwt_list),
        "fwt": compute_stats(fwt_list),
        "forgetting": compute_stats(fm_list),
    }


def fmt(stat, decimals=2):
    """mean ± std 포매팅"""
    if stat is None or stat["mean"] is None:
        return "N/A"
    return f"{stat['mean']:.{decimals}f} ± {stat['std']:.{decimals}f}"


# ─── Epoch Sweep 분석 ─────────────────────────────────────────────────────────

def analyze_epoch_sweep(include_baseline: bool = False):
    """Epoch sweep 결과 분석"""
    sweep_dir = RAW_DATA_DIR / "sweep_epochs"
    if not sweep_dir.exists():
        print("[WARNING] sweep_epochs/ 디렉토리가 없습니다. 먼저 실험을 실행하세요.")
        return []

    # epoch 값 수집
    epoch_dirs = sorted(sweep_dir.glob("epoch_*"), key=lambda p: int(p.name.split("_")[1]))
    epoch_values = [int(d.name.split("_")[1]) for d in epoch_dirs]

    if not epoch_values:
        print("[WARNING] sweep_epochs/ 에 결과가 없습니다.")
        return []

    # 기존 epoch=1 결과 포함 옵션
    if include_baseline:
        epoch_values = [1] + [v for v in epoch_values if v != 1]

    print(f"\nEpoch Sweep 분석")
    print(f"  Epoch 값: {epoch_values}")
    print(f"  모델: {MODELS}")

    all_rows = []

    for epoch_val in epoch_values:
        if epoch_val == 1 and include_baseline:
            base_path = RAW_DATA_DIR / "split_mnist"
        else:
            base_path = sweep_dir / f"epoch_{epoch_val}"

        print(f"\n--- Epoch = {epoch_val} ---")
        for model_name in MODELS:
            if epoch_val == 1 and include_baseline:
                # 기존 결과는 다른 경로 구조
                result = analyze_single_baseline(model_name)
            else:
                result = analyze_single(model_name, base_path)

            if result:
                result["epoch"] = epoch_val
                all_rows.append(result)
                print(f"  {model_name:<10} AA={fmt(result['aa'])}  BWT={fmt(result['bwt'])}")
            else:
                print(f"  {model_name:<10} 결과 없음")

    # CSV 저장
    if all_rows:
        save_epoch_sweep_csv(all_rows)
        save_epoch_sweep_json(all_rows)
        print_epoch_comparison_table(all_rows, epoch_values)

    return all_rows


def analyze_single_baseline(model_name: str) -> dict | None:
    """기존 epoch=1 결과 분석 (기존 디렉토리 구조)"""
    model_dir_name = model_name.replace("-", "_")
    logs_path = (RAW_DATA_DIR / "split_mnist" / model_name / "class-il"
                 / "seq-mnist" / model_dir_name / "logs.pyd")

    records = parse_logs_pyd(logs_path)
    if not records:
        return None

    aa_list, bwt_list, fwt_list, fm_list = [], [], [], []
    for rec in records:
        m = extract_metrics(rec)
        if m["aa"] is not None: aa_list.append(m["aa"])
        if m["bwt"] is not None: bwt_list.append(m["bwt"])
        if m["fwt"] is not None: fwt_list.append(m["fwt"])
        if m["forgetting"] is not None: fm_list.append(m["forgetting"])

    return {
        "model": model_name,
        "category": CATEGORIES.get(model_name, "Unknown"),
        "n_runs": len(records),
        "aa": compute_stats(aa_list),
        "bwt": compute_stats(bwt_list),
        "fwt": compute_stats(fwt_list),
        "forgetting": compute_stats(fm_list),
    }


def save_epoch_sweep_csv(rows: list):
    """Epoch sweep 결과 CSV 저장"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for path in [RESULTS_DIR / f"epoch_sweep_{timestamp}.csv",
                 RESULTS_DIR / "epoch_sweep_latest.csv"]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "model", "category", "n_runs",
                "aa_mean", "aa_std", "bwt_mean", "bwt_std",
                "fwt_mean", "fwt_std", "forgetting_mean", "forgetting_std",
            ])
            for r in rows:
                def g(stat, key):
                    return round(stat[key], 4) if stat and stat[key] is not None else ""
                writer.writerow([
                    r["epoch"], r["model"], r["category"], r["n_runs"],
                    g(r["aa"], "mean"), g(r["aa"], "std"),
                    g(r["bwt"], "mean"), g(r["bwt"], "std"),
                    g(r["fwt"], "mean"), g(r["fwt"], "std"),
                    g(r["forgetting"], "mean"), g(r["forgetting"], "std"),
                ])

    print(f"\n  CSV 저장: {RESULTS_DIR / 'epoch_sweep_latest.csv'}")


def save_epoch_sweep_json(rows: list):
    """Epoch sweep 결과 JSON 저장"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = []
    for r in rows:
        data.append({
            "epoch": r["epoch"], "model": r["model"], "category": r["category"],
            "n_runs": r["n_runs"],
            "aa": r["aa"], "bwt": r["bwt"], "fwt": r["fwt"],
            "forgetting": r["forgetting"],
        })

    for path in [RESULTS_DIR / f"epoch_sweep_{timestamp}.json",
                 RESULTS_DIR / "epoch_sweep_latest.json"]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  JSON 저장: {RESULTS_DIR / 'epoch_sweep_latest.json'}")


def print_epoch_comparison_table(rows: list, epoch_values: list):
    """Epoch별 AA 비교 테이블 출력"""
    print(f"\n{'='*90}")
    print(f"Epoch Sweep: AA (Average Accuracy) 비교")
    print(f"{'='*90}")

    # 헤더
    header = f"{'모델':<12} {'카테고리':<22}"
    for ep in epoch_values:
        header += f" {'ep=' + str(ep):>14}"
    print(header)
    print("-" * 90)

    # 모델별로 행 출력
    for model_name in MODELS:
        row = f"{model_name:<12} {CATEGORIES.get(model_name, ''):<22}"
        for ep in epoch_values:
            match = [r for r in rows if r["model"] == model_name and r["epoch"] == ep]
            if match and match[0]["aa"]["mean"] is not None:
                aa = match[0]["aa"]
                row += f" {aa['mean']:>6.2f}±{aa['std']:<5.2f}"
            else:
                row += f" {'N/A':>14}"
        print(row)

    print(f"{'='*90}")


# ─── Buffer Sweep 분석 ─────────────────────────────────────────────────────────

def analyze_buffer_sweep(include_baseline: bool = False):
    """Buffer sweep 결과 분석"""
    sweep_dir = RAW_DATA_DIR / "sweep_buffer"
    if not sweep_dir.exists():
        print("[WARNING] sweep_buffer/ 디렉토리가 없습니다. 먼저 실험을 실행하세요.")
        return []

    buffer_dirs = sorted(sweep_dir.glob("buffer_*"), key=lambda p: int(p.name.split("_")[1]))
    buffer_values = [int(d.name.split("_")[1]) for d in buffer_dirs]

    if not buffer_values:
        print("[WARNING] sweep_buffer/ 에 결과가 없습니다.")
        return []

    # 기존 buffer=200 결과 포함
    if include_baseline and 200 not in buffer_values:
        buffer_values = sorted(buffer_values + [200])

    print(f"\nBuffer Sweep 분석")
    print(f"  Buffer 값: {buffer_values}")
    print(f"  모델: {BUFFER_MODELS}")

    all_rows = []

    for buf_val in buffer_values:
        if buf_val == 200 and include_baseline and not (sweep_dir / f"buffer_{buf_val}").exists():
            base_path = RAW_DATA_DIR / "split_mnist"
            print(f"\n--- Buffer = {buf_val} (기존 baseline 결과) ---")
        else:
            base_path = sweep_dir / f"buffer_{buf_val}"
            print(f"\n--- Buffer = {buf_val} ---")

        for model_name in BUFFER_MODELS:
            if buf_val == 200 and include_baseline and not (sweep_dir / f"buffer_{buf_val}").exists():
                result = analyze_single_baseline(model_name)
            else:
                result = analyze_single(model_name, base_path)

            if result:
                result["buffer_size"] = buf_val
                all_rows.append(result)
                print(f"  {model_name:<10} AA={fmt(result['aa'])}  BWT={fmt(result['bwt'])}")
            else:
                print(f"  {model_name:<10} 결과 없음")

    if all_rows:
        save_buffer_sweep_csv(all_rows)
        save_buffer_sweep_json(all_rows)
        print_buffer_comparison_table(all_rows, buffer_values)

    return all_rows


def save_buffer_sweep_csv(rows: list):
    """Buffer sweep 결과 CSV 저장"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for path in [RESULTS_DIR / f"buffer_sweep_{timestamp}.csv",
                 RESULTS_DIR / "buffer_sweep_latest.csv"]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "buffer_size", "model", "category", "n_runs",
                "aa_mean", "aa_std", "bwt_mean", "bwt_std",
                "fwt_mean", "fwt_std", "forgetting_mean", "forgetting_std",
            ])
            for r in rows:
                def g(stat, key):
                    return round(stat[key], 4) if stat and stat[key] is not None else ""
                writer.writerow([
                    r["buffer_size"], r["model"], r["category"], r["n_runs"],
                    g(r["aa"], "mean"), g(r["aa"], "std"),
                    g(r["bwt"], "mean"), g(r["bwt"], "std"),
                    g(r["fwt"], "mean"), g(r["fwt"], "std"),
                    g(r["forgetting"], "mean"), g(r["forgetting"], "std"),
                ])

    print(f"\n  CSV 저장: {RESULTS_DIR / 'buffer_sweep_latest.csv'}")


def save_buffer_sweep_json(rows: list):
    """Buffer sweep 결과 JSON 저장"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = []
    for r in rows:
        data.append({
            "buffer_size": r["buffer_size"], "model": r["model"],
            "category": r["category"], "n_runs": r["n_runs"],
            "aa": r["aa"], "bwt": r["bwt"], "fwt": r["fwt"],
            "forgetting": r["forgetting"],
        })

    for path in [RESULTS_DIR / f"buffer_sweep_{timestamp}.json",
                 RESULTS_DIR / "buffer_sweep_latest.json"]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  JSON 저장: {RESULTS_DIR / 'buffer_sweep_latest.json'}")


def print_buffer_comparison_table(rows: list, buffer_values: list):
    """Buffer별 AA 비교 테이블 출력"""
    print(f"\n{'='*80}")
    print(f"Buffer Sweep: AA (Average Accuracy) 비교")
    print(f"{'='*80}")

    header = f"{'모델':<12} {'카테고리':<22}"
    for bv in buffer_values:
        header += f" {'buf=' + str(bv):>14}"
    print(header)
    print("-" * 80)

    for model_name in BUFFER_MODELS:
        row = f"{model_name:<12} {CATEGORIES.get(model_name, ''):<22}"
        for bv in buffer_values:
            match = [r for r in rows if r["model"] == model_name and r["buffer_size"] == bv]
            if match and match[0]["aa"]["mean"] is not None:
                aa = match[0]["aa"]
                row += f" {aa['mean']:>6.2f}±{aa['std']:<5.2f}"
            else:
                row += f" {'N/A':>14}"
        print(row)

    print(f"{'='*80}")


# ─── 메인 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split-MNIST Sweep 결과 분석기")
    parser.add_argument("type", choices=["epoch", "buffer", "all"],
                        help="분석 유형 (epoch/buffer/all)")
    parser.add_argument("--include-baseline", action="store_true",
                        help="기존 baseline 결과 (epoch=1, buffer=200) 포함")
    args = parser.parse_args()

    print(f"\n분석 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.type in ("epoch", "all"):
        analyze_epoch_sweep(include_baseline=args.include_baseline)

    if args.type in ("buffer", "all"):
        analyze_buffer_sweep(include_baseline=args.include_baseline)
