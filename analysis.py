"""
Split-MNIST 실험 결과 분석 스크립트
=====================================
raw_data/ 내의 logs.pyd 파일을 파싱하여
모델별 mean ± std 통계를 계산하고 CSV 및 표로 출력

사용법:
    python analysis.py                          # 전체 결과 분석
    python analysis.py --model sgd ewc-on       # 특정 모델만
    python analysis.py --output results/        # 결과 저장 경로 지정
"""

import ast
import os
import re
import json
import argparse
import numpy as np
import csv
from pathlib import Path
from datetime import datetime

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
RAW_DATA_DIR = BASE_DIR / "raw_data"
RESULTS_DIR = BASE_DIR / "results"

MODELS = ["sgd", "joint", "ewc-on", "si", "er", "derpp", "lwf", "agem"]
CATEGORIES = {
    "sgd": "Lower Bound",
    "joint": "Upper Bound",
    "ewc-on": "Regularization",
    "si": "Regularization",
    "er": "Replay",
    "derpp": "Replay",
    "lwf": "Knowledge Distillation",
    "agem": "Optimization",
}
N_TASKS = 5


def parse_logs_pyd(filepath: Path) -> list[dict]:
    """
    logs.pyd 파일 파싱 (각 줄이 Python dict literal).
    각 줄은 하나의 실험 결과.
    - np.float64(...) / np.float32(...) → float 값으로 변환
    - torch.device(...) → 문자열로 변환
    - enable_other_metrics=False 항목 제외 (테스트 런 중복 방지)
    - 동일 seed 중복 시 마지막 항목(최신)만 유지
    """
    records = []
    if not filepath.exists():
        return records

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # numpy 타입 및 torch.device 처리 (ast.literal_eval 호환)
    content = re.sub(r"np\.float64\(([^)]+)\)", r"\1", content)
    content = re.sub(r"np\.float32\(([^)]+)\)", r"\1", content)
    content = re.sub(r"np\.int64\(([^)]+)\)", r"\1", content)
    content = re.sub(r"device\(type='cuda',\s*index=\d+\)", "'cuda:0'", content)
    content = re.sub(r"device\(type='cpu'\)", "'cpu'", content)
    content = re.sub(r"device\('cuda:\d+'\)", "'cuda:0'", content)

    # 줄 단위로 분리하여 파싱
    seen_seeds = {}  # seed → 마지막 유효 record
    for line_num, line in enumerate(content.strip().split('\n'), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = ast.literal_eval(line)
            # enable_other_metrics=False 항목 건너뜀 (BWT/FWT 없는 테스트 런)
            if not record.get("enable_other_metrics", True):
                continue
            seed = record.get("seed", line_num)
            seen_seeds[seed] = record  # 동일 seed는 최신 항목으로 덮어씀
        except Exception as e:
            print(f"  [WARN] {filepath.name} 줄 {line_num} 파싱 실패: {e}")

    records = list(seen_seeds.values())
    return records


def extract_metrics(record: dict) -> dict:
    """
    하나의 실험 기록에서 핵심 지표 추출.
    - AA: accmean_task5 (마지막 태스크 이후 평균 정확도)
    - BWT: backward_transfer
    - FWT: forward_transfer
    - Forgetting: forgetting
    - 각 태스크별 최종 정확도 행렬
    """
    metrics = {
        "seed": record.get("seed"),
        "model": record.get("model"),
        "aa": None,
        "bwt": None,
        "fwt": None,
        "forgetting": None,
        "task_accs": [],  # 최종 각 태스크 정확도 (Class-IL)
    }

    # AA: 마지막 태스크 이후 평균 정확도
    for t in range(N_TASKS, 0, -1):
        key = f"accmean_task{t}"
        if key in record and record[key] is not None:
            metrics["aa"] = record[key]
            break

    # BWT, FWT, Forgetting
    metrics["bwt"] = record.get("backward_transfer")
    metrics["fwt"] = record.get("forward_transfer")
    metrics["forgetting"] = record.get("forgetting")

    # 최종 각 태스크 정확도 (accuracy_{j}_task{N_TASKS} 형태)
    for j in range(1, N_TASKS + 1):
        key = f"accuracy_{j}_task{N_TASKS}"
        if key in record:
            metrics["task_accs"].append(record[key])

    return metrics


def compute_stats(values: list) -> dict:
    """유효한 값들로 mean ± std 계산"""
    valid = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not valid:
        return {"mean": None, "std": None, "n": 0}
    arr = np.array(valid, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "n": len(arr),
    }


def analyze_model(model_name: str) -> dict:
    """특정 모델의 결과 분석"""
    # Mammoth는 모델명의 하이픈을 언더스코어로 변환하여 저장
    model_dir_name = model_name.replace("-", "_")
    # Class-IL 결과 경로
    logs_path = RAW_DATA_DIR / "split_mnist" / model_name / "class-il" / "seq-mnist" / model_dir_name / "logs.pyd"

    records = parse_logs_pyd(logs_path)
    if not records:
        print(f"  [{model_name}] 결과 없음: {logs_path}")
        return None

    print(f"  [{model_name}] {len(records)}개 실험 결과 파싱")

    aa_list, bwt_list, fwt_list, fm_list = [], [], [], []

    for rec in records:
        m = extract_metrics(rec)
        if m["aa"] is not None:
            aa_list.append(m["aa"])
        if m["bwt"] is not None:
            bwt_list.append(m["bwt"])
        if m["fwt"] is not None:
            fwt_list.append(m["fwt"])
        if m["forgetting"] is not None:
            fm_list.append(m["forgetting"])

    return {
        "model": model_name,
        "category": CATEGORIES.get(model_name, "Unknown"),
        "n_runs": len(records),
        "aa": compute_stats(aa_list),
        "bwt": compute_stats(bwt_list),
        "fwt": compute_stats(fwt_list),
        "forgetting": compute_stats(fm_list),
        "raw_aa": aa_list,
        "raw_bwt": bwt_list,
    }


def format_stat(stat: dict, decimals: int = 2) -> str:
    """mean ± std 형식으로 포매팅"""
    if stat is None or stat["mean"] is None:
        return "N/A"
    mean = round(stat["mean"], decimals)
    std = round(stat["std"], decimals)
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def print_results_table(results: list):
    """결과를 표 형식으로 출력"""
    print("\n" + "=" * 90)
    print("Split-MNIST 실험 결과 (Class-IL, 5 Tasks)")
    print("=" * 90)
    header = f"{'모델':<12} {'카테고리':<22} {'N':>3} {'AA (mean±std)':>18} {'BWT (mean±std)':>18} {'FWT':>12}"
    print(header)
    print("-" * 90)

    for r in results:
        if r is None:
            continue
        aa_str = format_stat(r["aa"])
        bwt_str = format_stat(r["bwt"])
        fwt_str = format_stat(r["fwt"])
        n = r["n_runs"]
        print(f"{r['model']:<12} {r['category']:<22} {n:>3} {aa_str:>18} {bwt_str:>18} {fwt_str:>12}")

    print("=" * 90)
    print("AA: Average Accuracy (높을수록 좋음)")
    print("BWT: Backward Transfer (0에 가까울수록, 음수는 망각 발생)")
    print("FWT: Forward Transfer (양수는 이전 학습이 새 태스크에 도움)")


def save_csv(results: list, output_path: Path):
    """결과를 CSV로 저장"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "category", "n_runs",
            "aa_mean", "aa_std",
            "bwt_mean", "bwt_std",
            "fwt_mean", "fwt_std",
            "forgetting_mean", "forgetting_std",
        ])
        for r in results:
            if r is None:
                continue
            def g(stat, key):
                return round(stat[key], 4) if stat and stat[key] is not None else ""
            writer.writerow([
                r["model"],
                r["category"],
                r["n_runs"],
                g(r["aa"], "mean"), g(r["aa"], "std"),
                g(r["bwt"], "mean"), g(r["bwt"], "std"),
                g(r["fwt"], "mean"), g(r["fwt"], "std"),
                g(r["forgetting"], "mean"), g(r["forgetting"], "std"),
            ])
    print(f"\n  CSV 저장: {output_path}")


def save_json(results: list, output_path: Path):
    """결과를 JSON으로 저장 (raw 값 포함)"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for r in results:
        if r is None:
            continue
        serializable.append({
            "model": r["model"],
            "category": r["category"],
            "n_runs": r["n_runs"],
            "aa": r["aa"],
            "bwt": r["bwt"],
            "fwt": r["fwt"],
            "forgetting": r["forgetting"],
            "raw_aa": r["raw_aa"],
            "raw_bwt": r["raw_bwt"],
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"  JSON 저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Split-MNIST 결과 분석기")
    parser.add_argument("--model", nargs="+", help="분석할 모델 (기본: 전체)")
    parser.add_argument("--output", default=str(RESULTS_DIR), help="결과 저장 경로")
    args = parser.parse_args()

    models_to_analyze = args.model if args.model else MODELS
    output_dir = Path(args.output)

    print(f"\n결과 분석 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"분석 모델: {models_to_analyze}")

    results = []
    for model_name in models_to_analyze:
        result = analyze_model(model_name)
        results.append(result)

    # 출력
    valid_results = [r for r in results if r is not None]
    if valid_results:
        print_results_table(valid_results)

        # 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_csv(valid_results, output_dir / f"split_mnist_results_{timestamp}.csv")
        save_json(valid_results, output_dir / f"split_mnist_results_{timestamp}.json")

        # 항상 최신 결과도 덮어쓰기
        save_csv(valid_results, output_dir / "split_mnist_latest.csv")
        save_json(valid_results, output_dir / "split_mnist_latest.json")
    else:
        print("\n[WARNING] 분석할 결과가 없습니다. 실험을 먼저 실행하세요:")
        print("  python run_experiments.py")


if __name__ == "__main__":
    main()
