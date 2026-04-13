"""
Split-MNIST Continual Learning 실험 결과 시각화 스크립트

생성되는 그래프:
1. figure_category_bar.png       - 1차 실험 카테고리별 AA 바 차트
2. figure_paper_comparison.png   - 논문 vs 본 실험 AA 비교 바 차트
3. figure_epoch_sweep.png        - Epoch sweep 라인 차트 (모델별)
4. figure_buffer_sweep.png       - Buffer sweep 라인 차트 (ER/DER++/A-GEM)
5. figure_bwt_aa_scatter.png     - BWT-AA 산점도 (Trade-off)
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ----- 공통 스타일 -----
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# 모델별 일관된 색상/마커
MODEL_STYLE = {
    "joint":  {"color": "#2ca02c", "marker": "s", "label": "Joint (Upper)"},
    "sgd":    {"color": "#7f7f7f", "marker": "x", "label": "SGD (Lower)"},
    "ewc-on": {"color": "#1f77b4", "marker": "o", "label": "EWC Online"},
    "si":     {"color": "#17becf", "marker": "o", "label": "SI"},
    "er":     {"color": "#d62728", "marker": "^", "label": "ER"},
    "derpp":  {"color": "#ff7f0e", "marker": "^", "label": "DER++"},
    "lwf":    {"color": "#9467bd", "marker": "D", "label": "LwF"},
    "agem":   {"color": "#8c564b", "marker": "v", "label": "A-GEM"},
}

CATEGORY_COLOR = {
    "Upper Bound": "#2ca02c",
    "Lower Bound": "#7f7f7f",
    "Regularization": "#1f77b4",
    "Replay": "#d62728",
    "Knowledge Distillation": "#9467bd",
    "Optimization": "#8c564b",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ----- 1. 카테고리별 바 차트 (1차 실험) -----
def plot_category_bar() -> None:
    rows = read_csv(RESULTS_DIR / "split_mnist_latest.csv")
    order = ["joint", "derpp", "er", "agem", "si", "ewc-on", "sgd", "lwf"]
    rows_sorted = sorted(rows, key=lambda r: order.index(r["model"]))

    names = [MODEL_STYLE[r["model"]]["label"] for r in rows_sorted]
    aa = [float(r["aa_mean"]) for r in rows_sorted]
    std = [float(r["aa_std"]) for r in rows_sorted]
    colors = [CATEGORY_COLOR[r["category"]] for r in rows_sorted]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, aa, yerr=std, color=colors, capsize=4,
                  edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, aa):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                f"{val:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Split-MNIST Class-IL Results (n_epochs=1, buffer=200, 5 seeds)")
    plt.xticks(rotation=25, ha="right")

    # 카테고리 범례
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=cat)
                      for cat, c in CATEGORY_COLOR.items()]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.9)

    out = FIGURES_DIR / "figure_category_bar.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved: {out}")


# ----- 2. 논문 vs 본 실험 비교 바 차트 -----
def plot_paper_comparison() -> None:
    # van de Ven et al. (2022) Nature MI, Table 2
    paper_data = {
        "SGD":     (19.89, 0.02),
        "Joint":   (98.17, 0.04),
        "EWC":     (20.64, 0.52),
        "SI":      (21.20, 0.57),
        "LwF":     (21.89, 0.32),
        "A-GEM":   (65.10, 3.64),
        "ER":      (88.79, 0.20),
    }
    our_data = {
        "SGD":     (19.50, 0.07),
        "Joint":   (93.91, 0.14),
        "EWC":     (19.51, 0.06),
        "SI":      (22.15, 4.19),
        "LwF":     (19.34, 0.47),
        "A-GEM":   (23.77, 1.17),
        "ER":      (81.38, 0.70),
    }
    models = list(paper_data.keys())
    paper_aa = [paper_data[m][0] for m in models]
    paper_std = [paper_data[m][1] for m in models]
    our_aa = [our_data[m][0] for m in models]
    our_std = [our_data[m][1] for m in models]

    x = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width / 2, paper_aa, width, yerr=paper_std,
                label="van de Ven et al. (2022)", color="#4c72b0",
                capsize=3, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + width / 2, our_aa, width, yerr=our_std,
                label="This work (ep=1, buf=200)", color="#dd8452",
                capsize=3, edgecolor="black", linewidth=0.5)

    for bars, vals in [(b1, paper_aa), (b2, our_aa)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                    f"{v:.1f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Literature vs This Work — Split-MNIST Class-IL")
    ax.legend(loc="upper left", fontsize=9)
    out = FIGURES_DIR / "figure_paper_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved: {out}")


# ----- 3. Epoch Sweep 라인 차트 -----
def plot_epoch_sweep() -> None:
    rows = read_csv(RESULTS_DIR / "epoch_sweep_latest.csv")
    epochs = sorted({int(r["epoch"]) for r in rows})
    models = ["joint", "derpp", "er", "agem", "si", "ewc-on", "lwf", "sgd"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in models:
        xs, ys, errs = [], [], []
        for ep in epochs:
            rec = next((r for r in rows if r["model"] == m and int(r["epoch"]) == ep), None)
            if rec:
                xs.append(ep)
                ys.append(float(rec["aa_mean"]))
                errs.append(float(rec["aa_std"]))
        style = MODEL_STYLE[m]
        ax.errorbar(xs, ys, yerr=errs, label=style["label"],
                    color=style["color"], marker=style["marker"],
                    markersize=7, linewidth=1.8, capsize=3)

    ax.set_xlabel("Number of Epochs per Task")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_xticks(epochs)
    ax.set_ylim(0, 105)
    ax.set_title("Epoch Sweep — Split-MNIST Class-IL (buffer=200, 5 seeds)")
    ax.legend(loc="center right", fontsize=9, ncol=1, framealpha=0.9)
    out = FIGURES_DIR / "figure_epoch_sweep.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved: {out}")


# ----- 4. Buffer Sweep 라인 차트 -----
def plot_buffer_sweep() -> None:
    rows = read_csv(RESULTS_DIR / "buffer_sweep_latest.csv")
    buffers = sorted({int(r["buffer_size"]) for r in rows})
    models = ["derpp", "er", "agem"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in models:
        xs, ys, errs = [], [], []
        for b in buffers:
            rec = next((r for r in rows if r["model"] == m and int(r["buffer_size"]) == b), None)
            if rec:
                xs.append(b)
                ys.append(float(rec["aa_mean"]))
                errs.append(float(rec["aa_std"]))
        style = MODEL_STYLE[m]
        ax.errorbar(xs, ys, yerr=errs, label=style["label"],
                    color=style["color"], marker=style["marker"],
                    markersize=8, linewidth=1.8, capsize=3)

    # Joint 상한선 (참고)
    ax.axhline(93.91, color="#2ca02c", linestyle="--", linewidth=1.2,
               alpha=0.7, label="Joint (Upper Bound, 93.91)")

    ax.set_xscale("log")
    ax.set_xticks(buffers)
    ax.set_xticklabels([str(b) for b in buffers])
    ax.set_xlabel("Memory Buffer Size (log scale)")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Buffer Size Sweep — Split-MNIST Class-IL (n_epochs=1, 5 seeds)")
    ax.legend(loc="center right", fontsize=9, framealpha=0.9)
    out = FIGURES_DIR / "figure_buffer_sweep.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved: {out}")


# ----- 5. BWT-AA 산점도 (trade-off) -----
def plot_bwt_aa_scatter() -> None:
    rows = read_csv(RESULTS_DIR / "split_mnist_latest.csv")
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in rows:
        m = r["model"]
        style = MODEL_STYLE[m]
        ax.errorbar(float(r["bwt_mean"]), float(r["aa_mean"]),
                    xerr=float(r["bwt_std"]), yerr=float(r["aa_std"]),
                    color=style["color"], marker=style["marker"],
                    markersize=11, linewidth=1.2, capsize=3,
                    label=style["label"])
        ax.annotate(style["label"],
                    (float(r["bwt_mean"]), float(r["aa_mean"])),
                    xytext=(8, 6), textcoords="offset points", fontsize=9)

    ax.axhline(19.5, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xlabel("Backward Transfer (BWT, %)")
    ax.set_ylabel("Average Accuracy (AA, %)")
    ax.set_title("BWT vs AA — Forgetting / Retention Trade-off (Phase 1)")
    out = FIGURES_DIR / "figure_bwt_aa_scatter.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved: {out}")


def main() -> None:
    print(f"생성 경로: {FIGURES_DIR}")
    plot_category_bar()
    plot_paper_comparison()
    plot_epoch_sweep()
    plot_buffer_sweep()
    plot_bwt_aa_scatter()
    print("완료")


if __name__ == "__main__":
    main()
