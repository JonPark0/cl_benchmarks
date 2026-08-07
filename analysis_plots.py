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
from matplotlib.patches import Rectangle

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
    "agem_fixed": {"color": "#e377c2", "marker": "P", "label": "A-GEM (patched)"},
    "gem":    {"color": "#2ca02c", "marker": "D", "label": "GEM"},
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
    for bar, val, s in zip(bars, aa, std):
        # 오차막대(val ± s) 위쪽 여유 공간에 배치해 막대가 레이블을 관통하지 않도록 함
        ax.text(bar.get_x() + bar.get_width() / 2, val + s + 2.2,
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
    # 가독성을 위해 정확도 오름차순으로 정렬
    models = sorted(paper_data.keys(), key=lambda m: our_data[m][0])
    paper_aa = [paper_data[m][0] for m in models]
    paper_std = [paper_data[m][1] for m in models]
    our_aa = [our_data[m][0] for m in models]
    our_std = [our_data[m][1] for m in models]
    deltas = [o - p for o, p in zip(our_aa, paper_aa)]

    y = np.arange(len(models)) * 1.3
    height = 0.42

    fig, ax = plt.subplots(figsize=(9.5, 7))
    b1 = ax.barh(y + height / 2, paper_aa, height, xerr=paper_std,
                 label="van de Ven et al. (2022)", color="#4c72b0",
                 capsize=3, edgecolor="black", linewidth=0.5)
    b2 = ax.barh(y - height / 2, our_aa, height, xerr=our_std,
                 label="This work (ep=1, buf=200)", color="#dd8452",
                 capsize=3, edgecolor="black", linewidth=0.5)

    for bars, vals, errs in [(b1, paper_aa, paper_std), (b2, our_aa, our_std)]:
        for bar, v, e in zip(bars, vals, errs):
            ax.text(v + e + 2.2, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}", va="center", fontsize=8)

    # A-GEM과 Joint의 격차(delta)를 각 쌍의 행 중앙에 표기 (인접 모델 막대와 겹치지 않도록
    # y[i] 자체를 앵커로 사용 — 기존 y[i]+height/2+0.16은 다음 행 쪽으로 치우쳐 있었음)
    for m, d in zip(models, deltas):
        if m in ("A-GEM", "Joint"):
            i = models.index(m)
            x_pos = max(paper_aa[i] + paper_std[i], our_aa[i] + our_std[i]) + 16
            ax.annotate(f"$\\Delta$={d:+.1f}%p", (x_pos, y[i]),
                        va="center", fontsize=9, fontweight="bold",
                        color="#c0392b" if abs(d) > 20 else "#555555")

    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_ylim(y[0] - 0.9, y[-1] + 0.9)
    ax.set_xlabel("Average Accuracy (%)")
    # 실제 데이터 범위(값+오차막대+레이블 폭)에 맞춰 여백을 계산 — 고정폭(0-130)이 남기던
    # 불필요한 공백을 제거
    max_extent = max(
        max(p + e for p, e in zip(paper_aa, paper_std)),
        max(o + e for o, e in zip(our_aa, our_std)),
    )
    ax.set_xlim(0, max_extent + 28)
    ax.set_title("Literature vs This Work: Split-MNIST Class-IL")
    ax.legend(loc="lower right", fontsize=9)
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
        xs_a, ys_a, errs_a = np.array(xs), np.array(ys), np.array(errs)
        ax.plot(xs_a, ys_a, label=style["label"], color=style["color"],
                marker=style["marker"], markersize=7, linewidth=1.8)
        ax.fill_between(xs_a, ys_a - errs_a, ys_a + errs_a,
                         color=style["color"], alpha=0.15, linewidth=0)

    # 실제 epoch 값 간격을 그대로 반영 (1, 5, 10, 20의 비례 위치)
    ax.set_xlabel("Number of Epochs per Task")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_xticks(epochs)
    ax.set_xlim(min(epochs) - 0.5, max(epochs) + 0.5)
    ax.set_ylim(0, 105)
    ax.set_title("Epoch Sweep: Split-MNIST Class-IL (buffer=200, 5 seeds, shaded = $\\pm$std)")
    # 범례를 축 바깥(우측)으로 배치 — 축 안쪽 어디에 두어도 8개 모델의 곡선이 전 구간에
    # 걸쳐 있어 겹침을 피할 수 없음. savefig.bbox="tight"가 캔버스를 자동으로 확장한다.
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9,
              ncol=1, framealpha=0.9, borderaxespad=0)
    out = FIGURES_DIR / "figure_epoch_sweep.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved: {out}")


# ----- 4. Buffer Sweep 라인 차트 -----
def plot_buffer_sweep() -> None:
    rows = read_csv(RESULTS_DIR / "buffer_sweep_latest.csv")
    wsl_rows = read_csv(RESULTS_DIR / "wsl" / "buffer_sweep_agem_fixed_gem.csv")
    buffers = sorted({int(r["buffer_size"]) for r in rows})
    models = ["derpp", "er", "gem", "agem", "agem_fixed"]
    wsl_models = {"gem", "agem_fixed"}
    low_cluster = ["agem", "agem_fixed"]        # A-GEM 계열: 버퍼 크기와 무관하게 낮은 구간에 정체
    high_cluster = ["derpp", "er", "gem"]        # 경험 재생 + 원본 GEM: 로그 스케일 선형 향상

    def draw_series(target_ax, model_list, markersize, linewidth):
        for m in model_list:
            xs, ys, errs = series[m]
            style = MODEL_STYLE[m]
            linestyle = "--" if m in wsl_models else "-"
            label = style["label"] + (" [WSL2]" if m in wsl_models else "")
            target_ax.errorbar(xs, ys, yerr=errs, label=label if target_ax is ax else None,
                                color=style["color"], marker=style["marker"],
                                markersize=markersize, linewidth=linewidth,
                                linestyle=linestyle, capsize=2.5)

    series = {}
    for m in models:
        src = wsl_rows if m in wsl_models else rows
        xs, ys, errs = [], [], []
        for b in buffers:
            rec = next((r for r in src if r["model"] == m and int(r["buffer_size"]) == b), None)
            if rec:
                xs.append(b)
                ys.append(float(rec["aa_mean"]))
                errs.append(float(rec["aa_std"]))
        series[m] = (xs, ys, errs)

    # 2행 그리드: 위쪽 전체 폭에 원본 그래프, 아래쪽에 두 확대 패널을 나란히 배치
    fig = plt.figure(figsize=(10, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.45, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    ax_low = fig.add_subplot(gs[1, 0])
    ax_high = fig.add_subplot(gs[1, 1])

    draw_series(ax, models, markersize=8, linewidth=1.8)
    ax.axhline(93.91, color="#7f7f7f", linestyle=":", linewidth=1.2,
               alpha=0.7, label="Joint (Upper Bound, 93.91)")

    ax.set_xscale("log")
    ax.set_xticks(buffers)
    ax.set_xticklabels([str(b) for b in buffers])
    ax.set_xlabel("Memory Buffer Size (log scale)")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Buffer Size Sweep: Split-MNIST Class-IL (n_epochs=1, 5 seeds)")

    # 두 확대 영역을 원본 그래프 위에 점선 사각형으로 표시 — 아래 패널과의 대응 관계를
    # 시각적으로 연결한다 (범례 A/B로 패널 제목과 일치시킴).
    x0, x1 = ax.get_xlim()
    low_band, high_band = (18, 30), (70, 95)
    ax.add_patch(Rectangle((x0, low_band[0]), x1 - x0, low_band[1] - low_band[0],
                            fill=False, edgecolor="#555555", linestyle="--", linewidth=1.1))
    ax.text(x0 * 1.05, low_band[1] + 1.5, "A", fontsize=10, fontweight="bold", color="#555555")
    ax.add_patch(Rectangle((x0, high_band[0]), x1 - x0, high_band[1] - high_band[0],
                            fill=False, edgecolor="#555555", linestyle="--", linewidth=1.1))
    ax.text(x0 * 1.05, high_band[1] + 1.5, "B", fontsize=10, fontweight="bold", color="#555555")
    ax.set_xlim(x0, x1)

    # 범례는 원본 그래프 우측 바깥에 배치 — 아래 확대 패널과 독립적으로 겹침 없이 표시
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8.5, framealpha=0.9,
              borderaxespad=0)

    # 패널 A: A-GEM 계열 확대
    draw_series(ax_low, low_cluster, markersize=6, linewidth=1.5)
    ax_low.set_xscale("log")
    ax_low.set_xticks(buffers)
    ax_low.set_xticklabels([str(b) for b in buffers], fontsize=8)
    ax_low.set_ylim(*low_band)
    ax_low.set_xlabel("Memory Buffer Size (log scale)", fontsize=9)
    ax_low.set_ylabel("Average Accuracy (%)", fontsize=9)
    ax_low.set_title("A. A-GEM family (zoom)", fontsize=10)
    ax_low.tick_params(labelsize=8)

    # 패널 B: 경험 재생 + 원본 GEM 확대
    draw_series(ax_high, high_cluster, markersize=6, linewidth=1.5)
    ax_high.set_xscale("log")
    ax_high.set_xticks(buffers)
    ax_high.set_xticklabels([str(b) for b in buffers], fontsize=8)
    ax_high.set_ylim(*high_band)
    ax_high.set_xlabel("Memory Buffer Size (log scale)", fontsize=9)
    ax_high.set_ylabel("Average Accuracy (%)", fontsize=9)
    ax_high.set_title("B. DER++ · ER · GEM (zoom)", fontsize=10)
    ax_high.tick_params(labelsize=8)

    out = FIGURES_DIR / "figure_buffer_sweep.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved: {out}")


# ----- 5. BWT-AA 산점도 (trade-off) -----
def plot_bwt_aa_scatter() -> None:
    rows = read_csv(RESULTS_DIR / "split_mnist_latest.csv")

    data = {}
    for r in rows:
        m = r["model"]
        data[m] = (float(r["bwt_mean"]), float(r["aa_mean"]),
                   float(r["bwt_std"]), float(r["aa_std"]))

    # 좌하단(SGD/EWC/LwF/SI/A-GEM)과 중앙(ER/DER++)은 본 플롯에서 서로 점이 가깝게 뭉쳐
    # 있어 본 플롯에는 라벨을 달지 않고, 아래 두 확대 패널에서만 개별 표기한다.
    low_cluster = ["sgd", "ewc-on", "lwf", "si", "agem"]
    mid_cluster = ["er", "derpp"]
    zoomed_models = set(low_cluster) | set(mid_cluster)
    # joint은 x축 우측 끝에 가까워 기본 오프셋(+8,+6)이 축 바깥, 즉 범례 자리와 겹침 —
    # 좌하단의 빈 공간으로 당겨서 배치
    main_label_offsets = {"joint": (-95, -8)}

    def draw_points(target_ax, model_list, markersize, with_legend_label):
        for m in model_list:
            bwt, aa, bwt_std, aa_std = data[m]
            style = MODEL_STYLE[m]
            target_ax.errorbar(bwt, aa, xerr=bwt_std, yerr=aa_std,
                                color=style["color"], marker=style["marker"],
                                markersize=markersize, linewidth=1.1, capsize=2.5,
                                label=style["label"] if with_legend_label else None)

    # 2행 그리드: 위쪽 전체 폭에 원본 산점도, 아래쪽에 두 확대 패널을 나란히 배치
    fig = plt.figure(figsize=(10, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.45, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    ax_low = fig.add_subplot(gs[1, 0])
    ax_mid = fig.add_subplot(gs[1, 1])

    draw_points(ax, list(MODEL_STYLE.keys() & data.keys()), markersize=11, with_legend_label=True)
    for m, (bwt, aa, _, _) in data.items():
        if m not in zoomed_models:
            ax.annotate(MODEL_STYLE[m]["label"], (bwt, aa),
                        xytext=main_label_offsets.get(m, (8, 6)),
                        textcoords="offset points", fontsize=9)

    # 파레토 프론티어: BWT를 높이면서 AA도 함께 높일 수 있는 비지배(non-dominated) 점들을 연결
    points = sorted((bwt, aa) for m, (bwt, aa, _, _) in data.items() if m != "joint")
    frontier = []
    best_aa = -np.inf
    for bwt, aa in reversed(points):  # BWT 내림차순으로 스캔하며 AA 최댓값 갱신
        if aa > best_aa:
            frontier.append((bwt, aa))
            best_aa = aa
    frontier.sort()
    if len(frontier) > 1:
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, linestyle="--", color="#444444", linewidth=1.3,
                alpha=0.7, zorder=0, label="Pareto frontier")

    ax.axhline(19.5, color="gray", linestyle=":", alpha=0.5, linewidth=1,
               label="SGD baseline (AA=19.5%)")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5, linewidth=1,
               label="No net forgetting (BWT=0)")
    ax.set_xlabel("Backward Transfer (BWT, %)")
    ax.set_ylabel("Average Accuracy (AA, %)")
    ax.set_title("BWT vs AA: Forgetting/Retention Trade-off (n_epochs=1, buffer=200)")

    # 두 확대 영역을 원본 산점도 위에 점선 사각형으로 표시 — 아래 패널과의 대응 관계를
    # 시각적으로 연결한다 (레이블 A/B로 패널 제목과 일치시킴).
    low_box = (-101, -85, 17, 27)     # x0, x1, y0, y1
    mid_box = (-26, -13, 77, 89)
    for (x0, x1, y0, y1), label in [(low_box, "A"), (mid_box, "B")]:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                edgecolor="#555555", linestyle="--", linewidth=1.1))
        ax.text(x0, y1 + 1.5, label, fontsize=10, fontweight="bold", color="#555555")

    # 범례는 축 바깥(우측)에 배치 — 모델 7개 + 프론티어 + 기준선 2개(총 10개 항목)를
    # 축 안쪽 어디에 두어도 점과 겹치므로 완전히 분리한다.
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8.5,
              framealpha=0.9, borderaxespad=0)

    # 패널 A: 하한선 인근 5개 모델 확대
    low_offsets = {
        "sgd": (6, -10), "ewc-on": (6, 8), "lwf": (6, -22),
        "si": (6, 20), "agem": (-45, 10),
    }
    draw_points(ax_low, low_cluster, markersize=9, with_legend_label=False)
    for m in low_cluster:
        bwt, aa, _, _ = data[m]
        ax_low.annotate(MODEL_STYLE[m]["label"], (bwt, aa),
                         xytext=low_offsets[m], textcoords="offset points", fontsize=8)
    ax_low.set_xlim(low_box[0], low_box[1])
    ax_low.set_ylim(low_box[2], low_box[3])
    ax_low.set_xlabel("Backward Transfer (BWT, %)", fontsize=9)
    ax_low.set_ylabel("Average Accuracy (AA, %)", fontsize=9)
    ax_low.set_title("A. Lower-bound cluster (zoom)", fontsize=10)
    ax_low.tick_params(labelsize=8)

    # 패널 B: DER++ · ER 확대
    mid_offsets = {"derpp": (8, 10), "er": (8, -16)}
    draw_points(ax_mid, mid_cluster, markersize=9, with_legend_label=False)
    for m in mid_cluster:
        bwt, aa, _, _ = data[m]
        ax_mid.annotate(MODEL_STYLE[m]["label"], (bwt, aa),
                         xytext=mid_offsets[m], textcoords="offset points", fontsize=8)
    ax_mid.set_xlim(mid_box[0], mid_box[1])
    ax_mid.set_ylim(mid_box[2], mid_box[3])
    ax_mid.set_xlabel("Backward Transfer (BWT, %)", fontsize=9)
    ax_mid.set_ylabel("Average Accuracy (AA, %)", fontsize=9)
    ax_mid.set_title("B. DER++ · ER (zoom)", fontsize=10)
    ax_mid.tick_params(labelsize=8)

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
