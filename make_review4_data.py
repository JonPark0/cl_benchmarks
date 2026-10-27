"""review4 실험 결과를 하나의 JSON과 그림용 CSV로 내보낸다. numpy, scipy 필요.

python make_review4_data.py
산출물: results/wsl/review4_numbers.json, results/wsl/e2_seeds20_summary.csv
논문 본문의 모든 수치는 이 JSON에서 옮긴다.
"""
import csv, glob, json, re
from pathlib import Path
import numpy as np
from scipy import stats

BASE = Path(__file__).parent
W = BASE / "raw_data" / "wsl"
OUT = BASE / "results" / "wsl"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS5 = [42, 123, 456, 789, 1024]
SEEDS20 = SEEDS5 + list(range(2001, 2016))
CAT = {"joint": "Upper Bound", "sgd": "Lower Bound", "ewc-on": "Regularization", "si": "Regularization",
       "er": "Replay", "derpp": "Replay", "lwf": "Knowledge Distillation", "agem": "Optimization",
       "agem_fixed": "Optimization"}
LIT = {"sgd": (19.89, 0.02), "joint": (98.17, 0.04), "ewc-on": (20.64, 0.52), "si": (21.20, 0.57),
       "lwf": (21.89, 0.32), "agem": (65.10, 3.64), "er": (88.79, 0.20)}  # van de Ven 2022 Table 2, s.e.m., n=20


def read(exp, model, seed):
    fs = list((W / exp / model / f"s{seed}" / "class-il").rglob("logs.pyd"))
    if not fs:
        return None
    line = [l for l in fs[0].read_text().splitlines() if l.strip()][-1]
    aa = re.search(r"'accmean_task5': (?:np\.float64\()?([-0-9.e]+)", line)
    bwt = re.search(r"'backward_transfer': (?:np\.float64\()?([-0-9.e]+)", line)
    if not aa:
        return None
    return float(aa.group(1)), float(bwt.group(1)) if bwt else None


def collect(exp, models, seeds):
    return {m: {s: r for s in seeds if (r := read(exp, m, s))} for m in models}


def summ(d):
    a = np.array([v[0] for v in d.values()])
    b = np.array([v[1] for v in d.values() if v[1] is not None])
    return {"n": len(a), "mean": a.mean(), "std": a.std(ddof=1) if len(a) > 1 else None,
            "bwt_mean": b.mean() if len(b) else None, "bwt_std": b.std(ddof=1) if len(b) > 1 else None,
            "raw": {str(k): v[0] for k, v in d.items()}}


def paired(d, a, b):
    s = sorted(set(d[a]) & set(d[b]))
    t = stats.ttest_rel([d[a][k][0] for k in s], [d[b][k][0] for k in s])
    return {"a": a, "b": b, "n": len(s), "t": float(t.statistic), "p": float(t.pvalue),
            "mean_diff": float(np.mean([d[a][k][0] - d[b][k][0] for k in s]))}


def holm(ps):
    idx = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for rank, i in enumerate(idx):
        run = max(run, (len(ps) - rank) * ps[i]); adj[i] = min(1, run)
    return adj.tolist()


R = {}
# ---------------- E2: 기본 실험 20시드 ----------------
mods = ["joint", "derpp", "er", "agem", "si", "ewc-on", "lwf", "sgd"]
e2 = collect("e2_seeds20", mods, SEEDS20)
R["e2"] = {m: summ(e2[m]) for m in mods}
non_bound = [m for m in mods if m not in ("joint", "sgd")]
order = sorted(non_bound, key=lambda m: -R["e2"][m]["mean"])
R["e2_order"] = order
pairs = list(zip(order, order[1:])) + [(order[-1], "sgd")]
tests = [paired(e2, a, b) for a, b in pairs]
for t_, h in zip(tests, holm([t["p"] for t in tests])):
    t_["holm_p"] = h
R["e2_tests"] = tests
# 기존 5시드에서 검정했던 쌍(1위-2위 등)과 A-GEM 대 EWC
R["e2_extra_tests"] = [paired(e2, "agem", "ewc-on"), paired(e2, "joint", "derpp")]
# SI 분산
si_std = R["e2"]["si"]["std"]
F = (si_std / 2.55) ** 2
R["si_variance"] = {"std20": si_std, "lit_std": 2.55, "F": F, "p_one_sided": float(1 - stats.f.cdf(F, 19, 19)),
                    "note": "문헌 환산 std는 n=20이므로 df=(19,19)로 다시 계산"}
F5 = (4.19 / 2.55) ** 2
R["si_variance"].update({"std5": 4.19, "F5": F5, "p5_one_sided": float(1 - stats.f.cdf(F5, 4, 19)),
                         "p5_two_sided": float(2 * (1 - stats.f.cdf(F5, 4, 19)))})
R["sigma_over_s_95_n5"] = np.sqrt(4 / stats.chi2.ppf([.975, .025], 4)).tolist()
# 기존 Windows 5시드 대비 시드별 차이
old = {m["model"]: m["raw_aa"] for m in json.load(open(BASE / "results" / "split_mnist_latest.json"))}
R["e2_vs_windows"] = {m: [round(e2[m][s][0] - o, 4) for s, o in zip(SEEDS5, old[m])] for m in mods}
R["e2_vs_windows_max"] = {m: max(abs(x) for x in v) for m, v in R["e2_vs_windows"].items()}
# 그림용 CSV (split_mnist_latest.csv와 같은 열)
with open(OUT / "e2_seeds20_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "category", "n_runs", "aa_mean", "aa_std", "bwt_mean", "bwt_std"])
    for m in mods:
        s = R["e2"][m]
        w.writerow([m, CAT[m], s["n"], round(s["mean"], 4), round(s["std"], 4), round(s["bwt_mean"], 4), round(s["bwt_std"], 4)])

# ---------------- E1: Adam 5축 ----------------
m1 = ["sgd", "joint", "ewc-on", "si", "lwf", "er", "agem", "agem_fixed"]
e1 = collect("e1_adam", m1, SEEDS20)
R["e1"] = {m: summ(e1[m]) for m in m1 if e1[m]}
R["e1_first5"] = {m: summ({s: e1[m][s] for s in SEEDS5 if s in e1[m]}) for m in ("agem", "agem_fixed", "er")}
R["e1_agem_vs_fixed"] = paired(e1, "agem", "agem_fixed")
R["e1_agem_vs_fixed_first5"] = paired({m: {s: e1[m][s] for s in SEEDS5} for m in ("agem", "agem_fixed")}, "agem", "agem_fixed")
lit = {}
for m, key in (("agem", "agem"), ("agem_fixed", "agem"), ("er", "er")):
    for tag, d in (("n20", R["e1"][m]), ("n5", R["e1_first5"][m])):
        n, mean, sd = d["n"], d["mean"], d["std"]
        se = np.sqrt(sd ** 2 / n + LIT[key][1] ** 2)
        lit[f"{m}_{tag}"] = {"mean": mean, "std": sd, "lit": LIT[key][0], "lit_sem": LIT[key][1],
                             "diff": mean - LIT[key][0], "t_approx": (mean - LIT[key][0]) / se, "n": n}
R["e1_vs_literature"] = lit
fails = (W / "e1_adam" / "failures.log").read_text() if (W / "e1_adam" / "failures.log").exists() else ""
R["e1_lwf"] = {"n_success": len(e1["lwf"]), "failed_seeds": [int(x) for x in re.findall(r"^lwf s(\d+)", fails, re.M)],
               "probes_seed42": [{"alpha": 1.0, "temp": 2.0, "result": "NaN"}, {"alpha": 0.5, "temp": 2.0, "result": "NaN", "task": 5},
                                 {"alpha": 0.1, "temp": 2.0, "result": "NaN", "task": 5},
                                 {"alpha": 1.0, "temp": 1.0, "result": "ok", "aa": 19.9785}],
               "note": "프로브 결과는 실행 로그(scratchpad lwf_probe.log)에서 옮긴 값이며 재실행 프로브 디렉터리 e1_lwf_probe에는 성공한 1건만 남는다"}

# ---------------- E3 ----------------
def e3_group(files, key):
    recs = [json.load(open(f)) for f in sorted(files)]
    recs = [r for r in recs if len(r["records"]) == 5]
    rows = []
    for t in range(5):
        prev, cur, fut, ratio, of = [], [], [], [], []
        for r in recs:
            v = np.array(r["records"][t][key])
            p = v[:2 * t].mean() if t > 0 else np.nan
            c = v[2 * t:2 * t + 2].mean()
            f_ = v[2 * t + 2:].mean() if t < 4 else np.nan
            prev.append(p); cur.append(c); fut.append(f_); ratio.append(p / c); of.append(p / f_)
        rows.append({"task": t + 1, "prev": float(np.nanmean(prev)) if t > 0 else None, "cur": float(np.mean(cur)),
                     "fut": float(np.nanmean(fut)) if t < 4 else None,
                     "prev_over_cur_mean": float(np.nanmean(ratio)) if t > 0 else None,
                     "prev_over_cur_min": float(np.nanmin(ratio)) if t > 0 else None,
                     "prev_over_cur_max": float(np.nanmax(ratio)) if t > 0 else None,
                     "prev_over_fut": float(np.nanmean(of)) if 0 < t < 4 else None})
    return {"n": len(recs), "seeds": [r["seed"] for r in recs], "rows": rows}

R["e3"] = {}
for name, pat in (("ep1", "ep1_s*.json"), ("ep20", "ep20_s*.json"), ("gamma0.5_ep1", "g0.5_ep1_s*.json")):
    fs = glob.glob(str(W / "e3_fisher" / pat))
    R["e3"][name] = {k: e3_group(fs, key) for k, key in (("task", "classifier_row_fisher_task"), ("accum", "classifier_row_fisher_accum"))}
aas = []
for sd in SEEDS5:
    x = read("e3_fisher/logs", f"g0.5_ep1_s{sd}", "") if False else None
    fs2 = list((W / "e3_fisher" / "logs" / f"g0.5_ep1_s{sd}" / "class-il").rglob("logs.pyd"))
    if fs2:
        line = [l for l in fs2[0].read_text().splitlines() if l.strip()][-1]
        m_ = re.search(r"'accmean_task5': (?:np\.float64\()?([-0-9.e]+)", line)
        if m_: aas.append(float(m_.group(1)))
R["e3"]["gamma0.5_aa"] = {"n": len(aas), "mean": float(np.mean(aas)), "std": float(np.std(aas, ddof=1)), "raw": aas}

# ---------------- 실행 횟수 (로그 기준) ----------------
def count(exp):
    return len(list((W / exp).rglob("class-il/**/logs.pyd")))
R["run_counts"] = {"e2": count("e2_seeds20"), "e1_success": count("e1_adam"), "e1_failed_lwf": len(R["e1_lwf"]["failed_seeds"]),
                   "e3_gamma1": len(glob.glob(str(W / "e3_fisher" / "ep*_s*.json"))),
                   "e3_gamma0.5": len(glob.glob(str(W / "e3_fisher" / "g0.5_ep1_s*.json"))),
                   "lwf_probe_runs": 3, "lwf_recheck_runs": 1, "previous_total_in_thesis": 345}
rc = R["run_counts"]
rc["new_total"] = rc["e2"] + rc["e1_success"] + rc["e1_failed_lwf"] + rc["e3_gamma1"] + rc["e3_gamma0.5"] + rc["lwf_probe_runs"] + rc["lwf_recheck_runs"]
rc["grand_total"] = rc["new_total"] + rc["previous_total_in_thesis"]

json.dump(R, open(OUT / "review4_numbers.json", "w"), indent=1, ensure_ascii=False, default=float)
print("saved", OUT / "review4_numbers.json")
print(json.dumps(R["run_counts"], ensure_ascii=False))
