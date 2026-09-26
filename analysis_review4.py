"""review4 실험(E1, E2, E3) 집계. scipy/numpy 필요.

python analysis_review4.py > ../review/review4_results_raw.txt  (numpy, scipy 필요: .venv에는 없으므로 별도 venv에 설치)
"""
import re, json, glob, itertools
from pathlib import Path
import numpy as np
from scipy import stats

W = Path(__file__).parent / "raw_data" / "wsl"


def read(exp, model, seed):
    fs = list((W / exp / model / f"s{seed}" / "class-il").rglob("logs.pyd"))
    if not fs:
        return None
    line = [l for l in fs[0].read_text().splitlines() if l.strip()][-1]
    aa = re.search(r"'accmean_task5': (?:np\.float64\()?([-0-9.e]+|nan)", line)
    bwt = re.search(r"'backward_transfer': (?:np\.float64\()?([-0-9.e]+|nan)", line)
    return (float(aa.group(1)), float(bwt.group(1)) if bwt else float('nan')) if aa else None


def collect(exp, models):
    out = {}
    for m in models:
        d = {}
        for sd in (W / exp / m).glob("s*"):
            r = read(exp, m, sd.name[1:])
            if r: d[int(sd.name[1:])] = r
        out[m] = d
    return out


def fmt(a):
    a = np.array(a); return f"{a.mean():.2f} ± {a.std(ddof=1):.2f} (n={len(a)})" if len(a) > 1 else f"{a[0]:.2f}"


print("## E1: Adam(lr=0.001) 5축 정렬, AA %")
e1 = collect("e1_adam", ["sgd", "joint", "ewc-on", "si", "lwf", "er", "agem", "agem_fixed"])
for m, d in e1.items():
    if d: print(f"{m:12s} {fmt([v[0] for v in d.values()])}   BWT {fmt([v[1] for v in d.values()])}")
if e1["agem"] and e1["agem_fixed"]:
    s = sorted(set(e1["agem"]) & set(e1["agem_fixed"]))
    t = stats.ttest_rel([e1["agem"][k][0] for k in s], [e1["agem_fixed"][k][0] for k in s])
    print(f"agem vs agem_fixed paired t({len(s)-1})={t.statistic:.2f} p={t.pvalue:.3f}")

print("\n## E2: 기본 실험 20시드 (WSL2), AA %")
mods = ["joint", "derpp", "er", "agem", "si", "ewc-on", "lwf", "sgd"]
e2 = collect("e2_seeds20", mods)
for m in mods:
    d = e2[m]
    if d:
        a = [v[0] for v in d.values()]
        print(f"{m:8s} {fmt(a)}   BWT {fmt([v[1] for v in d.values()])}")
ok = [m for m in mods if len(e2[m]) >= 2]
order = sorted([m for m in ok if m not in ("joint", "sgd")], key=lambda m: -np.mean([v[0] for v in e2[m].values()]))
print("순위(상/하한 제외):", order)
pairs = list(zip(order, order[1:])) + [(order[-1], "sgd")]
rows = []
for a, b in pairs:
    if b not in e2 or not e2[b]: continue
    s = sorted(set(e2[a]) & set(e2[b]))
    t = stats.ttest_rel([e2[a][k][0] for k in s], [e2[b][k][0] for k in s])
    rows.append((a, b, len(s), t.statistic, t.pvalue))
# Holm
ps = np.array([r[4] for r in rows]); idx = np.argsort(ps); adj = np.empty(len(ps)); run = 0
for rank, i in enumerate(idx):
    run = max(run, (len(ps) - rank) * ps[i]); adj[i] = min(1, run)
for (a, b, n, t, p), pa in zip(rows, adj):
    print(f"{a:7s} vs {b:7s} n={n} t({n-1})={t:7.2f} p={p:.4f} Holm={pa:.4f}")
if e2["si"]:
    v = np.array([x[0] for x in e2["si"].values()])
    print(f"SI 20시드 std={v.std(ddof=1):.2f}; 문헌 환산 2.55 대비 F={(v.std(ddof=1)/2.55)**2:.2f}")
print("\n시드별 SI:", sorted(e2['si'].items()) if e2['si'] else None)

print("\n## E3: Online EWC 분류층 출력 노드별 Fisher")
for ep in (1, 20):
    fs = sorted(glob.glob(str(W / "e3_fisher" / f"ep{ep}_s*.json")))
    recs = [json.load(open(f)) for f in fs]
    recs = [r for r in recs if len(r["records"]) == 5]
    if not recs: continue
    print(f"\nep={ep}, seeds={[r['seed'] for r in recs]}")
    for key in ("classifier_row_fisher_task", "classifier_row_fisher_accum"):
        print(f"[{key}] 작업 t 종료 시 노드 그룹별 평균 (시드 평균)")
        print("  t | 이전 클래스 | 현재 작업 | 미래 클래스 | 이전/현재")
        for t in range(5):
            prev, cur, fut = [], [], []
            for r in recs:
                v = np.array(r["records"][t][key])
                prev.append(v[:2 * t].mean() if t > 0 else np.nan)
                cur.append(v[2 * t:2 * t + 2].mean())
                fut.append(v[2 * t + 2:].mean() if t < 4 else np.nan)
            pm, cm, fm = np.nanmean(prev) if t > 0 else np.nan, np.mean(cur), np.nanmean(fut) if t < 4 else np.nan
            print(f"  {t+1} | {pm:.3e} | {cm:.3e} | {fm:.3e} | {pm/cm if t>0 else float('nan'):.3f}")

print("\n## E1 (20시드 확대: agem, agem_fixed, er) 문헌값 대비")
lit = {"agem": (65.10, 3.64, 20), "er": (88.79, 0.20, 20)}
for m in ("agem", "agem_fixed", "er"):
    a = np.array([v[0] for v in e1[m].values()])
    if len(a) >= 20:
        l = lit.get(m, lit["agem"])
        se = np.sqrt(a.var(ddof=1) / len(a) + l[1] ** 2)
        print(f"{m:10s} n={len(a)} {a.mean():.2f} ± {a.std(ddof=1):.2f}  s.e.m.={a.std(ddof=1)/np.sqrt(len(a)):.2f}  문헌 {l[0]}±{l[1]}(sem)  차이 {a.mean()-l[0]:+.2f}  t≈{(a.mean()-l[0])/se:.2f}")
s = sorted(set(e1["agem"]) & set(e1["agem_fixed"]))
if len(s) >= 20:
    t = stats.ttest_rel([e1["agem"][k][0] for k in s], [e1["agem_fixed"][k][0] for k in s])
    print(f"agem vs agem_fixed 20시드 paired t({len(s)-1})={t.statistic:.2f} p={t.pvalue:.4f}")

print("\n## E3 gamma=0.5 (ep=1)")
fs = sorted(glob.glob(str(W / "e3_fisher" / "g0.5_ep1_s*.json")))
recs = [json.load(open(f)) for f in fs]; recs = [r for r in recs if len(r["records"]) == 5]
print("seeds", [r["seed"] for r in recs])
for key in ("classifier_row_fisher_accum",):
    for t in range(5):
        prev, cur, fut = [], [], []
        for r in recs:
            v = np.array(r["records"][t][key])
            prev.append(v[:2 * t].mean() if t > 0 else np.nan); cur.append(v[2 * t:2 * t + 2].mean())
            fut.append(v[2 * t + 2:].mean() if t < 4 else np.nan)
        ratios = np.array(prev) / np.array(cur)
        print(f"  t={t+1} 이전 {np.nanmean(prev):.3e} 현재 {np.mean(cur):.3e} 미래 {np.nanmean(fut):.3e} 이전/현재 {np.nanmean(ratios):.3f}")
aas = []
for r in recs:
    x = read("e3_fisher/logs", f"g0.5_ep1_s{r['seed']}", "") if False else None
for sd in (42, 123, 456, 789, 1024):
    fs2 = list((W / "e3_fisher" / "logs" / f"g0.5_ep1_s{sd}" / "class-il").rglob("logs.pyd"))
    if fs2:
        line = [l for l in fs2[0].read_text().splitlines() if l.strip()][-1]
        m = re.search(r"'accmean_task5': (?:np\.float64\()?([-0-9.e]+)", line)
        if m: aas.append(float(m.group(1)))
print("AA (gamma=0.5):", fmt(aas) if aas else None)
