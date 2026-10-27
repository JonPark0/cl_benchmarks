"""Linux 재실험 결과의 표, 통계 검정, 그림을 만든다.

python report_linux.py [base|epoch|buffer|penalty|fisher|lwf|figs ...]
출력: 표준 출력(표와 검정 결과), 그림은 ../figures/*_linux.png
"""
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import analysis_linux as A

FIG = Path(__file__).resolve().parent.parent / 'figures'
SEEDS20 = [42, 123, 456, 789, 1024] + list(range(2001, 2016))
SEEDS10 = SEEDS20[:10]
NAME = {'sgd': 'SGD', 'joint': 'Joint', 'ewc_on': 'Online EWC', 'si': 'SI', 'er': 'ER',
        'derpp': 'DER++', 'lwf': 'LwF', 'gem': 'GEM'}
CAT = {'sgd': 'Lower Bound', 'joint': 'Upper Bound', 'ewc_on': 'Regularization', 'si': 'Regularization',
       'er': 'Replay', 'derpp': 'Replay', 'lwf': 'Knowledge Distillation', 'gem': 'Optimization'}
CAT_COLOR = {'Upper Bound': '#2ca02c', 'Lower Bound': '#7f7f7f', 'Regularization': '#1f77b4',
             'Replay': '#d62728', 'Knowledge Distillation': '#9467bd', 'Optimization': '#8c564b'}
STYLE = {'joint': ('#2ca02c', 's'), 'sgd': ('#7f7f7f', 'x'), 'ewc_on': ('#1f77b4', 'o'),
         'si': ('#17becf', 'o'), 'er': ('#d62728', '^'), 'derpp': ('#ff7f0e', '^'),
         'lwf': ('#9467bd', 'D'), 'gem': ('#8c564b', 'v')}
plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 200, 'savefig.bbox': 'tight', 'font.size': 11,
                     'axes.grid': True, 'grid.alpha': 0.3, 'axes.spines.top': False,
                     'axes.spines.right': False})


def vals(runs, key='aa', seeds=None, setting='cil'):
    return np.array([runs[s][setting][key] for s in (seeds or sorted(runs)) if s in runs], dtype=float)


def ms(a, nd=2):
    return f'${a.mean():.{nd}f} \\pm {a.std(ddof=1):.{nd}f}$'


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def base():
    d = A.collect('base')
    print('n per model:', {m: len(r) for m, r in d.items()})
    ranked = sorted([m for m in d if m not in ('sgd', 'joint')], key=lambda m: -vals(d[m]).mean())
    print('\n% Table 4.1 rows (Joint top, SGD bottom)')
    for i, m in enumerate(['joint'] + ranked + ['sgd']):
        rk = '---' if m in ('sgd', 'joint') else str(ranked.index(m) + 1)
        bwt = '---' if m == 'joint' else ms(vals(d[m], 'bwt'))
        print(f"        {rk:5s} & {NAME[m]:10s} & {ms(vals(d[m]))} & {bwt} \\\\")
    print('\nTask-IL AA:', {NAME[m]: f"{vals(d[m], 'aa', setting='til').mean():.2f}" for m in d})
    pairs = list(zip(ranked, ranked[1:])) + [(ranked[-1], 'sgd')]
    common = sorted(set.intersection(*[set(d[m]) for m in d]))
    ts, ps = [], []
    for a, b in pairs:
        t, p = stats.ttest_rel(vals(d[a], seeds=common), vals(d[b], seeds=common))
        ts.append(t), ps.append(p)
    adj = holm(np.array(ps))
    print(f'\nadjacent paired t-tests (n={len(common)}), Holm-adjusted:')
    for (a, b), t, p, q in zip(pairs, ts, ps, adj):
        diff = vals(d[a], seeds=common).mean() - vals(d[b], seeds=common).mean()
        print(f'  {NAME[a]:10s} vs {NAME[b]:10s} diff={diff:+.2f} t({len(common)-1})={t:.2f} p={p:.3g} holm={q:.3g}')
    for a, b in [('si', 'sgd'), ('ewc_on', 'sgd'), ('lwf', 'sgd'), ('gem', 'er'), ('gem', 'derpp'), ('derpp', 'er')]:
        if a in d and b in d:
            t, p = stats.ttest_rel(vals(d[a], seeds=common), vals(d[b], seeds=common))
            print(f'  extra {NAME[a]} vs {NAME[b]}: t={t:.2f} p={p:.3g}')
    return d


def fig_base(d):
    order = ['joint'] + sorted([m for m in d if m not in ('sgd', 'joint')], key=lambda m: -vals(d[m]).mean()) + ['sgd']
    aa = [vals(d[m]).mean() for m in order]
    sd = [vals(d[m]).std(ddof=1) for m in order]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([NAME[m] for m in order], aa, yerr=sd, capsize=4, edgecolor='black', linewidth=0.6,
                  color=[CAT_COLOR[CAT[m]] for m in order])
    for b, v, s in zip(bars, aa, sd):
        ax.text(b.get_x() + b.get_width() / 2, v + s + 2.2, f'{v:.2f}', ha='center', fontsize=9)
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_ylim(0, 110)
    ax.set_title(f'Split-MNIST Class-IL (Adam, learning rate 0.001, 1 epoch per task, buffer size 200, {len(d["sgd"])} seeds)', fontsize=10)
    plt.xticks(rotation=25, ha='right')
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=k) for k, c in CAT_COLOR.items()],
              loc='upper right', fontsize=9, framealpha=0.9)
    fig.savefig(FIG / 'figure_category_bar_linux.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for m in [x for x in order if x != 'joint']:  # Joint의 BWT는 망각 지표가 아니므로 제외
        c, mk = STYLE[m]
        b, a = vals(d[m], 'bwt'), vals(d[m])
        ax.errorbar(b.mean(), a.mean(), xerr=b.std(ddof=1), yerr=a.std(ddof=1), color=c, marker=mk,
                    markersize=9, capsize=2.5, linewidth=1.1, label=NAME[m])
    low = [m for m in order if m != 'joint' and vals(d[m]).mean() < 30]
    if low:  # 하한선 부근 모델은 한 점으로 겹치므로 확대 삽입창에 다시 그린다
        ins = ax.inset_axes([0.25, 0.22, 0.42, 0.42])
        for m in low:
            c, mk = STYLE[m]
            b, a = vals(d[m], 'bwt'), vals(d[m])
            ins.errorbar(b.mean(), a.mean(), xerr=b.std(ddof=1), yerr=a.std(ddof=1), color=c, marker=mk,
                         markersize=7, capsize=2, linewidth=1)
            ins.annotate(NAME[m], (b.mean(), a.mean()), xytext=(6, 4), textcoords='offset points', fontsize=8)
        ins.set_title('Lower-bound cluster (zoom)', fontsize=9)
        ins.tick_params(labelsize=7)
        ins.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.indicate_inset_zoom(ins, edgecolor='#555555')
    ax.set_xlabel('Backward Transfer (BWT, %)')
    ax.set_ylabel('Average Accuracy (AA, %)')
    ax.set_title('BWT vs AA (Adam, learning rate 0.001, 1 epoch per task, buffer size 200)', fontsize=10)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9, borderaxespad=0)
    fig.savefig(FIG / 'figure_bwt_aa_scatter_linux.png')
    plt.close(fig)


def epoch():
    b, e = A.collect('base'), A.collect('epoch')
    rows = {}
    for m in NAME:
        cols = [vals(b[m], seeds=SEEDS10)] + [vals(e.get(f'{m}_ep{k}', {}), seeds=SEEDS10) for k in (5, 10, 20)]
        rows[m] = cols
        print(f"        {NAME[m]:10s} & " + ' & '.join(ms(c) if len(c) > 1 else '-' for c in cols) + ' \\\\'
              + f"   n={[len(c) for c in cols]}")
    return rows


def fig_epoch(rows):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    eps = np.array([1, 5, 10, 20])
    for m, cols in rows.items():
        if any(len(c) < 2 for c in cols):
            continue
        y = np.array([c.mean() for c in cols]); s = np.array([c.std(ddof=1) for c in cols])
        c, mk = STYLE[m]
        ax.plot(eps, y, color=c, marker=mk, markersize=7, linewidth=1.8, label=NAME[m])
        ax.fill_between(eps, y - s, y + s, color=c, alpha=0.15, linewidth=0)
    ax.set_xticks(eps); ax.set_xlabel('Number of Epochs per Task'); ax.set_ylabel('Average Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.set_title('Epoch Sweep (Adam, learning rate 0.001, buffer size 200, 10 seeds; shaded: $\\pm$std)', fontsize=10)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9, borderaxespad=0)
    fig.savefig(FIG / 'figure_epoch_sweep_linux.png')
    plt.close(fig)


def buffer():
    b, f = A.collect('base'), A.collect('buffer')
    rows = {}
    for m in ('er', 'derpp', 'gem'):
        cols = [vals(f.get(f'{m}_buf100', {}), seeds=SEEDS10), vals(b[m], seeds=SEEDS10),
                vals(f.get(f'{m}_buf500', {}), seeds=SEEDS10), vals(f.get(f'{m}_buf1000', {}), seeds=SEEDS10)]
        rows[m] = cols
        print(f"        {NAME[m]:6s} & " + ' & '.join(ms(c) if len(c) > 1 else '-' for c in cols) + ' \\\\'
              + f"   n={[len(c) for c in cols]}")
    for bs, i in ((100, 0), (200, 1), (500, 2), (1000, 3)):
        for x, y in (('derpp', 'er'), ('gem', 'er'), ('gem', 'derpp')):
            if len(rows[x][i]) == len(rows[y][i]) > 1:
                t, p = stats.ttest_rel(rows[x][i], rows[y][i])
                print(f'  buf={bs} {NAME[x]}-{NAME[y]} diff={rows[x][i].mean()-rows[y][i].mean():+.2f} t={t:.2f} p={p:.3g}')
    return rows


def fig_buffer(rows, joint):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bs = [100, 200, 500, 1000]
    for m, cols in rows.items():
        if any(len(c) < 2 for c in cols):
            continue
        c, mk = STYLE[m]
        ax.errorbar(bs, [x.mean() for x in cols], yerr=[x.std(ddof=1) for x in cols], color=c, marker=mk,
                    markersize=8, linewidth=1.8, capsize=2.5, label=NAME[m])
    ax.axhline(joint, color='#2ca02c', linestyle=':', linewidth=1.2, label=f'Joint (Upper Bound, {joint:.2f})')
    ax.set_xscale('log'); ax.set_xticks(bs); ax.set_xticklabels([str(x) for x in bs])
    ax.set_xlabel('Memory Buffer Size (log scale)'); ax.set_ylabel('Average Accuracy (%)')
    ax.set_title('Buffer Size Sweep (Adam, learning rate 0.001, 1 epoch per task, 10 seeds)', fontsize=10)
    ax.legend(loc='lower right', fontsize=9)
    fig.savefig(FIG / 'figure_buffer_sweep_linux.png')
    plt.close(fig)


def penalty():
    b, p = A.collect('base'), A.collect('penalty')
    for m, key, grid in (('ewc_on', 'lambda', ('0.9', '9', '90', '900', '9000', '90000')),
                         ('si', 'c', ('0.01', '0.1', '1.0', '10', '100', '1000'))):
        for v in grid:
            r = b[m] if v in ('90', '1.0') else p.get(f'{m}_{key}{v}', {})
            a, w = vals(r, seeds=SEEDS10), vals(r, 'bwt', seeds=SEEDS10)
            nan = sum(1 for s in SEEDS10 if (A.L / 'penalty' / f'{m}_{key}{v}' / f's{s}').exists()
                      and s not in r) if v not in ('90', '1.0') else 0
            print(f'  {NAME[m]} {key}={v:6s} n={len(a):2d} fail={nan} AA {ms(a) if len(a)>1 else "-"} '
                  f'BWT {ms(w) if len(w)>1 else "-"} TIL {vals(r, setting="til", seeds=SEEDS10).mean() if len(r) else float("nan"):.2f}')


def fisher():
    for tag in ('ep1', 'ep20', 'ep1_gamma0.5'):
        fs = sorted((A.L / 'fisher' / tag).glob('s*.json'))
        if not fs:
            continue
        recs = [r for r in (json.loads(f.read_text())['records'] for f in fs) if len(r) == 5]  # 끝난 시드만
        print(f'== fisher {tag} n={len(recs)}')
        for t in range(1, 5):
            out = {}
            for kind in ('task', 'accum'):
                g = {'prev': [], 'cur': [], 'fut': [], 'ratio': []}
                for r in recs:
                    row = np.array(r[t][f'classifier_row_fisher_{kind}'])
                    prev, cur, fut = row[:2 * t], row[2 * t:2 * t + 2], row[2 * t + 2:]
                    g['prev'].append(prev.mean()); g['cur'].append(cur.mean())
                    g['fut'].append(fut.mean() if len(fut) else np.nan)
                    g['ratio'].append(prev.mean() / cur.mean())
                out[kind] = {k: np.nanmean(v) for k, v in g.items()}
            print(f"  task{t+1}: task prev={out['task']['prev']:.1e} cur={out['task']['cur']:.1e} fut={out['task']['fut']:.1e}"
                  f" | accum prev={out['accum']['prev']:.1e} cur={out['accum']['cur']:.1e} fut={out['accum']['fut']:.1e}"
                  f" ratio={out['accum']['ratio']:.2f}")
        aa = [A.read(d)['aa'] for d in sorted((A.L / 'fisher' / tag).glob('s*')) if d.is_dir() and A.read(d)]
        if aa:
            print(f'  AA {np.mean(aa):.2f} ± {np.std(aa, ddof=1):.2f} (n={len(aa)})')


def lwf(exp='lwf'):
    root = A.L / exp
    if not root.exists():
        return
    for tag in sorted(p.name for p in root.iterdir() if p.is_dir()):
        aa, nan, firsts, minp, gap = [], 0, [], [], []
        for sd in sorted((root / tag).glob('s*')):
            if not (sd / 'run_output.log').exists():  # 아직 실행 중
                continue
            r = A.read(sd)
            pj = sd / 'probe.json'
            pr = json.loads(pj.read_text()) if pj.exists() else None
            if r:
                aa.append(r['aa'])
            else:
                nan += 1
            if pr:
                if pr['first_bad']:
                    firsts.append((pr['first_bad']['task'], pr['first_bad']['step']))
                tr = [x for x in pr['trace'] if 'min_p' in x]
                if tr:
                    minp.append(min(x['min_p'] for x in tr)); gap.append(max(x['max_gap'] for x in tr))
        a = np.array(aa)
        print(f"  {tag:22s} ok={len(aa):2d} diverged={nan:2d} AA {ms(a) if len(a)>1 else (f'{a[0]:.2f}' if len(a) else '-'):>18s}"
              f" min_p={min(minp) if minp else float('nan'):.1e} max_gap={max(gap) if gap else float('nan'):.1f}"
              f" first_bad={firsts[:3]}")


def lwf_table(exp='lwf'):
    """조건별 (성공 시드 수, 발산 시드 수, AA 평균, 최소 확률, 최대 로짓 간격)."""
    out = {}
    root = A.L / exp
    for tag in sorted(q.name for q in root.iterdir() if q.is_dir()):
        aa, div, minp, gap, first = [], 0, [], [], []
        for sd in sorted((root / tag).glob('s*')):
            pj = sd / 'probe.json'
            if not pj.exists() or not (sd / 'run_output.log').exists():  # 실행 중이면 건너뜀
                continue
            pr = json.loads(pj.read_text())
            r = A.read(sd)
            if r:
                aa.append(r['aa'])
            else:
                div += 1
            if pr['first_bad']:
                first.append(pr['first_bad'])
            tr = [x for x in pr['trace'] if 'min_p' in x]
            if tr:
                minp.append(min(x['min_p'] for x in tr)); gap.append(max(x['max_gap'] for x in tr))
        out[tag] = dict(ok=len(aa), div=div, aa=np.array(aa), minp=min(minp) if minp else np.nan,
                        gap=max(gap) if gap else np.nan, first=first)
    return out


def fig_lwf(exp='lwf'):
    t = lwf_table(exp)
    alphas, temps, lrs = ('0.5', '1.0', '3.0'), ('1.0', '2.0', '4.0'), ('0.0001', '0.001', '0.003')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, a in zip(axes, alphas):
        m = np.array([[t.get(f'a{a}_t{T}_lr{lr}', {}).get('div', np.nan) for lr in lrs] for T in temps], dtype=float)
        im = ax.imshow(m, cmap='Reds', vmin=0, vmax=10)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, '' if np.isnan(m[i, j]) else f'{int(m[i, j])}', ha='center', va='center', fontsize=11)
        ax.set_xticks(range(3)); ax.set_xticklabels(lrs); ax.set_yticks(range(3)); ax.set_yticklabels(temps)
        ax.set_xlabel('learning rate'); ax.set_title(f'alpha = {a}'); ax.grid(False)
    axes[0].set_ylabel('softmax_temp T')
    fig.colorbar(im, ax=axes, shrink=0.8, label='diverged seeds (of 10)')
    fig.savefig(FIG / 'figure_lwf_divergence_linux.png')
    plt.close(fig)


def fig_lwf_trace(tag='vdv_a1.0_t2.0', seed=1024):
    """발산한 실행(원본 계산)과 같은 시드의 수치 안정 계산을 단계별로 비교한다."""
    runs = {}
    for exp, lab in (('lwf', 'Mammoth LwF (probability-power form)'), ('lwf_stable', 'log-softmax form (same seed)')):
        pj = A.L / exp / tag / f's{seed}' / 'probe.json'
        if pj.exists():
            runs[lab] = json.loads(pj.read_text())
    if not runs:
        return
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    for (lab, pr), c in zip(runs.items(), ('#d62728', '#1f77b4')):
        tr = [x for x in pr['trace'] if 'min_p' in x]
        st = np.array([x['step'] for x in tr])
        mp = np.array([max(x['min_p'], 1e-46) for x in tr])
        gp = np.array([x['max_gap'] for x in tr])
        z = 3 if pr['first_bad'] else 2  # 발산한 원본 곡선을 위에 그린다
        a1.semilogy(st, mp, color=c, linewidth=0.6, label=lab, alpha=0.85, zorder=z)
        a2.plot(st, gp, color=c, linewidth=0.6, label=lab, alpha=0.85, zorder=z)
        if pr['first_bad']:
            fb = pr['first_bad']['step']
            for ax in (a1, a2):
                ax.axvline(fb, color=c, linestyle='--', linewidth=1)
            a2.annotate(f'NaN/inf at step {fb}', (fb, 140), xytext=(-150, 0), textcoords='offset points',
                        fontsize=8, color=c, arrowprops=dict(arrowstyle='->', color=c))
    a1.axhline(np.finfo(np.float32).tiny, color='gray', linestyle=':', linewidth=1)
    a1.text(st[0], np.finfo(np.float32).tiny * 3, 'smallest normal float32', fontsize=8, color='gray')
    a1.text(st[0], 3e-46, 'exact 0 (plotted at 1e-46)', fontsize=8, color='gray', va='bottom')
    a1.set_ylabel('min softmax prob.\n(previous classes)')
    a1.set_title(f'LwF under the van de Ven protocol ($\\lambda$ = {tag.split("_a")[1].split("_t")[0]}, T = {float(tag.split("_t")[1]):g}, seed {seed})')
    a1.legend(fontsize=8, loc='upper right')
    a2.set_ylabel('max logit gap\n(previous classes)')
    a2.set_xlabel('update step (all tasks)')
    fig.savefig(FIG / 'figure_lwf_trace_linux.png')
    plt.close(fig)


if __name__ == '__main__':
    what = sys.argv[1:] or ['base']
    for w in what:
        print(f'##### {w}')
        if w == 'base':
            fig_base(base())
        elif w == 'epoch':
            fig_epoch(epoch())
        elif w == 'buffer':
            fig_buffer(buffer(), vals(A.collect('base')['joint'], seeds=SEEDS10).mean())
        elif w == 'penalty':
            penalty()
        elif w == 'fisher':
            fisher()
        elif w in ('lwf', 'lwf_stable'):
            lwf(w)
            if w == 'lwf_stable':
                fig_lwf_trace()
