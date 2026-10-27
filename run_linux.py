"""교정 1 이후 재실험 실행기 (Linux 단일 환경).

설계 (2026-10-03 지도교수 면담 결정):
- 실행 환경: Linux(WSL2 Ubuntu 24.04) 하나로 통일한다.
- 비교 모델 8개: SGD(하한선), Joint(상한선), Online EWC, SI, ER, DER++, LwF, GEM. A-GEM은 제외한다.
- 공통 조건: Adam(lr=0.001, Kingma & Ba의 기본값), 배치 64(Mammoth seq-mnist 기본값), 에폭 1, 버퍼 200.
- 모델 고유 인자: Mammoth에 기본값이 있으면 그 값(LwF alpha=0.5, softmax_temp=2; GEM gamma=0.5),
  기본값이 없는 필수 인자는 Buzzega et al.(2020) 표 9의 Seq-MNIST 값
  (Online EWC e_lambda=90, gamma=1.0; SI c=1.0, xi=0.9; DER++ alpha=0.2, beta=1.0).
- Mammoth 0.2.2 (커밋 2025-11-22).

사용법:
    python run_linux.py base --seeds 42 --workers 1          # 시운전
    python run_linux.py base                                  # 기본 실험 8모델 x 20시드
    python run_linux.py epoch | buffer | penalty | fisher
    python run_linux.py lwf                                   # LwF 심층 분석(계측 모델)
    python run_linux.py lwf_stable --tags vdv                 # 수치 안정 변형으로 같은 조건 재실행
결과는 raw_data/linux/<exp>/<tag>/s<seed>/ 아래에 저장된다.
"""
import argparse
import itertools
import json
import platform
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path(__file__).parent.resolve()
PY = BASE / '.venv' / 'bin' / 'python'
MAMMOTH = BASE / 'mammoth'
OUT = BASE / 'raw_data' / 'linux'

SEEDS20 = [42, 123, 456, 789, 1024] + list(range(2001, 2016))
SEEDS10 = SEEDS20[:10]

COMMON = ['--dataset', 'seq-mnist', '--optimizer', 'adam', '--lr', '0.001',
          '--enable_other_metrics', '1', '--non_verbose', '1', '--device', '0']
MODEL_ARGS = {
    'sgd': [], 'joint': [],
    'ewc_on': ['--e_lambda', '90', '--gamma', '1.0'],
    'si': ['--c', '1.0', '--xi', '0.9'],
    'er': [],
    'derpp': ['--alpha', '0.2', '--beta', '1.0'],
    'lwf': [],   # Mammoth 기본값 alpha=0.5, softmax_temp=2
    'gem': [],   # Mammoth 기본값 gamma=0.5
}
MODELS = list(MODEL_ARGS)
BUFFER_MODELS = {'er', 'derpp', 'gem'}


def job(exp, tag, model, seed, n_epochs=1, buffer=200, extra=()):
    """(exp, tag, seed, argv) 하나를 만든다. extra는 MODEL_ARGS와 COMMON의 값을 덮어쓴다."""
    args = dict(zip(COMMON[::2], COMMON[1::2]))
    args.update(zip(MODEL_ARGS[model][::2], MODEL_ARGS[model][1::2]))
    args.update(dict(zip(extra[::2], extra[1::2])))
    argv = [str(PY), str(MAMMOTH / 'main.py'), '--model', model, '--seed', str(seed),
            '--n_epochs', str(n_epochs), '--base_path', str(OUT) + '/',
            '--results_path', f'{exp}/{tag}/s{seed}']
    if model in BUFFER_MODELS:
        argv += ['--buffer_size', str(buffer), '--minibatch_size', '64']
    for k, v in args.items():
        argv += [k, str(v)]
    return exp, tag, seed, argv


def plan(exp, seeds=None):
    if exp == 'base':
        return [job('base', m, m, s) for m in MODELS for s in (seeds or SEEDS20)]
    if exp == 'epoch':      # ep=1은 base의 처음 10시드를 쓴다
        return [job('epoch', f'{m}_ep{e}', m, s, n_epochs=e)
                for m in MODELS for e in (5, 10, 20) for s in (seeds or SEEDS10)]
    if exp == 'buffer':     # buf=200은 base의 처음 10시드를 쓴다
        return [job('buffer', f'{m}_buf{b}', m, s, buffer=b)
                for m in ('er', 'derpp', 'gem') for b in (100, 500, 1000) for s in (seeds or SEEDS10)]
    if exp == 'penalty':    # 기준값의 0.01, 0.1, 10, 100, 1000배
        jobs = [job('penalty', f'ewc_on_lambda{v}', 'ewc_on', s, extra=('--e_lambda', v))
                for v in ('0.9', '9', '900', '9000', '90000') for s in (seeds or SEEDS10)]
        jobs += [job('penalty', f'si_c{v}', 'si', s, extra=('--c', v))
                 for v in ('0.01', '0.1', '10', '100', '1000') for s in (seeds or SEEDS10)]
        return jobs
    if exp == 'fisher':     # 출력 노드별 Fisher: ep1, ep20, gamma=0.5
        jobs = []
        for tag, ep, gamma in (('ep1', 1, '1.0'), ('ep20', 20, '1.0'), ('ep1_gamma0.5', 1, '0.5')):
            for s in (seeds or SEEDS10[:5]):
                e, t, sd, argv = job('fisher', tag, 'ewc_on', s, n_epochs=ep, extra=('--gamma', gamma))
                argv[argv.index('ewc_on')] = 'ewc_on_fisher'
                argv += ['--fisher_out', str(OUT / 'fisher' / tag / f's{s}.json')]
                jobs.append((e, t, sd, argv))
        return jobs
    if exp in ('lwf', 'lwf_stable'):
        # alpha x T x lr 격자(본 연구 프로토콜)와 van de Ven(2022) 5축 정렬 프로토콜.
        # 계측 모델 lwf_probe는 lwf_stable=0일 때 원본 LwF와 같은 계산을 한다.
        stable = '1' if exp == 'lwf_stable' else '0'
        cells = [(f'a{a}_t{t}_lr{lr}', ('--alpha', a, '--softmax_temp', t, '--lr', lr))
                 for a, t, lr in itertools.product(('0.5', '1.0', '3.0'), ('1.0', '2.0', '4.0'),
                                                   ('0.0001', '0.001', '0.003'))]
        vdv = ('--mlp_hidden_size', '400', '--fitting_mode', 'iters', '--n_iters', '2000',
               '--batch_size', '128', '--permute_classes', '1')
        cells += [(f'vdv_a{a}_t{t}', ('--alpha', a, '--softmax_temp', t) + vdv)
                  for a, t in (('1.0', '2.0'), ('1.0', '1.0'), ('0.5', '2.0'))]
        jobs = []
        for tag, extra in cells:
            for s in (seeds or SEEDS10):
                e, t_, sd, argv = job(exp, tag, 'lwf', s, extra=extra + ('--lwf_stable', stable))
                argv[argv.index('lwf')] = 'lwf_probe'
                argv += ['--probe_out', str(OUT / exp / tag / f's{s}' / 'probe.json')]
                jobs.append((e, t_, sd, argv))
        return jobs
    if exp in ('lwfep', 'lwfep_stable'):
        # 에폭 탐색에서 발산한 LwF(Mammoth 기본값, n_epochs=10/20)를 계측 모델로 재실행한다.
        stable = '1' if exp == 'lwfep_stable' else '0'
        jobs = []
        for ep in (10, 20):
            for s in (seeds or SEEDS10):
                e, t_, sd, argv = job(exp, f'ep{ep}', 'lwf', s, n_epochs=ep, extra=('--lwf_stable', stable))
                argv[argv.index('lwf')] = 'lwf_probe'
                argv += ['--probe_out', str(OUT / exp / f'ep{ep}' / f's{s}' / 'probe.json')]
                jobs.append((e, t_, sd, argv))
        return jobs
    raise SystemExit(f'unknown experiment {exp}')


def run(j):
    exp, tag, seed, argv = j
    d = OUT / exp / tag / f's{seed}'
    if list(d.rglob('logs.pyd')):
        return exp, tag, seed, 'skip', 0
    d.mkdir(parents=True, exist_ok=True)
    t = time.time()
    p = subprocess.run(argv, cwd=MAMMOTH, capture_output=True, text=True)
    (d / 'run_output.log').write_text(' '.join(argv) + '\n\n' + p.stdout + p.stderr)
    return exp, tag, seed, p.returncode, round(time.time() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exp')
    ap.add_argument('--seeds', type=lambda s: [int(x) for x in s.split(',')])
    ap.add_argument('--models', type=lambda s: s.split(','))
    ap.add_argument('--tags', type=lambda s: s.split(','), help='태그에 이 문자열이 들어간 조건만 실행')
    ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()
    if a.exp == 'fisher':
        shutil.copy(BASE / 'models_patch' / 'ewc_on_fisher.py', MAMMOTH / 'models' / 'ewc_on_fisher.py')
    if a.exp.startswith('lwf'):  # lwf, lwf_stable, lwfep, lwfep_stable
        shutil.copy(BASE / 'models_patch' / 'lwf_probe.py', MAMMOTH / 'models' / 'lwf_probe.py')
    jobs = plan(a.exp, a.seeds)
    if a.models:
        jobs = [j for j in jobs if any(j[3][j[3].index('--model') + 1].startswith(m) for m in a.models)]
    if a.tags:
        jobs = [j for j in jobs if any(t in j[1] for t in a.tags)]
    OUT.mkdir(parents=True, exist_ok=True)
    import torch
    (OUT / 'environment.json').write_text(json.dumps({
        'platform': platform.platform(), 'python': platform.python_version(),
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'mammoth': '0.2.2 (commit ea967c3, 2025-11-22)'}, indent=1))
    print(f'{a.exp}: {len(jobs)} jobs, {a.workers} workers', flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(run, jobs):
            print(*r, flush=True)


if __name__ == '__main__':
    main()
