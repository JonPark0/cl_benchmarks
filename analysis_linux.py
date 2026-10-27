"""Linux 재실험(raw_data/linux) 집계.

python analysis_linux.py [exp ...]   # 기본: 모든 실험의 조건별 AA, BWT, Task-IL AA 요약
"""
import re
import sys
from pathlib import Path

import numpy as np

L = Path(__file__).parent / 'raw_data' / 'linux'
NUM = r"(?:np\.float64\()?([-0-9.e]+|nan)"


def read(run_dir, setting='class-il'):
    """한 실행 폴더에서 마지막 작업 기준 AA, BWT, 작업별 정확도를 읽는다."""
    fs = list((run_dir / setting).rglob('logs.pyd'))
    if not fs:
        return None
    line = [l for l in fs[0].read_text().splitlines() if l.strip()][-1]
    get = lambda k: (m := re.search(rf"'{k}': {NUM}", line)) and float(m.group(1))
    t = max(int(k) for k in re.findall(r"'accmean_task(\d+)'", line))
    return {'aa': get(f'accmean_task{t}'), 'bwt': get('backward_transfer'),
            'acc': [get(f'accuracy_{i}_task{t}') for i in range(1, t + 1)]}


def collect(exp):
    """{tag: {seed: {'cil': ..., 'til': ...}}}"""
    out = {}
    for tag in sorted(p.name for p in (L / exp).iterdir() if p.is_dir()):
        runs = {}
        for sd in (L / exp / tag).glob('s*'):
            c = read(sd)
            if c:
                runs[int(sd.name[1:])] = {'cil': c, 'til': read(sd, 'task-il')}
        out[tag] = runs
    return out


def ms(a):
    a = np.array([x for x in a if x is not None], dtype=float)
    if len(a) == 0:
        return '-'
    return f'{a.mean():.2f} ± {a.std(ddof=1):.2f}' if len(a) > 1 else f'{a[0]:.2f}'


def main():
    for exp in sys.argv[1:] or sorted(p.name for p in L.iterdir() if p.is_dir()):
        print(f'== {exp}')
        for tag, runs in collect(exp).items():
            r = list(runs.values())
            print(f"{tag:22s} n={len(r):2d}  AA {ms([x['cil']['aa'] for x in r]):>15s}"
                  f"  BWT {ms([x['cil']['bwt'] for x in r]):>16s}"
                  f"  TIL {ms([x['til']['aa'] for x in r if x['til']]):>15s}")


if __name__ == '__main__':
    main()
