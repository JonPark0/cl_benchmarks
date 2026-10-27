"""DER++ alpha/beta sensitivity: Seq-MNIST buf-200 values from Buzzega et al. (2020) Table 9
(alpha=0.2, beta=1.0) under the thesis's unified base protocol (ep=1, buf=200, SGD lr=0.1, batch 64)."""
import sys, subprocess, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/home/yeochoon/Documents/grad_thesis/cl_benchmarks')
import run_review4 as R
EXP = 'e4_derpp_alpha02_beta10'
def cmd(seed):
    c = R.cmd_for('e2', 'derpp', seed)
    c = [x for x in c]
    i = c.index('--results_path'); c[i + 1] = f'{EXP}/derpp/s{seed}'
    a = c.index('--alpha'); c[a + 1] = '0.2'
    b = c.index('--beta'); c[b + 1] = '1.0'
    return c
def run(seed):
    d = R.BASE_PATH / EXP / 'derpp' / f's{seed}'
    if list(d.rglob('logs.pyd')):
        return seed, 'skip', 0
    t = time.time()
    p = subprocess.run(cmd(seed), cwd=R.MAMMOTH, capture_output=True, text=True)
    (d).mkdir(parents=True, exist_ok=True)
    (d / 'run_output.log').write_text(p.stdout + p.stderr)
    return seed, p.returncode, round(time.time() - t)
if __name__ == '__main__':
    seeds = [int(s) for s in sys.argv[2].split(',')] if len(sys.argv) > 2 else R.SEEDS20
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(' '.join(cmd(seeds[0])))
    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(run, seeds):
            print(r, flush=True)
