import sys, subprocess, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/home/yeochoon/Documents/grad_thesis/cl_benchmarks')
import run_review4 as R
EXP = 'e4_derpp_alpha02_beta10_bufsweep'
SEEDS = [42, 123, 456, 789, 1024]
def cmd(buf, seed):
    c = list(R.cmd_for('e2', 'derpp', seed))
    i = c.index('--results_path'); c[i + 1] = f'{EXP}/buf{buf}/s{seed}'
    c[c.index('--alpha') + 1] = '0.2'; c[c.index('--beta') + 1] = '1.0'
    c[c.index('--buffer_size') + 1] = str(buf)
    return c
def run(job):
    buf, seed = job
    d = R.BASE_PATH / EXP / f'buf{buf}' / f's{seed}'
    if list(d.rglob('logs.pyd')): return job, 'skip', 0
    t = time.time(); p = subprocess.run(cmd(buf, seed), cwd=R.MAMMOTH, capture_output=True, text=True)
    d.mkdir(parents=True, exist_ok=True); (d / 'run_output.log').write_text(p.stdout + p.stderr)
    return job, p.returncode, round(time.time() - t)
if __name__ == '__main__':
    jobs = [(b, s) for b in (100, 500, 1000) for s in SEEDS]
    with ThreadPoolExecutor(3) as ex:
        for r in ex.map(run, jobs): print(r, flush=True)
