"""review4 대응 실험 실행기 (WSL2). E1: 5축 정렬(Adam), E2: 기본 실험 20시드.

사용법:
    python run_review4.py e1 --seeds 42 --models agem      # 시운전
    python run_review4.py e1 --workers 3
    python run_review4.py e2 --workers 3
결과는 raw_data/wsl/<exp>/<model>/s<seed>/ 아래 logs.pyd 로 저장된다.
"""
import argparse, subprocess, sys, time, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path(__file__).parent.resolve()
PY = BASE / ".venv" / "bin" / "python"
MAMMOTH = BASE / "mammoth"
BASE_PATH = BASE / "raw_data" / "wsl"

SEEDS20 = [42, 123, 456, 789, 1024] + list(range(2001, 2016))

# 모델별 고유 인자 (기존 run_experiments.py와 동일)
MODEL_ARGS = {
    "sgd": [], "joint": [],
    "ewc-on": ["--e_lambda", "0.7", "--gamma", "1.0"],
    "si": ["--c", "0.5", "--xi", "0.001"],
    "er": [], "derpp": ["--alpha", "0.1", "--beta", "0.5"],
    "lwf": ["--alpha", "1.0", "--softmax_temp", "2.0"],
    "agem": [], "agem_fixed": [],
}
BUFFER_MODELS = {"er", "derpp", "agem", "agem_fixed"}


def cmd_for(exp, model, seed):
    c = [str(PY), str(MAMMOTH / "main.py"), "--model", model, "--dataset", "seq-mnist",
         "--seed", str(seed), "--device", "0", "--base_path", str(BASE_PATH) + "/",
         "--results_path", f"{exp}/{model}/s{seed}", "--enable_other_metrics", "1",
         "--non_verbose", "1"]
    if exp.startswith("e1"):
        # van de Ven (2022) 프로토콜 5축 정렬: 은닉 400, 2000 iter, 배치 128, 버퍼 1000,
        # 클래스 무작위 배정, Adam lr=0.001
        c += ["--mlp_hidden_size", "400", "--fitting_mode", "iters", "--n_iters", "2000",
              "--batch_size", "128", "--permute_classes", "1",
              "--optimizer", "adam", "--lr", "0.001"]
        if model in BUFFER_MODELS:
            c += ["--buffer_size", "1000", "--minibatch_size", "128"]
        if model == "joint":
            c += ["--n_epochs", "21"]  # joint.py는 fitting_mode를 무시하므로 5x2000iter = 약 21 epoch
    else:  # e2: 기본 실험 (ep=1, buf=200, SGD lr=0.1)
        c += ["--n_epochs", "1", "--lr", "0.1"]
        if model in BUFFER_MODELS:
            c += ["--buffer_size", "200"]
            if model in ("er", "agem", "agem_fixed"):
                c += ["--minibatch_size", "64"]
    return c + MODEL_ARGS[model]


def run(job):
    exp, model, seed = job
    d = BASE_PATH / exp / model / f"s{seed}"
    if list(d.rglob("logs.pyd")):
        return job, "skip", 0.0
    t = time.time()
    p = subprocess.run(cmd_for(exp, model, seed), cwd=MAMMOTH, capture_output=True, text=True,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    st = "ok" if p.returncode == 0 else "FAIL"
    if st == "FAIL":
        (BASE_PATH / exp).mkdir(parents=True, exist_ok=True)
        with open(BASE_PATH / exp / "failures.log", "a") as f:
            f.write(f"{model} s{seed}\n{p.stderr[-1500:]}\n")
    return job, st, time.time() - t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["e1", "e2"])
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    exp = {"e1": "e1_adam", "e2": "e2_seeds20"}[a.exp] + a.tag
    models = a.models or (["sgd", "joint", "ewc-on", "si", "lwf", "er", "agem", "agem_fixed"] if a.exp == "e1"
                          else ["sgd", "joint", "ewc-on", "si", "er", "derpp", "lwf", "agem"])
    seeds = a.seeds or (SEEDS20[:5] if a.exp == "e1" else SEEDS20)
    jobs = [(exp, m, s) for m in models for s in seeds]
    with ThreadPoolExecutor(a.workers) as ex:
        for job, st, el in ex.map(run, jobs):
            print(f"[{time.strftime('%H:%M:%S')}] {job[1]} s{job[2]} {st} {el:.0f}s", flush=True)
