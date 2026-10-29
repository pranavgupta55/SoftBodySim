"""Checks for ospool_evolve.py, the OSPool job's entry script, run as a job would run it. From the repo root:

    .venv/bin/python3 tests/ospool_evolve_check.py

A temp dir stands in for the job's scratch dir: flat copies of every repo module chunk_evolve can import (the static
import closure of ospool_evolve.py, lazy imports included), maps/diamonds.json (as diamonds.json) and the base map
as base-v1.json; no maps/ or fonts/ folder.
gpu_backend.py, mlx.py and moderngl.py there are blockers: importing one leaves a marker file and raises ImportError.

1. A tiny real evolution (1 round, 1 variant / 1 seed per chunk, a 3 s spot check on 1 seed; OMP_NUM_THREADS=2 so
   2 workers; ~1-3 min): exit 0; outputs/map.json is a valid map of the base's tiling; layouts/r1.json; runs.jsonl
   has the start, round, job and spot records; summary.json has the schema's fields (status ok, 1 round, the
   spot check vs the base, env), workers from OMP_NUM_THREADS, round and spot seeds derived from --seed; nothing
   imported a blocker.
2. A missing base: exit 1, summary.json says error with the traceback.
3. --workers among the extra flags is refused.
Exit status 1 on any failure.
"""
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import numpy as np  # noqa: E402

FAILS = []
BASE = os.path.join(REPO, "maps", "penrose", "evolved 1 halloween.json")
BLOCKED = ("gpu_backend", "mlx", "moderngl")
SEED = 7
SUMMARY_KEYS = {"schema", "tag", "seed", "status", "exit_code", "error", "args", "workers", "started", "updated",
                "deadline", "wall_seconds", "base", "metric", "rounds_done", "rounds", "spot_seeds", "spots",
                "metric_start", "metric_end_est", "spot_last", "files", "env"}
ROUND_KEYS = {"round", "metric_in", "metric_out_est", "score_in", "worst", "worst_metric", "changed", "reverted",
              "dropped", "tiles", "jobs", "secs", "layout"}
ENV_KEYS = {"host", "cpu_model", "omp_num_threads", "os_cpu_count", "affinity", "python", "numpy", "numba", "pygame",
            "jit", "platform"}


def check(ok, what):
    print(f"{'ok  ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        FAILS.append(what)


def closure(start="ospool_evolve"):
    """The repo modules `start` can import (statically, imports inside functions too)."""
    local = {f[:-3] for f in os.listdir(REPO) if f.endswith(".py")}
    seen, todo = set(), [start]
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        with open(os.path.join(REPO, m + ".py")) as f:
            tree = ast.parse(f.read())
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                names = [n.module]
            else:
                continue
            todo += [x.split(".")[0] for x in names if x.split(".")[0] in local]
    return sorted(seen)


def scratch():
    """A flat job dir: the modules, the base map, the import blockers."""
    d = tempfile.mkdtemp(prefix="ospool_evolve_check_")
    for m in closure():
        if m not in BLOCKED:
            shutil.copy(os.path.join(REPO, m + ".py"), d)
    shutil.copy(BASE, os.path.join(d, "base-v1.json"))
    shutil.copy(os.path.join(REPO, "maps", "diamonds.json"), d)  # the tilings' palette (diamonds_map.MAP_PATH)
    for m in BLOCKED:
        with open(os.path.join(d, m + ".py"), "w") as f:
            f.write("import os\nopen(os.path.join(os.path.dirname(os.path.abspath(__file__)), "
                    f"'IMPORTED_{m}'), 'a').close()\nraise ImportError('{m} is blocked on the OSPool path')\n")
    return d


def run(d, *args, threads=2):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "SDL_VIDEODRIVER")}
    env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS="1", SDL_VIDEODRIVER="dummy")
    cmd = ["nice", "-n", "10", sys.executable, "-u", "ospool_evolve.py", *args]
    p = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True, timeout=1200)
    return p


def summary(d):
    with open(os.path.join(d, "outputs", "summary.json")) as f:
        return json.load(f)


def check_run():
    from diamonds_map import load_layout
    d = scratch()
    try:
        p = run(d, "t1", "--base", "base-v1.json", "--hours", "1", "--seed", str(SEED), "--rounds", "1",
                "--variants", "1", "--seeds", "1", "--recheck-seeds", "0", "--spot-every", "1", "--spot-seconds", "3",
                "--spot-seeds", "1")
        print(p.stdout[-2000:], p.stderr[-3000:] if p.returncode else "", sep="\n", flush=True)
        check(p.returncode == 0, f"tiny evolution exits 0 (got {p.returncode})")
        out = os.path.join(d, "outputs")
        keys, size = load_layout(os.path.join(out, "map.json"))
        base_keys, base_size = load_layout(BASE)
        check(len(keys) > 0.8 * len(base_keys) and tuple(size) == tuple(base_size),
              f"map.json is a map ({len(keys)} tiles vs base {len(base_keys)}, size {size})")
        with open(os.path.join(out, "map.json")) as f, open(BASE) as g:
            m, b = json.load(f), json.load(g)
        check(all(m.get(k) == b.get(k) for k in ("spawn", "tiling")), "map.json keeps the base's spawn and tiling")
        check(os.path.isfile(os.path.join(out, "layouts", "r1.json")), "layouts/r1.json")
        with open(os.path.join(out, "runs.jsonl")) as f:
            recs = [json.loads(line) for line in f]
        kinds = {r["type"] for r in recs}
        check({"start", "job", "round", "spot"} <= kinds, f"runs.jsonl record types {sorted(kinds)}")
        s = summary(d)
        check(SUMMARY_KEYS <= set(s), f"summary keys (missing {sorted(SUMMARY_KEYS - set(s))})")
        check(s["status"] == "ok" and s["exit_code"] == 0 and s["error"] is None, f"status {s['status']}")
        check(s["tag"] == "t1" and s["seed"] == SEED and s["workers"] == 2, "tag, seed, workers = OMP_NUM_THREADS")
        check(s["rounds_done"] == 1 and len(s["rounds"]) == 1 and ROUND_KEYS <= set(s["rounds"][0]),
              f"one round {s['rounds']}")
        r = s["rounds"][0]
        check(0 <= r["metric_in"] <= 1 and r["metric_out_est"] >= r["metric_in"] and r["jobs"] >= 32,
              f"round metric {r['metric_in']:.4f} -> {r['metric_out_est']:.4f}, {r['jobs']} tests")
        check(s["metric_start"] == r["metric_in"] and s["metric_end_est"] == r["metric_out_est"], "metric start / end")
        sp = s["spot_last"]
        check(len(s["spots"]) == 1 and sp == s["spots"][0] and sp["round"] == 1
              and {"score", "base_score"} <= set(sp), f"spot check vs the base {sp}")
        spot_seed = int(np.random.default_rng([SEED, 29]).integers(1, 1 << 30))
        round_seed = int(np.random.default_rng([SEED, 23]).integers(1, 1 << 30))
        check(s["spot_seeds"] == [spot_seed] and sp["seeds"] == [spot_seed], "spot seeds derived from --seed")
        rnd = [x for x in recs if x["type"] == "round"][0]
        check(rnd["seeds"] == [round_seed], "round seeds derived from --seed")
        check(ENV_KEYS <= set(s["env"]) and s["env"]["omp_num_threads"] == "2",
              f"env (missing {sorted(ENV_KEYS - set(s['env']))}) {s['env']}")
        check(s["files"] == {"map": "map.json", "layouts": ["r1.json"], "log": "runs.jsonl"}, f"files {s['files']}")
        check(s["base"] == {"file": "base-v1.json", "tiles": len(base_keys)}, f"base {s['base']}")
        check(s["args"]["hours"] == 1 and "--variants" in s["args"]["extra"], "args recorded")
        marks = [f for f in os.listdir(d) if f.startswith("IMPORTED_")]
        check(not marks, f"no gpu_backend / mlx / moderngl import ({marks})")
        check(not os.path.exists(os.path.join(d, "maps")), "no maps/ folder made")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def check_error():
    d = scratch()
    try:
        p = run(d, "bad", "--base", "nope.json", "--hours", "1", "--seed", "1", threads=1)
        check(p.returncode == 1, f"missing base exits 1 (got {p.returncode})")
        s = summary(d)
        check(s["status"] == "error" and s["exit_code"] == 1 and "nope.json" in (s["error"] or ""),
              f"summary records the error ({s['status']})")
        p = run(d, "bad", "--base", "base-v1.json", "--hours", "1", "--seed", "1", "--workers", "3", threads=1)
        check(p.returncode == 2 and "--workers" in p.stderr, "--workers refused")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    check_error()
    check_run()
    print("all passed" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
