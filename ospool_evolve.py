"""One complete chunk_evolve map evolution as an OSPool (HTCondor) job: the job's entry script.

    python3 -u ospool_evolve.py TAG --base base-v1.json --hours H --seed S [chunk_evolve flags, e.g. --variants 6
                                --seeds 10 --seam-threshold 0.04 --metric free --spot-every 2 --rounds N]

Runs where the job lands: every repo .py it needs and the base map flat in the cwd (no maps/ folder), headless
(SDL_VIDEODRIVER=dummy; the CPU physics backend, nothing imports gpu_backend / mlx / moderngl). It sets, and so
rejects as extra flags:
  --workers   int(OMP_NUM_THREADS), else 1 (HTCondor sets it to request_cpus; os.cpu_count() is the whole machine)
  --deadline  now + H hours - PACK_MARGIN (time to pack and return ./outputs/), to the minute
  --log-dir   ./outputs
and --rounds defaults to ROUNDS (the deadline is what ends the run). Every other flag goes to chunk_evolve.main as is.
--seed is required: the round seeds, the mutations and the spot-check seeds all derive from it (chunk_evolve's own
default is the clock), so a job rerun after an eviction repeats the same search.

Writes into ./outputs/ (created; run.sh tars it, the only thing that comes back):
  map.json          the evolved map (a normal map file: the base's spawn, tiling, clusters and lighting), saved after
                    every round
  layouts/rN.json   the map after round N
  runs.jsonl        chunk_evolve's log: one record per chunk test ("job"), round ("round") and whole-map spot run
                    ("spot"); see chunk_evolve.evolve
  summary.json      written at the start and rewritten (atomically) after every round and spot check and at the end,
                    so a job killed mid-run still leaves a usable one. Schema (SCHEMA = 1):

    schema          1
    tag, seed       as given
    status          "running" (also what a killed job leaves) | "ok" | "error" | "terminated" (SIGTERM / SIGINT)
    exit_code       None while running, else the process's exit status (0 ok, 1 error, 143 terminated)
    error           None, or the traceback text
    args            {"hours", "base", "extra" (the pass-through flags), "argv" (chunk_evolve.main's full argv)}
    workers         processes used (OMP_NUM_THREADS)
    started, updated  UTC ISO times; deadline: the local "YYYY-MM-DD HH:MM" passed to chunk_evolve
    wall_seconds    since the start, at the last write
    base            {"file", "tiles"}
    metric          the chunk metric ("free" or "flow")
    rounds_done     rounds finished (and saved in map.json)
    rounds          per round: {"round", "metric_in" (mean chunk metric of the incumbent over the 16 grid chunks),
                    "metric_out_est" (the same after the accepted variants, estimated from their scores), "score_in"
                    (mean chunk score of the incumbent), "worst" (chunk id), "worst_metric", "changed" (chunks that took
                    a variant), "reverted" (picked but reverted at the seams), "dropped" (border tiles), "tiles",
                    "jobs" (chunk tests), "secs", "layout" (id)}
    spot_seeds      the whole-map spot-check seeds (from --seed)
    spots           per spot check: {"round", "seeds", "score" (mean whole-map score of the evolved map), "base_score"
                    (the base on the same seeds), and when present "unstuck" / "base_unstuck", "spread" /
                    "base_spread"}
    metric_start    rounds[0].metric_in (None before round 1 ends); metric_end_est: rounds[-1].metric_out_est
    spot_last       spots[-1] or None
    files           {"map", "layouts" [names], "log"}: paths relative to outputs/ (map is None before round 1)
    env             {"host", "cpu_model", "omp_num_threads", "os_cpu_count", "affinity", "python", "numpy", "numba",
                    "pygame", "jit" (numba kernels available), "platform"}

Exit status: 0 when the evolution finished (deadline or --rounds), 1 on any exception (traceback in summary.json
and stderr), 143 on SIGTERM / SIGINT. Run it from a script file only (the pool's spawn workers re-import it).
"""
import argparse
import json
import os
import platform
import shutil
import signal
import socket
import sys
import time
import traceback

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

SCHEMA = 1
OUT = "outputs"
PACK_MARGIN = 300.0  # s before the job's end kept for packing outputs/ (chunk_evolve keeps its own margin on top)
ROUNDS = 100000  # --rounds default: effectively unlimited, the deadline stops the run
OWN_FLAGS = ("--workers", "--deadline", "--log-dir")


class Terminated(BaseException):
    """SIGTERM / SIGINT (HTCondor's eviction, removal or hold): unwinds like KeyboardInterrupt."""


def _utc(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def env_info():
    import numpy
    out = {"host": socket.gethostname(), "cpu_model": cpu_model(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
           "os_cpu_count": os.cpu_count(),
           "affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
           "python": sys.version.split()[0], "numpy": numpy.__version__, "platform": platform.platform()}
    for mod in ("numba", "pygame"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # missing or broken: the run says so (numba missing = numpy physics, ~10x slower)
            out[mod] = None
    try:
        import solver_jit
        out["jit"] = bool(solver_jit.AVAILABLE)
    except Exception:
        out["jit"] = None
    return out


class Summary:
    """outputs/summary.json (schema in the module doc), refreshed from runs.jsonl's new lines on every write."""

    def __init__(self, out_dir, tag, seed, args, workers, deadline):
        self.dir = out_dir
        self.log_path = os.path.join(out_dir, "runs.jsonl")
        self.t0 = time.time()
        self.offset = 0  # runs.jsonl bytes read so far
        self.base_spot = []  # the base's spot results (run once, with the first spot check)
        self.spot_runs = {}  # round -> the evolved map's spot results
        self.data = {"schema": SCHEMA, "tag": tag, "seed": seed, "status": "running", "exit_code": None, "error": None,
                     "args": args, "workers": workers, "started": _utc(self.t0), "updated": None,
                     "deadline": deadline, "wall_seconds": 0.0, "base": None, "metric": None, "rounds_done": 0,
                     "rounds": [], "spot_seeds": None, "spots": [], "metric_start": None, "metric_end_est": None,
                     "spot_last": None, "files": {"map": None, "layouts": [], "log": "runs.jsonl"}, "env": None}

    def _read_log(self):
        if not os.path.isfile(self.log_path):
            return
        if os.path.getsize(self.log_path) < self.offset:  # truncated (evolve opens it afresh)
            self.offset = 0
        with open(self.log_path) as f:
            f.seek(self.offset)
            while True:
                line = f.readline()
                if not line.endswith("\n"):  # none left, or a half-written last line: read it next time
                    break
                self.offset += len(line.encode())
                self._take(json.loads(line))

    def _take(self, rec):
        d, kind = self.data, rec.get("type")
        if kind == "start":
            d["metric"], d["spot_seeds"] = rec.get("metric"), rec.get("spot_seeds")
            d["base"] = {"file": d["args"]["base"], "tiles": rec.get("tiles")}
        elif kind == "round":
            d["rounds"].append({"round": rec["round"], "metric_in": rec["flow_in"],
                                "metric_out_est": rec["flow_out_est"], "score_in": rec["score_in"],
                                "worst": rec["worst"], "worst_metric": rec["worst_flow"],
                                "changed": len(rec["changed"]) - len(rec["reverted"]), "reverted": len(rec["reverted"]),
                                "dropped": len(rec["dropped"]), "tiles": rec["tiles"], "jobs": rec["jobs"],
                                "secs": rec["secs"], "layout": rec["layout"]})
            d["rounds_done"] = rec["round"]
        elif kind == "spot":
            if rec.get("who") == "base":
                self.base_spot.append(rec)
            else:
                self.spot_runs.setdefault(rec["round"], []).append(rec)

    def _spots(self):
        def mean(rs, k):
            return sum(r[k] for r in rs) / len(rs)
        out = []
        for r, mine in sorted(self.spot_runs.items()):
            if not self.base_spot:
                continue
            rec = {"round": r, "seeds": [x["seed"] for x in mine], "score": mean(mine, "score"),
                   "base_score": mean(self.base_spot, "score")}
            for k in ("unstuck", "spread"):
                if all(k in x for x in mine + self.base_spot):
                    rec[k], rec["base_" + k] = mean(mine, k), mean(self.base_spot, k)
            out.append(rec)
        return out

    def write(self, **fields):
        self.data.update(fields)
        try:
            self._read_log()
        except Exception:  # a log problem must not end the run: keep what was read
            self.data.setdefault("log_error", traceback.format_exc())
        d = self.data
        d["spots"] = self._spots()
        d["spot_last"] = d["spots"][-1] if d["spots"] else None
        d["metric_start"] = d["rounds"][0]["metric_in"] if d["rounds"] else None
        d["metric_end_est"] = d["rounds"][-1]["metric_out_est"] if d["rounds"] else None
        lay = os.path.join(self.dir, "layouts")
        d["files"]["layouts"] = sorted(os.listdir(lay), key=lambda n: (len(n), n)) if os.path.isdir(lay) else []
        d["files"]["map"] = "map.json" if os.path.isfile(os.path.join(self.dir, "map.json")) else None
        now = time.time()
        d["updated"], d["wall_seconds"] = _utc(now), round(now - self.t0, 1)
        tmp = os.path.join(self.dir, ".summary.json.tmp")
        with open(tmp, "w") as f:
            json.dump(d, f, indent=1)
        os.replace(tmp, os.path.join(self.dir, "summary.json"))


def make_backend(base, out_dir, summary):
    """chunk_evolve.MapBackend on the base file, saving the evolved map into out_dir (as map.json: main names it) and
    refreshing summary.json after every save (each round, each spot check)."""
    from chunk_evolve import MapBackend

    class Backend(MapBackend):
        def save(self, path, keys, note):
            super().save(path, keys, note)
            summary.write()

    be = Backend(base)
    be.folder = out_dir  # chunk_evolve.main saves to <folder>/NAME.json
    return be


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("tag", help="the job's tag (from params.txt)")
    p.add_argument("--base", required=True, help="the base map file (flat in the cwd on the OSPool)")
    p.add_argument("--hours", type=float, required=True, help="wall hours the job may run (keep under 20: held)")
    p.add_argument("--seed", type=int, required=True, help="the search's seed (round, mutation and spot seeds)")
    args, extra = p.parse_known_args(argv)
    for flag in extra:
        if flag.split("=")[0] in OWN_FLAGS:
            p.error(f"{flag.split('=')[0]} is set by ospool_evolve (see its doc)")
    return args, extra


def main(argv=None):
    args, extra = parse(sys.argv[1:] if argv is None else argv)
    t0 = time.time()
    out_dir = os.path.abspath(OUT)
    os.makedirs(out_dir, exist_ok=True)
    shutil.rmtree(os.path.join(out_dir, "layouts"), ignore_errors=True)  # a rerun's stale rounds
    workers = max(1, int(os.environ.get("OMP_NUM_THREADS") or 1))
    deadline = time.strftime("%Y-%m-%d %H:%M", time.localtime(t0 + args.hours * 3600.0 - PACK_MARGIN))
    base = os.path.abspath(args.base)  # (whole-map spot runs chdir to the code's folder)
    ce_argv = ["map", "--base", base, "--seed", str(args.seed), "--workers", str(workers), "--deadline", deadline,
               "--log-dir", out_dir] + ([] if any(a.split("=")[0] == "--rounds" for a in extra) else
                                        ["--rounds", str(ROUNDS)]) + extra
    summary = Summary(out_dir, args.tag, args.seed, {"hours": args.hours, "base": os.path.basename(args.base),
                                                     "extra": extra, "argv": ce_argv}, workers, deadline)

    def on_signal(signum, frame):
        raise Terminated(signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    status, code, error = "ok", 0, None
    try:
        summary.write(env=env_info())
        print(f"ospool_evolve {args.tag}: seed {args.seed}, {workers} workers, deadline {deadline} (local), host "
              f"{summary.data['env']['host']}, jit {summary.data['env']['jit']}", flush=True)
        import chunk_evolve
        chunk_evolve.main(ce_argv, backend=make_backend(base, out_dir, summary))
    except Terminated as e:
        status, code, error = "terminated", 143, f"signal {e}"
    except BaseException:
        status, code, error = "error", 1, traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
    try:
        summary.write(status=status, exit_code=code, error=error)
    except Exception:
        traceback.print_exc()
        code = code or 1
    print(f"ospool_evolve {args.tag}: {status}, {summary.data['rounds_done']} rounds, "
          f"{time.time() - t0:.0f} s", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
