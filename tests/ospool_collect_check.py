"""Checks for ospool_collect.py against a fake OSPool results folder (no network, no physics; runs in a second). From
the repo root:

    .venv/bin/python3 tests/ospool_collect_check.py

The fake job folder (as after the runbook's block F pull): sweep.json with 4 rows in 2 settings; tarballs made the
way run.sh makes them (shutil.make_archive of outputs/) for 3 jobs:
  v6-s01  ok, 3 rounds, spots at rounds 2 and 3 (gain +0.03); evicted once (2 starts); an older cluster's failed log
        for the same tag must lose to the newer one
  v6-s02  ok, 3 rounds, last spot at round 2 (gain +0.05): the export must take layouts/r2.json, not map.json
  v8-s01  failed (return value 1): 1 round, no spot, a traceback in .err and summary.json
  v8-s02  no tarball; its log has no termination event
1. Parsing: status / exit / rounds / chunk metric start -> end / spot / gain / host / CPU / evictions per job; tags
   matched to logs via the .out line; settings from the sweep's flags.
2. Per setting mean +- sd of the gain; ranking v6-s02, v6-s01, v8-s01.
3. The report lists the missing and the failed job with the .err tail.
4. Unpacking is skipped for folders already unpacked; the plots page is written with the data and Chart.js.
5. --export: valid map files named "ospool <tag>.json" with the scored layout's tiles; an existing file is never
   overwritten (its content unchanged), on a second run nothing is.
Exit status 1 on any failure.
"""
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.getcwd())
import ospool_collect as oc  # noqa: E402
from diamonds_map import load_layout, save_layout  # noqa: E402

FAILS = []


def check(ok, what):
    print(f"{'ok  ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        FAILS.append(what)


SIZE = (1800.0, 1169.0)
V6 = ["--variants", "6", "--seeds", "10", "--margin", "0.005"]  # summary.json's args.extra (the sweep's flags)
V8 = ["--variants", "8", "--seeds", "10", "--margin", "0.005"]


def tiles(n, off):
    return [(10.0 + off + 20 * i, 30.0) for i in range(n)]


def condor_log(cluster, proc, host, site, exit_code=None, evictions=0):
    j = f"({cluster}.{proc:03d}.000)"
    ev = [f"000 {j} 2026-10-29 01:00:00 Job submitted from host: <1.2.3.4:9618?alias=ap40.uw.osg-htc.org>", "..."]
    for _ in range(evictions):
        ev += [f"001 {j} 2026-10-29 01:01:00 Job executing on host: <10.0.0.1:1?alias=other.host&noUDP>",
               "\tSlotName: slot1@glidein_1@other.host", '\tGLIDEIN_ResourceName = "Other-Site"', "...",
               f"004 {j} 2026-10-29 01:02:00 Job was evicted. Code 1008 Subcode 0", "\tReason: x", "..."]
    ev += [f"001 {j} 2026-10-29 01:03:00 Job executing on host: <10.0.0.2:1?alias={host}&noUDP>",
           f"\tSlotName: slot1_2@glidein_2@{host}", f'\tGLIDEIN_ResourceName = "{site}"', "\tCpus = 8", "..."]
    if exit_code is not None:
        ev += [f"005 {j} 2026-10-29 03:03:00 Job terminated.", f"\t(1) Normal termination (return value {exit_code})",
               "\tPartitionable Resources :    Usage  Request Allocated ",
               "\t   Cpus                 :        7        8         8 ",
               "\t   Memory (MB)          :      900     4096      4096 ",
               "\t   TimeExecute (s)      :     7200                    ", "..."]
    return "\n".join(ev) + "\n"


def make_job(root, tag, seed, n_rounds, flows, est, spots, cpu, exit_status=0, traceback=None, args=None):
    """outputs/ like ospool_evolve.py writes it, then outputs_<tag>.tar.gz in results/ (run.sh's make_archive)."""
    work = tempfile.mkdtemp(dir=root)
    out = os.path.join(work, "outputs")
    os.makedirs(os.path.join(out, "layouts"))
    for r in range(1, n_rounds + 1):
        save_layout(os.path.join(out, "layouts", f"r{r}.json"), None, SIZE, note=f"{tag} r{r}",
                    keys=tiles(5 + r, seed), tiling={"tiling": "penrose", "edge": 20.16})
    if n_rounds:
        shutil.copy(os.path.join(out, "layouts", f"r{n_rounds}.json"), os.path.join(out, "map.json"))
    recs = [{"type": "start", "metric": "free", "seed": seed, "spot_seeds": [11, 12]},
            {"type": "job", "round": 1, "role": "inc", "chunk": "grid r0c0", "free": 0.9}]
    for r in range(1, n_rounds + 1):
        recs.append({"type": "round", "round": r, "flow_in": flows[r - 1],
                     "flow_out_est": est if r == n_rounds else flows[r - 1] + 0.004, "changed": [], "reverted": []})
        for rr, who, scores in spots:
            if rr == r:
                recs += [{"type": "spot", "round": rr, "who": who, "seed": s, "score": v}
                         for s, v in zip((11, 12), scores)]
    with open(os.path.join(out, "runs.jsonl"), "w") as f:
        f.write("".join(json.dumps(x) + "\n" for x in recs) + '{"type": "round", "torn')  # a torn last line
    json.dump({"schema": 1, "tag": tag, "seed": seed, "status": "ok" if not exit_status else "error",
               "exit_code": exit_status, "error": traceback, "args": {"hours": 2, "base": "base-v1.json", "extra": args,
                                                                       "argv": []},
               "rounds_done": n_rounds, "env": {"host": "h", "cpu_model": cpu}, "wall_seconds": 7100.0},
              open(os.path.join(out, "summary.json"), "w"))
    shutil.make_archive(os.path.join(root, "results", f"outputs_{tag}"), "gztar", out)
    shutil.rmtree(work)


def build(root):
    os.makedirs(os.path.join(root, "results"))
    os.makedirs(os.path.join(root, "logs"))
    # agent C's schema (ospool_stage.py); "full" as a generator: v6-s01, v6-s02, v8-s01, v8-s02
    json.dump({"image": "x-v1.sif", "request": {"cpus": 8, "memory": "4GB", "disk": "3GB"},
               "fixed": {"--margin": 0.005},
               "columns": ["hours", "seed", "variants", "seeds"],
               "smoke": [{"tag": "smoke1", "hours": 0.4, "seed": 9, "variants": 6, "seeds": 10}],
               "full": {"tag": "v{variants}-s{seed:02d}", "hours": 2, "seed": {"from": 1, "to": 2}, "variants": [6, 8],
                        "seeds": 10}}, open(os.path.join(root, "sweep.json"), "w"))
    epyc, xeon = "AMD EPYC 7763 64-Core Processor", "Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz"
    make_job(root, "v6-s01", 1, 3, [0.950, 0.960, 0.970], 0.975,
             [(2, "base", [0.78, 0.80]), (2, "K", [0.80, 0.82]), (3, "K", [0.81, 0.83])], epyc, args=V6)
    make_job(root, "v6-s02", 2, 3, [0.940, 0.955, 0.965], 0.972, [(2, "base", [0.70, 0.72]), (2, "K", [0.76, 0.76])],
             xeon, args=V6)
    make_job(root, "v8-s01", 1, 1, [0.930], 0.936, [], epyc, exit_status=1,
             traceback="Traceback (most recent call last):\n  File \"x.py\", line 1\nRuntimeError: boom\n", args=V8)
    logs = os.path.join(root, "logs")
    jobs = [("v6-s01", 0, "epyc016.rci.montana.edu", "MTState", 0, 1), ("v6-s02", 1, "comp-ex-0344", "PSU-LIGO", 0, 0),
            ("v8-s01", 2, "epyc016.rci.montana.edu", "MTState", 1, 0), ("v8-s02", 3, "gpu03", "Somewhere", None, 0)]
    for tag, proc, host, site, code, ev in jobs:
        stem = os.path.join(logs, f"job_200_{proc}")
        open(stem + ".log", "w").write(condor_log(200, proc, host, site, code, ev))
        open(stem + ".out", "w").write(f"tag={tag} host={host.split('.')[0]} start=2026-10-29T06:03:00Z\n")
        open(stem + ".err", "w").write("" if code != 1 else
                                       "Traceback (most recent call last):\n  File \"/srv/ospool_evolve.py\", "
                                       "line 9, in <module>\nRuntimeError: boom\n")
    # an older cluster for v6-s01 that failed: the newer one (200) must win
    open(os.path.join(logs, "job_150_0.log"), "w").write(condor_log(150, 0, "old.host", "Old", 1))
    open(os.path.join(logs, "job_150_0.out"), "w").write("tag=v6-s01 host=old\n")
    open(os.path.join(logs, "job_150_0.err"), "w").write("old failure\n")


def run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = oc.main(argv)
    return res, buf.getvalue()


def main():
    root = tempfile.mkdtemp(prefix="ospool_collect_check_")
    try:
        build(root)
        maps = os.path.join(root, "maps")
        os.makedirs(maps)
        html = os.path.join(root, "page", "plots.html")
        res, text = run([root, "--no-open", "--html", html, "--maps-dir", maps])
        J = {j["tag"]: j for j in res["jobs"]}

        # 1. parsing
        check(sorted(J) == ["v6-s01", "v6-s02", "v8-s01", "v8-s02"], f"4 jobs from the sweep: {sorted(J)}")
        check([J[t]["status"] for t in ("v6-s01", "v6-s02", "v8-s01", "v8-s02")] == ["ok", "ok", "failed", "missing"],
              "statuses ok / ok / failed / missing")
        check(J["v6-s01"]["exit"] == 0 and J["v6-s01"]["condor"]["cluster"] == 200,
              "newest cluster's log wins for v6-s01")
        check(J["v8-s01"]["exit"] == 1 and J["v8-s02"]["exit"] is None, "exit codes 1 / none")
        check(J["v6-s01"]["condor"]["evictions"] == 1 and J["v6-s01"]["condor"]["starts"] == 2, "eviction + restart")
        check(J["v6-s01"]["host"] == "epyc016.rci.montana.edu" and J["v6-s01"]["site"] == "MTState", "last host / site")
        check(J["v6-s01"]["condor"]["run_secs"] == 7200 and J["v6-s01"]["condor"]["mem_mb"] == 900, "usage table")
        check([J[t]["rounds"] for t in ("v6-s01", "v6-s02", "v8-s01", "v8-s02")] == [3, 3, 1, 0], "rounds done")
        check(J["v6-s01"]["curve"] == [0.950, 0.960, 0.970, 0.975] and J["v6-s01"]["curve_est"],
              "chunk metric after 0..3 rounds (torn runs.jsonl line skipped)")
        check(abs(J["v6-s01"]["gain"] - 0.03) < 1e-9 and J["v6-s01"]["spot_round"] == 3, "v6-s01 gain +0.03 at r3")
        check(abs(J["v6-s02"]["gain"] - 0.05) < 1e-9 and J["v6-s02"]["spot_round"] == 2, "v6-s02 gain +0.05 at r2")
        check(J["v8-s01"]["gain"] is None and J["v8-s01"]["traceback"], "v8-s01: no spot, a traceback")
        check(J["v6-s02"]["scored_map"].endswith(os.path.join("layouts", "r2.json")), "v6-s02's scored map is r2")
        check(J["v6-s01"]["cpu"].startswith("AMD EPYC") and J["v6-s01"]["wall"] == 7100.0, "summary cpu / wall")
        check(J["v6-s01"]["sid"] == J["v6-s02"]["sid"] == "v6" and J["v8-s01"]["sid"] == J["v8-s02"]["sid"] == "v8",
              f"settings from the sweep rows: {res['settings']}")
        check(res["settings"]["v8"] == "variants=8" and "variants=6" in res["settings"]["v6"], "setting labels")
        check("smoke1" not in J, "the smoke section (no results here) not expected")

        # 2. per setting, ranking
        st = {s["sid"]: s for s in oc.per_setting(res["jobs"], res["settings"])}
        m, sd, n = st["v6"]["gain"]
        check(abs(m - 0.04) < 1e-9 and abs(sd - math.sqrt(2) * 0.01) < 1e-9 and n == 2, f"v6 gain {m} +- {sd} n {n}")
        check(st["v8"]["gain"] == (None, None, 0) and st["v8"]["n"] == 2, "v8: no gains")
        check([j["tag"] for j in oc.ranked(res["jobs"])] == ["v6-s02", "v6-s01", "v8-s01"],
              "ranking v6-s02, v6-s01, v8-s01")

        # 3. report text
        check("v8-s02: no tarball" in text and "no termination event" in text, "missing job reported")
        check("v8-s01: exit 1" in text and "| RuntimeError: boom" in text, "failed job with the .err tail")
        check("old failure" not in text, "the old cluster's .err not shown")
        check("+0.0400" in text and "0.0141" in text, "per setting mean / sd in the report")
        check(all(len(ln) <= 120 for ln in text.split("\n") if not ln.startswith(("sweep:", "plots:"))),
              "report lines <= 120 chars (bar the path lines)")

        # 4. unpack skip, page
        r1 = os.path.join(root, "results", "outputs_v6-s01", "layouts", "r1.json")
        os.remove(r1)
        res2, _ = run([root, "--no-open", "--html", html, "--maps-dir", maps])
        check(not os.path.exists(r1), "already unpacked folder left alone (a removed file not re-extracted)")
        check(not any(p.endswith(".unpacking") for p in os.listdir(os.path.join(root, "results"))), "no temp dirs")
        page = open(html).read()
        check("Chart.js/4.4.1/chart.umd.min.js" in page and '"v6-s02"' in page and "__DATA__" not in page,
              "plots page written with the data")

        # 5. export
        sentinel = os.path.join(maps, "ospool v6-s01.json")
        open(sentinel, "w").write("SENTINEL")
        res3, text3 = run([root, "--no-open", "--html", html, "--maps-dir", maps, "--export", "3"])
        what = {t: w for t, _, w in res3["exported"]}
        check(what == {"v6-s02": "copied", "v6-s01": "skipped: already exists", "v8-s01": "copied"}, f"export: {what}")
        check(open(sentinel).read() == "SENTINEL", "existing file not overwritten")
        keys, size = load_layout(os.path.join(maps, "ospool v6-s02.json"))
        check(size == SIZE and keys == [tuple(k) for k in tiles(7, 2)], "v6-s02 export = its r2 layout, valid map")
        note = json.load(open(os.path.join(maps, "ospool v6-s02.json")))["note"]
        check(note.startswith("ospool v6-s02 (spot gain +0.0500 at round 2"), "export note names tag and gain")
        check("v6-s02: copied -> ospool v6-s02.json" in text3, "export printed")
        before = {f: open(os.path.join(maps, f)).read() for f in os.listdir(maps)}
        res4, _ = run([root, "--no-open", "--html", html, "--maps-dir", maps, "--export", "3"])
        check(all(w == "skipped: already exists" for _, _, w in res4["exported"]), "second export skips all")
        check(before == {f: open(os.path.join(maps, f)).read() for f in os.listdir(maps)}, "maps unchanged")

        # without a sweep file: settings from summary.json's args (the missing job is then unknown, its log kept)
        os.rename(os.path.join(root, "sweep.json"), os.path.join(root, "sweep.off"))
        res5 = oc.collect(root, os.path.join(root, "nonexistent.json"))
        J5 = {j["tag"]: j for j in res5["jobs"]}
        check(J5["v6-s01"]["sid"] == J5["v6-s02"]["sid"] == "v6" and J5["v8-s01"]["sid"] == "v8",
              f"no sweep: settings from summary args {res5['settings']}")
        check(J5["v6-s01"]["values"] == {"hours": 2, "variants": "6", "seeds": "10", "margin": "0.005"},
              f"summary args values {J5['v6-s01']['values']}")
        os.rename(os.path.join(root, "sweep.off"), os.path.join(root, "sweep.json"))

        # CLI as a subprocess
        p = subprocess.run([sys.executable, "ospool_collect.py", root, "--no-open", "--html", html],
                           capture_output=True, text=True)
        check(p.returncode == 0 and "top 3 maps" in p.stdout, "CLI runs")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("ALL OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
