"""Bring an OSPool chunk_evolve sweep home: unpack, report, plot, and export the best maps.

    python3 ospool_collect.py [RESULTS_DIR] [--sweep ospool/sweep.json] [--top 5] [--export N]
                              [--maps-dir maps/penrose] [--html PATH] [--no-open]

RESULTS_DIR (default ~/Desktop/PennAir/OSPoolTesting/jobs/softbody-evolve-v1) is a job folder after the runbook's
block F pull: results/outputs_<tag>.tar.gz (one per job, made by run.sh even when the job fails) and logs/job_<cluster>_
<proc>.{log,out,err}. Each tarball is unpacked next to itself into results/outputs_<tag>/ (skipped if that folder is
already there; members are extracted with tarfile's "data" filter, so nothing lands outside it) and is expected to
hold {map.json, layouts/rN.json, runs.jsonl, summary.json} (ospool_evolve.py's contract).

A job's tag comes from the tarball name; its logs are matched by the "tag=<tag>" line run.sh prints first in .out
(or, failing that, by the proc number = the tag's row in params.txt). With several clusters for one tag (a
resubmit) the newest cluster wins. From the HTCondor .log (logs/job_* and logs/smoke_*): exit code (or
signal), execute host and site, starts, evictions, holds, run seconds, peak memory.

From runs.jsonl (chunk_evolve's log): the chunk metric per round, "after k rounds" = round k+1's measured flow_in
(fresh seeds), and after the last round its flow_out_est (an estimate, marked ~); the whole-map spot checks (the
evolved layout K vs the base map on the job's fixed spot seeds). A job's gain is its last spot's K score minus the
base score (paired: the same seeds). summary.json (ospool_evolve.py's schema 1) supplies env.cpu_model / host,
wall_seconds, status / exit_code / error, and the curve and spots if runs.jsonl lacks them. A returned job is failed
if its exit code is nonzero, it died of a signal, or its summary says error / terminated / running (killed).

A job's setting is its sweep row's columns minus the seed (ospool/sweep.json, agent C's schema: "columns", and
"smoke" / "full" as row lists or generators, expanded by ospool_stage.expand_rows; looked for at --sweep, then
RESULTS_DIR/sweep.json, then the repo's ospool/sweep.json); without a row, summary.json's args (hours + the
pass-through flags) minus the per-job ones; without those, the tag's name. Replicates are the jobs of one setting;
it is named after their tag minus the replicate suffix (base-s01 -> base) and labelled with the knobs that differ
from the most common value. Expected jobs: every row of each sweep section with at least one tarball or log here.

Stdout: per job (tag, setting, exit, rounds, chunk metric start -> end, last spot K vs base); missing / failed jobs
with the tail of .err (or the summary's traceback); per setting mean +- sd of the spot gain over its replicates; the
top --top maps by spot gain (jobs without a spot after, by end metric). The plots page (Chart.js: chunk metric vs
round per job, spot gain per setting, rounds per job vs CPU model, a sortable job table) is written to --html
(default <tmp>/ospool_collect/<job folder name>.html) and opened with `open` unless --no-open.

--export N copies the top N maps into --maps-dir as "ospool <tag>.json" (the layout the ranking spot check scored:
layouts/r<spot round>.json, = map.json unless rounds ran after the last spot), the note prefixed with the tag and
gain. Existing files are never overwritten: those are skipped and reported.
"""
import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

REPO = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.expanduser("~/Desktop/PennAir/OSPoolTesting/jobs/softbody-evolve-v1")
CHARTJS = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"
ERR_TAIL = 12  # lines of .err shown for a failed job
# per-job arguments, not part of a sweep setting (argparse dest names; "--x-y" flags normalise to x_y)
PER_JOB_ARGS = {"seed", "name", "out", "out_name", "log_dir", "outdir", "out_dir", "deadline", "workers",
                "cpus", "tag", "base_path"}


# ---------------------------------------------------------------------------------------------------------------
# Small helpers
def pick(d, *names, default=None):
    """The first of `names` present (not None) in dict d."""
    if isinstance(d, dict):
        for n in names:
            if d.get(n) is not None:
                return d[n]
    return default


def mean_sd(vals):
    """(mean, sample sd or None for n < 2, n) of the non-None vals; (None, None, 0) if none."""
    v = [float(x) for x in vals if x is not None]
    if not v:
        return None, None, 0
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else None
    return m, sd, len(v)


def fmt(x, nd=4, sign=False):
    if x is None:
        return "-"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def read_json(path):
    if not path:
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def tail(path, n=ERR_TAIL):
    try:
        with open(path, errors="replace") as f:
            lines = f.read().rstrip("\n").split("\n")
    except OSError:
        return []
    return [ln for ln in lines[-n:] if ln.strip()] if any(ln.strip() for ln in lines) else []


def find(root, name):
    """The shallowest file called `name` under root (None if none)."""
    best = None
    for d, _, files in os.walk(root):
        if name in files:
            p = os.path.join(d, name)
            if best is None or p.count(os.sep) < best.count(os.sep):
                best = p
    return best


# ---------------------------------------------------------------------------------------------------------------
# Unpacking
def unpack(results):
    """{tag: unpacked folder} for every results/outputs_<tag>.tar.gz (unpacked now unless its folder exists) and
    every outputs_<tag>/ folder without a tarball. Second value: [(tag, error)] for tarballs that failed."""
    out, bad = {}, []
    for tgz in sorted(glob.glob(os.path.join(results, "outputs_*.tar.gz"))):
        tag = os.path.basename(tgz)[len("outputs_"):-len(".tar.gz")]
        dest = tgz[:-len(".tar.gz")]
        if not os.path.isdir(dest):
            tmp = dest + ".unpacking"
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp)
            try:
                with tarfile.open(tgz) as tf:
                    tf.extractall(tmp, filter="data")
            except (tarfile.TarError, OSError, EOFError) as e:
                shutil.rmtree(tmp, ignore_errors=True)
                bad.append((tag, f"{type(e).__name__}: {e}"))
                continue
            os.rename(tmp, dest)  # only a complete unpack gets the final name
        out[tag] = dest
    for d in sorted(glob.glob(os.path.join(results, "outputs_*"))):
        tag = os.path.basename(d)[len("outputs_"):]
        if os.path.isdir(d) and not d.endswith(".unpacking") and tag not in out:
            out[tag] = d
    return out, bad


# ---------------------------------------------------------------------------------------------------------------
# HTCondor logs
EVENT = re.compile(r"^(\d{3}) \((\d+)\.(\d+)\.\d+\) (\S+ \S+) (.*)$")


def _when(text):
    for f in ("%Y-%m-%d %H:%M:%S", "%m/%d %H:%M:%S"):
        try:
            return time.mktime(time.strptime(text, f))
        except ValueError:
            pass
    return None


def parse_condor_log(path):
    """The job's user log -> {cluster, proc, exit, signal, terminated, aborted, host, site, starts, evictions,
    holds, disconnects, run_secs, mem_mb, hold_reason}."""
    info = {"cluster": None, "proc": None, "exit": None, "signal": None, "terminated": False, "aborted": False,
            "host": None, "site": None, "starts": 0, "evictions": 0, "holds": 0, "disconnects": 0,
            "run_secs": None, "mem_mb": None, "hold_reason": None}
    try:
        with open(path, errors="replace") as f:
            text = f.read()
    except OSError:
        return info
    events, cur = [], None
    for line in text.split("\n"):
        m = EVENT.match(line)
        if m:
            cur = {"code": m.group(1), "when": _when(m.group(4)), "head": m.group(5), "body": []}
            events.append(cur)
            info["cluster"], info["proc"] = int(m.group(2)), int(m.group(3))
        elif cur is not None and line.strip() != "...":
            cur["body"].append(line)
    last_start = None
    for e in events:
        body = "\n".join(e["body"])
        c = e["code"]
        if c == "001":
            info["starts"] += 1
            last_start = e["when"]
            m = re.search(r"SlotName: \S*@([^\s@]+)\s*$", body, re.M) or re.search(r"alias=([^&>]+)", e["head"])
            info["host"] = m.group(1) if m else info["host"]
            m = re.search(r'GLIDEIN_ResourceName = "([^"]*)"', body)
            info["site"] = m.group(1) if m else info["site"]
        elif c == "004":
            info["evictions"] += 1
        elif c == "012":
            info["holds"] += 1
            m = re.search(r"^\s*(\S.*)$", body, re.M)
            info["hold_reason"] = m.group(1).strip() if m else None
        elif c == "022":
            info["disconnects"] += 1
        elif c == "009":
            info["aborted"] = True
        elif c == "005":
            info["terminated"] = True
            m = re.search(r"return value (-?\d+)", body)
            info["exit"] = int(m.group(1)) if m else None
            m = re.search(r"signal (\d+)", body)
            info["signal"] = int(m.group(1)) if m else None
            m = re.search(r"TimeExecute \(s\)\s*:\s*(\d+)", body)
            if m:
                info["run_secs"] = float(m.group(1))
            elif e["when"] is not None and last_start is not None:
                info["run_secs"] = e["when"] - last_start
            m = re.search(r"Memory \(MB\)\s*:\s*(\d+)", body)
            info["mem_mb"] = int(m.group(1)) if m else None
    return info


def out_tag(path):
    """The tag in run.sh's first .out line "tag=<tag> host=..." (None if absent)."""
    try:
        with open(path, errors="replace") as f:
            for _, line in zip(range(20), f):
                m = re.match(r"tag=(\S+)", line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return None


def logs_by_tag(logs, params):
    """{tag: {"log", "out", "err", "cluster", "proc"}}, newest cluster per tag, from logs/<prefix>_<cluster>_<proc>.log
    (prefix job or smoke). params: {prefix: [tags in queue order]} (proc number -> tag, used when .out names none)."""
    found = {}
    for log in glob.glob(os.path.join(logs, "*_*_*.log")):
        m = re.match(r"(.+)_(\d+)_(\d+)\.log$", os.path.basename(log))
        if not m:
            continue
        prefix, cluster, proc = m.group(1), int(m.group(2)), int(m.group(3))
        stem = log[:-len(".log")]
        tag = out_tag(stem + ".out")
        if tag is None and proc < len(params.get(prefix, [])):
            tag = params[prefix][proc]
        if tag is None:
            continue
        if tag not in found or cluster > found[tag]["cluster"]:
            found[tag] = {"log": log, "out": stem + ".out", "err": stem + ".err", "cluster": cluster, "proc": proc}
    return found


# ---------------------------------------------------------------------------------------------------------------
# Sweep settings
def _expand(section, columns):
    """A sweep section (a list of rows or one generator dict) -> [row]; ospool_stage.expand_rows does the
    generators (the stager's own rules); a list is taken as is."""
    if isinstance(section, list):
        return [r for r in section if isinstance(r, dict) and "tag" in r]
    if isinstance(section, dict):
        try:
            from ospool_stage import expand_rows
            return expand_rows(section, columns, "sweep")
        except Exception as e:  # noqa: BLE001 - a bad generator: report, carry on without it
            print(f"warning: sweep generator not expanded ({type(e).__name__}: {e})", file=sys.stderr)
    return []


def load_sweep(path):
    """{tag: (section, row)} from ospool/sweep.json's "smoke" and "full" (agent C's schema; "rows" also read), and
    the per-row columns ([] if none)."""
    sw = read_json(path)
    if not isinstance(sw, dict):
        return {}, []
    columns = [c for c in sw.get("columns", []) if isinstance(c, str)]
    out = {}
    for section in ("smoke", "full", "rows"):
        for r in _expand(sw.get(section), columns):
            out.setdefault(str(r["tag"]), (section, r))
    return out, columns


def setting_name(tag):
    """The tag without its replicate suffix: base-s01 -> base."""
    return re.sub(r"[_-](s|seed|r|rep)\d+$", "", tag) or tag


def _argv_values(argv):
    """{flag name (no dashes, '-' -> '_'): value} of a flag list (a bare flag: True)."""
    out, i = {}, 0
    argv = [str(a) for a in argv]
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            k, eq, v = a[2:].partition("=")
            if not eq and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                v, i = argv[i + 1], i + 1
            out[k.replace("-", "_")] = v if (eq or v) else True
        i += 1
    return out


def setting_values(tag, row, columns, summary):
    """The job's setting as {knob: value}: the sweep row's columns minus the per-job ones (seed), else summary.json's
    args (hours + the pass-through flags) minus them, else {} (the setting is then the tag's name alone)."""
    if row is not None:
        cols = columns or [k for k in row if k not in ("tag", "flags")]
        vals = {c: row[c] for c in cols if c in row and c not in PER_JOB_ARGS}
        if "flags" in row:
            vals.update(_argv_values(str(row["flags"]).split()))
        return {k: v for k, v in vals.items() if k not in PER_JOB_ARGS}
    args = pick(summary, "args")
    if isinstance(args, dict):
        vals = {"hours": args["hours"]} if "hours" in args else {}
        vals.update(_argv_values(args.get("extra") or []))
        return {k: v for k, v in vals.items() if k not in PER_JOB_ARGS}
    return {}


def name_settings(jobs):
    """Give every job its setting id ("sid"; the replicates' shared tag name, or S<n> if that's long or taken) and
    return {sid: label}: the knobs that differ from the most common value across settings (all knobs for a setting
    that differs in none)."""
    groups = {}
    for j in jobs:
        key = tuple(sorted((k, json.dumps(v)) for k, v in j["values"].items())) or ("name", setting_name(j["tag"]))
        groups.setdefault(key, []).append(j)
    common = {}
    for key in groups:
        for k, v in (key if key and key[0] != "name" else ()):
            common.setdefault(k, {}).setdefault(v, 0)
            common[k][v] += 1
    mode = {k: max(c, key=c.get) for k, c in common.items()}
    labels, used = {}, set()
    for i, (key, js) in enumerate(groups.items(), 1):
        names = {setting_name(j["tag"]) for j in js}
        sid = names.pop() if len(names) == 1 else f"S{i}"
        if len(sid) > 10 or sid in used:
            sid = f"S{i}"
        used.add(sid)
        vals = js[0]["values"]
        diff = {k: v for k, v in vals.items() if mode.get(k) != json.dumps(v)}
        labels[sid] = " ".join(f"{k}={v}" for k, v in (diff or vals).items()) or sid
        if not diff and vals and len(groups) > 1:
            labels[sid] = "common values: " + labels[sid]
        for j in js:
            j["sid"], j["setting"] = sid, labels[sid]
    return labels


# ---------------------------------------------------------------------------------------------------------------
# One job
def read_runs(path):
    """runs.jsonl -> (start record, [round records], [spot records]); torn / bad lines skipped."""
    start, rounds, spots = None, [], []
    if not path:
        return start, rounds, spots
    try:
        with open(path, errors="replace") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                t = r.get("type") if isinstance(r, dict) else None
                if t == "start" and start is None:
                    start = r
                elif t == "round":
                    rounds.append(r)
                elif t == "spot":
                    spots.append(r)
    except OSError:
        pass
    rounds.sort(key=lambda r: r.get("round", 0))
    return start, rounds, spots


def spot_series(spots):
    """[{round, score, base_score, gain}] per spot round of K, base = the mean of all base spot runs."""
    base = [s["score"] for s in spots if s.get("who") == "base" and s.get("score") is not None]
    if not base:
        return []
    b = sum(base) / len(base)
    by = {}
    for s in spots:
        if s.get("who") == "K" and s.get("score") is not None:
            by.setdefault(s["round"], []).append(s["score"])
    return [{"round": r, "score": sum(v) / len(v), "base_score": b, "gain": sum(v) / len(v) - b}
            for r, v in sorted(by.items())]


def summary_spots(summary):
    """The summary's spot list in spot_series' shape (when runs.jsonl has none)."""
    sp = pick(summary, "spots", "spot")
    sp = [sp] if isinstance(sp, dict) else sp if isinstance(sp, list) else []
    out = []
    for s in sp:
        k, b = pick(s, "score", "evolved", "K"), pick(s, "base_score", "base")
        if isinstance(k, (int, float)) and isinstance(b, (int, float)):
            out.append({"round": pick(s, "round", default=0), "score": k, "base_score": b, "gain": k - b})
    return out


def summary_curve(summary):
    """The chunk metric after 0..n rounds from the summary (when runs.jsonl has no rounds): [value], est flag."""
    rows = pick(summary, "rounds", "per_round", "chunk_metric", "history")
    if not isinstance(rows, list) or not rows:
        return [], False
    if all(isinstance(x, (int, float)) for x in rows):
        return [float(x) for x in rows], False
    ins = [pick(r, "flow_in", "metric_in", "metric") for r in rows if isinstance(r, dict)]
    last = pick(rows[-1], "flow_out_est", "metric_out_est", "metric_out") if isinstance(rows[-1], dict) else None
    curve = [float(x) for x in ins if x is not None]
    return (curve + [float(last)], True) if last is not None and curve else (curve, False)


def collect_job(tag, folder, logs, row, columns=()):
    """Everything known about one job (folder: its unpacked tarball or None; logs: logs_by_tag entry or None)."""
    summary = read_json(find(folder, "summary.json")) if folder else None
    summary = summary if isinstance(summary, dict) else {}
    start, rounds, spots = read_runs(find(folder, "runs.jsonl") if folder else None)
    condor = parse_condor_log(logs["log"]) if logs else parse_condor_log("")
    env = pick(summary, "env") if isinstance(pick(summary, "env"), dict) else {}
    job = {"tag": tag, "folder": folder, "has_tarball": folder is not None, "logs": logs,
           "values": setting_values(tag, row, columns, summary),
           "seed": pick(row or {}, "seed", default=pick(summary, "seed", default=pick(start, "seed"))),
           "metric": pick(start, "metric") or pick(summary, "metric") or pick(pick(summary, "args"), "metric")
           or "free",
           "host": condor["host"] or pick(env, "host"),
           "site": condor["site"],
           "cpu": pick(env, "cpu_model") or "unknown",
           "wall": pick(summary, "wall_seconds"), "summary_status": pick(summary, "status"),
           "condor": condor, "summary": summary}
    if rounds:
        curve = [float(r["flow_in"]) for r in rounds] + [float(rounds[-1]["flow_out_est"])]
        job["curve"], job["curve_est"] = curve, True
        job["rounds"] = len(rounds)
    else:
        job["curve"], job["curve_est"] = summary_curve(summary)
        job["rounds"] = max(0, len(job["curve"]) - (1 if job["curve_est"] else 0))
    rd = pick(summary, "rounds_done")
    if isinstance(rd, int):
        job["rounds"] = rd
    job["metric_start"] = job["curve"][0] if job["curve"] else None
    job["metric_end"] = job["curve"][-1] if job["curve"] else None
    job["spots"] = spot_series(spots) or summary_spots(summary)
    last = job["spots"][-1] if job["spots"] else None
    job["spot_round"] = last["round"] if last else None
    job["spot"], job["spot_base"] = (last["score"], last["base_score"]) if last else (None, None)
    job["gain"] = last["gain"] if last else None
    # exit: HTCondor's return value first, then the summary's
    sx = pick(summary, "exit_code")
    job["exit"] = condor["exit"] if condor["exit"] is not None else (sx if isinstance(sx, int) else None)
    tb = pick(summary, "error")
    job["traceback"] = tb if isinstance(tb, str) and tb.strip() else None
    st = job["summary_status"]
    if not job["has_tarball"]:
        job["status"] = "missing"
    elif (job["exit"] not in (None, 0)) or condor["signal"] is not None or job["traceback"] \
            or st in ("error", "terminated", "running") or not summary:
        job["status"] = "failed"  # "running" in a returned tarball: killed before ospool_evolve could finish
    elif logs and not condor["terminated"]:
        job["status"] = "unfinished"
    else:
        job["status"] = "ok"
    job["err_tail"] = tail(logs["err"]) if logs else []
    # the map the ranking spot check scored, and the final map
    job["map"] = find(folder, "map.json") if folder else None
    scored = os.path.join(os.path.dirname(job["map"] or ""), "layouts", f"r{job['spot_round']}.json") \
        if job["spot_round"] is not None and job["map"] else None
    if scored and not os.path.isfile(scored) and folder:
        scored = find(folder, f"r{job['spot_round']}.json")
    job["scored_map"] = scored if scored and os.path.isfile(scored) else job["map"]
    return job


# ---------------------------------------------------------------------------------------------------------------
# Whole sweep
def params_tags(path):
    """The tags (first column) of a params file, in queue order ([] if there is none)."""
    tags = []
    try:
        with open(path) as f:
            for line in f:
                parts = line.replace(",", " ").split()
                if parts and not parts[0].startswith("#"):
                    tags.append(parts[0])
    except OSError:
        pass
    return tags


def find_sweep(results_dir, given):
    for p in (given, os.path.join(results_dir, "sweep.json"), os.path.join(REPO, "ospool", "sweep.json")):
        if p and os.path.isfile(p):
            return p
    return None


def collect(results_dir, sweep_path=None):
    """{"jobs": [job], "bad": [(tag, error)], "sweep": path or None, "settings": {sid: label}}. The expected jobs:
    every row of each sweep section (smoke / full) that has at least one tarball or log here."""
    folders, bad = unpack(os.path.join(results_dir, "results"))
    sweep_path = find_sweep(results_dir, sweep_path)
    rows, columns = load_sweep(sweep_path)
    by_section = {}
    for tag, (section, _) in rows.items():
        by_section.setdefault(section, []).append(tag)
    params = {"job": params_tags(os.path.join(results_dir, "params.txt")) or by_section.get("full", []),
              "smoke": params_tags(os.path.join(results_dir, "params-smoke.txt")) or by_section.get("smoke", [])}
    logs = logs_by_tag(os.path.join(results_dir, "logs"), params)
    seen = set(folders) | set(logs)
    tags = [t for sec, ts in by_section.items() if seen & set(ts) for t in ts]
    tags += sorted(seen - set(tags))
    jobs = [collect_job(t, folders.get(t), logs.get(t), rows[t][1] if t in rows else None, columns) for t in tags]
    for t, err in bad:
        for j in jobs:
            if j["tag"] == t:
                j["status"], j["bad_tarball"] = "failed", err
    settings = name_settings(jobs)
    return {"jobs": jobs, "bad": bad, "sweep": sweep_path, "settings": settings}


def per_setting(jobs, settings):
    """[{sid, setting, n, done, gain (mean, sd, n), dmetric (mean, sd, n), rounds mean}] in setting order."""
    out = []
    for sid, label in settings.items():
        js = [j for j in jobs if j["sid"] == sid]
        dm = [j["metric_end"] - j["metric_start"] for j in js if j["metric_start"] is not None]
        out.append({"sid": sid, "setting": label, "n": len(js), "ok": sum(j["status"] == "ok" for j in js),
                    "gain": mean_sd([j["gain"] for j in js]), "dmetric": mean_sd(dm),
                    "rounds": mean_sd([j["rounds"] for j in js if j["has_tarball"]])})
    return out


def ranked(jobs):
    """Jobs with a map, best first: by spot gain; jobs without a spot after them, by end chunk metric."""
    have = [j for j in jobs if j["scored_map"]]
    return sorted(have, key=lambda j: (j["gain"] is None, -(j["gain"] or 0.0), -(j["metric_end"] or 0.0), j["tag"]))


# ---------------------------------------------------------------------------------------------------------------
# Report
def cpu_short(name):
    """'AMD EPYC 7763 64-Core Processor' -> 'EPYC 7763', 'Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz' -> 'Xeon Gold
    6248'."""
    s = re.sub(r"\((R|TM|tm|r)\)|@.*$|\b(CPU|Processor|AMD|Intel|\d+-Core)\b", " ", str(name))
    return " ".join(s.split()) or str(name)


def report(res, top):
    jobs, settings = res["jobs"], res["settings"]
    lines = []
    say = lines.append
    n = {s: sum(j["status"] == s for j in jobs) for s in ("ok", "failed", "missing", "unfinished")}
    say(f"{len(jobs)} jobs: {n['ok']} ok, {n['failed']} failed, {n['missing']} missing a tarball, "
        f"{n['unfinished']} without a termination event")
    say(f"sweep: {res['sweep'] or '(none found: settings from summary args / tags)'}")
    say("")
    say("settings:")
    for sid, label in settings.items():
        say(f"  {sid:<10} {label}"[:118])
    say("")
    hdr = (f"{'tag':<16} {'set':<10} {'seed':>6} {'status':<10} {'exit':>4} {'rnds':>4} {'chunk metric':<16} "
           f"{'spot K / base':<15} {'gain':>8} {'ev':>2} cpu")
    say(hdr)
    say("-" * 119)
    for j in jobs:
        cm = (f"{fmt(j['metric_start'])}->{'~' if j['curve_est'] else ''}{fmt(j['metric_end'])}"
              if j["metric_start"] is not None else "-")
        sp = f"{fmt(j['spot'])} / {fmt(j['spot_base'])}" if j["spot"] is not None else "-"
        ex = "-" if j["exit"] is None else str(j["exit"])
        say(f"{j['tag'][:16]:<16} {j['sid']:<10} {str(j['seed'] if j['seed'] is not None else '-'):>6} "
            f"{j['status']:<10} {ex:>4} {j['rounds']:>4} {cm:<16} {sp:<15} {fmt(j['gain'], sign=True):>8} "
            f"{j['condor']['evictions']:>2} {cpu_short(j['cpu'])[:18]}")
    bad = [j for j in jobs if j["status"] != "ok"]
    if bad:
        say("")
        say("problems:")
        for j in bad:
            c = j["condor"]
            if j["status"] == "missing":
                why = ("no tarball; " + ("log: " + ("terminated, exit " + str(c["exit"]) if c["terminated"] else
                                                    "removed" if c["aborted"] else "no termination event (still "
                                                    "queued / running / held?)") if j["logs"] else "no log either"))
            elif j["status"] == "unfinished":
                why = "tarball but no termination event in the log"
            else:
                why = j.get("bad_tarball") or (f"signal {c['signal']}" if c["signal"] is not None else
                                               f"exit {j['exit']}")
                why += f", summary status {j['summary_status'] or '(no summary.json)'}"
            extra = [f"{c['starts']} starts", f"{c['evictions']} evictions"] + \
                ([f"{c['holds']} holds ({c['hold_reason']})"] if c["holds"] else [])
            say(f"  {j['tag']}: {why}; {', '.join(extra)}")
            src = j["err_tail"] or (j["traceback"] or "").rstrip().split("\n")[-ERR_TAIL:]
            for ln in src:
                if ln.strip():
                    say(f"      | {ln[:110]}")
    say("")
    say("per setting (gain = last spot K - base, paired seeds; d metric = chunk metric end - start):")
    say(f"  {'set':<10} {'jobs':>4} {'ok':>3} {'gain mean':>10} {'sd':>8} {'n':>3} {'d metric':>9} {'sd':>8} "
        f"{'rounds':>6}  setting")
    for s in per_setting(jobs, settings):
        g, d, r = s["gain"], s["dmetric"], s["rounds"]
        say(f"  {s['sid']:<10} {s['n']:>4} {s['ok']:>3} {fmt(g[0], sign=True):>10} {fmt(g[1]):>8} {g[2]:>3} "
            f"{fmt(d[0], sign=True):>9} {fmt(d[1]):>8} {fmt(r[0], 1):>6}  {s['setting'][:38]}")
    say("")
    top_jobs = ranked(jobs)[:top]
    say(f"top {len(top_jobs)} maps (by spot gain):")
    for i, j in enumerate(top_jobs, 1):
        say(f"  {i}. {j['tag']} ({j['sid']}, {j['status']}): gain {fmt(j['gain'], sign=True)} at round "
            f"{j['spot_round'] if j['spot_round'] is not None else '-'}, chunk metric {fmt(j['metric_end'])}, "
            f"{os.path.relpath(j['scored_map'], os.path.dirname(os.path.dirname(j['folder'])))}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------
# Export
def export(jobs, n, maps_dir):
    """Copy the top n maps to maps_dir/"ospool <tag>.json" (never overwriting). [(tag, dest, "copied" | why)]."""
    out = []
    os.makedirs(maps_dir, exist_ok=True)
    for j in ranked(jobs)[:n]:
        dest = os.path.join(maps_dir, f"ospool {j['tag']}.json")
        data = read_json(j["scored_map"])
        if not (isinstance(data, dict) and isinstance(data.get("tiles"), list) and data["tiles"] and "size" in data):
            out.append((j["tag"], dest, f"skipped: {j['scored_map']} is not a valid map file"))
            continue
        data["note"] = (f"ospool {j['tag']} (spot gain {fmt(j['gain'], sign=True)} at round {j['spot_round']}, "
                        f"{j['setting']}): " + str(data.get("note", "")))
        try:
            with open(dest, "x") as f:  # "x": fails if it exists, so never overwrites (even racing another run)
                json.dump(data, f, indent=0)
        except FileExistsError:
            out.append((j["tag"], dest, "skipped: already exists"))
            continue
        out.append((j["tag"], dest, "copied"))
    return out


# ---------------------------------------------------------------------------------------------------------------
# Plots page
def page_data(res):
    jobs = res["jobs"]
    return {
        "settings": [{"sid": sid, "label": label} for sid, label in res["settings"].items()],
        "stats": [{"sid": s["sid"], "mean": s["gain"][0], "sd": s["gain"][1], "n": s["gain"][2]}
                  for s in per_setting(jobs, res["settings"])],
        "jobs": [{"tag": j["tag"], "sid": j["sid"], "seed": j["seed"], "status": j["status"], "exit": j["exit"],
                  "rounds": j["rounds"], "curve": j["curve"], "est": j["curve_est"], "start": j["metric_start"],
                  "end": j["metric_end"], "spot": j["spot"], "base": j["spot_base"], "gain": j["gain"],
                  "spot_round": j["spot_round"], "host": j["host"], "site": j["site"], "cpu": j["cpu"],
                  "cpu_s": cpu_short(j["cpu"]),
                  "wall": j["wall"], "run_secs": j["condor"]["run_secs"], "evictions": j["condor"]["evictions"],
                  "starts": j["condor"]["starts"], "mem_mb": j["condor"]["mem_mb"]} for j in jobs],
        "metric": next((j["metric"] for j in jobs if j["metric"]), "free"),
        "made": time.strftime("%Y-%m-%d %H:%M"), "sweep": res["sweep"],
    }


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OSPool sweep results</title>
<script src="__CHARTJS__"></script>
<style>
:root { color-scheme: light; --bg: #fcfcfb; --card: #ffffff; --ink: #0b0b0b; --ink2: #52514e; --line: #e4e3df;
  --grid: #ecebe8; --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300;
  --s7: #6250d6; --s8: #e34948; --other: #9a9994; --bad: #c62828; }
@media (prefers-color-scheme: dark) { :root { color-scheme: dark; --bg: #1a1a19; --card: #222220; --ink: #f0efec;
  --ink2: #c3c2b7; --line: #3a3936; --grid: #2e2d2b; --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
  --s5: #d55181; --s6: #008300; --s7: #9085e9; --s8: #e66767; --other: #77766f; --bad: #ef6b6b; } }
body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 -apple-system, system-ui, sans-serif; }
main { max-width: 1200px; margin: 0 auto; padding: 20px 16px 48px; }
h1 { font-size: 20px; margin: 0 0 4px; } h2 { font-size: 15px; margin: 0 0 8px; }
.sub { color: var(--ink2); margin: 0 0 16px; }
.tiles { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
.tile { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 8px 14px; }
.tile b { font-size: 20px; display: block; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 16px; }
@media (max-width: 600px) { .grid { grid-template-columns: 1fr; } }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 14px; min-width: 0; }
.card p { color: var(--ink2); margin: 0 0 8px; font-size: 13px; }
.chart { position: relative; height: 320px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12px; color: var(--ink2); margin: 8px 0 0; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; }
.tablewrap { overflow-x: auto; margin-top: 16px; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { cursor: pointer; user-select: none; color: var(--ink2); position: sticky; top: 0; background: var(--card); }
th:hover { color: var(--ink); } td.num { text-align: right; } .bad { color: var(--bad); font-weight: 600; }
#settab td { white-space: normal; } #settab td:first-child { white-space: nowrap; }
</style></head>
<body><main>
<h1>OSPool chunk_evolve sweep</h1>
<p class="sub" id="sub"></p>
<div class="tiles" id="tiles"></div>
<div class="grid">
  <div class="card"><h2>Chunk metric vs round</h2><p>Mean chunk metric of the layout after k rounds, one line per job
    (the last point is chunk_evolve's estimate).</p><div class="chart"><canvas id="curve"></canvas></div>
    <div class="legend" id="leg1"></div></div>
  <div class="card"><h2>Spot gain per setting</h2><p>Last whole-map spot check: evolved minus base, same seeds.
    Dots are replicates; the bar is mean &plusmn; sd.</p><div class="chart"><canvas id="gain"></canvas></div></div>
  <div class="card"><h2>Rounds per job vs CPU model</h2><p>Rounds finished within the time budget, by the execute
    host's CPU; dots colored by setting.</p><div class="chart"><canvas id="cpu"></canvas></div></div>
  <div class="card"><h2>Settings</h2><div class="tablewrap" style="margin:0"><table id="settab"></table></div></div>
</div>
<div class="card tablewrap"><h2>Jobs (click a header to sort)</h2><table id="jobs"></table></div>
</main>
<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const sidx = Object.fromEntries(D.settings.map((s, i) => [s.sid, i]));
const color = sid => (sidx[sid] < 8 ? css('--s' + (sidx[sid] + 1)) : css('--other'));
const f = (x, n = 4) => (x === null || x === undefined ? '-' : (+x).toFixed(n));
const fs = (x, n = 4) => (x === null || x === undefined ? '-' : (x >= 0 ? '+' : '') + (+x).toFixed(n));
const ink2 = css('--ink2'), gridc = css('--grid');
Chart.defaults.color = ink2; Chart.defaults.borderColor = gridc; Chart.defaults.font.size = 12;
const el = (tag, text, cls) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text;
  if (cls) e.className = cls; return e; };

const count = s => D.jobs.filter(j => j.status === s).length;
document.getElementById('sub').textContent = `${D.jobs.length} jobs, metric "${D.metric}", collected ${D.made}` +
  (D.sweep ? ` - sweep ${D.sweep}` : '');
for (const [k, v] of [['ok', count('ok')], ['failed', count('failed')], ['missing', count('missing')],
                      ['unfinished', count('unfinished')], ['settings', D.settings.length]]) {
  const t = el('div', undefined, 'tile'); t.append(el('b', String(v)), el('span', k));
  document.getElementById('tiles').append(t);
}
const leg = document.getElementById('leg1');
for (const s of D.settings) { const sp = el('span'); const i = el('i'); i.style.background = color(s.sid);
  sp.append(i, document.createTextNode(s.sid + ': ' + s.label)); leg.append(sp); }

// 1. chunk metric vs round
new Chart(document.getElementById('curve'), { type: 'line',
  data: { datasets: D.jobs.filter(j => j.curve.length).map(j => ({ label: j.tag, sid: j.sid,
    data: j.curve.map((y, k) => ({ x: k, y })), borderColor: color(j.sid), backgroundColor: color(j.sid),
    borderWidth: 2, pointRadius: 3, pointHoverRadius: 6, tension: 0,
    segment: { borderDash: c => (j.est && c.p1DataIndex === j.curve.length - 1 ? [4, 3] : undefined) } })) },
  options: { maintainAspectRatio: false, parsing: false, interaction: { mode: 'nearest', intersect: false },
    plugins: { legend: { display: false }, tooltip: { callbacks: {
      label: c => `${c.dataset.label} (${c.dataset.sid}): ${f(c.parsed.y)} after ${c.parsed.x} rounds` } } },
    scales: { x: { type: 'linear', title: { display: true, text: 'rounds done' }, ticks: { precision: 0 } },
              y: { title: { display: true, text: 'chunk metric (' + D.metric + ')' } } } } });

// 2. spot gain per setting: replicate dots + mean +- sd
const jit = (i, n) => (n > 1 ? (i / (n - 1) - 0.5) * 0.3 : 0);
const dots = D.settings.map(s => { const js = D.jobs.filter(j => j.sid === s.sid && j.gain !== null);
  return { label: s.sid, data: js.map((j, i) => ({ x: sidx[s.sid] + jit(i, js.length), y: j.gain, tag: j.tag })),
    backgroundColor: color(s.sid), borderColor: css('--card'), borderWidth: 2, pointRadius: 6, pointHoverRadius: 8 };
});
const stats = D.stats.filter(s => s.mean !== null);
const meanSet = { type: 'line', label: 'mean', data: [], showLine: false, pointStyle: 'line', pointRadius: 18,
  borderWidth: 3, borderColor: ink2 };
const sdSets = stats.filter(s => s.sd !== null).map(s => ({ type: 'line', label: 'sd ' + s.sid, borderColor: ink2,
  borderWidth: 1.5, pointRadius: 0,
  data: [{ x: sidx[s.sid], y: s.mean - s.sd }, { x: sidx[s.sid], y: s.mean + s.sd }] }));
meanSet.data = stats.map(s => ({ x: sidx[s.sid], y: s.mean, n: s.n, sd: s.sd }));
new Chart(document.getElementById('gain'), { type: 'scatter', data: { datasets: [...dots, meanSet, ...sdSets] },
  options: { maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: {
      label: c => (c.raw.tag ? `${c.raw.tag}: ${fs(c.parsed.y)}` : c.dataset.label === 'mean' ?
        `mean ${fs(c.raw.y)} ± ${f(c.raw.sd)} (n=${c.raw.n})` : `${c.dataset.label}: ${fs(c.parsed.y)}`) } } },
    scales: { x: { min: -0.6, max: D.settings.length - 0.4, ticks: { stepSize: 1,
      callback: v => (Number.isInteger(v) && D.settings[v] ? D.settings[v].sid : '') }, grid: { display: false } },
      y: { title: { display: true, text: 'spot gain (evolved - base)' },
           grid: { color: c => (c.tick.value === 0 ? ink2 : gridc) } } } } });

// 3. rounds per job vs CPU model (horizontal strip: CPU names are long)
const cpus = [...new Set(D.jobs.filter(j => j.status !== 'missing').map(j => j.cpu_s))].sort();
const cdots = D.settings.map(s => { const js = D.jobs.filter(j => j.sid === s.sid && j.status !== 'missing');
  return { label: s.sid, data: js.map((j, i) => ({ x: j.rounds, y: cpus.indexOf(j.cpu_s) + jit(i % 5, 5) * 0.8,
    tag: j.tag, cpu: j.cpu })), backgroundColor: color(s.sid), borderColor: css('--card'), borderWidth: 2,
    pointRadius: 6, pointHoverRadius: 8 }; });
const cpuChart = document.getElementById('cpu');
cpuChart.parentElement.style.height = Math.max(240, 60 + cpus.length * 44) + 'px';
new Chart(cpuChart, { type: 'scatter', data: { datasets: cdots },
  options: { maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: {
      label: c => `${c.raw.tag} (${c.dataset.label}): ${c.raw.x} rounds on ${c.raw.cpu}` } } },
    scales: { x: { title: { display: true, text: 'rounds done' }, ticks: { precision: 0 }, beginAtZero: true },
      y: { min: -0.6, max: cpus.length - 0.4, reverse: true, ticks: { stepSize: 1, autoSkip: false,
        callback: v => (Number.isInteger(v) && cpus[v] !== undefined ? cpus[v].slice(0, 38) : '') },
        grid: { display: false } } } } });

// settings table
const st = document.getElementById('settab');
const sh = el('tr'); for (const h of ['set', 'knobs', 'n', 'gain mean', 'sd']) sh.append(el('th', h)); st.append(sh);
for (const s of D.settings) { const g = D.stats.find(x => x.sid === s.sid); const tr = el('tr');
  const c0 = el('td'); const i = el('i'); i.style.cssText = `display:inline-block;width:10px;height:10px;` +
    `border-radius:2px;margin-right:6px;background:${color(s.sid)}`; c0.append(i, document.createTextNode(s.sid));
  tr.append(c0, el('td', s.label), el('td', String(g.n), 'num'), el('td', fs(g.mean), 'num'), el('td', f(g.sd), 'num'));
  st.append(tr); }

// 4. sortable job table
const cols = [['tag', 'tag'], ['set', 'sid'], ['seed', 'seed'], ['status', 'status'], ['exit', 'exit'],
  ['rounds', 'rounds'], ['metric start', 'start', 4], ['metric end', 'end', 4], ['spot K', 'spot', 4],
  ['spot base', 'base', 4], ['gain', 'gain', 4, true], ['spot rnd', 'spot_round'], ['cpu', 'cpu'], ['host', 'host'],
  ['site', 'site'], ['run s', 'run_secs', 0], ['wall s', 'wall', 0], ['starts', 'starts'], ['evict', 'evictions'],
  ['mem MB', 'mem_mb']];
let sortKey = 'gain', sortDir = -1;
function drawJobs() {
  const t = document.getElementById('jobs'); t.replaceChildren();
  const hr = el('tr');
  for (const [h, k] of cols) { const th = el('th', h + (k === sortKey ? (sortDir < 0 ? ' ▼' : ' ▲') : ''));
    th.onclick = () => { sortDir = k === sortKey ? -sortDir : -1; sortKey = k; drawJobs(); }; hr.append(th); }
  t.append(hr);
  const rows = [...D.jobs].sort((a, b) => { const x = a[sortKey], y = b[sortKey];
    if (x === y) return 0; if (x === null || x === undefined) return 1; if (y === null || y === undefined) return -1;
    return (x < y ? -1 : 1) * sortDir; });
  for (const j of rows) { const tr = el('tr');
    for (const [, k, nd, sg] of cols) { const v = j[k];
      const txt = v === null || v === undefined ? '-' : nd !== undefined ? (sg ? fs(v, nd) : f(v, nd)) : String(v);
      const td = el('td', txt, typeof v === 'number' ? 'num' : undefined);
      if (k === 'status' && v !== 'ok') td.className = 'bad'; tr.append(td); }
    t.append(tr); }
}
drawJobs();
</script></body></html>
"""


def write_page(res, path):
    data = json.dumps(page_data(res)).replace("</", "<\\/")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(PAGE.replace("__CHARTJS__", CHARTJS).replace("__DATA__", data))
    return path


# ---------------------------------------------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("results_dir", nargs="?", default=DEFAULT_RESULTS, help="the job folder (results/ and logs/)")
    p.add_argument("--sweep", default=None, help="sweep.json (default: RESULTS_DIR/sweep.json, then ospool/)")
    p.add_argument("--top", type=int, default=5, help="maps listed in the ranking")
    p.add_argument("--export", type=int, default=0, metavar="N", help="copy the top N maps into --maps-dir")
    p.add_argument("--maps-dir", default=os.path.join(REPO, "maps", "penrose"))
    p.add_argument("--html", default=None, help="plots page path (default <tmp>/ospool_collect/<folder>.html)")
    p.add_argument("--no-open", action="store_true", help="write the plots page but don't open it")
    args = p.parse_args(argv)
    rd = os.path.abspath(os.path.expanduser(args.results_dir))
    if not os.path.isdir(os.path.join(rd, "results")) and not os.path.isdir(os.path.join(rd, "logs")):
        p.error(f"{rd} has neither results/ nor logs/ (run the runbook's block F first)")
    res = collect(rd, args.sweep)
    print(report(res, max(args.top, args.export)))
    if args.export:
        print("")
        print(f"export to {args.maps_dir}:")
        res["exported"] = export(res["jobs"], args.export, args.maps_dir)
        for tag, dest, what in res["exported"]:
            print(f"  {tag}: {what} -> {os.path.basename(dest)}")
    html = args.html or os.path.join(tempfile.gettempdir(), "ospool_collect", os.path.basename(rd) + ".html")
    res["html"] = write_page(res, html)
    print("")
    print(f"plots: {res['html']}")
    if not args.no_open and sys.platform == "darwin":
        subprocess.run(["open", res["html"]], check=False)
    return res


if __name__ == "__main__":
    main()
