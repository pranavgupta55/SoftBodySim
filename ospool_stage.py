"""Build / refresh the OSPool job folder for chunk_evolve map evolutions (one complete evolution per job).

    python3 ospool_stage.py [--sweep ospool/sweep.json] [--dest DIR] [--files ospool_files.txt]
                            [--env-def ospool/env.def] [--base "maps/penrose/evolved 1 halloween.json"]
                            [--commands-out ospool/COMMANDS.md] [--repo DIR] [--clean]

Follows OSPoolTesting/AGENT_RUNBOOK.md sections 4-7. The job folder (default
~/Desktop/PennAiR/OSPoolTesting/jobs/softbody-evolve-v1/) is made from templates/python-job (read only):
  - every file named in --files (repo-relative paths or globs, # comments), copied flat (inputs arrive by basename;
    run.sh moves files listed from a subfolder back into it), skipping the base map if listed,
    plus ospool_evolve.py if the list leaves it out;
  - the base map as base-v1.json, --env-def as env.def;
  - run.sh: the template with the script filled in, plus NUMBA_CACHE_DIR in the scratch dir;
  - job.sub (queue from params.txt, the full sweep) and smoke.sub (queue from params-smoke.txt, logs/smoke_*): the
    template with image, arguments, inputs, request_cpus / memory / disk filled in (+JobDurationCategory "Medium");
  - params.txt / params-smoke.txt from the sweep spec (see the _comment in ospool/sweep.json);
  - .staged: the list of files this script wrote, so a later run can remove the ones no longer wanted.
It also writes the runbook section 7 command blocks for this job to --commands-out.

Re-running is idempotent: unchanged files are not rewritten (mtimes kept, so rsync sends nothing). results/ and
logs/ are never touched. __pycache__ and .DS_Store are removed. A previously staged file that's no longer wanted is
removed; any other unknown file makes it refuse (nothing changed) unless --clean, which removes unknown top-level
files only in a folder this script staged before (.staged present); unknown folders always refuse. Nothing outside
the job folder is ever deleted. Ends with the runbook 4.7 checks (grep '@@', bash -n run.sh, py_compile *.py).
Exit status 1 on any error.
"""
import argparse
import filecmp
import glob
import itertools
import json
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
OSPOOL = os.path.expanduser("~/Desktop/PennAiR/OSPoolTesting")
TEMPLATES = os.path.join(OSPOOL, "templates", "python-job")
JOB_NAME = "softbody-evolve-v1"
SCRIPT = "ospool_evolve.py"
BASE_NAME = "base-v1.json"
BASE_MAP = os.path.join("maps", "penrose", "evolved 1 halloween.json")
DATA_DIR = "/ospool/ap40/data/pranav.gupta"
MANIFEST = ".staged"
KEEP_DIRS = ("results", "logs")
JUNK = ("__pycache__", ".DS_Store")
GENERATED = ("env.def", "run.sh", "job.sub", "smoke.sub", "params.txt", "params-smoke.txt", BASE_NAME, MANIFEST)
TAG_RE = re.compile(r"^[A-Za-z0-9_-]+$")
COLUMN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
SIZE_RE = re.compile(r"^\d+(\.\d+)?[KMGT]B$")
VALUE_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")  # no spaces, commas or quotes: a params column / an argument word


class StageError(Exception):
    pass


def fmt(v):
    return repr(v) if isinstance(v, float) else str(v)


# ---- sweep spec ------------------------------------------------------------------------------------------------

def expand_rows(section, columns, name):
    """A list of rows, or one generator dict -> [{tag, col: value}]."""
    if isinstance(section, list):
        rows = section
    elif isinstance(section, dict):
        axes = []
        for c in columns:
            if c not in section:
                raise StageError(f"sweep {name}: generator lacks column {c!r}")
            v = section[c]
            if isinstance(v, dict):
                if set(v) != {"from", "to"} or not all(isinstance(v[k], int) for k in v):
                    raise StageError(f"sweep {name}.{c}: a range is {{\"from\": int, \"to\": int}}")
                v = list(range(v["from"], v["to"] + 1))
            axes.append(v if isinstance(v, list) else [v])
        pattern = section.get("tag")
        if not isinstance(pattern, str):
            raise StageError(f"sweep {name}: generator needs a \"tag\" format string, e.g. \"e{{seed:03d}}\"")
        rows = []
        for combo in itertools.product(*axes):
            row = dict(zip(columns, combo))
            try:
                row["tag"] = pattern.format(**row)
            except (KeyError, ValueError) as e:
                raise StageError(f"sweep {name}: bad tag pattern {pattern!r}: {e}")
            rows.append(row)
    else:
        raise StageError(f"sweep {name}: a list of rows or a generator dict")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"tag", *columns}:
            raise StageError(f"sweep {name}: row {row!r} must have exactly tag + {columns}")
        if not TAG_RE.match(str(row["tag"])):
            raise StageError(f"sweep {name}: tag {row['tag']!r} (letters, digits, _ and - only)")
        for c in columns:
            if isinstance(row[c], bool) or not VALUE_RE.match(fmt(row[c])):
                raise StageError(f"sweep {name}: {c}={row[c]!r} in {row['tag']} (no spaces, commas or quotes)")
    if not rows:
        raise StageError(f"sweep {name}: no rows")
    return rows


def load_sweep(path):
    try:
        with open(path) as f:
            spec = json.load(f)
    except (OSError, ValueError) as e:
        raise StageError(f"sweep spec {path}: {e}")
    for key in ("image", "request", "fixed", "columns", "smoke", "full"):
        if key not in spec:
            raise StageError(f"sweep spec {path}: missing {key!r}")
    image = spec["image"]
    if not isinstance(image, str) or not re.match(r"^[A-Za-z0-9_.-]+-v\d+\.sif$", image):
        raise StageError(f"sweep image {image!r}: want <name>-vN.sif (versioned, never overwritten)")
    req = spec["request"]
    if not isinstance(req.get("cpus"), int) or req["cpus"] < 1:
        raise StageError("sweep request.cpus: a positive int")
    for k in ("memory", "disk"):
        if not SIZE_RE.match(str(req.get(k, ""))):
            raise StageError(f"sweep request.{k} {req.get(k)!r}: needs a unit, e.g. 4GB (bare = MB / KB)")
    columns = spec["columns"]
    if not columns or not all(isinstance(c, str) and COLUMN_RE.match(c) and c != "tag" for c in columns):
        raise StageError(f"sweep columns {columns!r}: identifiers, not 'tag'")
    if len(set(columns)) != len(columns):
        raise StageError(f"sweep columns {columns!r}: duplicates")
    fixed = spec["fixed"]
    col_flags = {"--" + c.replace("_", "-") for c in columns}
    for flag, v in fixed.items():
        if not re.match(r"^--[a-z][a-z0-9-]*$", flag):
            raise StageError(f"sweep fixed flag {flag!r}: want --name")
        if flag in col_flags or flag == "--base":
            raise StageError(f"sweep fixed flag {flag}: set per row (a column) or by the stager (--base)")
        if v is not None and (isinstance(v, bool) or not VALUE_RE.match(fmt(v))):
            raise StageError(f"sweep fixed {flag}={v!r}: no spaces, commas or quotes")
    smoke = expand_rows(spec["smoke"], columns, "smoke")
    full = expand_rows(spec["full"], columns, "full")
    tags = [r["tag"] for r in smoke + full]
    dup = sorted({t for t in tags if tags.count(t) > 1})
    if dup:
        raise StageError(f"sweep: duplicate tags {dup} (tags must be unique across smoke and full)")
    return {"image": image, "cpus": req["cpus"], "memory": req["memory"], "disk": req["disk"],
            "fixed": fixed, "columns": columns, "smoke": smoke, "full": full}


# ---- file list -------------------------------------------------------------------------------------------------

def read_file_list(path, repo, base_map=None):
    """ospool_files.txt -> [(source path, basename, repo subfolder or "")], SCRIPT first and included; an entry that
    is the base map is dropped (it ships as BASE_NAME)."""
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError as e:
        raise StageError(f"file list {path}: {e}")
    repo_real = os.path.realpath(repo)
    out = []
    for n, line in enumerate(lines, 1):
        entry = re.sub(r"(^|\s)#.*$", "", line).strip()
        if not entry:
            continue
        if os.path.isabs(entry):
            raise StageError(f"{path}:{n}: {entry!r} must be repo-relative")
        matches = sorted(glob.glob(os.path.join(repo, entry))) if glob.has_magic(entry) else [os.path.join(repo, entry)]
        if not matches:
            raise StageError(f"{path}:{n}: {entry!r} matches nothing")
        for src in matches:
            if not os.path.realpath(src).startswith(repo_real + os.sep):
                raise StageError(f"{path}:{n}: {entry!r} is outside the repo")
            if not os.path.isfile(src):
                raise StageError(f"{path}:{n}: {src} is not a file (folders aren't shipped; list the files)")
            if base_map and os.path.realpath(src) == os.path.realpath(base_map):
                print(f"note: {path}:{n}: the base map is shipped as {BASE_NAME}; entry skipped")
                continue
            sub = os.path.relpath(os.path.dirname(os.path.abspath(src)), os.path.abspath(repo))
            sub = "" if sub == "." else sub
            if sub and not re.match(r"^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*$", sub):
                raise StageError(f"{path}:{n}: folder {sub!r} has characters unsafe for run.sh")
            out.append((src, os.path.basename(src), sub))
    if SCRIPT not in [b for _, b, _ in out]:
        src = os.path.join(repo, SCRIPT)
        if not os.path.isfile(src):
            raise StageError(f"{SCRIPT} not found in {repo}")
        print(f"note: {SCRIPT} not in {path}; shipping it anyway")
        out.insert(0, (src, SCRIPT, ""))
    seen = {}
    for src, b, _ in out:
        if not re.match(r"^[A-Za-z0-9_.+-]+$", b):
            raise StageError(f"{src}: basename {b!r} has characters unsafe for transfer_input_files")
        if b in GENERATED:
            raise StageError(f"{src}: basename {b!r} clashes with a generated job file")
        if b in KEEP_DIRS or b in JUNK:
            raise StageError(f"{src}: basename {b!r} is reserved")
        if b in seen and os.path.realpath(seen[b]) != os.path.realpath(src):
            raise StageError(f"basename clash (inputs arrive flattened): {seen[b]} and {src}")
        seen[b] = src
    uniq, done = [], set()
    for src, b, sub in out:
        if b not in done:
            done.add(b)
            uniq.append((src, b, sub))
    return sorted(uniq, key=lambda e: e[1] != SCRIPT)


# ---- generated content -----------------------------------------------------------------------------------------

def template(name):
    with open(os.path.join(TEMPLATES, name)) as f:
        return f.read()


def fill(text, values, what):
    for k, v in values.items():
        token = f"@@{k}@@"
        if token not in text:
            raise StageError(f"template {what}: no {token} (has the template changed?)")
        text = text.replace(token, v)
    return text


def replace_line(text, prefix, new, what):
    lines = text.split("\n")
    hits = [i for i, ln in enumerate(lines) if ln.startswith(prefix)]
    if len(hits) != 1:
        raise StageError(f"template {what}: expected one line starting {prefix!r}, found {len(hits)}")
    lines[hits[0]] = new
    return "\n".join(lines)


def make_run_sh(shipped):
    text = fill(template("run.sh"), {"SCRIPT": SCRIPT}, "run.sh")
    anchor = "mkdir -p outputs\n"
    if anchor not in text:
        raise StageError("template run.sh: no 'mkdir -p outputs' line")
    extra = ('# numba writes its JIT cache next to the modules by default; keep it in the scratch dir explicitly.\n'
             'export NUMBA_CACHE_DIR="${_CONDOR_SCRATCH_DIR:-$PWD}/numba_cache"\n'
             'export SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1\n')
    moves = relocation(shipped)
    if moves:
        extra += ("# Inputs arrive flattened; put the files the code opens by a subfolder path back in their repo\n"
                  "# folders (from ospool_files.txt).\n")
        extra += "".join(m + "\n" for m in moves)
    return text.replace(anchor, anchor + extra, 1)


def relocation(shipped):
    """Shell commands that move the shipped files listed from a repo subfolder back into it (run in the flat dir)."""
    subs = {}
    for _, b, sub in shipped:
        if sub:
            subs.setdefault(sub, []).append(b)
    if not subs:
        return []
    return [f"mkdir -p {' '.join(sorted(subs))}"] + [f"mv -f {' '.join(subs[d])} {d}/" for d in sorted(subs)]


def make_sub(sweep, inputs, params, log_prefix):
    cols = sweep["columns"]
    args = ["$(tag)", "--base", BASE_NAME] + [f"--{c.replace('_', '-')} $({c})" for c in cols]
    for flag, v in sweep["fixed"].items():
        args.append(flag if v is None else f"{flag} {fmt(v)}")
    text = template("job.sub")
    text = fill(text, {"IMAGE": sweep["image"], "ARGS": " ".join(args), "INPUTS": ", ".join(inputs),
                       "MEMORY": sweep["memory"], "DISK": sweep["disk"]}, "job.sub")
    text = replace_line(text, "request_cpus", f"request_cpus   = {sweep['cpus']}", "job.sub")
    text = replace_line(text, "+JobDurationCategory", '+JobDurationCategory = "Medium"', "job.sub")
    text = replace_line(text, "queue ", f"queue tag, {', '.join(cols)} from {params}", "job.sub")
    for kind in ("log", "output", "error"):
        ext = {"log": "log", "output": "out", "error": "err"}[kind]
        text = replace_line(text, f"{kind} ", f"{kind:<6} = logs/{log_prefix}_$(Cluster)_$(Process).{ext}", "job.sub")
    head =(f"# GENERATED by SoftBodySim/ospool_stage.py from ospool/sweep.json: edit that and re-stage, not this.\n"
            f"# $(tag) appears twice in arguments: run.sh eats the first, {SCRIPT} takes the second as TAG.\n")
    return head + text


def make_params(rows, cols):
    return "".join(", ".join([r["tag"]] + [fmt(r[c]) for c in cols]) + "\n" for r in rows)


def commands_md(job, image, smoke, full, shipped):
    n_s, n_f = len(smoke), len(full)
    moves = "".join(f"  {m}\n" for m in relocation(shipped))
    smoke_tags = " ".join(r["tag"] for r in smoke)
    return f"""# OSPool commands: {job}

GENERATED by `ospool_stage.py` (re-staging rewrites it). These are `OSPoolTesting/AGENT_RUNBOOK.md` section 7
blocks A-F filled in for this job. **The human runs them; agents never SSH.** Every `== ` line marks a section to
paste back. Order: stage locally, A, B (new image only), C1 smoke, D, E1, (F), then C2 full batch, D, E2, F.

- Job folder: `OSPoolTesting/jobs/{job}/` -> `~/jobs/{job}/` on the AP.
- Image: `{image}` (in `{DATA_DIR}/images/`). Bump `-vN` in `ospool/sweep.json`
  whenever `ospool/env.def` changes.
- Smoke: {n_s} job(s) from `smoke.sub` / `params-smoke.txt` (tags: {smoke_tags}), logs in `logs/smoke_*`.
- Full: {n_f} job(s) from `job.sub` / `params.txt`, logs in `logs/job_*`.

**0. Laptop: stage (re-run after any code / sweep change, then repeat block A)**
```bash
cd ~/PycharmProjects/SoftBodySim && .venv/bin/python3 ospool_stage.py
```

**A. Laptop: send the job folder** (results/ and logs/ stay out: they only ever come back)
```bash
cd ~/Desktop/PennAiR/OSPoolTesting
rsync -av --exclude .DS_Store --exclude __pycache__ --exclude results --exclude logs \\
  jobs/{job} ospool:~/jobs/
ssh ospool
```

**B. AP: build and check the image** (only for a new image; skip it if `ls {DATA_DIR}/images/` lists
`{image}`)
```bash
mkdir -p $HOME/tmp /ospool/ap40/data/$USER/images
export TMPDIR=$HOME/tmp APPTAINER_TMPDIR=$HOME/tmp APPTAINER_CACHEDIR=$HOME/tmp
cd ~/jobs/{job}
echo "== BUILD"; apptainer build --ignore-proot {image} env.def 2>&1 | tail -8
echo "== IMAGE CHECK"; apptainer exec {image} python3 -c \\
  "import sys, numpy, numba, pygame; print(sys.version, numpy.__version__, numba.__version__, pygame.version.ver)"
# job imports, in a throwaway copy laid out the way run.sh lays out the job's scratch dir:
rm -rf $HOME/tmp/importcheck && mkdir -p $HOME/tmp/importcheck
grep -v '^#' {MANIFEST} | while read -r f; do cp "$f" $HOME/tmp/importcheck/; done
( cd $HOME/tmp/importcheck
{moves}  echo "== JOB IMPORTS"; SDL_VIDEODRIVER=dummy apptainer exec \\
    ~/jobs/{job}/{image} python3 -c "import {SCRIPT[:-3]}, chunk_evolve; print('ok')" )
rm -rf $HOME/tmp/importcheck
ls -la {image} && mv {image} /ospool/ap40/data/$USER/images/
```

**C1. AP: dry run both submit files, then submit the smoke job only**
```bash
cd ~/jobs/{job}
ls -la /ospool/ap40/data/$USER/images/{image}     # must exist
mkdir -p logs results
echo "== DRY RUN SMOKE"; condor_submit -dry-run dry.ad smoke.sub && \\
  grep -E '^(Args|RequestCpus|RequestMemory|RequestDisk)=' dry.ad; rm -f dry.ad
echo "== DRY RUN FULL"; condor_submit -dry-run dry.ad job.sub && grep -cE '^Args=' dry.ad; rm -f dry.ad
condor_submit smoke.sub
```
The smoke dry run must print `{n_s} job(s) dry-run` and {n_s} `Args=` line(s); the full one `{n_f} job(s) dry-run`
and the count {n_f}. If not, stop and paste the output.

**D. AP: monitor** (`condor_watch_q` printed nothing in this terminal, so don't use it)
```bash
condor_q -nobatch          # re-run every few min; done when your jobs disappear
condor_q -held -af HoldReason      # only if any job shows ST = H
```

**E1. AP: verify the smoke job, then paste everything from here down**
```bash
cd ~/jobs/{job}
echo "== EXIT CODES"; grep -h -A1 'Job terminated' logs/smoke_*.log | grep -o 'return value [0-9]*' | sort | uniq -c
echo "== RESULTS"; ls -la results/
echo "== TARBALL"; for t in {smoke_tags}; do tar tzvf results/outputs_$t.tar.gz | head -20; done
echo "== STDERR"; for f in logs/smoke_*.err; do [ -s "$f" ] && {{ echo "-- $f"; tail -20 "$f"; }}; done
echo "== STDOUT"; tail -40 "$(ls logs/smoke_*.out | tail -1)"
echo "== USAGE"; grep -h -A7 'Partitionable' "$(ls logs/smoke_*.log | tail -1)"
echo "== HISTORY"; condor_history $USER -limit {n_s} -af ClusterId ProcId ExitCode RemoteWallClockTime
exit
```
Go on to C2 only if the smoke job returned 0 with a sane tarball and the usage table fits the requests.

**C2. AP: submit the full batch**
```bash
cd ~/jobs/{job}
mkdir -p logs results
echo "== DRY RUN FULL"; condor_submit -dry-run dry.ad job.sub && grep -E '^Args=' dry.ad | head -3; rm -f dry.ad
condor_submit job.sub
```
Then monitor with block D.

**E2. AP: verify the full batch, then paste everything from here down**
```bash
cd ~/jobs/{job}
echo "== EXIT CODES"; grep -h -A1 'Job terminated' logs/job_*.log | grep -o 'return value [0-9]*' | sort | uniq -c
echo "== RESULTS"; ls results/*.tar.gz | wc -l; du -sh results/
echo "== STDERR"; for f in logs/job_*.err; do [ -s "$f" ] && {{ echo "-- $f"; tail -20 "$f"; }}; done
echo "== STDOUT (first job)"; tail -40 "$(ls logs/job_*.out | head -1)"
echo "== USAGE (first job)"; grep -h -A7 'Partitionable' "$(ls logs/job_*.log | head -1)"
echo "== HISTORY"; condor_history $USER -limit {n_f} -af ClusterId ProcId ExitCode RemoteWallClockTime
exit
```
Expect {n_f} tarballs (plus {n_s} smoke). Missing tags: see runbook section 8.

**F. Laptop: bring the results home** (after E1 for the smoke job, after E2 for the batch; re-running is fine)
```bash
cd ~/Desktop/PennAiR/OSPoolTesting
# no trailing slash on the source: creates jobs/{job}/results/
rsync -av ospool:~/jobs/{job}/results jobs/{job}/
rsync -av ospool:~/jobs/{job}/logs    jobs/{job}/
for f in jobs/{job}/results/*.tar.gz; do d="${{f%.tar.gz}}"; mkdir -p "$d" && tar xzf "$f" -C "$d"; done
ls -R jobs/{job}/results | head -50
```
"""


# ---- staging ---------------------------------------------------------------------------------------------------

def write_if_changed(path, text, mode=None):
    old = None
    if os.path.isfile(path) and not os.path.islink(path):
        with open(path) as f:
            old = f.read()
    changed = old != text
    if changed:
        if os.path.islink(path):
            os.unlink(path)
        with open(path, "w") as f:
            f.write(text)
    if mode is not None and (os.stat(path).st_mode & 0o777) != mode:
        os.chmod(path, mode)
    return changed


def copy_if_changed(src, dst):
    if os.path.isfile(dst) and not os.path.islink(dst) and filecmp.cmp(src, dst, shallow=False):
        return False
    if os.path.islink(dst):
        os.unlink(dst)
    shutil.copy2(src, dst)
    return True


def remove_entry(dest, name):
    """Delete dest/name (a top-level entry of the job folder only; symlinks are unlinked, never followed)."""
    path = os.path.join(dest, name)
    if os.sep in name or name in ("", ".", "..") or os.path.dirname(os.path.abspath(path)) != os.path.abspath(dest):
        raise StageError(f"refusing to delete {path!r}")
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def stage(repo, sweep_path, files_path, env_def, base_map, dest, commands_out, clean=False):
    """Stage the job folder; returns {"written": [...], "removed": [...], "files": [...]}. Raises StageError."""
    sweep = load_sweep(sweep_path)
    shipped = read_file_list(files_path, repo, base_map)
    if not os.path.isfile(env_def):
        raise StageError(f"env.def source {env_def} not found")
    if not os.path.isfile(base_map):
        raise StageError(f"base map {base_map} not found")
    for t in ("job.sub", "run.sh"):
        if not os.path.isfile(os.path.join(TEMPLATES, t)):
            raise StageError(f"template {t} not found in {TEMPLATES}")
    dest = os.path.abspath(dest)
    if os.path.realpath(dest) in (os.path.realpath(p) for p in (TEMPLATES, OSPOOL, repo, os.path.expanduser("~"), "/")):
        raise StageError(f"refusing to stage into {dest}")

    inputs = [b for _, b, _ in shipped] + [BASE_NAME]
    with open(env_def) as f:
        env_text = f.read()
    gen = {
        "env.def": (env_text, 0o644),
        "run.sh": (make_run_sh(shipped), 0o755),
        "job.sub": (make_sub(sweep, inputs, "params.txt", "job"), 0o644),
        "smoke.sub": (make_sub(sweep, inputs, "params-smoke.txt", "smoke"), 0o644),
        "params.txt": (make_params(sweep["full"], sweep["columns"]), 0o644),
        "params-smoke.txt": (make_params(sweep["smoke"], sweep["columns"]), 0o644),
    }
    copies = {b: src for src, b, _ in shipped}
    copies[BASE_NAME] = base_map
    wanted = set(gen) | set(copies) | {MANIFEST}

    # Look before touching anything.
    removed, unknown, stale = [], [], []
    if os.path.exists(dest) and not os.path.isdir(dest):
        raise StageError(f"{dest} exists and is not a folder")
    previous = set()
    has_manifest = os.path.isfile(os.path.join(dest, MANIFEST))
    if has_manifest:
        with open(os.path.join(dest, MANIFEST)) as f:
            previous = {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
    if os.path.isdir(dest):
        for name in sorted(os.listdir(dest)):
            path = os.path.join(dest, name)
            if name in JUNK:
                stale.append(name)
            elif name in KEEP_DIRS and os.path.isdir(path) and not os.path.islink(path):
                continue
            elif name in wanted:
                if os.path.isdir(path) and not os.path.islink(path):
                    unknown.append(name + "/")
            elif name in previous and not (os.path.isdir(path) and not os.path.islink(path)):
                stale.append(name)
            else:
                unknown.append(name + ("/" if os.path.isdir(path) and not os.path.islink(path) else ""))
    if unknown:
        dirs = [u for u in unknown if u.endswith("/")]
        if not clean or dirs or not has_manifest:
            why = ("folders are never removed" if dirs else
                   "--clean only works in a folder staged before (.staged present)" if clean else "pass --clean")
            raise StageError(f"unknown entries in {dest}: {unknown} ({why}); nothing was changed")
        stale += unknown

    os.makedirs(dest, exist_ok=True)
    for name in stale:
        remove_entry(dest, name)
        removed.append(name)
    written = []
    for name, (text, mode) in gen.items():
        if write_if_changed(os.path.join(dest, name), text, mode):
            written.append(name)
    for name, src in copies.items():
        if copy_if_changed(src, os.path.join(dest, name)):
            written.append(name)
    manifest = "# written by SoftBodySim/ospool_stage.py: files it owns here (removed when no longer staged)\n"
    manifest += "".join(n + "\n" for n in sorted(wanted - {MANIFEST}))
    if write_if_changed(os.path.join(dest, MANIFEST), manifest, 0o644):
        written.append(MANIFEST)
    if commands_out:
        os.makedirs(os.path.dirname(os.path.abspath(commands_out)), exist_ok=True)
        if write_if_changed(commands_out, commands_md(os.path.basename(dest), sweep["image"], sweep["smoke"],
                                                      sweep["full"], shipped)):
            written.append(commands_out)
    local_checks(dest)
    return {"written": written, "removed": removed, "files": sorted(wanted), "sweep": sweep}


def local_checks(dest):
    """Runbook 4.7: grep '@@' (must print nothing; -I skips binaries like the fonts), bash -n run.sh,
    py_compile *.py, then drop __pycache__."""
    names = [n for n in sorted(os.listdir(dest)) if os.path.isfile(os.path.join(dest, n))]
    g = subprocess.run(["grep", "-nI", "@@"] + names, cwd=dest, capture_output=True, text=True)
    if g.stdout.strip():
        raise StageError(f"leftover @@ in {dest}:\n{g.stdout}")
    b = subprocess.run(["bash", "-n", "run.sh"], cwd=dest, capture_output=True, text=True)
    if b.returncode:
        raise StageError(f"bash -n run.sh failed:\n{b.stderr}")
    pys = [n for n in names if n.endswith(".py")]
    try:
        if pys:
            p = subprocess.run([sys.executable, "-m", "py_compile"] + pys, cwd=dest, capture_output=True, text=True)
            if p.returncode:
                raise StageError(f"py_compile failed:\n{p.stderr}{p.stdout}")
    finally:
        cache = os.path.join(dest, "__pycache__")
        if os.path.isdir(cache) and not os.path.islink(cache):
            shutil.rmtree(cache)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sweep", default=os.path.join(REPO, "ospool", "sweep.json"))
    p.add_argument("--files", default=os.path.join(REPO, "ospool_files.txt"))
    p.add_argument("--env-def", default=os.path.join(REPO, "ospool", "env.def"))
    p.add_argument("--base", default=None, help=f"the base map (default: <repo>/{BASE_MAP}); shipped as {BASE_NAME}")
    p.add_argument("--dest", default=os.path.join(OSPOOL, "jobs", JOB_NAME), help="the job folder (override: tests)")
    p.add_argument("--commands-out", default=os.path.join(REPO, "ospool", "COMMANDS.md"),
                   help="where to write the filled-in command blocks ('' to skip)")
    p.add_argument("--repo", default=REPO, help="root the --files paths are relative to")
    p.add_argument("--clean", action="store_true", help="remove unknown top-level files in a previously staged folder")
    a = p.parse_args(argv)
    try:
        res = stage(a.repo, a.sweep, a.files, a.env_def, a.base or os.path.join(a.repo, BASE_MAP), a.dest,
                    a.commands_out, a.clean)
    except StageError as e:
        print(f"ospool_stage: {e}", file=sys.stderr)
        return 1
    sw = res["sweep"]
    print(f"staged {a.dest}: {len(res['files'])} files, {len(sw['smoke'])} smoke + {len(sw['full'])} full jobs, "
          f"image {sw['image']}, {sw['cpus']} cpus / {sw['memory']} / {sw['disk']}")
    print(f"  written: {', '.join(res['written']) or 'nothing (unchanged)'}")
    if res["removed"]:
        print(f"  removed: {', '.join(res['removed'])}")
    print("  checks ok: no @@, bash -n run.sh, py_compile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
