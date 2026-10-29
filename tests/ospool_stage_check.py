"""Checks for ospool_stage.py (the OSPool job-folder stager). Run from the repo root:

    .venv/bin/python3 tests/ospool_stage_check.py

Everything stages into temp folders (--dest / --commands-out overrides); the real job folder is never touched.
1. Stub repo (a stub ospool_evolve.py, a module, a data file in a subfolder, a base map with spaces in its name, a
   stub env.def and file list): the folder is complete (shipped files flat, base-v1.json, env.def, run.sh +x,
   job.sub, smoke.sub, both params files, .staged), placeholder-free, and job.sub / smoke.sub / run.sh / params say
   what the sweep spec says; COMMANDS.md is filled in.
2. Idempotent: a second run writes nothing and keeps mtimes; results/ and logs/ survive; __pycache__ / .DS_Store go.
3. Stale files: a file dropped from the list is removed; an unknown file refuses (nothing changed) until --clean;
   an unknown folder always refuses; --clean in a folder never staged refuses and deletes nothing.
4. Bad input refuses: basename clash, missing file, bad / duplicate tags, memory without a unit, a fixed flag that
   duplicates a column, a syntax error in a shipped .py (py_compile).
5. The real repo (ospool_files.txt, ospool/env.def, ospool/sweep.json), if those files exist yet; skipped otherwise.
Exit status 1 on any failure.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.getcwd())
import ospool_stage as st  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []


def check(ok, what):
    print(f"{'ok  ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        FAILS.append(what)


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def readb(path):
    with open(path, "rb") as f:
        return f.read()


def read(path):
    with open(path) as f:
        return f.read()


SWEEP = {
    "image": "softbody-test-v1.sif",
    "request": {"cpus": 4, "memory": "4GB", "disk": "3GB"},
    "fixed": {"--variants": 6, "--seam-threshold": 0.04, "--metric": "free"},
    "columns": ["hours", "seed"],
    "smoke": [{"tag": "smoke1", "hours": 0.33, "seed": 1}],
    "full": {"tag": "e{seed:03d}", "hours": 4, "seed": {"from": 1, "to": 5}},
}


def make_stub(root):
    write(os.path.join(root, "ospool_evolve.py"), "import mod_a\nprint('stub', mod_a.X)\n")
    write(os.path.join(root, "mod_a.py"), "X = 1\n")
    write(os.path.join(root, "data", "table.json"), "{}\n")
    write(os.path.join(root, "maps", "penrose", "evolved 1 halloween.json"), '{"tiles": []}\n')
    write(os.path.join(root, "ospool", "env.def"), "Bootstrap: docker\nFrom: python:3.13-slim\n")
    write(os.path.join(root, "ospool_files.txt"), "# stub list\nospool_evolve.py\nmod_a.py  # inline comment\n"
                                                  "data/table.json\nmaps/penrose/evolved 1 halloween.json\n")
    write(os.path.join(root, "sweep.json"), json.dumps(SWEEP))


def run(root, dest, *extra, sweep=None):
    cmd = [sys.executable, os.path.join(REPO, "ospool_stage.py"), "--repo", root, "--dest", dest,
           "--files", os.path.join(root, "ospool_files.txt"), "--env-def", os.path.join(root, "ospool", "env.def"),
           "--sweep", sweep or os.path.join(root, "sweep.json"), "--commands-out", os.path.join(root, "COMMANDS.md"),
           *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def snapshot(dest):
    return {n: (os.stat(os.path.join(dest, n)).st_mtime_ns, readb(os.path.join(dest, n)))
            for n in os.listdir(dest) if os.path.isfile(os.path.join(dest, n))}


def check_folder(dest, shipped, sweep, label):
    names = set(os.listdir(dest)) - set(st.KEEP_DIRS)
    want = set(shipped) | {"base-v1.json", "env.def", "run.sh", "job.sub", "smoke.sub", "params.txt",
                           "params-smoke.txt", ".staged"}
    check(names == want, f"{label}: folder holds exactly the expected files (extra {sorted(names - want)}, "
                         f"missing {sorted(want - names)})")
    leftovers = []
    for n in names:
        with open(os.path.join(dest, n), "rb") as f:
            data = f.read()
        if b"\0" not in data and b"@@" in data:  # text files only (the fonts are binary)
            leftovers.append(n)
    check(not leftovers, f"{label}: no @@ placeholders ({leftovers})")
    check(os.access(os.path.join(dest, "run.sh"), os.X_OK), f"{label}: run.sh is executable")
    run_sh = read(os.path.join(dest, "run.sh"))
    check("python3 -u ospool_evolve.py \"$@\"" in run_sh and "NUMBA_CACHE_DIR=" in run_sh
          and run_sh.index("NUMBA_CACHE_DIR") < run_sh.index("python3 -u"), f"{label}: run.sh runs the script, "
                                                                           f"numba cache set first")
    sub = read(os.path.join(dest, "job.sub"))
    smoke = read(os.path.join(dest, "smoke.sub"))
    lines = {ln.split("=")[0].strip(): ln.split("=", 1)[1].strip() for ln in sub.splitlines()
             if "=" in ln and not ln.startswith("#")}
    inputs = [s.strip() for s in lines["transfer_input_files"].split(",")]
    check(inputs[0] == "ospool_evolve.py" and sorted(inputs) == sorted(set(shipped) | {"base-v1.json"}),
          f"{label}: transfer_input_files = shipped files + base-v1.json, script first ({inputs})")
    check(lines["container_image"] == f"osdf:///ospool/ap40/data/pranav.gupta/images/{sweep['image']}",
          f"{label}: container_image")
    cols = sweep["columns"]
    per_row = " ".join(f"--{c.replace('_', '-')} $({c})" for c in cols)
    fixed = " ".join(k if v is None else f"{k} {v}" for k, v in sweep["fixed"].items())
    check(lines["arguments"] == f"$(tag) $(tag) --base base-v1.json {per_row} {fixed}".rstrip(),
          f"{label}: arguments ({lines['arguments']})")
    req = sweep["request"]
    check((lines["request_cpus"], lines["request_memory"], lines["request_disk"])
          == (str(req["cpus"]), req["memory"], req["disk"]), f"{label}: request_cpus / memory / disk")
    check(lines["+JobDurationCategory"] == '"Medium"', f"{label}: +JobDurationCategory Medium")
    q = "queue tag, " + ", ".join(cols)
    check(sub.rstrip().endswith(q + " from params.txt") and smoke.rstrip().endswith(q + " from params-smoke.txt"),
          f"{label}: queue lines")
    check("logs/job_$(Cluster)_$(Process).log" in sub and "logs/smoke_$(Cluster)_$(Process).err" in smoke,
          f"{label}: job / smoke logs apart")
    check(sub.replace("params.txt", "P").replace("logs/job_", "L") ==
          smoke.replace("params-smoke.txt", "P").replace("logs/smoke_", "L"), f"{label}: smoke.sub = job.sub "
                                                                              f"but for params and log names")
    return sub


def part1_2_3_4(tmp):
    root = os.path.join(tmp, "stub")
    dest = os.path.join(tmp, "jobs", "softbody-evolve-v1")
    make_stub(root)
    shipped = ["ospool_evolve.py", "mod_a.py", "table.json"]

    r = run(root, dest)
    check(r.returncode == 0, f"stub: stages ({r.stderr.strip()[-300:]})")
    sub = check_folder(dest, shipped, SWEEP, "stub")
    run_sh = read(os.path.join(dest, "run.sh"))
    check("mkdir -p data\nmv -f table.json data/\n" in run_sh and run_sh.index("mv -f") < run_sh.index("python3 -u"),
          "stub: run.sh moves data/table.json back into data/ before the script")
    check("halloween" not in read(os.path.join(dest, "job.sub")), "stub: the listed base map isn't shipped twice")
    params = read(os.path.join(dest, "params.txt")).splitlines()
    check(params == [f"e{s:03d}, 4, {s}" for s in range(1, 6)], f"stub: params.txt rows ({params})")
    check(read(os.path.join(dest, "params-smoke.txt")) == "smoke1, 0.33, 1\n", "stub: params-smoke.txt")
    check(read(os.path.join(dest, "env.def")) == read(os.path.join(root, "ospool", "env.def")), "stub: env.def copy")
    check(read(os.path.join(dest, "base-v1.json")) == '{"tiles": []}\n', "stub: base map shipped as base-v1.json")
    cmds = read(os.path.join(root, "COMMANDS.md"))
    check("softbody-test-v1.sif" in cmds and "jobs/softbody-evolve-v1" in cmds and "<" + "job-name>" not in cmds
          and "5 job(s) dry-run" in cmds and "1 job(s) dry-run" in cmds and "import sys, numpy, numba, pygame" in cmds,
          "stub: COMMANDS.md filled in (image, job, sweep sizes, image check)")
    long_lines = [ln for ln in cmds.splitlines() if len(ln) > 120]
    check(not long_lines, f"stub: COMMANDS.md lines <= 120 chars ({long_lines[:2]})")
    check(subprocess.run(["bash", "-n"], input="\n".join(
        b.split("```")[0] for b in cmds.split("```bash\n")[1:]), text=True).returncode == 0,
        "stub: COMMANDS.md bash blocks parse (bash -n)")

    # 2. idempotent; kept folders; junk removed
    os.makedirs(os.path.join(dest, "results", "outputs_e001"))
    write(os.path.join(dest, "results", "outputs_e001.tar.gz"), "x")
    write(os.path.join(dest, "logs", "job_1_0.log"), "x")
    os.makedirs(os.path.join(dest, "__pycache__"))
    write(os.path.join(dest, ".DS_Store"), "x")
    before = snapshot(dest)
    r = run(root, dest)
    check(r.returncode == 0 and "nothing (unchanged)" in r.stdout, f"rerun: writes nothing ({r.stdout.strip()})")
    check(snapshot(dest) == {k: v for k, v in before.items() if k != ".DS_Store"}, "rerun: files and mtimes kept")
    check(not os.path.exists(os.path.join(dest, "__pycache__")) and not os.path.exists(os.path.join(dest, ".DS_Store")),
          "rerun: __pycache__ and .DS_Store removed")
    check(os.path.isfile(os.path.join(dest, "results", "outputs_e001.tar.gz"))
          and os.path.isdir(os.path.join(dest, "results", "outputs_e001"))
          and os.path.isfile(os.path.join(dest, "logs", "job_1_0.log")), "rerun: results/ and logs/ untouched")

    # 3. stale and unknown entries
    write(os.path.join(root, "ospool_files.txt"), "ospool_evolve.py\nmod_a.py\n")
    r = run(root, dest)
    check(r.returncode == 0 and not os.path.exists(os.path.join(dest, "table.json")) and "table.json" in r.stdout,
          "stale: a file dropped from the list is removed")
    check_folder(dest, ["ospool_evolve.py", "mod_a.py"], SWEEP, "stale")
    write(os.path.join(dest, "notes.txt"), "mine")
    before = snapshot(dest)
    r = run(root, dest)
    check(r.returncode == 1 and "notes.txt" in r.stderr and snapshot(dest) == before,
          "unknown file: refuses, nothing changed")
    r = run(root, dest, "--clean")
    check(r.returncode == 0 and not os.path.exists(os.path.join(dest, "notes.txt")), "unknown file: --clean removes")
    os.makedirs(os.path.join(dest, "outputs"))
    r = run(root, dest, "--clean")
    check(r.returncode == 1 and os.path.isdir(os.path.join(dest, "outputs")), "unknown folder: refuses even with "
                                                                             "--clean, kept")
    shutil.rmtree(os.path.join(dest, "outputs"))
    foreign = os.path.join(tmp, "foreign")
    write(os.path.join(foreign, "precious.txt"), "keep")
    r = run(root, foreign, "--clean")
    check(r.returncode == 1 and os.listdir(foreign) == ["precious.txt"],
          "never-staged folder: --clean refuses, deletes nothing, writes nothing")

    # 4. bad input
    def refuses(what, files=None, sweep=None, needle=""):
        if files is not None:
            write(os.path.join(root, "ospool_files.txt"), files)
        sp = None
        if sweep is not None:
            sp = os.path.join(root, "bad_sweep.json")
            write(sp, json.dumps(sweep))
        before = snapshot(dest)
        r = run(root, dest, sweep=sp)
        check(r.returncode == 1 and needle in r.stderr and snapshot(dest) == before,
              f"refuses {what} ({r.stderr.strip()[:110]})")
        write(os.path.join(root, "ospool_files.txt"), "ospool_evolve.py\nmod_a.py\n")

    write(os.path.join(root, "other", "mod_a.py"), "X = 2\n")
    refuses("a basename clash", files="ospool_evolve.py\nmod_a.py\nother/mod_a.py\n", needle="clash")
    refuses("a missing file", files="ospool_evolve.py\nnope.py\n", needle="nope.py")
    refuses("a bad tag", sweep={**SWEEP, "smoke": [{"tag": "a b", "hours": 1, "seed": 1}]}, needle="tag")
    refuses("a duplicate tag", sweep={**SWEEP, "smoke": [{"tag": "e001", "hours": 1, "seed": 1}]}, needle="duplicate")
    refuses("memory without a unit", sweep={**SWEEP, "request": {"cpus": 1, "memory": "4", "disk": "3GB"}},
            needle="unit")
    refuses("a fixed flag duplicating a column", sweep={**SWEEP, "fixed": {"--seed": 3}}, needle="--seed")
    write(os.path.join(root, "broken.py"), "def f(:\n")
    write(os.path.join(root, "ospool_files.txt"), "ospool_evolve.py\nmod_a.py\nbroken.py\n")
    r = run(root, dest)
    check(r.returncode == 1 and "py_compile" in r.stderr and not os.path.exists(os.path.join(dest, "__pycache__")),
          "py_compile failure is reported, __pycache__ removed")
    check("@@" not in sub, "stub: job.sub placeholder-free")


def part5(tmp):
    need = [os.path.join(REPO, p) for p in ("ospool_files.txt", os.path.join("ospool", "env.def"),
                                            os.path.join("ospool", "sweep.json"), "ospool_evolve.py")]
    missing = [os.path.relpath(p, REPO) for p in need if not os.path.isfile(p)]
    if missing:
        print(f"skip real-repo staging: not there yet: {', '.join(missing)}", flush=True)
        return
    dest = os.path.join(tmp, "real", "softbody-evolve-v1")
    cmds = os.path.join(tmp, "real", "COMMANDS.md")
    r = subprocess.run([sys.executable, os.path.join(REPO, "ospool_stage.py"), "--dest", dest, "--commands-out", cmds],
                       capture_output=True, text=True)
    check(r.returncode == 0, f"real: stages ({(r.stdout + r.stderr).strip()[-400:]})")
    if r.returncode:
        return
    listed = st.read_file_list(os.path.join(REPO, "ospool_files.txt"), REPO, os.path.join(REPO, st.BASE_MAP))
    shipped = [b for _, b, _ in listed]
    with open(os.path.join(REPO, "ospool", "sweep.json")) as f:
        sweep = json.load(f)
    check_folder(dest, shipped, sweep, "real")
    check(read(os.path.join(dest, "base-v1.json")) == read(os.path.join(REPO, st.BASE_MAP)),
          "real: base-v1.json is maps/penrose/evolved 1 halloween.json")
    run_sh = read(os.path.join(dest, "run.sh"))
    subs = sorted({sub for _, _, sub in listed if sub})
    check(all(f"mv -f {' '.join(b for _, b, s in listed if s == sub)} {sub}/" in run_sh for sub in subs),
          f"real: run.sh puts the subfolder files back ({subs})")


def main():
    with tempfile.TemporaryDirectory(prefix="ospool_stage_") as tmp:
        part1_2_3_4(tmp)
        part5(tmp)
    print(f"\n{'FAILED: ' + str(len(FAILS)) if FAILS else 'all ok'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
