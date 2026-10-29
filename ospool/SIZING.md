# OSPool sizing and sweep for chunk_evolve (option 1)

One OSPool job = one complete `chunk_evolve` evolution (`ospool_evolve.py`), looping rounds until its deadline.
Measured 2026-10-29 on the laptop (Apple M4, 4 performance + 6 efficiency cores, 16 GB), load average 4.5-5.7
from other apps during the runs. Scripts and logs: the session scratchpad `ospool_d/` (not in the repo).

## sweep.json schema

`ospool/sweep.json` uses **agent C's schema**, the one `ospool_stage.py` (`load_sweep`) validates. (The brief's
`defaults/hours/cpus/rows` layout was dropped because C's stager already existed and reads this one; the file
validates with `load_sweep`.)

| Key | Meaning |
|---|---|
| `image` | container image file name, `<name>-vN.sif` |
| `request` | `{"cpus": int, "memory": "4GB", "disk": "3GB"}` (units required). `cpus` also sets `--workers`: `ospool_evolve` reads `OMP_NUM_THREADS`, which HTCondor sets to `request_cpus` |
| `fixed` | chunk_evolve flags given to every job, `{"--flag": value}` (`null` = bare flag). `--base` is set by the stager |
| `columns` | per-row values, each passed as `--<column>` (`_` becomes `-`). Here: `hours` (job wall budget; `ospool_evolve` sets the deadline to start + hours - 5 min), `seed` (search seed), then the swept knobs `metric, seam_threshold, variants, seeds, remove, add, targeted_share` |
| `smoke` | list of rows `{tag, <every column>}`: here one 0.4 h row |
| `full` | list of rows (or one generator dict) for the full batch: here 80 rows |

Tags: letters, digits, `_`, `-`, unique across smoke and full. Regenerate with `ospool_d/make_sweep.py`
(scratchpad), or edit the JSON and re-run `ospool_stage.py`, which re-validates it.

## 1. Measurements

| What | Result | How |
|---|---|---|
| Per chunk test, 1 process, warm JIT | **0.55 s** (0.49-0.66 over the 16 grid chunks; first test after load 0.92 s) | `single.py`, 17 tests, `nice -n 10` |
| Per chunk test inside chunk_evolve, 2 workers | **0.44 s** mean test wall; round wall = sum of test walls / workers + ~2 s main-process work (3 %) | 1 round, `--variants 2 --seeds 2`: 244 tests in 56 s |
| Earlier local figure (brief) | 0.83 s/core idle Mac with 6 workers (some land on efficiency cores), 2.97 s contended | not re-measured |
| Numba JIT, cold cache | **~10.4 s** extra on the first chunk test (10.9 s vs 0.46 s). Cache written: 3.3 MB, 28 index files | `NUMBA_CACHE_DIR` = empty dir |
| Whole-map spot run | **0.35 s wall per sim second** (30 s sim: 10.4-10.9 s); a 120 s spot is ~42 s on the Mac | `--spot-seconds 30` |
| Peak RSS, main process | **~540 MB** for ~1 s at startup (MapBackend / TileUniverse build, before the pool starts), then **~100 MB** steady; `/usr/bin/time -l` max RSS 564 MB | ps sampling every 1 s + `/usr/bin/time -l` |
| Peak RSS, pool worker | **~160 MB** steady (warm cache); **~330 MB** while compiling with a cold cache (single-process cold run peak 331 MB, warm 189 MB) | ps sampling; `single.py` cold vs warm |
| Peak RSS, spot worker | **~190 MB** (separate spawn processes; the chunk pool workers stay resident meanwhile) | ps sampling |
| `runs.jsonl` per round | **~561 B per chunk-test line**; ~2,000 tests/round at defaults → **~1.1 MB/round** raw, gzip ~6x → ~0.19 MB | 244-test run: 140 KB raw, 22.7 KB gzipped |
| `layouts/rN.json` per round | **18 KB** | |
| Map file | 18 KB | |

Caveats: macOS compresses memory under pressure, so RSS readings dip and may understate Linux usage; sampling at
1 s can miss spikes (the OSPool log's usage table misses them too, runbook T5). The Mac per-test numbers above
ran mostly on performance cores; under full load with 6+ workers the Mac is closer to 0.83 s.

Other finding: the code opens `fonts/Montserrat-*.ttf` by a **relative path** (`fontDict.py`, pulled in via
`scene` → `palettes` → `ui_style`), so the job's cwd must contain `fonts/` (agent B's env.def note says it is
recreated; keep it that way).

## 2. Recommendation

**Assumption:** an OSPool x86 core is 1.0-1.5x slower than the Mac's loaded per-core figure (0.83 s/test), so
**0.83-1.25 s per chunk test, planned at 1.0 s** (`--test-wall 1.0`). A "core" there is often one hyperthread
on a busy machine; the Mac's isolated 0.55 s is not a fair comparison. Check against the smoke job's
`runs.jsonl` test walls and `summary.json` (`cpu_model`) before the full batch.

| Setting | Value | Why |
|---|---|---|
| `request_cpus` | **4** | Rounds are sequential, so depth needs cores per job: at 1 cpu a 6 h job gets ~9 rounds, at 4 ~38. Local runs only reached 1-3 rounds, and the gain we have came from 1 round. 4-core slots are common; 8 is the docs' upper limit and matches far fewer slots; the main process's serial share (~3 % at 2 workers) grows with more workers. |
| `request_memory` | **4GB** (1 GB/core, a common slot shape) | Realistic peak ~1.5-2.3 GB at 4 workers: main ~100 MB + 4 pool workers at 160-330 MB (cold JIT) + 4 spot workers at ~190-330 MB while the pool idles. All transient peaks coinciding would be ~3.2 GB. Tighten only after a few jobs; the log table understates spikes. |
| `request_disk` | **3GB** | Image (agent B's; numpy + numba + llvmlite + pygame on python:3.13-slim, a few hundred MB, copied into scratch) + inputs (a few MB) + outputs (~45 MB raw for 40 rounds, tarball ~8 MB) + numba cache (~5 MB). Lower to 2GB once the `.sif` size is known. |
| Hours per job | **6** (`hours` column; deadline = start + 6 h - 5 min) | Medium category: expected under 10 h, held at 20 h. An eviction restarts from scratch (`ospool_evolve` clears `layouts/`), losing on average half the elapsed run; with an assumed eviction rate p per hour the expected loss is ~p·H²/2 (p = 0.03/h: 4 h → 0.24 h, 6 h → 0.54 h, 10 h → 1.5 h). 6 h keeps the loss under ~10 % while giving 25-45 rounds. Measure the real rate from `NumJobStarts` in the first batch. |
| Rounds per job | **~38** at 1.0 s/test (range **26-46** for 1.25-0.83 s/test) | Round ≈ 2,000 tests × t / 4 × 1.05 = 7.3-13.1 min (`--estimate`: 460-560 s at 1.0 s). Usable ≈ 6 h - 5 min packing - ~10 min deadline margin - ~10 min spot checks - start-up (JIT ~10 s) ≈ 5.6 h. `v12s5` rows run ~1,500 tests/round, so ~25 % more rounds. |
| Spot checks | **every 5 rounds, 4 seeds, 120 s** (`--spot-every 5 --spot-seeds 4 --spot-seconds 120`) plus the final one chunk_evolve always does | 4 seeds = one wave on 4 workers (~45-65 s; the first one also runs the base: 2 waves). ~7 checks per job ≈ 8-10 min, ~3 % of the job. 4 seeds halve the 2-seed noise (~0.03 → ~0.02). These are only a trajectory monitor; ranking is option 2's job. |
| `--test-wall` | 1.0 | Only used for the first round's deadline check (and `--estimate`). |
| First full batch | **80 jobs** (grid below) | |
| Core-hours | **80 × 4 cpus × 6 h = 1,920 core-hours** (~2,100-2,300 with eviction reruns); smoke 1.6 core-hours | |

Scale-up order (docs/07): the smoke row alone; then 3-10 full rows (e.g. `base-s01..s03` plus one row from each
of `flow`, `edit6`, `v12s5`, `seam08`, which have the most different costs); then all 80.

Smoke row `smoke1`: base settings, seed 999, 0.4 h. Its deadline leaves ~19 min, enough for one round
(1.1 × 520 s + 120 s ≈ 11.5 min needed before it starts) plus the final spot check if the round is quick.
At 1.25 s/test the round takes ~13 min and the spot check is skipped; that's still a valid smoke test.

## 3. Sweep design (ospool/sweep.json)

One factor at a time around a base setting, with replicate search seeds. Replicate `sNN` uses `--seed NN` in
every setting, so spot-check seeds (derived from the search seed) are paired across settings.

| Setting | Changed from base | Replicates | Question |
|---|---|---|---|
| `base` | none: `--metric free --seam-threshold 0.04 --variants 6 --seeds 10 --remove 3 --add 3 --targeted-share 0.5` | 16 | the reference; its spread is the seed-noise estimate for everything else |
| `flow` | `--metric flow` | 8 | does draining speed beat settled share as the chunk metric? |
| `seam02` | `--seam-threshold 0.02` | 8 | stricter seam reverts |
| `seam08` | `--seam-threshold 0.08` | 8 | looser seam reverts |
| `edit1` | `--remove 1 --add 1` | 8 | small local edits |
| `edit6` | `--remove 6 --add 6` | 8 | big edits (still clipped by chunk_mutate's count band) |
| `share02` | `--targeted-share 0.2` | 8 | mostly uniform draws |
| `share09` | `--targeted-share 0.9` | 8 | mostly stuck-targeted draws |
| `v12s5` | `--variants 12 --seeds 5` | 8 | more variants on fewer seeds (similar cost per round) |

Fixed for every job: `--margin 0.005 --recheck-seeds 2 --spot-every 5 --spot-seeds 4 --spot-seconds 120
--test-wall 1.0`; `--base`, `--workers`, `--deadline`, `--log-dir` and `--rounds` come from the stager /
`ospool_evolve`.

Why 16 / 8: the outcome spread across search seeds is unknown; guess it's similar to the 2-seed whole-map noise
(~0.02-0.03). The SE of a setting-minus-base difference is σ·√(1/8 + 1/16) ≈ 0.43σ, so a difference of ~0.9σ
(~0.02-0.03) shows at 2 SE. A full factorial was too costly for a first batch; once a knob clearly wins, the
next batch can cross the winners.

Not swept (minor calls): `--margin` (0.005), `--recheck-seeds` (2), the base map (`evolved 1 halloween`), and
`--seeds` beyond the `v12s5` trade.

## 4. Option 2: the final whole-map scoring batch (not built)

- **What:** every returned map (80, plus the base as a control, and the smoke map if wanted) scored by
  `chunk_test.whole_map_score`, the same scorer as the spot checks.
- **Seeds:** the **same 16 spot seeds for every map** (paired), none of them used during evolution. At a per-seed
  sd of ~0.04 (from 0.03 noise at 2 seeds), 16 seeds give an SE of ~0.01 per map.
- **Seconds:** 120 s of sim (matching the spot checks; the score uses the last 10 s). Then a second pass on the
  top ~10 maps and the base: 32-64 fresh seeds at 300 s, to check the ranking holds over a longer run.
- **Shape:** 1 cpu, 1-2 GB, one job per map (16 runs in sequence ≈ 16 × 42-63 s ≈ 11-17 min). Pass 1 ≈ 81 maps ×
  16 runs × ~1 min ≈ 22 core-hours; pass 2 ≈ 11 maps × 48 runs × ~2.5 min ≈ 22 core-hours.
- **Analysis:** per setting, the mean over replicates of (map score - base score) on paired seeds; also each
  job's `summary.json` trajectory (rounds done, spot scores over time) to see whether rounds past ~20 still help,
  which sets the hours for the next batch.
