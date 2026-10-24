# 08 Parallelism on the M4 (P/E cores, GIL, processes, free threading)

Status: cycle 1 complete. All measurements are mine unless marked; scripts and raw output in
`/private/tmp/claude-501/research-08/` (`gil_threads.py`, `overlap.py`, `mp_worlds.py`, `pipeline.py`,
`qos_threads.py`, `gil_release.py`, `numba_par.py`, `blas_ops.py`, free-threaded venv `ft314/`).

## Summary

1. **Threads under the GIL make things worse, not better.** Two worlds stepped in two threads ran at 0.78x the
   serial throughput for the demo scene (1.36x at 200 bodies, 1.6x at 500, where big numpy calls release the GIL).
   A physics thread next to a pygame drawing thread was starved almost completely (one `World.step` in 3 s while
   the draw loop ran), and total work done fell to 0.85x of serial. The cause is the "convoy effect": every numpy call
   that releases the GIL has to wait up to the 5 ms switch interval to get it back from a busy Python thread.
   pygame 2.6.1 drawing and blits, and moderngl `tex.write` + `ctx.finish`, all **hold** the GIL (measured), so the
   render thread never yields it while it works. **Don't use a physics thread in this codebase on GIL CPython.**
2. **A physics worker process with shared memory works today and keeps results bit-identical.** Prototype
   (`pipeline.py`): the worker steps frame n+1 while the main process draws frame n, then the main process copies
   P/PREV/V (about 100 KB at 6k points) from shared memory into its own `World`. The IPC round trip costs 10-20 us.
   Measured frame time for step + `World.draw`: pile200 23.9 -> 18.6 ms (1.29x), pile500 58.6 -> 46.3 ms (1.26x),
   pile500 on the GPU backend 41.4 -> 31.1 ms (1.33x), demo 2.20 -> 2.05 ms. The CPU trajectory was **bit-identical**
   to the serial loop. Frame time becomes max(physics, main-side work), so how much it helps depends on how big the
   main side is (draw + UI + upload + GPU wait: 3-16 ms per 09). The cost: one frame (16.7 ms) of added input latency,
   and editor edits need care (see Applicability).
3. **Free-threaded CPython 3.14t scales almost perfectly for physics, but the app can't use it yet.** In a scratch
   3.14t venv (numpy 2.5.4, numba 0.68, mlx 0.32.4 all ship `cp314t` macOS arm64 wheels and keep the GIL off), one
   thread runs as fast as GIL 3.13 (1.97 vs 1.91 ms/step for the demo), and K worlds in K threads give 1.87x (K=2) /
   3.28x (K=4) for the demo and 1.95x / 3.81x at 200 bodies. Blockers: **pygame and pygame-ce have no `cp314t`
   wheels** (pygame-ce fails to build from source without SDL2 dev libraries), and **moderngl's `cp314t` wheel turns the
   GIL back on** at import (it doesn't declare `Py_MOD_GIL_NOT_USED`). Forcing `PYTHON_GIL=0` is possible but unsafe.
4. **QoS matters far more than thread count.** Measured ms/step for the demo in a thread at each QoS class:
   USER_INTERACTIVE 1.91, DEFAULT 1.91, UTILITY 2.39, **BACKGROUND 83 (44x slower)** with the E cores loaded by other
   jobs. Background QoS is confined to the E cores at about 1.0 GHz, against 4.5 GHz for a P core. A new Python thread
   starts at DEFAULT (0x15), which runs at P-core speed. A per-thread QoS call from ctypes
   (`pthread_set_qos_class_self_np`) works: physics can be pinned to a class, or background worlds and batch jobs
   can be pushed to the E cores on purpose.
5. **Accelerate (BLAS) is irrelevant here.** numpy 2.5.4 in the venv is built against Accelerate (BLAS + LAPACK, NEON
   / ASIMD baseline), but the solver uses only elementwise ufuncs, `bincount`, `sort`, `searchsorted`, fancy indexing
   and tiny batched `matmul`s, none of which go to BLAS threads. The rigid-body covariance `matmul` costs 2-6 us
   (B = 50-125), and setting `VECLIB_MAXIMUM_THREADS` changed nothing that matters. `OMP_NUM_THREADS` has no effect on
   numpy's core.
6. **Inside one world, compiling the code matters more than threads, though threads add a real bonus at 4
   threads.** One contact Jacobi pass with numba: serial numba is already **3.2-4.0x faster than numpy**
   (bit-identical). Naive `prange` adds only 1.3-1.9x, capped by the serial order-preserving scatter. A CSR gather
   (each point sums its pairs in bincount's order, built once per substep with a compiled counting sort) is still
   **bit-identical**, and per substep it gives **1.6x (200 bodies), 1.8x (500), 2.6x (2000)** over serial numba, so
   5-10x over today's numpy. Going past 4 threads onto the E cores makes it slower (static chunks wait for the slowest
   core). Islands rarely help: a settled pile touches the ground, so it is one island.

Overall: the two real wins for this machine are (a) **keep everything interactive on the P cores** (never run the app
or its physics at background QoS; set the physics thread or process QoS explicitly), and (b) **a physics worker
process pipelined with rendering** for heavy scenes (1.25-1.35x now, close to 2x when physics and the main side cost
about the same). Free threading is the cleaner long-term route (threads, a shared editor `World`, K worlds in
parallel), but it waits on a pygame build with free-threading support.

## Findings

### F1. Environment facts (measured)

- `np.show_config()`: numpy 2.5.4, BLAS = LAPACK = **accelerate** (system), clang 15, SIMD baseline NEON / NEON_FP16 /
  NEON_VFPV4 / ASIMD, found ASIMDHP / ASIMDDP, missing ASIMDFHM. Python 3.13.7 (python.org framework build, GIL).
- CPU: 4 P cores (16 MB L2 shared), 6 E cores (4 MB L2), 16 GB. P cores run up to 4.512 GHz alone and about 3.85 GHz
  with the whole cluster busy. E cores run at 1.02-2.59 GHz; low-QoS work stays near 1.02 GHz, and high-QoS work that
  spills onto the E cores gets 2.59 GHz ([eclecticlight: M4 frequency](https://eclecticlight.co/2024/12/02/inside-m4-chips-controlling-frequency/),
  [M4 core management](https://eclecticlight.co/2024/12/05/inside-m4-chips-cpu-core-management/)). Background-QoS
  threads run **only** on E cores, and higher classes can use both
  ([eclecticlight: M1 scheduling](https://eclecticlight.co/2022/01/25/scheduling-of-threads-on-m1-series-chips-second-draft/)).
  macOS has no thread-to-core affinity API, only QoS ([Apple forum 674456](https://developer.apple.com/forums/thread/674456)).
- Free-threaded wheels on PyPI (checked through the JSON API, macOS arm64): numpy, numba + llvmlite, mlx (`cp313t` +
  `cp314t`; mlx-metal is `py3-none`), torch, moderngl + glcontext: yes. **pygame 2.6.1: no `cp313t`/`cp314t`; pygame-ce
  2.5.8: no**; taichi: no. `uv` offers `cpython-3.14.4+freethreaded` (and 3.13.13t) as a download. Import test in 3.14t:
  numpy, numba and mlx keep `sys._is_gil_enabled() == False`; **importing moderngl re-enables the GIL** with a
  RuntimeWarning. CPython re-enables the GIL for any extension that doesn't declare `Py_mod_gil = Py_MOD_GIL_NOT_USED`
  ([CPython free-threading extension howto](https://docs.python.org/3.13/howto/free-threading-extensions.html)).
  numpy calls its own free-threaded support "experimental" since 2.1; arrays shared between threads need user-side
  locking, and `np.errstate` is thread-local, which suits `_ccd_detect`
  ([numpy thread safety](https://numpy.org/doc/stable/reference/thread_safety.html)). Trackers:
  [py-free-threading](https://py-free-threading.github.io/), [free-threaded wheels](https://hugovk.dev/free-threaded-wheels/),
  [Quansight rollout notes](https://labs.quansight.org/blog/free-threaded-python-rollout).

### F2. Threads on GIL CPython (measured, `gil_threads.py`, P cores, short runs)

K independent worlds, one per thread, against the same K steps run serially:

| scene | serial ms/step | K=2 speedup | K=4 speedup |
|---|---|---|---|
| demo (66 pts) | 1.91 | **0.78x** | **0.55x** |
| 200 bodies (2412 pts) | 18.4 | 1.36x | 1.14x |
| 500 bodies (6012 pts) | 45.3 | 1.60x | 2.08x |

Small scenes are pure dispatch (09 F7: ~1200 numpy C calls per frame, 0.2-1 us each, all GIL-held), so threads only
add GIL handoffs. Big scenes spend more time inside long numpy kernels (sort, gathers over 10^5 elements), which
release the GIL, so some overlap appears.

`overlap.py`: physics in a worker thread and `World.draw(filled)` + `fill` in the main thread, the shape a
"physics thread" design would take:

| scene | draw alone | step alone | concurrent: draw | concurrent: step | total work vs serial |
|---|---|---|---|---|---|
| demo | 0.25 ms | 1.92 ms | 0.31 ms | **~3000 ms** (starved) | 0.83 |
| 200 | 5.38 | 18.5 | 6.34 | **~3000** | 0.86 |
| 500 | 13.2 | 45.2 | 15.8 | **~3000** | 0.85 |

Lowering `sys.setswitchinterval` from 5 ms to 0.5 ms did not help. This is the convoy effect
([Beazley, "Understanding the Python GIL"](https://www.dabeaz.com/python/UnderstandingGIL.pdf)): each numpy call that
releases the GIL in the physics thread must win it back from a thread that runs Python almost all the time.

`gil_release.py` (a pure-Python victim thread's speed while the main thread loops one call; 1.0 = the call releases
the GIL, ~0.5 = it holds it, so the two threads time-slice):

| call | victim keeps |
|---|---|
| `time.sleep`, `mx.eval` (GPU wait), `np.sort` / `np.add` on 400k elements | 0.98 (released) |
| 50 small numpy ops (200 elements) | 0.50 (held: the per-call Python overhead) |
| moderngl `tex.write` (6 MB) + `ctx.finish()` | 0.47 (held) |
| pygame `fill` + 200 `draw.polygon` | 0.49 (held) |
| pygame 3 full-surface blits | 0.49 (held) |

The same table also shows the convoy cost from the other side: a 0.5 ms `time.sleep` took 7.8 ms per call, and a
400k `np.sort` 25 ms, because they had to get the GIL back from the busy victim. So even "GIL-releasing" waits (the
GPU backend's `mx.eval`) don't help a GIL-threaded design when the other thread is busy in Python.

### F3. Worker processes with shared memory (measured, `mp_worlds.py`, `pipeline.py`)

Lockstep K worker processes (spawn), each owning a `World`, publishing `P` into `multiprocessing.shared_memory`
after every step:

| scene | K=1 | K=2 | K=4 | K=6 (spills to E cores) | IPC round trip + publish |
|---|---|---|---|---|---|
| demo | 1.95 ms | 2.01 | 3.22 | 3.64 | 9-17 us |
| 200 bodies | 18.8 | 19.3 | 27.4 | 32.4 | 14-18 us |
| 500 bodies | 47.4 | 46.6 | 52.9 | 73.5 | 17-23 us |

K=2 is free and K=4 costs +15-70% (other agents were using the P cores during these runs; the machine's load average
was 28-34). A spawned worker has to re-import numpy, pygame and the solver (and mlx for the GPU backend) before its
first step, a startup cost I didn't time separately, so workers should be long-lived.

Pipelined loop (`pipeline.py`): the main process sends "step", draws the previous frame, waits, then copies
P/PREV/V from shared memory into its mirror `World` (whose bodies are views into those arrays, so `World.draw` and
the editor see the new frame without other changes):

| scene | serial step + draw | pipelined | speedup | main process waiting | trajectory |
|---|---|---|---|---|---|
| demo (cpu) | 2.20 ms | 2.05 | 1.07x | 1.75 ms | bit-identical |
| 200 (cpu) | 23.85 | 18.55 | 1.29x | 13.1 | bit-identical |
| 500 (cpu) | 58.56 | 46.32 | 1.26x | 32.8 | bit-identical |
| 500 (gpu) | 41.36 | 31.12 | 1.33x | 17.5 | differs (GPU float atomics aren't deterministic; expected) |

The main process sits idle most of each frame, so the gain is capped by physics alone. With 09's whole-frame numbers
(pile500 CPU: physics 22.9, draw 11.9, ui 0.9, upload 0.6, GPU wait 1.8 ms; 38.2 ms/frame) a pipelined frame would be
~max(22.9, 15.2) = ~23 ms (1.65x). With the GPU backend (physics 13.7) it is ~max(13.7, 15.2) = ~15 ms (1.9x), close
to 60 fps. Once 09's 3.4x faster draw lands, the main side drops to ~6-7 ms, and pipelining is worth ~6-7 ms/frame on
heavy scenes. pile200 CPU (18.3 ms/frame) would drop under the 16.7 ms budget either way. Light presets (default
6.1 ms/frame) are already at 60 fps; there the only gain is lower frame time, not fps.

### F4. Free-threaded CPython 3.14t (measured, `ft314/` venv, pygame stubbed out since physics doesn't use it)

| scene | serial ms/step (3.14t) | serial (3.13 GIL) | K=2 threads | K=4 threads |
|---|---|---|---|---|
| demo | 1.97 | 1.91 | **1.87x** | **3.28x** |
| 200 bodies | 18.1 | 18.4 | 1.95x | 3.81x |
| 500 bodies | 46.2 | 45.3 | 1.96x | 3.64x |

The single-thread cost of the free-threaded build is ~3%, and scaling across the 4 P cores is near perfect. With free
threading, a plain physics thread would give the same pipelining as F3 without copying through shared memory, and
background or parallel worlds would get the speedups in the table. What blocks it in the app: pygame (no wheel, the GUI
depends on it: `pygame.draw`, surfaces, events), moderngl (re-enables the GIL), and **thread-unsafe global state in
the solver** (F6).

### F5. QoS / P vs E cores (measured, `qos_threads.py`, `qos_default.py`)

| thread QoS class | demo ms/step |
|---|---|
| USER_INTERACTIVE (0x21) | 1.91 |
| DEFAULT (0x15) | 1.91 |
| UTILITY (0x11) | 2.39 |
| BACKGROUND (0x09) | **83.4** |

- A process started from the terminal has a USER_INTERACTIVE main thread. New `threading.Thread`s start at DEFAULT,
  which still gets P cores. `taskpolicy -b` doesn't change `qos_class_self()`; it applies a process-wide darwin
  background policy, which is just as slow.
- The BACKGROUND figure is inflated because the E cores were busy with other agents' background jobs (09 saw 10-19x for
  the same reason). Even on idle E cores, background QoS means ~1.0 GHz against 4.5 GHz, so expect at least 4-5x
  slower. Topic 04 measured that background QoS also makes each MLX GPU sync 4x more expensive (733 vs 186 us).
- So: **the interactive app must never run at background QoS**, and benchmark numbers taken at background QoS on this
  shared machine are not absolute timings (09 reached the same conclusion). For batch work (sweeps, test suites,
  pre-settling preset worlds), background or UTILITY QoS in worker processes is the right, quiet choice.

### F6. Thread-safety audit of the solver (code reading)

Problems that would break threads (free-threaded or not) or several worlds in one process:

- `World.step` applies per-world `World.env` overrides by **writing the class attributes of `EnvConfig`** and restoring
  them afterwards (`solver.py:212-222`). Two worlds stepping at once in threads would see each other's settings.
  The world-selector branch (`world_selector.py`, not on main) also saves and restores `EnvConfig.gravity` per world.
  Fix: read config through a per-world resolved object (`self.cfg = ChainMap(world.env, EnvConfig)` or a small
  dataclass made at the top of `_step`), not through the class.
- `EnvConfig.backend` is read in `World._rebuild` through `_make_backend`. That's fine as long as no other thread
  changes it.
- `World.draw` reads `self.P` while `step` writes it in place (`self.P[:] = ...`, `+=`). Drawing during a step would
  show torn frames, so a threaded design needs a snapshot (double buffer) for the renderer.
- MLX: one GpuBackend per world uses the default stream. Whether two threads can evaluate on the same device at once
  wasn't tested; keep one GPU world per process to be safe.
- Processes avoid all of this: each process has its own `EnvConfig`.

### F7. Accelerate / numpy internal threading (measured, `blas_ops.py`)

| op (us) | value |
|---|---|
| 2-vector `a @ b` (BLAS ddot path) | 0.34 |
| `a[0]*b[0] + a[1]*b[1]` | 0.16 |
| `np.hypot(*a)` (used per hit in `_ccd_impulses`) | 0.60 |
| `float` math (`(x*x + y*y) ** 0.5`) | 0.14 |
| rigid covariance `matmul` B=125 (n=12) | 4.1 (einsum 11.3; broadcast-sum 8.8) |
| rigid covariance `matmul` B=2000 | 57 (`VECLIB_MAXIMUM_THREADS=1`: 81) |
| 1024^3 dgemm (control: Accelerate/AMX) | 6-7 ms (~290-360 GFLOP/s) |

No solver op is big enough for Accelerate's threads or the AMX unit to matter. The only BLAS-adjacent detail is
per-call overhead in the Python loops (`_ccd_impulses` uses `@` and `np.hypot` on 2-vectors, about 0.3-0.6 us each
where plain floats cost 0.15 us). That belongs to topic 05 (compile the loop), not to threading.

### F8. Parallelism inside one world (measured, `numba_par.py`; code reading for islands)

One contact Jacobi pass (`_solve_contacts` without friction; real candidates from `World._candidates` on settled
piles), with numba's default workqueue threading layer:

| scene | candidates | numpy | numba serial | prange 2 thr | prange 4 thr | prange 6 | prange 10 |
|---|---|---|---|---|---|---|---|
| 200 bodies | 19.7k | 0.58 ms | 0.18 | 0.15 | **0.14** | 0.15 | 0.16 |
| 500 bodies | 47.5k | 1.58 | 0.44 | 0.29 | **0.26** | 0.30 | 0.27 |
| 2000 bodies | 195k | 7.47 | 1.87 | 1.17 | **1.01** | 0.97 | 0.98 |

- numba matches numpy bitwise (max difference 0) with the same bincount accumulation order.
- The per-pair phase scales 2.2-2.7x on 4 threads. The serial, order-preserving scatter (needed for bit-identical
  Jacobi sums) is what's left.
- **Follow-up, measured: deterministic parallel scatter through a CSR gather.** CSR means compressed sparse rows:
  for each point, the list of (pair, role) entries that write to it. Build it once per candidate set (once per
  substep, reused by the 4 iterations) with a stable counting sort, so each point sums its entries in exactly
  bincount's input order. Then the pass runs as two `prange` loops (per pair, then per point), and the result is still
  **bit-identical to numpy** (max difference 0 in all three scenes):

  | scene | numba serial pass | CSR pass, 4 threads | CSR build (numba counting sort) | per substep (build + 4 passes): serial numba -> CSR | vs numpy (4 passes) |
  |---|---|---|---|---|---|
  | 200 bodies | 0.18 ms | 0.09 | 0.10 | 0.73 -> 0.46 ms (1.6x) | 2.32 ms -> 0.46 (5.0x) |
  | 500 bodies | 0.44 | 0.18 | 0.24 | 1.76 -> 0.96 (1.8x) | 6.3 -> 0.96 (6.6x) |
  | 2000 bodies | 1.84 | 0.46 | 1.01 | 7.35 -> 2.85 (2.6x) | 28.4 -> 2.85 (10x) |

  Building the same CSR with `np.argsort(kind="stable")` costs 1.4 / 3.7 / 17 ms, which would wipe out the gain, so
  the build has to be compiled too. Caveat: today's `_solve_contacts` switches to the smaller `reach` subset after
  the first pass. A CSR for the full set plus a mask keeps the indexing fixed but gives up part of that saving, so
  either rebuild the CSR for the reach set (cheap: 0.1-1 ms) or apply 03's skin/Verlet-list idea so that one
  candidate list and one CSR last several substeps.
- More than 4 threads doesn't help: E cores join, and static scheduling waits on the slowest chunk. Use
  `numba.set_num_threads(4)` (or `NUMBA_NUM_THREADS=4`) when `prange` is used on this machine.
- Islands: `_resolve_ccd`, the contact pass and `_impact_zones` are global vectorized passes, so stepping islands
  separately under the GIL would only add Python calls. Islands only pay with real threads or processes, and the heavy
  scenes (settled piles) are a single island, because every body is linked through contacts and the ground (fixed bodies
  can be shared read-only, but the pile itself is connected). The demo and diamonds scenes have a few islands, but they
  are already cheap. Not worth building.

## Applicability to this codebase

| idea | files / functions | what changes | verdict |
|---|---|---|---|
| Never run interactive at background QoS; optionally set QoS explicitly | `main_opengl.main` (top), any physics worker | `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)` via ctypes in the physics thread or process; docs note for benchmarks | do (trivial) |
| Physics worker process, pipelined with rendering | new `physics_server.py`; `main_opengl.main` loop (the `profiler.switch("physics")` block); `World` gets `export_state()` / `import_state()` and a `structure_version` counter | see design below | best measured win for heavy scenes |
| Free-threaded physics thread | same split, but a thread and a snapshot | blocked by pygame/moderngl; needs F6 fixes first | later |
| Per-world config instead of mutating `EnvConfig` | `solver.World.step/_step` and every `EnvConfig.x` read in solver.py, entities.py, looping.py, gpu_backend.py | resolve config once per step into `self.cfg` | needed before any threaded design; harmless now |
| Several worlds in parallel | `world_selector.WorldSession` (branch) | only if inactive worlds should keep running, or to pre-build / pre-settle preset worlds in background processes | optional feature |
| Parallel test / benchmark suites | harnesses in `/private/tmp/claude-501/*` | `ProcessPoolExecutor` over scenes at UTILITY QoS | do (cheap) |
| numba `prange` inside kernels | topic 05's compiled contact / CCD kernels | `parallel=True` with 4 threads, CSR scatter | add-on to 05, +1.3-1.9x |
| Islands | solver.py | none | skip |

### Design: pipelined physics process (the main recommendation)

```
main process (P core, USER_INTERACTIVE)              physics process (P core, USER_INTERACTIVE)
------------------------------------------           -------------------------------------------
events -> editor edits on the mirror World            loop:
if mirror.structure_version changed:                    recv(msg)
    send ("load", pickled World minus backend)          if "load": replace world, rebuild backend
send ("step", dt, scale, grab, grab_target, env)        world.step(dt, scale)
draw frame n from the mirror World                      write P, PREV, V (+ zone_count, last_wraps,
UI, upload, render, flip                                 looping/pending walls) to shared memory
wait for "done" -> copy shared P/PREV/V into mirror     send done
```

- The mirror `World` in the main process is a full `World` (bodies are views into `P/PREV/V`, `solver.py:104-105`),
  so `World.draw`, the editor, trails and selection keep working unchanged. Only the three arrays (and a few scalars)
  cross over each frame: 96 KB at 6k points, ~10-20 us.
- **Edits and the zero-overlap guarantee.** The editor checks a placement against the state it sees (frame n). If the
  worker is already computing n+1, adding the body into n+1 could create an overlap that no CCD ever checked. The rule
  that keeps the guarantee: **an edit throws away the in-flight step.** The main process waits for (and ignores) the
  pending result, applies the edit to frame n, and sends the whole world ("load") before the next step. The worker
  always starts from a state the main process validated, and every motion still goes through the unchanged
  `World.step` + CCD, so the guarantee is exactly today's. Edits are rare (clicks), so a dropped frame of physics per
  edit doesn't matter. Pickling a 500-body `World` happens only on edits (needs `__getstate__` that drops `backend`,
  `profiler` and `debug`).
- Grab/drag: `grab`/`release` change only `World.grabbed` (an index) and `grab_target`, so they go with each "step"
  message. Debug overlays (`World.debug`, a step recorder) must run inside the worker; send the recorder's output back
  with the frame, or run the world in-process while an overlay is on.
- Latency: what you see is one frame behind the input (+16.7 ms). Pausing, frame-stepping (`timectl`) and
  `max_frames` tests can keep the synchronous path (step, wait, draw).
- Small scenes: switch automatically. Pipeline only when the last N frames' physics took more than ~4-5 ms, otherwise
  step in-process (no latency, nothing to gain).
- GPU backend: the worker owns the MLX device, and the main process keeps only moderngl. That separation also removes
  any question of MLX and GL sharing the GPU queue from one thread.

## Expected gains and risks

| change | scene | now | expected | risk to the zero-overlap guarantee |
|---|---|---|---|---|
| pipelined physics process | pile200 cpu | 18.3 ms/frame (09) | ~10-12 ms (60 fps) | none if edits discard the in-flight step; physics code unchanged |
| pipelined physics process | pile500 cpu / gpu | 38.2 / 28.7 ms | ~23 / ~15 ms (1.65x / 1.9x) | same |
| pipelined physics process | default demo | 6.1 ms | ~4 ms (fps already capped at 60) | same |
| QoS hygiene | all | unknown in practice | avoids 5-40x slowdowns if launched from a background-QoS parent | none |
| per-world config (F6) | all | 0 | 0 (enables threads) | none (pure refactor; compare trajectories bitwise) |
| free-threaded 3.14t + physics thread | heavy | - | like the process design, no copies; K worlds 3.3-3.8x | blocked: pygame/moderngl; data races on shared arrays |
| numba prange + CSR gather on top of 05 | pile200 / 500 / 2000 contacts, per substep | numpy 2.3 / 6.3 / 28.4 ms; serial numba 0.73 / 1.76 / 7.35 ms | 0.46 / 0.96 / 2.85 ms | none: bit-identical to numpy (measured) |

Other risks:
- Processes: two copies of the scene in memory (trivial at these sizes); pickling custom body subclasses (Rope,
  Chain, Hook, FixedBody) must round-trip, so test with every preset. A worker crash needs a fallback to in-process
  stepping.
- Spawn start method: workers must be started from a module (not `-c` / stdin), with
  `if __name__ == "__main__"` guards. `main_opengl.py` already has one.
- Determinism: the CPU pipeline is bit-identical to serial (measured); the GPU path was never deterministic.

## Effort

- QoS helper + docs: ~20 lines.
- Per-world config refactor: small-medium (mechanical: replace `EnvConfig.x` reads in solver/looping/gpu_backend
  with `cfg.x`; test that trajectories are bitwise identical on every preset).
- Pipelined physics process: medium (~250-400 lines: server loop, World pickling, state import/export, main-loop
  integration, edit/rollback rule, auto on/off, tests).
- Free-threaded path: large and externally blocked (pygame build).

## Build plan (ranked)

1. **Pipelined physics worker process (`physics_server.py`).** Change: a long-lived spawn worker owning the `World`,
   with shared-memory `P/PREV/V`. `main_opengl.main` sends step parameters and draws the previous frame. The rule
   "edit = discard the in-flight step + reload world" (see Design). The worker sets USER_INTERACTIVE QoS, and the
   pipeline turns itself on only above ~4-5 ms of physics per frame. Metric: whole-frame ms on pile200 / pile500 /
   diamonds / default (09's `frame.py` harness in `/private/tmp/claude-501/research-09/`), P-core runs interleaved
   ABAB. Correctness: CPU trajectories bit-identical to in-process stepping over 600 frames on every preset, and the
   brute-force overlap checker (`/private/tmp/claude-501/hud-layout-agent/suite/brute.py` `checked()`) wrapped
   inside the worker with 0 bad substeps, including a scripted run that adds and deletes bodies every 10 frames while
   piled up. Expected: 1.3-1.9x on heavy scenes (pile200 under 16.7 ms). Risk: medium (editor and overlay integration,
   1 frame of input latency). Effort: medium.
2. **Per-world config (`World.cfg`) instead of mutating `EnvConfig`, plus a thread-safety test.** Change:
   `solver.World.step/_step`, `looping.py`, `gpu_backend.py`, entities read config through the world. Metric:
   bitwise-identical trajectories on all presets. A test steps two worlds with different `env` in two threads (in the
   `ft314` venv at `/private/tmp/claude-501/research-08/ft314`) and checks each matches its serial run. Expected: no
   speed change. It unlocks threaded and multi-world designs and makes the free-threaded path a drop-in later. Risk:
   low. Effort: small-medium.
3. **numba `prange` for topic 05's compiled kernels, with a deterministic CSR gather.** Change: in 05's compiled
   `_solve_contacts` (and later the CCD narrow phase, which is per-pair independent and needs no scatter at all), a
   parallel per-pair phase plus a per-point gather over a CSR built once per substep with a compiled counting sort.
   `numba.set_num_threads(4)`. The prototype is `numba_par.py` in `/private/tmp/claude-501/research-08/`
   (`_pair_phase_par`, `_csr_build`, `_gather_par`). Metric: physics ms/frame on pile200 / pile500 / pile1000
   settled against serial numba, bit-identical P, and topic 10's `cpu_demo_repeat_identical` + `overlap_substeps = 0`.
   Expected (measured on the kernel): 1.6-2.6x per substep over serial numba (5-10x over numpy). Keep the serial path
   below ~10-20k candidates, where thread fork/join (~10-20 us per `prange`) costs more than it saves. Risk: low
   (determinism testable bitwise). Effort: small once 05's kernel exists.
4. **(Exploratory) free-threaded spike.** Build pygame-ce from source for 3.14t (Homebrew SDL2 + SDL2_image/ttf/mixer)
   in a scratch venv, run `main_opengl.py` with `PYTHON_GIL=0` and a physics thread plus snapshot double buffer.
   Metric: the same as #1, and the brute checker. Expected: like #1 without copies, and parallel worlds 3-3.8x. Risk:
   high (unsupported, possible crashes in pygame/moderngl without the GIL). Effort: medium. Do it only if #1's
   process boundary proves painful.

## Cross-links

- 09 (profiling): whole-frame breakdown used for the pipelining estimates. Its draw speedup (F6/B2) shrinks the
  main-process side and so lowers what pipelining gains. Do both: draw first (no latency cost), then pipelining for
  what remains. 09 and I both found background-QoS numbers inflated 10-44x by E-core contention.
- 04 (Metal/MLX): `mx.eval` releases the GIL (measured here), but that doesn't help a GIL-threaded design (F2 convoy
  effect). A worker process owning the GPU backend keeps MLX and moderngl apart.
- 05 (compiled loops): compiling is worth more than threads inside one world. Serial numba is 3.2-4x over numpy for
  the contact pass. A naive prange adds at most 1.9x, and the CSR-gather prange 1.6-2.6x per substep, both
  bit-identical. 05's file was still a skeleton when I wrote this, so these kernel numbers are an independent data
  point for it.
- 10 (accuracy benchmarks): its harness (`/private/tmp/claude-501/research-10/bench/run_bench.py --jobs 6`) already
  runs test cases in parallel processes, so "parallel test suites" is covered. Its `cpu_demo_repeat_identical` and
  state-hash gates are the right acceptance tests for build plans #1-#3, all of which claim bit-identical CPU
  trajectories. 10 also saw ~35x slowdowns at background QoS (same cause as F5).
- 03 (broad phase): per-substep motion is tiny (p99 0.3 px), so a skin/Verlet pair list can last several substeps.
  That also spreads the CSR build cost of build plan #3 over more passes.
- 01 / 07 (solvers): graph-coloured Gauss-Seidel (Vivace; 07's 2-colour batched ropes) is the standard way to
  parallelise GS. Here the only GS loops (`_ccd_impulses`, ~35 hits/substep at pile1000; ropes) are far too small for
  CPU threads; compile them (05) instead of threading them.
