# 05: Compiled loops (numba / Cython / C extension / mypyc / numpy restructuring)

Status: cycle 1 complete (overlap-validation runs: see F9). Prototype, harnesses and logs in
`/private/tmp/claude-501/research-05/`; the repo is untouched.

| file | what |
|---|---|
| `nbk.py` | the numba kernels (contacts, springs, shape matching, ropes, broad phase, CCD detect / impulses, impact zones, `_bounce`, `warmup()`) |
| `nbworld.py` | `install(World_or_instance, features)`: monkeypatches the kernels into the `World` class or one instance |
| `np_alt.py` | numpy shape matching without `matmul` / trig (the reference the kernels match bit for bit, F4) |
| `compare2.py` | interleaved A/B harness: numpy world and numba world stepped alternately, bitwise check, brute-force overlap check |
| `pcore.py`, `phases.py` | short P-core timing (`nice -n 10`), per-phase breakdown |
| `bench_kern.py`, `cext/`, `fma_test.py` | numpy vs numba vs C (cffi) per kernel; FMA contraction test |
| `zones_test.py`, `ccd_nb.py`, `code_nb/` | forced impact zones; suite `ccd_test.py` and research-10's `run_bench.py` on a numba copy |

Terms: **JIT** = just-in-time compilation (numba compiles a function the first time it is called with given argument
types); **AOT** = ahead-of-time compilation (an ordinary compiled extension module); **FMA** = fused multiply-add
(a*b+c with one rounding instead of two); **GS** = Gauss-Seidel; **CSR** = compressed sparse rows; **P-core / E-core**
= the M4's performance / efficiency cores; **QoS** = macOS quality-of-service class.

Timing rules (shared with 04, 08, 09, 10): under `taskpolicy -b` the E-cores run at ~1 GHz and are shared by 10
agents, so background numbers are ~10-40x inflated and noisy. All **absolute ms below are P-core runs** (`nice -n 10`,
a few seconds each, interleaved A/B, median); background runs were used only for correctness (bitwise, overlaps).

## Summary

1. **numba makes the CPU physics 5-13x faster end to end, bit for bit.** A numba port of every per-pair /
   per-contact hot path (contact pass, springs, shape matching, ropes, both broad phases incl. `_filter_pairs`, CCD
   pre-reject + narrow phase, CCD impulses, impact zones, the `_bounce` restitution pass) gives, P-core ms/frame:

   | scene (points) | numpy | numba | speedup |
   |---|---|---|---|
   | default demo, dragged (78) | 2.67 | 0.48 | 5.6x |
   | random walls, dragged (97) | 2.61 | 0.46 | 5.7x |
   | looping (35) | 1.16 | 0.14 | 8.1x |
   | Newton's cradle (124) | 6.0 | 0.47 | 12.8x |
   | pile50 settled (330) | 4.2 | 0.47 | 9.0x |
   | diamonds (872) | 5.8 | 1.03 | 5.6x |
   | pile200 settled / dragged (1262) | 11.6 / 12.9 | 1.92 / 2.30 | 6.1x / 5.6x |
   | pile500 settled (3137) | 33.5 | 6.35 | 5.3x |
   | pile1000 settled (6262) | 90.6 | 20.4 | 4.4x |

   For comparison, the GPU backend (09) is 19.6 ms at pile500 and 43.5 ms at pile1000: **compiled CPU code beats the
   default GPU backend at every size measured**, and makes the small presets dispatch-free.
2. **The zero-overlap guarantee is untouched, and provably so: the numba world's trajectory is bitwise identical to
   numpy's** (every frame of pile200 dragged 150 frames, demo dragged 200, pile500 120, diamonds, cradle; 662 forced
   impact-zone fallbacks). Same float64 operations in the same order, same pair sets in the same order, same impulse
   order, so the CCD makes exactly the same decisions. Getting there needed three non-obvious rules (F4): no FMA
   contraction, numpy's pairwise-summation order wherever numpy reduces a contiguous run, and no `np.sin` (numpy's
   SIMD sin differs from libm in 0.06% of inputs), which is fixed by computing the shape-matching rotation without
   trig (also slightly faster in numpy).
3. **The loops named in the brief are not where the time is.** The sequential GS CCD impulse loop, impact-zone
   union-find and looping's clearance checks are 0-1% of a frame in every benchmark scene (CCD fires on 0.1-0.4 hits
   per substep; impact zones only under aggressive drags, 10 saw 47 of 2,400 substeps). The time is in the
   *vectorized* numpy passes (contacts 40-55%, CCD broad + narrow 25-35%, constraints 15-20%) and, for small scenes,
   in numpy call overhead (09: ~1,200 numpy C calls per frame). Fusing each pass into one compiled loop removes both
   the temporaries and the call overhead. The rare paths are still worth porting for spikes: one forced impact-zone
   fallback costs 0.91 ms in numpy vs 0.11 ms in numba (8.5x).
4. **numba is the right tool here; C / Cython would run at the same speed but cost a build system.** A cffi C port
   of the CCD narrow phase ran at the same speed as numba (1.77 vs 1.80 ms, E-core) with the same per-call overhead
   (~8 us E-core, ~1-2 us P-core). But Apple clang contracts `a*b+c` into FMA by default (`-ffp-contract=on`): the
   default `-O3` build changed 42% of CCD hit times (by up to 8e-12), so any C/Cython route must build with
   `-ffp-contract=off` to stay bit-identical. Rust/pyo3 has no toolchain installed; mypyc cannot speed up numpy code.
5. **Costs and risks of numba, all measured:** cold compile of all kernels 4.0 s on a P-core (138 s at background
   QoS) the first time only; cached load 0.1 s; `import numba` ~0.1 s warm. A first-use JIT mid-simulation stalls a
   frame, so call `warmup()` at startup. Two numba pitfalls hit during the prototype: a self-recursive jitted
   function segfaults when loaded from the on-disk cache (fixed by an iterative version), and a hot-loop helper that
   takes a dozen arrays costs ~4x even with `inline="always"` (inline by hand).

## Findings

### F1. Where the CPU time goes (code reading + measured)

cProfile and per-call timings (background QoS; shares, not ms). Pile500 settled, per frame: `_solve_contacts` 41%,
`_ccd_detect` 18%, `_solve_body_constraints` 10%, `_filter_pairs` 6%, sorts 6%, `_ccd_boxes` 4%, `_cell_pairs` +
`_cells` 6%; `_ccd_impulses` 0.8%; `_impact_zones` 0 (never fired). Default demo dragged: the same three passes, with
no CCD impulses in the top 20 at all. Array sizes per substep (pile500): 29k contact candidates (5.3k within reach),
7.9k CCD pairs on average (22.6k max), 0.4 CCD hits. Agent 02 measured the same: 85-100% of substeps have zero CCD hits
and impulses take <0.5% of the time.

The looping pass is ~1% even in the diamonds preset (pure-Python `rope_groups` is the biggest part of it); `spot_free` /
`bodies_clear` run only when something actually wraps. Not worth compiling.

### F2. The numba port, kernel by kernel (measured)

All kernels are `@njit(cache=True, error_model="numpy", nogil=True)`, no `fastmath`, no `parallel`. Per call at a
settled pile500 state (E-core thread time, ratios only):

| kernel | replaces | numpy | numba | speedup | identical? |
|---|---|---|---|---|---|
| `solve_contacts` | `_solve_contacts` (one full Jacobi pass, 28k candidates) | 7.6 ms | 1.26 | 6.0x | bitwise |
| `cell_pairs` + `contact_cells` | `_candidates` (`_cells`, `_cell_pairs`, `_filter_pairs`) | 6.2 | 1.05 | 5.9x | same pairs, same order |
| `ccd_boxes`, `floor_cells`, `cell_pairs` | `_ccd_cells` + `_ccd_pairs` | 5.0 | ~1.0 | ~5x | same pairs, same order |
| `ccd_detect` | `_ccd_detect` (pre-reject + quadratic, 19.7k pairs) | 6.0 | 0.67 | 9.0x | bitwise |
| `springs` + `shape_match` | `_solve_body_constraints` | 1.5 | 0.18 | 8.6x | bitwise (F4) |
| `ropes` | `Rope.solve_constraints` loop (GS, list order) | per rope ~10 us | one call | - | bitwise |
| `ccd_impulses` | `_ccd_impulses` (Python loop per hit) | - | - | - | bitwise |
| `zone_round` | `_impact_zones` inner work (union-find + rigidify) | 0.91 ms/fallback | 0.11 | 8.5x | bitwise (662 fallbacks) |
| `bounce` | `World._bounce` (Python pair loop, 5 balls, 8 substeps) | 0.137 ms/call (1.1 ms/frame) | 0.005 ms/call | 27x | bitwise |

P-core per-phase breakdown, numpy -> numba (ms/frame, `phases.py`):

| phase | demo dragged | pile200 settled | pile500 settled |
|---|---|---|---|
| contacts (broad + 4 passes) | 1.33 -> 0.21 | 6.38 -> 0.98 | 17.6 -> 3.28 |
| constraints (springs, shapes, ropes) | 0.55 -> 0.04 | 1.78 -> 0.33 | 3.86 -> 0.87 |
| CCD broad | 0.38 -> 0.13 | 1.84 -> 0.22 | 5.80 -> 0.98 |
| CCD narrow + impulses | 0.33 -> 0.01 | 1.71 -> 0.19 | 5.73 -> 1.01 |
| **physics** | **2.68 -> 0.48** | **11.8 -> 1.83** | **33.3 -> 6.39** |

What is left in the demo (0.48 ms) is mostly Python glue (`_step`, `_resolve_ccd`, per-substep wrappers) and ~20
numba dispatches per substep at ~1-2 us each. Fusing the whole substep into one kernel would save maybe 0.1-0.2 ms;
not worth the loss of structure.

Design notes that mattered:
- **Fusing beats porting line by line.** The numpy contact pass computes 6 gathers and ~20 temporaries over all
  candidates, then filters to the 5-20% that touch; the kernel does one loop that skips non-touching pairs right after
  the distance. The CCD kernel applies the swept-box pre-reject inline before the quadratic (no `keep` arrays).
- **Broad phase:** a counting-sort grid (count, prefix sum, scatter in edge order, so each cell's edge list is already
  sorted) gives the numpy code's exact `(point, edge)` order with no sort; multi-cell (swept) points collect edges with
  a per-edge stamp (dedupe) and an insertion sort. Only cells inside the points' cell range are registered (exact: no
  point can share any other cell). The world-box walls run `ext` past the screen, so their *points* make that range
  huge; the kernel then switches to a sorted list of occupied cells with binary search instead of a dense grid (both
  paths verified identical to numpy on demo, random walls, pile200, contact and swept CCD boxes, `test_cp.py`).
  Without that switch the demo silently fell back to numpy and stayed at 0.88 ms instead of 0.43 ms.
- **numba call overhead** is ~0.13 us + ~0.13 us per array argument (E-core; 2 us for 12 arrays), about one small
  numpy call. One kernel per pass removes dozens of numpy calls, which is why small scenes gain as much as big ones.
- **Helpers in hot loops:** `_filter_pairs` as an `@njit(inline="always")` helper taking 13 arrays cost 1.8 ms vs
  0.5 ms written inline (38k checks); the int64 `%` in the neighbour test was not the problem. Inline by hand.

### F3. Startup, caching and robustness of numba on this machine (measured + docs)

- Versions: numba 0.68.0 / llvmlite 0.50 with numpy 2.5.4 on Python 3.13.7, macOS arm64, all from wheels. numba has
  supported Python 3.13 since 0.61 (Jan 2025, not free-threaded then) ([0.61 notes](https://numba.readthedocs.io/en/stable/release/0.61.0-notes.html));
  0.68 (Sep 2026) adds 3.15 and switches cache freshness stamps to a **content hash** of the source
  ([0.68 notes](https://numba.readthedocs.io/en/stable/release/0.68.0-notes.html)). Agent 08 found `cp314t`
  (free-threaded) wheels for numba too.
- Cold compile of all kernels (17 jitted functions): **4.0 s P-core** (first run after install or after editing the kernel file), 138 s at
  background QoS. Cached: **0.1 s** for import + loading every kernel + the first frames. The cache lives in
  `__pycache__` next to the source (already gitignored); it is keyed by source content, CPU model and features, and is
  pickled, so it must stay in a trusted directory ([caching notes](https://numba.readthedocs.io/en/latest/developer/caching.html)).
  Editing a kernel file recompiles every kernel in it, so keep kernels in one module that rarely changes.
- **Lazy compilation stalls mid-run:** with JIT-on-first-call, the first CCD hit or first impact zone compiles a kernel
  inside a frame (seconds cold, ms cached). `nbk.warmup()` calls every kernel once with dummy arrays of the real types
  at startup. Signatures given up front (eager compilation) would do the same.
- **Pitfall 1, observed:** a self-recursive `@njit(cache=True)` function (numpy's pairwise sum written recursively)
  compiled and ran fine, then **segfaulted when loaded from the cache in the next process**. An iterative version with
  an explicit stack fixed it. Caching docs list only ufuncs/gufuncs/parallel as explicitly verified; avoid recursion
  in cached kernels.
- **Pitfall 2:** numba does no bounds checking by default; `NUMBA_BOUNDSCHECK=1` is the debugging switch (it found no
  out-of-bounds access here). Keep a numpy fallback path and the bitwise A/B test in CI so a wrong index can't hide.
- AOT is not an option: `numba.pycc` is deprecated with no replacement yet
  ([deprecation notices](https://numba.readthedocs.io/en/latest/reference/deprecation.html)). The on-disk cache is
  the practical "AOT".
- Disk/RAM: numba 29 MB + llvmlite 125 MB in the venv (already installed).

### F4. Bit-for-bit identity with numpy: what it takes (measured)

Bitwise identity is what lets the speedup be validated by hash (10's `cpu_demo_repeat_identical`, state hashes) and is
the strongest argument that the guarantee is unchanged. Rules found:

1. **No FMA contraction.** numba (no `fastmath`) does not contract; Apple clang does by default (F5).
2. **Same operation order** as the numpy expression, including `bincount`'s accumulation order (all point shares,
   then all edge-start shares, then all edge-end shares, each in pair order) and `(1 - t) ** 2` as `x * x`.
3. **Same summation order where numpy reduces.** `np.sum` over a contiguous run uses pairwise summation (8
   accumulators, blocks of <= 128); the kernels replicate it exactly (`_pw`, verified on 3,000 random lengths up to
   3,000). But the order depends on memory layout: `P[:, idx]` with a 2-D `idx` is **not** C-contiguous (strides
   `(8, 8n, 16)`), so numpy's `mean` over it is a plain sequential sum, while the following products are contiguous
   and summed pairwise. This only shows for bodies with >= 8 points (the cradle's 20-point balls).
4. **Accelerate BLAS uses FMA.** `np.matmul` on the stacked 2x2 covariances equals an FMA-sequential dot product, so
   the stock shape matching can't be matched by plain code. The rotation `matmul` matched no simple FMA order.
5. **`np.sin` is not libm `sin`:** over 2M angles in [-pi, pi], `np.arctan2` and `np.cos` matched `math.atan2` /
   `math.cos` exactly but `np.sin` differed in 1,268 cases (numpy's own SIMD float64 sin). `np.hypot` == `math.hypot`.

Fix for 4 and 5, applied to both sides: compute the best-fit rotation without `matmul` and without trig:
`num = sum(q_y r_x) - sum(q_x r_y)`, `den = sum(q_x r_x) + sum(q_y r_y)`, `c = den / hypot(num, den)`,
`s = num / hypot(num, den)` (`c, s = 1, 0` when both are 0, which is what `atan2(0, 0) = 0` gives). This is the same
rotation (cos and sin of `atan2(num, den)`), the same speed in numpy (218 vs 217 us at B=125), deterministic across
machines (no BLAS), and the numba kernel then matches it bit for bit. `np_alt.py` is the numpy version.

Verified bitwise identical, every frame, numpy (with `np_alt` shape matching) vs numba: demo dragged 200 frames,
pile200 dragged 150, pile500 settled 120, diamonds 120, Newton's cradle 120, plus 240 frames of the demo and pile200
with `ccd_passes = 1` (8 and 662 impact-zone fallbacks).

### F5. C, Cython, C++ and Rust extensions (measured + sources)

- **Speed is the same as numba.** cffi (API mode) C port of the CCD narrow phase, `-O3 -ffp-contract=off`: 1.77 ms
  vs numba 1.80 ms on the same 19.7k pairs, identical outputs. Both are LLVM underneath (clang vs numba's llvmlite).
- **Call overhead is the same order:** cffi with `ffi.from_buffer` for 12 arrays ~8.7 us vs numba ~7.6 us (E-core).
  Going through `ndarray.ctypes.data` is 4x worse (32 us). nanobind claims ~10x lower binding overhead than pybind11
  ([nanobind docs](https://nanobind.readthedocs.io/en/latest/why.html)), but at 20 calls per substep the overhead is
  already noise next to the work once the passes are fused.
- **FMA trap:** Apple clang 17 (`clang-1700`) contracts `a*b+c` into FMA at `-O3` by default; the default build gave
  the same number of CCD hits but 1,314 of 3,094 hit times differed (max 7.8e-12). LLVM has defaulted to
  `-ffp-contract=on` since clang 14, and projects that need reproducible results on Apple Silicon add
  `-ffp-contract=off` ([LLVM discourse](https://discourse.llvm.org/t/fp-contraction-fma-on-by-default/64975),
  [VTK issue](https://gitlab.kitware.com/vtk/vtk/-/issues/19418)). Cython inherits this (it compiles through clang),
  and Cython's `cdivision=True` changes `%` semantics for negative operands, which `_filter_pairs`' neighbour test
  relies on.
- **Build cost:** Cython is not installed; Rust (`cargo`, `rustc`, `maturin`) is not installed; the repo has no
  packaging (`pyproject.toml`), so any AOT route adds a build step and a per-Python-version binary. Cython 3.1 supports
  3.13 and free-threaded 3.13 experimentally ([Cython free-threading](https://cython.readthedocs.io/src/userguide/freethreading.html)).
  The upside of AOT is no JIT warmup (4 s once) and no 150 MB numba/llvmlite dependency. Not worth it while numba's
  cache gives a 0.1 s start.
- **mypyc** compiles type-annotated Python to C; calls into numpy stay numpy calls, and element access on numpy
  arrays stays boxed, so it cannot speed up this code. Only `_bounce`-style pure-Python loops on Python floats would
  gain, and those are better in numba.

### F6. Pure-numpy restructuring (measured)

The numpy code was already tuned hard (perf-engine agent's v1-v8: 403 -> 37 ms at pile500). What is left without a
compiler: the trig-free, matmul-free shape matching (F4, same speed, deterministic); a `rope_groups` cache in looping
(~1% of diamonds); 09's exact swept-box filter right after the cell join (7-12% on CPU); 03's fat-box pair list
(10-32% in numpy). Agent 09's numbers put the fixed per-frame cost at ~1,200 numpy calls; no numpy rewrite removes
that, only fewer passes or a compiler does.

### F7. How this combines with the other topics

- **08 (parallelism):** serial numba is the base; 08's `prange` + CSR gather on top adds 1.6-2.6x to the contact pass
  (bit-identical). After this port, contacts are 51% of pile500 (3.3 of 6.4 ms), so the remaining gain is ~1.3x
  end to end at 500 bodies, less for small scenes (fork/join ~10-20 us per `prange`). The kernels are `nogil=True`,
  so a physics thread would now spend most of its time without the GIL, which weakens 08's convoy-effect objection to
  a physics thread (the Python glue still needs the GIL ~20 times per substep; untested).
- **03 (broad phase):** 03's per-call exact numba broad phase (pairs already box-filtered) and fat-box list are
  complementary to F2: my `cell_pairs` reproduces the grid pairs exactly so CCD results are bitwise; 03's variant emits
  12.7x fewer pairs and lets `ccd_detect` skip the pre-reject. Either way the broad phase drops to ~1 ms at pile500.
  The latent `_cell_pairs` empty-join `IndexError` 03 found does not exist in the kernel (no empty-array indexing).
- **01 (XPBD / substeps):** 01 found the per-substep fixed cost F (broad phase + first contact pass + CCD) is what
  makes many small substeps expensive. numba cuts F ~6x, so 16x1 or 20x1 schedules come within today's frame budget
  even before hoisting the broad phase / CCD out of the substep loop.
- **06 / 07 (soft and rigid models):** ARAP triangles (3-5x the constraint phase in numpy) and soft shape matching
  (~2x) are cheap in numba: the constraint phase is now 0.04 ms (demo) to 0.87 ms (pile500). 07's batched,
  graph-coloured rigid-aware ropes are a natural kernel too.
- **09 / 04 (GPU):** with numba the CPU wins at every measured size (pile1000: 20.4 vs 43.5 ms GPU). 09's backend
  auto-selection threshold should be re-measured against numba; probably "cpu" for everything below several thousand
  points. 04's GPU CCD prefilter still matters only if the GPU path is kept.
- **10 (accuracy):** bitwise identity means 10's hash-based determinism checks validate the port directly; the suite
  ran on a numba copy of the repo (F9).

### F8. What was not done

- `parallel=True` / `prange` (covered by 08), `fastmath` (would break bitwise identity and risks the CCD's careful
  rounding-level tests), float32 (CCD must stay float64).
- `_ccd_new_pairs` and `_ccd_redetect`'s bookkeeping still use numpy (small, after impulses only).
- Free-threaded 3.14t (08: blocked by pygame/moderngl wheels).

### F9. Overlap validation of the numba world (measured)

(being filled: suite `ccd_test.py` stress drags with the kernels installed class-wide; brute-force checker on every
substep of demo / pile50 / cradle / random walls, every 5th frame of looping / diamonds / pile200 dragged; research-10
`run_bench.py --skip-perf` on `code_nb/`, a copy of the repo with the kernels installed.)

Already measured: demo dragged 200 frames, brute-force check on all 800 substeps: 0 bad substeps, 0 impact zones
(earlier kernel version). Since every trajectory tested is bitwise identical to numpy, overlap results are those of
the numpy code by construction.

## Applicability to this codebase

Proposed shape of the change (what a worktree agent would build):

- New `kernels.py` (from `nbk.py`): the jitted functions, `warmup()`, and a module flag `AVAILABLE` (False if numba
  can't be imported; everything then runs the numpy code as today).
- `entities.EnvConfig.compiled = True` (per world via `World.env` like other settings). `solver.World` methods keep
  their numpy bodies as the reference and dispatch to kernels when `compiled` and available:
  `_solve_contacts`, `_solve_body_constraints`, `_candidates` / `_ccd_cells` / `_ccd_pairs` (via one
  `_pairs(plo, phi, elo, ehi, contact)` helper), `_ccd_detect`, `_ccd_impulses`, `_impact_zones`, `_bounce`, and the
  rope loop in `_pbd_substep` (with the packed rope ends cached per `_rebuild` + rope list, anchors / rest lengths /
  stiffness re-read each substep because the editor can move hooks).
- Topology arrays the kernels need contiguous (`edges[0]`, `edges[1]`, `edge_local` rows, bool `body_fixed` /
  `body_self`, spring rows) are built once in `_rebuild` / `_pack_constraints`.
- Shape matching in numpy switches to the trig-free, matmul-free form (F4) so numpy and numba stay bit-identical;
  this changes trajectories once at ~1e-9 px level (a new baseline for hash tests), not physics.
- `main_opengl.py` calls `kernels.warmup()` at startup (0.1 s cached).
- `gpu_backend.py` untouched; the CCD it relies on gets the kernels automatically.
- Tests: a bitwise A/B (numpy vs compiled) over a few presets, plus 10's suite.

## Expected gains and risks

Gains (P-core, measured on the prototype): physics 5.6x on the default demo (2.7 -> 0.48 ms), 12.8x on the cradle,
5-9x on piles up to 500 bodies, 4.4x at 1000. Whole-frame for the demo: 09 measured 5.3 ms with 2.3 ms physics, so
~3.4 ms after (draw/UI/present now dominate; 09's draw fix is the next lever). Big piles: pile500 33 -> 6.4 ms puts a
500-body pile at 60 FPS on the CPU.

Risks:
- **Zero-overlap guarantee: none added**, as long as the bitwise A/B stays green (it proves the CCD makes identical
  decisions). If a later kernel change breaks bitwise identity on purpose (e.g. 08's `prange`), the guarantee still
  rests on the same exact-arithmetic argument, but validation falls back to 10's brute-force checker.
- Silent wrong-index bugs (no bounds checks): mitigated by the A/B test and `NUMBA_BOUNDSCHECK=1` in tests.
- Dependency: numba pins numpy ranges; a numpy upgrade can break `import numba` until numba catches up. Mitigation:
  numpy fallback when the import fails.
- First run after install / kernel edit: 4 s compile. Cache corruption or recursion-in-cache segfaults (F3): no
  recursion in kernels; cache dir trusted.
- Two code paths to keep in sync (numpy reference + kernel). The A/B test is what keeps them honest.

## Effort

Medium: ~700 lines of kernels already prototyped (`nbk.py`, plus 235 lines of glue in `nbworld.py`) + ~150 lines of dispatch in `solver.py`. One agent, about a
day including the A/B test and 10's suite. The broad-phase kernel is the trickiest (dense / sparse switch, multi-cell
dedupe); the rest are direct ports.

## Build plan

Ranked; 1 first, the others can run in parallel worktrees on top of 1 (or independently against numpy).

1. **Compiled physics core (numba) with numpy fallback, bitwise-tested.** Files: new `kernels.py` (from
   `/private/tmp/claude-501/research-05/nbk.py`), `solver.py` (dispatch in the methods listed above, contiguous
   topology in `_rebuild`), `entities.EnvConfig.compiled`, shape matching -> trig-free form in `solver.py` (and
   `entities.RigidBody.solve_constraints` for solo bodies), `main_opengl.py` -> `kernels.warmup()`.
   Metric: physics ms/frame on demo, cradle, diamonds, pile200, pile500 (P-core, `nice -n 10`, interleaved A/B:
   `/private/tmp/claude-501/research-05/pcore.py` / `phases.py`), bitwise-identical state every frame vs the numpy
   path (`compare2.py`), 0 overlaps on `/private/tmp/claude-501/hud-layout-agent/suite/ccd_test.py` and research-10's
   `run_bench.py --code <worktree>`. Expected: 5.6x demo, 12.8x cradle, 5-9x piles. Risk: low (bitwise A/B).
   Effort: medium (1 day).
2. **Deterministic `prange` for the contact pass and CCD detect (with 08).** On top of 1: 08's CSR gather
   (`/private/tmp/claude-501/research-08/numba_par.py`) for `solve_contacts`, plain `prange` for `ccd_detect`
   (independent per pair, compact hits afterwards), `numba.set_num_threads(4)`, serial below ~10k candidates.
   Metric: pile500 / pile1000 physics ms, bitwise vs serial. Expected: ~1.3x at pile500 (contacts are 51% after 1),
   ~1.5x at pile1000. Risk: low. Effort: small.
3. **Hoist and shrink the broad phase in compiled code (with 03 / 01).** On top of 1: 03's fat-box pair list (or the
   box-filtered exact pairs) rebuilt by the `cell_pairs` kernel, shared by contacts and CCD; optionally 01's "broad
   phase once per frame". Metric: physics ms at pile200/500 and on a 16x1 schedule; 0 overlaps on 10's suite.
   Expected: broad phase is 2 of 6.4 ms at pile500 after 1, so up to ~1.4x there and makes 16x1 affordable.
   Risk: medium (new cached state; guarantee argument is 03's superset proof). Effort: medium.
4. **Cheap physics upgrades now that constraints are ~free (with 06 / 07).** On top of 1: 06's soft shape matching
   or ARAP triangles and 07's rigid-aware ropes as numba kernels. Metric: 06/07 accuracy metrics on 10's suite at
   <= 1.2x the compiled baseline's ms. Risk: medium (changes physics, not the guarantee). Effort: medium.

Not recommended: Cython / C / Rust extensions (same speed as numba, need a build system and `-ffp-contract=off`),
mypyc (no gain on numpy code), `fastmath`.
