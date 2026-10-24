# 04 Metal / MLX custom kernels and float32 robustness

Status: cycle 1 complete (prototype verified: 0 overlaps in 1,260 checked substeps). Scratch harness and prototype: `/private/tmp/claude-501/research-04/` (no repo edits).
All timings at normal QoS (`nice -n 10`, P-cores) unless stated, because GPU sync latency is what is being measured and
background QoS distorts it ~4x (F1).

Terms: **sync** = one CPU wait for GPU completion (`mx.eval`, which ends in Metal's `waitUntilCompleted`);
**dispatch** = one compute kernel launch; **TG** = threadgroup (threads that can barrier and share memory, max 1024);
**CCD** = continuous collision detection; **PBD** = position-based dynamics; **pre-reject** = the swept-box overlap test at
the top of `World._ccd_detect`. **pileN** in this file = `build_scaled(N)` (N 12-gons, 12 points each, 25% rigid, in a
walled box; from the earlier GPU agent's `scale.py`), not 09's mixed-shape pileN (~6.3 points per body); 09's pile1000
has about as many points as my pile500.

## Summary

1. **The biggest available GPU-mode win is in the CCD, which is on the CPU.** Only 3-4% of CCD grid pairs survive the
   swept-box pre-reject. A GPU kernel that builds exactly that small set from the float32 boxes, **in the same command
   buffer as the PBD solve (no extra sync)**, plus a **float64 containment check on the CPU** (every exact swept box must
   lie inside its GPU box, else that item is patched on the CPU) keeps the zero-overlap guarantee independent of
   float32 rounding. End-to-end prototype (`proto/`, with fix 2 below): **pile200 13.7 -> 7.6 ms/frame (1.8x),
   pile500 44.4 -> 16.7 (2.7x), pile1000 131 -> 44 (3.0x)**; 0 overlaps in 1,260 independently checked substeps (settled and drop scenes), 0 impact zones.
2. **The GPU contact broad phase was slow because of 12 wall edges, not atomics.** The `pairs` kernel runs one thread per
   edge; the ground and side walls span 50-200 grid cells, so their threads walk the whole floor while 12k others are
   done (1000 bodies: 667 us with walls, 151 us without). One thread per (edge, cell) entry plus a point-to-segment
   distance filter (keep < 2r + 1 px) gives the identical pair set 7x faster (200 bodies: 261 -> 35 us) and 7x fewer
   pairs for the 4 contact passes. SIMD-aggregated atomics, the textbook fix, changed nothing.
3. **The sync floor is ~150 us per round trip and the GPU backend pays it every substep.** MLX, torch MPS and Taichi
   all sit at 150-250 us per sync (wgpu-py ~1.8 ms); each extra dependent kernel in a command buffer costs 3-8 us.
   With CCD on the CPU between substeps that is >= 0.6 ms/frame, so the GPU can't win on the small presets (09:
   crossover ~1200 points). **Fusing all PBD iterations into one dispatch** (one TG, barriers between phases) halves the
   small-scene substep (13 bodies 532 -> 301 us) but loses above ~150 bodies (it uses 1 of 10 GPU cores). If 01's
   "CCD once per frame" lands, the GPU can run all 4 substeps' PBD in one command buffer: **1 sync per frame**.
4. **float32 is fine for the proposal, never for the proof.** The CCD stays float64. In the GPU solve, the only real
   float32 error is world-space `PREV` rounded to float32 (<= 1.2e-4 px at 2048 px). It does not accumulate, but it grows
   relative to the physics as substeps shrink (gravity moves a point g h^2 = 5e-4 px per substep at 16 substeps), so
   per-body origins are needed if 01 raises substep counts on the GPU path.
5. **Determinism is cheap to get:** int32 fixed-point atomics (2^-20 px) cost +3-5% GPU time and make springs and
   contacts bit-reproducible. The rigid shape-matching sums also need to be order-independent, via a per-body loop.
   10 can't hash-compare GPU runs today (two GPU runs end 142 px apart after 4 s).

## Findings

### F1. Dispatch and sync overhead on this M4 (`overhead.py`)

Median of 5 x 300 calls:

| operation | n = 5k floats | n = 50k |
|---|---|---|
| MLX: 1 custom kernel + `mx.eval` | 186 us | 151 us |
| MLX: 4 / 20 / 64 chained kernels, 1 eval | 232 / 292 / 349 | 197 / 462 / 494 |
| MLX: graph build per `metal_kernel` call (Python only) | 0.7 | 0.7 |
| MLX: `mx.compile` of 20 kernels + eval | 237 | 473 |
| MLX: idle `mx.synchronize()` | 12 | 12 |
| torch MPS: 1 op / 40 ops + `torch.mps.synchronize()` | 252 / 462 | 172 / 615 |
| Taichi Metal: 1 / 20 kernels + `ti.sync()` | 218 / 573 | 178 / 765 |
| wgpu-py: 1 dispatch + `read_buffer` | 1816 | 1983 |

- **Background QoS:** the same MLX "1 kernel + eval" costs **733 us** under `taskpolicy -b` (graph build 5.7 us).
  Background work is confined to the E-cores and its wakeups are coalesced, which hurts a sleep-then-wake sync most.
  08 measured the same effect on plain CPU work (44x at background QoS with loaded E-cores). The app runs at default
  QoS, so the numbers above are the relevant ones.
- The floor matches what others report for Metal `waitUntilCompleted` (~120 us for an empty kernel on an M2 Max,
  [Apple forum 767853](https://developer.apple.com/forums/thread/767853)). Apple GPUs also clock down for short, bursty
  work, and there is no API to make them ramp up ([Apple forum 702651](https://developer.apple.com/forums/thread/702651)).
  That fits the 3-8 us we see per dependent dispatch on tiny grids.
- MLX `mx.fast.metal_kernel` ([docs](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)) is the
  cheapest Python route for many small custom kernels. Taichi's sync costs about the same, but each launch costs ~2x
  more. torch MPS has no custom-kernel story this simple, and wgpu-py's readback path is ~10x slower.
  **Stay on MLX.**
- CCD must run in float64 on the CPU between substeps, so each substep needs at least one round trip: **>= 4 syncs/frame
  ~= 0.6 ms**. Only a change to *when* CCD runs can remove this (F7).

### F2. Where GPU time goes in `GpuBackend.pbd_substep` (`bench_pbd.py`, `pairs_bench.py`, `longedge.py`)

Settled piles of 12-gons (`build_scaled`, 25% rigid). Times are per substep.

| bodies (points) | pbd_substep | pair build + sync | 16 solver kernels + sync | GPU per extra 16-kernel pass | CPU `_pbd_substep` |
|---|---|---|---|---|---|
| 13 (168) | 499 us | 418 | 305 | ~110 us | 472 us |
| 50 (612) | 842 | 597 | 423 | ~140 | 867 |
| 200 (2412) | 1389 | 795 | 414 | ~150 | 3222 |
| 500 (6012) | 2531 | 1608 | 501 | ~190 | 15537 |
| 1000 (12012) | 2894 | 2501 | 890 | ~330 | 75106 |

(My 1000-body state is a heavily compressed pile, so its CPU numbers are higher than 09's pile1000.)

- The `pairs` kernel dominates the pair build. **Cause: load imbalance from the walls.** Edge cell counts are mean
  3.0-3.2 but max 96 (200 bodies) or 200 (1000 bodies); 6 edges span > 20 cells. Timing the kernel on subsets:

  | bodies | all edges | without the 12 wall edges | only the 12 wall edges |
  |---|---|---|---|
  | 200 | 154 us | 74 us | 186 us |
  | 1000 | 667 us | 151 us | 730 us |

- **Not atomic contention:** with SIMD-group aggregation (`simd_prefix_exclusive_sum` + one atomic per 32 lanes, the
  MLX docs' advice), the pair set was identical and the time was not lower (200 bodies 240 vs 234 us; 1000: 1076 vs 725).
- **Fix measured (`flatpairs.py`):** one thread per (edge, cell) entry (7.8k entries at 200 bodies, 36.5k at 1000)
  gives the **identical pair set** in 35 us instead of 261 (200 bodies) and 119 instead of 586 (1000). The entry to edge
  map is a binary search over the exclusive prefix sum of per-edge cell counts. The CPU already has those counts from
  the cell ranges it computes, so there is no extra GPU pass. A point-to-segment distance filter (keep d < 2r + 1 px at
  the predicted positions) shrinks the list from 38.4k to 5.5k pairs (200 bodies) and from 285k to 47k (1000). That is
  the same set the CPU path effectively solves: `_solve_contacts` keeps only `dist < r + slack = 2r` in its `reach`
  cache after the first pass, and its cell candidates are only guaranteed complete within 2r anyway.
- Launching `contact_acc` over `contact_cap` threads instead of the pair count wastes 20-35% of the contact pass
  (`capgrid.py`: 4 passes 131 -> 88 us at 200 bodies). That falls out naturally once the pair list is smaller.
- Uploads (`mx.array` from numpy, 10-360 us) and downloads (`np.array`, 3-18 us) are cheap: memory is unified.

### F3. CCD swept-box pair set on the GPU, verified in float64 (`ccdgpu.py`, `margin.py`, prototype `proto/`)

What survives each CCD filter on the CPU (15 frames per row):

| bodies | CCD grid pairs | pass swept-box test (margin 0) | margin 0.5 px | margin 1 px | real hits / substep |
|---|---|---|---|---|---|
| 200 | 21,067 | 636 (3.0%) | 1,303 | 2,683 | 0-2 |
| 1000 | 211,049 | 7,937 (3.8%) | 15,864 | 31,734 | 66-127 |

So ~97% of `_ccd_pairs` + pre-reject work builds pairs that are thrown away. 03 finds the same (7.9% survive at pile500).
Design:

1. **GPU, same command buffer as the PBD iterations** (`proto/gpu_backend.py` `_ccd_graph`). One thread per point and
   per edge forms its swept box from float32 `PREV` and the final PBD displacement `x` (already on the GPU). Edges grow by
   the existing 1.0 px, then every box grows by margin m (0.5 px) and a rounding slack (1e-3 px + 1e-5 |coord|).
   Integer cell ranges are `floor(box / cell)` of those same float32 values; a hash grid holds the point ranges; one
   thread per edge emits (point, edge) pairs whose float32 boxes overlap (exact comparisons), using `_filter_pairs`'
   CCD rules. Completeness needs only integer logic and comparisons of identical values: floor is monotone, so
   overlapping boxes always share a cell.
2. **CPU after the existing sync** (`proto/solver.py` `_box_patch`). Download the boxes as float64 (exact conversion)
   and the sorted pairs, compute the exact float64 swept boxes (`World._ccd_boxes(D)`, D after `_limit_speed`) and
   check each lies inside its GPU box. The speed clamp only shortens paths, so clamped boxes sit inside unclamped
   ones. Any item outside its box gets B = exact box +- m and is brute-force paired against every box (k items x N or E,
   vectorized). Invariant: **`pairs` holds every filtered pair whose boxes in B overlap, and every exact box is inside
   B**. So every pair whose exact boxes overlap is in `pairs`, and `_ccd_detect` can only report a hit for such a pair.
3. **Impulse passes:** the same check on the new `D` replaces `_ccd_new_pairs` and the "fresh broad phase" confirmation.
   If no item left its box and the redetect found no hits, the substep is proven clean. Impulses moved path ends
   <= 0.40 px at 200 bodies (all 60 substeps) and <= 1.05 px at 1000 (1 of 60 substeps above 1 px), so m = 0.5 px
   rarely needs a patch (2 items in 160 substeps at pile500, 24 at pile1000).
4. **`_impact_zones`** keeps calling the full CPU broad phase (`_ccd_detect(V * h)` with `pairs=None`). It's the rare
   fail-safe and needs no change.
5. One behaviour difference: today's pre-reject also keeps pairs whose edge collapses to ~zero length mid-substep,
   which produce spurious far "hits". The box set drops them. Safety is unaffected (a real hit on a collapsed edge is
   inside its box), but for bit-identical results add grid pairs for those few edges on the CPU (`collapses` is O(E)).

Micro-benchmark of the pieces (`ccdgpu.py`):

| bodies | GPU set superset of CPU set (m = 0 / 0.5) | GPU cost inside a batch | CPU after sync (download, verify, float64 narrow) | CPU today (`_ccd_detect` from scratch) | hits equal |
|---|---|---|---|---|---|
| 13 | yes / yes | ~170-240 us | 140 us | 212 us | 0 = 0 |
| 200 | yes / yes | ~150 us | 430-480 us | 2,120 us | 2 = 2 |
| 500 | yes / yes | ~150 us | 1,400-1,700 us | 9,570 us | 47 = 47 |
| 1000 | yes / yes | ~500 us | 2,450-3,040 us | 17,000 us | 66 = 66 |

This is the CCD analogue of a Verlet / skin neighbour list (store pairs within cutoff + skin, reuse until something
moves more than the skin; [LAMMPS](https://docs.lammps.org/Developer_par_neigh.html)). The difference is that validity
is checked exactly per item in float64 rather than assumed from a max-displacement bound.

A float32 conservative *narrow*-phase filter on the GPU, in the spirit of provably conservative floating-point CCD
(Tight Inclusion and its GPU version, [arXiv 2112.06300](https://arxiv.org/abs/2112.06300)), isn't worth building:
after the box filter the float64 quadratic runs on a few hundred to a few thousand pairs on the CPU (< 1-2 ms), and an
error-bounded float32 root test would bring a proof burden of its own.

### F4. End-to-end prototype (`proto/`, `e2e.py`)

`proto/` holds copies of `solver.py` and `gpu_backend.py` with F2's balanced, distance-filtered contact pairs and F3's
CCD box set. Diffs: `/private/tmp/claude-501/research-04/proto_solver.diff` (92 lines) and `proto_gpu_backend.diff`
(270 lines). Settled piles, 40 frames, median ms/frame (physics only):

| bodies (points) | main, GPU backend | prototype | speed-up |
|---|---|---|---|
| 13 (168) | 4.04 | 3.99 | 1.0x |
| 50 (612) | 5.42 | 4.85 | 1.1x |
| 200 (2412) | 13.69 | 7.63 | 1.8x |
| 500 (6012) | 44.36 | 16.66 | 2.7x |
| 1000 (12012) | 131.12 | 44.42 | 3.0x |

Overlap check: an independent segment-crossing checker (`e2e.py crossings`: its own grid, no solver code, verified to
flag an injected crossing) ran on every substep's end state: **0 overlapping substeps** in pile200 (172 substeps) and
pile500 (172), pile1000 (172), and drop scenes pile200 / pile1000 (372 each); see the status note. Impact zones: 0 everywhere.

Where the prototype's time goes now (profiler, per frame): pile200: GPU PBD + CCD set 6.2 ms, CPU CCD 1.85 (broad 0.37,
narrow + impulses 1.46). pile1000: GPU 18 ms, CPU CCD 30 (broad 4.3, narrow + impulses 25.6). The remaining CPU cost at
scale is the sequential Python Gauss-Seidel impulse loop `_ccd_impulses` and repeated redetects, which is 05's
territory (numba), not the GPU's.

### F5. Fusing the PBD iterations into one dispatch (`fused.py`)

One TG of 1024 threads loops over springs, rigid points, pairs and points, with `threadgroup_barrier(
mem_flags::mem_device)` between phases, all 4 iterations in one kernel. Results match the 16-dispatch version within
float-atomic noise (max |dx| 4e-6 to 8e-6 px, the same spread as rerunning the current version).

| bodies | current: synced / GPU-only | fused float32: synced / GPU-only | fused fixed-point |
|---|---|---|---|
| 13 | 532 / 105 us | 301 / 74 us | 305 / 77 |
| 50 | 533 / 122 | 247 / 47 | 248 / 54 |
| 200 | 328 / 88 | 475 / 272 | 500 / 277 |
| 500 | 710 / 143 | 1589 / 1313 | 1639 / 1351 |

Metal has no grid-wide barrier. A multi-TG "persistent kernel" with an atomic spin barrier assumes all TGs are
co-resident, which Metal does not guarantee (risk: GPU hang). So the rule is: fused single-TG kernel below ~1500 points,
the current multi-dispatch path above. With the 4-syncs/frame floor, this takes small-scene GPU physics from ~2 to
~1.2 ms/frame. That is still slower than the CPU path (0.5-2.3 ms, 09), so its value is to soften the switch-over of an
"auto" backend and to pair with F7.

### F6. float32 robustness

- Current design: CPU float64 `PREV`, GPU float32 substep-local `x = P - PREV`, differences formed as
  `(PREV_j - PREV_i) + (x_j - x_i)`, `P = PREV + float64(x)`. Nothing accumulates across substeps. Per-substep error:
  `x` carries ~1e-7 relative error on <= 10 px of motion (~1e-6 px); world-space `PREV` rounded to float32 is off by
  <= 2^-24 |coord| (1.2e-4 px at 2048 px, 6e-4 px at 10^4 px).
- The `PREV` rounding acts as a constant offset in every spring and contact for one substep, then a different one next
  substep. That is position noise delta ~1e-4 px, or velocity noise delta / h: 0.03 px/s at 4 substeps, **0.1 px/s at
  16 substeps**. Meanwhile the physics per substep shrinks as h^2 (gravity sag per substep 3e-2 px at h = 1/240, 5e-4 px
  at 1/960). Invisible at 4 substeps; measurable in 10's "dead still stack" metric and comparable to the per-substep
  physics if 01's small-steps schedules (16+ substeps) are run on the GPU.
- Cheap fix: upload `PREV` relative to a per-body origin (the body's first point, sent as float32 from a float64
  subtraction) and form cross-body differences as `(origin_b - origin_a) + (local differences)`. Contacting bodies are
  close together, so the origin difference is small and accurate. Error drops ~30x (ulp of ~36 px instead of
  ~1500 px), and worlds with |coord| ~1e5 px (where ulp ~ 0.008 px ~ `ccd_gap`) become safe. Double-single
  ("float-float") arithmetic would also work but costs ~10-20 flops per op and isn't needed.
- XPBD (01, 06) on the GPU: alpha / h^2 stays O(1e2-1e3) at these h, which is fine in float32. The thing to watch is the
  constraint-error rounding above, which the per-body origin fixes.
- CCD never uses a float32 result without a float64 check (F3).

### F7. Interaction with 01's "CCD once per frame": one sync per frame

01 proposes running the contact broad phase and CCD once per frame on the frame's start -> end straight-line motion
(Bridson et al. 2002 permits it; the guarantee only needs the frame's motion checked from an intersection-free start).
For the GPU backend that changes the cost model completely. The 4 substeps' PBD (predict, iterations, velocity update)
no longer need the CPU between them, so all of it can go into **one command buffer with one sync per frame**,
followed by the F3 box set for the frame's motion. The sync floor drops from ~0.6 to ~0.15 ms/frame. Predict moves to
the GPU (trivial kernel), and the substep velocity update `V = (P - PREV) / h` would run in float32 on the GPU, which
is fine because the final CCD-checked motion is recomputed in float64 on the CPU. Estimate for small scenes with the
fused kernel: ~0.15 ms sync + 4 x ~75 us = ~0.45 ms/frame GPU physics plus CPU CCD. The crossover with the CPU path
would then drop from ~1200 points to a few hundred. Caveat: frame-long sweeps are 4x longer, so boxes grow and there are
more CCD pairs and hits; 01 is measuring this.

### F8. Determinism: alternatives to float atomics

Float atomic adds commit in scheduling order, so the GPU backend is not reproducible (rerunning one substep differs by
up to 6e-6 px; 10 measured runs ending 142 px apart after 4 s). Integer atomic adds are associative, so fixed-point
accumulation is deterministic ([NVIDIA Warp, deterministic execution](https://nvidia.github.io/warp/latest/user_guide/execution_and_performance/deterministic_execution.html);
[Defour & Collange 2015](https://annals-csis.org/proceedings/2015/pliks/86.pdf);
[arXiv 2408.05148](https://arxiv.org/pdf/2408.05148)).

- Prototype (`fused.py` fixed variant): int32 at 2^-20 px (range +-2048 px, each contribution clamped to +-64 px),
  counts as plain int atomics. Reruns were bit-identical at 13 bodies. At 50+ bodies they still differed because the
  rigid shape-matching sums (`rsum`) were left as float atomics. Fix: one thread per rigid body summing its points in
  order (12 points: trivial). Cost: +3-5% GPU time.
- Static topology (springs) needs no atomics at all: a per-point CSR (compressed sparse row) gather of incident springs
  is deterministic and fuses `con_acc` + `con_apply` into one kernel.
- Pair lists come out of atomic counters in arbitrary order. With fixed-point sums the order doesn't matter for
  contacts; the CCD set is sorted on the CPU (already done in the prototype).

## Applicability to this codebase

- `gpu_backend.py`
  - `GpuBackend.pbd_substep`: append the CCD box set kernels (`_CCD_BOXES`, `_CCD_GCOUNT`, `_CCD_GFILL`,
    `_CCD_BPAIRS` in `proto/gpu_backend.py`) to the same graph and evaluate them in the same `mx.eval`. Store
    `(boxes, pts, eids)` for `World` (`take_ccd_set`). Overflow handling mirrors `contact_cap` (`ccdset_cap`,
    `ccdset_ent`, one extra sync on overflow only).
  - `_pairs` mode 0 -> `_FLAT_PAIRS`: one thread per (edge, cell) entry, distance filter at `2 * contact_radius + 1`.
    The mode 1 (`gpu_ccd_broadphase`) path becomes obsolete.
  - Optional: fused single-TG iteration kernel when N < ~1500; fixed-point accumulation and per-body rigid sums behind
    a flag; `PREV` relative to per-body origins.
- `solver.py`
  - `World._resolve_ccd`: if the backend supplied a set, use it as `pairs` after `_box_patch`; in the "confirm" branch
    replace `_ccd_new_pairs` / the fresh `_ccd_pairs` with `_box_patch` (prototype: `proto_solver.diff`).
    `_impact_zones` unchanged.
  - Any change to `_solve_contacts` (10's momentum fix: scale each contact by 1/max(count) over its 3 points; 07's
    rigid-aware contacts) must be mirrored in `_CONTACT_ACC` / `_CONTACT_APPLY`. 10's fix needs the counts before the
    corrections, so it costs one more kernel per iteration (count pass, then correction pass), or a fused-kernel phase.
- `entities.py` `EnvConfig.backend`: the default `"gpu"` is a net loss on every preset (09). An `"auto"` mode (CPU
  below ~1200 points or when ropes / solo bodies force CPU PBD anyway, GPU above) is the companion change.

## Expected gains and risks

| change | gain (measured or projected) | risk to zero overlap | other risk |
|---|---|---|---|
| F3 CCD box set + F2 flat contact pairs (prototype) | **1.8x / 2.7x / 3.0x** at 200 / 500 / 1000 bodies (measured) | none if `_box_patch`'s invariant is kept (GPU proposes, CPU verifies in float64); bugs would be in the impulse-loop bookkeeping, so test with the brute-force checker on every substep | contact pair set differs slightly from today (distance-filtered), so trajectories differ |
| F2 alone (flat pairs + distance filter) | ~0.3-1.5 ms/substep GPU at 200-1000 bodies | none (contacts only shape the proposed motion) | none |
| F5 fused small-N kernel | ~0.8 ms/frame on small scenes in GPU mode | none | still slower than CPU there; useful with F7 or "auto" |
| F7 one sync per frame (needs 01) | sync floor 0.6 -> 0.15 ms/frame; crossover maybe a few hundred points | relies on 01's frame-level CCD being correct | larger CCD sweeps |
| F8 fixed-point determinism | reproducible GPU runs, +3-5% GPU time | none | contributions > 64 px per iteration saturate (never seen; add an overflow flag) |
| F6 per-body origins | float32 error ~30x lower | none | small code churn |

## Effort

- F3 + F2 productionized from the prototype: 1-2 days (kernels exist; the work is tests, overflow paths, profiler
  sections, the collapsing-edge corner case, and the `gpu_ccd_broadphase` cleanup).
- F5 fused small-N kernel: 0.5-1 day. F8 determinism: 0.5 day. F6 origins: 0.5 day. F7: 1-2 days on top of 01.

## Cross-links

- 01 (XPBD, substeps): F7 depends on 01's frame-level CCD; F6 is the float32 caveat 01 flagged for high substep counts on
  the MLX path.
- 03 (broad phase): same observation (only ~3-8% of CCD pairs survive the pre-reject) and a CPU fat-box pair cache. F3
  is the GPU version of that idea, built in the PBD command buffer and verified per item instead of by a global
  displacement bound. My whole-list skin test (`skin.py`) found a global max-displacement rule nearly useless for
  contacts in moving piles (skin 4 px: rebuild in 111/240 substeps at 200 bodies, 239/240 at 1000), so per-item
  verification or patching, as in `_box_patch`, is the variant that pays.
- 05 (compiled loops): after F3 the GPU-mode frame of my compressed 1000-body pile is dominated by the sequential
  `_ccd_impulses` loop and redetects (25.6 of 44 ms). That state has 66-127 CCD hits per substep; 05 and 02 see only
  0.1-0.4 hits per substep in ordinary piles, where impulses are not hot. Either way numba fixes it, and the two
  compose (GPU PBD + verified box set, numba narrow phase + impulses). 05's numba kernels (contact pass 6x, contact
  broad phase 6x, CCD detect 9x) also make the CPU backend much stronger, so the GPU/CPU crossover has to be re-measured
  once both land. Don't pick a default backend from today's numbers.
- 08 (parallelism): `mx.eval` releases the GIL, but a physics thread doesn't help (GIL convoy). A worker process that
  owns MLX, as in 08's pipeline, also keeps MLX and moderngl apart; F3 shortens the worker's frame.
- 02 (CCD accuracy): agrees that CCD time is almost all "proving nothing hits" (85-100% of substeps have zero hits),
  so a cheaper no-hit proof is the lever. F3 is such a proof: the box set shrinks the candidate list ~25x, and the
  float64 quadratic stays exactly as 02 analyses it.
- 09 (profiling): agrees on the crossover (~1200 points) and that CPU float64 CCD is 76% of the GPU-mode frame at
  pile1000.
- 10 (accuracy): F8 makes GPU runs hash-comparable; F3 must pass 10's per-substep overlap checker.
  10's momentum fix needs mirroring in the GPU contact kernels.

## Build plan

Ranked. Each runs in its own git worktree; the starting code is in `/private/tmp/claude-501/research-04/proto/` (diffs
`proto_solver.diff`, `proto_gpu_backend.diff`). Common validation: 10's `run_bench.py --code <worktree>` (overlap on
every substep, determinism, GPU-vs-CPU) plus the timing rule (normal QoS, settled cached states, median of >= 40 frames).
Scenes: `build_scaled` piles 13/50/200/500/1000 (`/private/tmp/claude-501/research-04/bench_pbd.py`, states cached in
`.../research-04/states/`), 09's presets, plus `e2e.py` (frame time + independent crossing checker).

1. **GPU CCD swept-box set + balanced contact pairs** (highest value; prototype exists).
   Change: `gpu_backend.py` (`_FLAT_PAIRS`, CCD box kernels, `take_ccd_set`, overflow retries), `solver.py`
   (`_box_patch`, `_merge_pairs`, `_resolve_ccd` confirm branch); delete the `gpu_ccd_broadphase` path; move
   `_CCD_BPAIRS` to one thread per (edge, cell) entry (the walls make the per-edge version cost ~0.8-1.5 ms/substep).
   Success: pile200/500/1000 GPU-mode ms/frame <= 8 / 17 / 45 (from 13.7 / 44 / 131); 0 overlaps and 0 impact zones in
   10's suite and in `e2e.py check` (settled and drop, 200/1000 bodies); presets unchanged or faster.
   Risk: low for the guarantee (float64 verification), medium for bookkeeping bugs. Effort: 1-2 days.
2. **Auto backend + fused small-N PBD kernel.** Change: `EnvConfig.backend = "auto"` (`solver._make_backend`: CPU
   below a measured point threshold or when ropes / solo bodies exist), `gpu_backend.py` single-TG fused kernel below
   ~1500 points. Success: no preset slower than the CPU backend (09's table); GPU-mode small scenes ~2 -> ~1.2 ms/frame;
   threshold chosen from a 13..1000-body sweep. Risk: none to overlap. Effort: 1 day.
3. **Deterministic GPU accumulation + per-body float32 origins.** Change: `_CON_ACC` / `_CONTACT_ACC` /
   `_CONTACT_APPLY` to int32 fixed point (2^-20 px, clamp + overflow flag -> rerun in float), rigid sums by one thread
   per body, `PREV` uploaded relative to body origins. Success: two GPU runs of the same scene hash-identical (10's
   determinism test extended to GPU); GPU time +<= 5%; 10's rest-jitter metric for GPU no worse than CPU at 4 and 16
   substeps. Risk: none to overlap. Effort: 1 day.
4. **One sync per frame (after 01's frame-level CCD).** Change: GPU runs predict + PBD for all substeps in one graph,
   then the F3 box set for the frame's motion; CPU does float64 CCD once. Success: GPU-mode physics floor <= 0.5
   ms/frame on small scenes; crossover with the CPU path below 500 points; 0 overlaps. Risk: depends on 01's
   correctness argument. Effort: 1-2 days after 01 lands.

## Status note (final verification)

Independent crossing checker (including self-crossings of self-colliding soft bodies) on every substep end state,
prototype `proto/`: settled pile200 0/172, pile500 0/172, pile1000 0/172; drop (built fresh, falling and landing)
pile200 0/372, pile1000 0/372. **0 overlapping substeps out of 1,260.** 0 impact zones. Patched items (exact box
outside its GPU box) were rare: 0-3 per run at 200-500 bodies, 13-26 per run at 1000.

Substep breakdown of the prototype (`protosub.py`, synced `pbd_substep`, includes the CCD set download and sort):

| bodies | main | + flat contact pairs | + CCD set only | both |
|---|---|---|---|---|
| 200 | 1015 us | 720 | 2138 | 1510 |
| 1000 | 2725 | 1118 | 3651 | 2579 |

The CCD set adds ~0.8-1.5 ms per substep on the GPU side, far more than the ~150 us measured in isolation. The cause is
the same wall-edge imbalance as F2: `_CCD_BPAIRS` is still one thread per edge, and the walls' swept boxes span many
cells. Moving it to the (edge, cell) entry scheme of `_FLAT_PAIRS` should recover most of that, so the measured
1.8-3.0x speed-ups are conservative. Build plan item 1 should include that change.
