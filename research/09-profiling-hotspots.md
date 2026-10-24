# 09 Profiling-driven hotspot analysis (measured)

Status: cycle-1 data complete (whole frame, physics phases for all presets + piles on both backends, cProfile,
line-level timing, allocation churn, scaling exponents, work-funnel counts, and two validated prototypes: numpy-free
body drawing, 3.4x and pixel-identical; CCD swept-box test before dedupe, -27% GPU-backend physics at 1000 bodies,
bit-identical). Raw data and
scripts: `/private/tmp/claude-501/research-09/` (`phys.py`, `frame.py`, `cprof.py`, `lprof.py`, `alloc.py`,
`counts.py`, `cellsweep.py`, `micro*.py`, `drawproto.py`, `b3proto.py`; results in `*.jsonl`, `prof/`).

## Summary

1. **The default GPU backend never helps a shipped preset, and hurts the small ones.** It has a ~2 ms/frame floor:
   one synced GPU substep (pair build + 16 solver kernels + `mx.eval`) costs ~0.45 ms even for an 8-point scene
   (line profile: 85% of `GpuBackend.pbd_substep` is the `mx.eval` wait), times 4 substeps. Interleaved A/B (3 runs
   each): blank 0.52 vs 1.8-3.1 ms (3.5x slower), looping 1.24 vs 2.3 (1.9x), random walls 2.47 vs 3.5 (1.4x), diamonds (872 points) break-even (0.92x,
   GPU noisy). default and newtons cradle contain ropes, so "gpu" silently runs CPU PBD there. GPU wins clearly from
   ~600 points in contact-heavy scenes (pile100 0.68x, pile200 0.53x, pile500 0.52x, pile1000 93.6 -> 43.5 ms,
   pile2000 189 -> 80). `MLX_METAL_FAST_SYNCH=1` does not move the floor (2.71 vs 2.72 ms). CPU timings repeat within
   2%; GPU ones vary +-30%. Fix: pick the backend automatically by point count (~600-900 points), and/or make ropes
   GPU-capable so the big rope/chain scenes can use it.
2. **Drawing is a bigger hotspot than expected, and it is pure Python/numpy overhead.** `Body.draw_filled` +
   `draw_border` cost ~20-25 us per body (12 tiny numpy calls incl. two `np.roll`s per body; the pygame line calls
   are only ~15% of it). diamonds: draw 4.0 ms = as much as 73% of physics; pile500: draw 11.9 ms vs GPU physics
   13.7 ms. **A numpy-free prototype is pixel-identical and 3.4x faster** (pile500: 10.3 -> 3.0 ms).
3. **Small scenes are dispatch-bound, big settled piles are pair-count-bound.** Fitted CPU cost: ~1.9-2.5 ms fixed
   per frame (~3000 Python calls, ~1200 numpy C calls, 0.2-2.4 us each) + 4 us/point (drop) to 15 us/point
   (settled pile). Over 1000 points, settled-pile cost grows as N^1.23 (CCD narrow N^1.35) because pile depth
   compresses bodies: contact candidates per point go from 8.6 (pile200) to 16 (pile1000). The grid broad phase is
   loose: only 24% of contact candidates are inside the edge's padded box and **only 5-8% of CCD broad-phase pairs
   survive the swept-box overlap test**. Once PBD is on the GPU, **float64 CPU CCD is 76% of the frame** (pile1000:
   33 of 43.5 ms). Applying the exact swept-box test right after the cell join, before the dedupe sort (scratch
   prototype, bit-identical results), cuts GPU-backend physics 45.6 -> 33.2 ms at 1000 bodies (-27%) and 7-12% on CPU.

Other measured facts others should know: physics is only 16-38% of the whole frame for the light presets
(blank, looping, default, random walls; default 6.1 ms/frame on Retina: physics 2.2, draw 0.7, UI/sidebar 0.75, texture upload 0.5, `ctx.finish` wait 1.75);
newtons cradle's `_bounce` is 1.2 ms (26% of physics); zero impact-zone fallbacks occurred in any run; temporaries
above ~1 MB cost 2.6-3.2x more per element on macOS (fresh pages), but chunking the contact solve only gained 10-14%.

## Measurement setup (read this before quoting numbers)

- Harness: `phys.py` = physics only, `World._step` re-instrumented with extra profiler sections (same operations
  in the same order); `frame.py` = whole frame through `main_opengl.main(display=..., max_frames=..., on_frame=...)`
  with a standalone moderngl GL 4.1 core context and an offscreen FBO (Retina 2x unless noted), no frame cap
  (`pygame.time.Clock` replaced), fake clock so dt is exactly 1/60 s per frame. World size = what `main_opengl`
  builds for a 1512x982 window (1354 x 982 world units). Piles: `pileN` = N mixed bodies (soft dodecagons, soft
  hexagons, rigid squares, rigid triangles; ~6.3 points/body) in a box, same builder as
  `/private/tmp/claude-501/perf-engine-agent/scenes.py`. "drop" = frames 5.. after build; "settled" = state after
  300 frames (cached in `states/`; for diamonds/looping it is 5 s into a never-ending fall).
- **QoS: background numbers are not usable for absolute timing right now.** With 10 research agents loading the 6
  efficiency cores (load average 10.7), the same default-scene physics run took 22-40 ms/frame at background QoS
  (`taskpolicy -b`) vs 2.13 ms on a P-core, i.e. **10-19x slower and drifting by 2x within minutes** (topic 10 saw
  up to ~35x on its pendulum scene; topic 04 saw a GPU sync inflate 4x more than numpy work). The inflation
  is not uniform: whole frame 69 vs 5.3 ms, with texture upload 8.5 vs 0.52 ms (16x) and UI 12.2 vs 0.78 ms (16x)
  vs physics 32.7 vs 2.34 ms (14x). So every table below is a **P-core run (`nice -n 10`, short: seconds each)**;
  section *ratios* agree with the background runs. Anyone comparing to the 16.7 ms frame budget must use P-core
  numbers, and A/B comparisons should be interleaved in one session (ABAB).
- GPU timings carry extra noise (another agent may be using the GPU; p95 is often 1.5-2x the mean). The CPU vs
  GPU crossover was re-measured with interleaved repeats (section "A/B repeats").
- Profilers used: `profiler.Profiler` sections (sub-us overhead), cProfile, line_profiler 5.x (installed into a
  scratch `--target` dir, not the repo venv), tracemalloc, `resource.getrusage`.

## Findings

### F1. Whole frame by phase (P-core, Retina 2x framebuffer, 1512x982 window, filled view, grid on)

`frame.py`, 120 timed frames after 10 warm-up frames, ms/frame from the app's own `Profiler` sections.
"gpu wait" is the GPU-tagged `ctx.finish()` before the (no-op) flip: the CPU blocking until the GPU has done this
frame's 3 layer uploads + composite at 3024x1964 (it would overlap or merge with vsync on a real display).

| backend | scene | frame wall | physics | draw | ui | upload | gpu wait (ctx.finish) |
|---|---|---|---|---|---|---|---|
| cpu | default | **6.06** | 2.24 | 0.73 | 0.75 | 0.50 | 1.75 |
| gpu | default | **5.83** | 2.18 | 0.71 | 0.72 | 0.48 | 1.66 |
| cpu | newtons cradle | **9.73** | 5.89 | 0.93 | 0.75 | 0.48 | 1.59 |
| gpu | newtons cradle | **9.83** | 5.92 | 0.93 | 0.75 | 0.49 | 1.67 |
| cpu | blank | **3.33** | 0.52 | 0.41 | 0.69 | 0.47 | 1.18 |
| gpu | blank | **4.66** | 2.00 | 0.41 | 0.70 | 0.48 | 1.01 |
| cpu | random walls | **5.83** | 2.20 | 0.77 | 0.72 | 0.48 | 1.59 |
| gpu | random walls | **5.53** | 2.62 | 0.75 | 0.71 | 0.48 | 0.89 |
| cpu | looping | **4.74** | 1.47 | 0.40 | 0.74 | 0.50 | 1.53 |
| gpu | looping | **4.81** | 2.22 | 0.37 | 0.71 | 0.48 | 0.96 |
| cpu | diamonds | **12.34** | 5.50 | 4.05 | 0.81 | 0.50 | 1.38 |
| gpu | diamonds | **12.30** | 5.52 | 4.00 | 0.80 | 0.51 | 1.37 |
| cpu | pile200 | **18.34** | 9.89 | 5.44 | 0.96 | 0.56 | 1.37 |
| gpu | pile200 | **15.18** | 6.97 | 5.22 | 0.85 | 0.54 | 1.50 |
| cpu | pile500 | **38.23** | 22.91 | 11.87 | 0.93 | 0.56 | 1.84 |
| gpu | pile500 | **28.68** | 13.69 | 11.74 | 0.91 | 0.55 | 1.67 |

View options on the default scene (cpu): points view instead of filled saves 0.1 ms of draw; grid off saves
0.37 ms of draw (`World.draw_grid` loops over occupied cells in Python: 0.85 ms in diamonds); trails on adds 0.2 ms.
1x vs 2x framebuffer changes only the GPU wait (0.91 vs 1.85 ms).

Reads:
- Physics is 16-45% of the frame for the light presets (blank 16%, looping 31%, default 37%, random walls 38%) and
  ~45-60% for diamonds / piles / the cradle. Fixed non-physics cost is ~2.7-3.2 ms/frame (UI 0.7-0.9, upload 0.5,
  GPU wait 1-1.8, draw >= 0.4).
- **draw grows with body count at ~20-25 us/body** (diamonds 147 bodies: 4.0 ms; pile200: 5.4; pile500: 11.9). See F6.
- **ui (sidebar) is a constant ~0.75 ms**: `HudLayer.draw` re-lays-out and redraws every card every frame
  (Controls card 0.35-0.43 ms, gravity dial rhombi 0.29-0.34 ms, compute card 0.22-0.28 ms in cProfile), even when
  nothing on them changed.
- upload is ~0.5 ms for 3 full layers (756x491x4 B each; `Texture.write` from `get_buffer`, no conversion) and is
  already lean.

### F2. Physics by phase: every preset + synthetic piles, both backends (P-core)

contacts = PBD broad phase (`_candidates`) + 4 Jacobi contact passes, constraints = springs + shape matching + ropes
(4 iterations x 4 substeps), GPU pbd = the whole PBD part on MLX (incl. its broad phase, synced), CCD = float64 CPU.

| backend | scene | state | pts | ms/frame | p95 | predict | contacts | constraints | GPU pbd (+upload) | CCD broad | CCD narrow+imp | impact zones | bounce | loop | vel/finish | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu | default | drop | 78 | **2.13** | 2.30 | 0.01 | 1.08 | 0.55 | - | 0.30 | 0.12 | - | - | - | 0.02 | 0.04 |
| gpu | default | drop | 78 | **2.31** | 3.30 | 0.01 | 1.18 | 0.58 | - | 0.32 | 0.13 | - | - | - | 0.04 | 0.05 |
| cpu | default | settled | 78 | **2.30** | 2.57 | 0.01 | 1.26 | 0.56 | - | 0.28 | 0.13 | - | - | - | 0.02 | 0.04 |
| gpu | default | settled | 78 | **2.35** | 2.92 | 0.01 | 1.28 | 0.57 | - | 0.29 | 0.13 | - | - | - | 0.02 | 0.04 |
| cpu | newtons cradle | drop | 124 | **6.06** | 6.57 | 0.03 | 2.43 | 1.40 | - | 0.59 | 0.26 | - | 1.20 | - | 0.04 | 0.09 |
| gpu | newtons cradle | drop | 124 | **6.01** | 6.34 | 0.03 | 2.42 | 1.39 | - | 0.59 | 0.26 | - | 1.19 | - | 0.04 | 0.09 |
| cpu | newtons cradle | settled | 124 | **4.55** | 4.69 | 0.03 | 1.04 | 1.34 | - | 0.56 | 0.25 | - | 1.20 | - | 0.04 | 0.09 |
| gpu | newtons cradle | settled | 124 | **4.58** | 4.71 | 0.03 | 1.04 | 1.35 | - | 0.57 | 0.26 | - | 1.20 | - | 0.04 | 0.09 |
| cpu | blank | drop | 8 | **0.52** | 0.53 | 0.01 | 0.25 | - | - | 0.21 | - | - | - | - | 0.02 | 0.03 |
| gpu | blank | drop | 8 | **2.62** | 4.65 | 0.01 | - | - | 2.25 | 0.27 | - | - | - | - | 0.03 | 0.06 |
| cpu | blank | settled | 8 | **0.54** | 0.56 | 0.01 | 0.26 | - | - | 0.21 | - | - | - | - | 0.02 | 0.03 |
| gpu | blank | settled | 8 | **2.23** | 4.44 | 0.01 | - | - | 1.90 | 0.24 | - | - | - | - | 0.02 | 0.05 |
| cpu | random walls | drop | 81 | **2.18** | 2.57 | 0.01 | 0.97 | 0.56 | - | 0.30 | 0.26 | - | - | - | 0.02 | 0.04 |
| gpu | random walls | drop | 81 | **3.46** | 5.67 | 0.01 | - | - | 2.72 | 0.33 | 0.28 | - | - | - | 0.03 | 0.08 |
| cpu | random walls | settled | 81 | **2.50** | 2.56 | 0.01 | 1.24 | 0.58 | - | 0.28 | 0.32 | - | - | - | 0.02 | 0.04 |
| gpu | random walls | settled | 81 | **4.85** | 6.28 | 0.01 | - | - | 4.09 | 0.30 | 0.32 | - | - | - | 0.03 | 0.09 |
| cpu | looping | drop | 35 | **1.46** | 2.13 | 0.01 | 0.44 | 0.56 | - | 0.22 | 0.16 | - | - | 0.02 | 0.02 | 0.04 |
| gpu | looping | drop | 35 | **2.49** | 3.33 | 0.01 | - | - | 1.95 | 0.24 | 0.16 | - | - | 0.02 | 0.03 | 0.07 |
| cpu | looping | settled | 35 | **1.25** | 1.31 | 0.01 | 0.27 | 0.55 | - | 0.22 | 0.11 | - | - | 0.02 | 0.04 | 0.04 |
| gpu | looping | settled | 35 | **3.09** | 4.76 | 0.01 | - | - | 2.56 | 0.25 | 0.12 | - | - | 0.02 | 0.05 | 0.08 |
| cpu | diamonds | drop | 872 | **4.96** | 5.25 | 0.02 | 2.01 | 0.98 | - | 1.06 | 0.72 | - | - | 0.08 | 0.03 | 0.05 |
| gpu | diamonds | drop | 872 | **6.02** | 7.95 | 0.02 | - | - | 4.00 | 1.07 | 0.70 | - | - | 0.08 | 0.05 | 0.10 |
| cpu | diamonds | settled | 872 | **6.38** | 7.66 | 0.02 | 2.66 | 0.98 | - | 1.31 | 1.00 | - | - | 0.28 | 0.08 | 0.05 |
| gpu | diamonds | settled | 872 | **7.40** | 8.84 | 0.02 | - | - | 4.60 | 1.32 | 1.00 | - | - | 0.29 | 0.08 | 0.09 |
| cpu | pile50 | drop | 330 | **3.29** | 4.50 | 0.02 | 1.39 | 0.87 | - | 0.53 | 0.41 | - | - | - | 0.03 | 0.04 |
| gpu | pile50 | drop | 330 | **4.63** | 6.74 | 0.02 | - | - | 3.48 | 0.57 | 0.44 | - | - | - | 0.03 | 0.08 |
| cpu | pile50 | settled | 330 | **4.36** | 4.66 | 0.02 | 2.28 | 0.89 | - | 0.46 | 0.63 | - | - | - | 0.03 | 0.05 |
| gpu | pile50 | settled | 330 | **5.35** | 7.03 | 0.02 | - | - | 4.10 | 0.47 | 0.63 | - | - | - | 0.04 | 0.09 |
| cpu | pile200 | drop | 1262 | **5.82** | 5.98 | 0.03 | 1.94 | 1.77 | - | 1.38 | 0.63 | - | - | - | 0.04 | 0.04 |
| gpu | pile200 | drop | 1262 | **5.76** | 7.28 | 0.03 | - | - | 3.58 | 1.38 | 0.63 | - | - | - | 0.04 | 0.09 |
| cpu | pile200 | settled | 1262 | **11.79** | 12.15 | 0.03 | 6.28 | 1.81 | - | 1.87 | 1.69 | - | - | - | 0.04 | 0.06 |
| gpu | pile200 | settled | 1262 | **9.74** | 11.32 | 0.03 | - | - | 5.83 | 1.97 | 1.74 | - | - | - | 0.05 | 0.12 |
| cpu | pile500 | drop | 3137 | **13.70** | 13.97 | 0.04 | 4.72 | 3.83 | - | 3.49 | 1.43 | - | - | - | 0.13 | 0.05 |
| gpu | pile500 | drop | 3137 | **9.61** | 10.48 | 0.04 | - | - | 4.80 | 3.18 | 1.41 | - | - | - | 0.07 | 0.10 |
| cpu | pile500 | settled | 3137 | **33.63** | 36.17 | 0.04 | 17.84 | 4.02 | - | 5.84 | 5.72 | - | - | - | 0.06 | 0.10 |
| gpu | pile500 | settled | 3137 | **19.57** | 23.13 | 0.05 | - | - | 7.75 | 5.87 | 5.55 | - | - | - | 0.08 | 0.27 |
| cpu | pile1000 | drop | 6262 | **29.05** | 34.86 | 0.08 | 10.45 | 7.45 | - | 7.62 | 3.15 | - | - | - | 0.17 | 0.11 |
| gpu | pile1000 | drop | 6262 | **15.90** | 17.58 | 0.07 | - | - | 5.86 | 6.80 | 2.78 | - | - | - | 0.14 | 0.24 |
| cpu | pile1000 | settled | 6262 | **93.57** | 99.11 | 0.09 | 51.80 | 7.30 | - | 15.57 | 17.79 | - | - | - | 0.22 | 0.80 |
| gpu | pile1000 | settled | 6262 | **43.47** | 46.97 | 0.10 | - | - | 9.81 | 15.39 | 17.66 | - | - | - | 0.16 | 0.35 |

Extra pile sizes (scaling runs):

| backend | scene | state | pts | ms/frame | p95 | predict | contacts | constraints | GPU pbd (+upload) | CCD broad | CCD narrow+imp | impact zones | bounce | loop | vel/finish | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu | pile10 | drop | 80 | **2.04** | 2.63 | 0.01 | 0.86 | 0.60 | - | 0.26 | 0.25 | - | - | - | 0.02 | 0.04 |
| gpu | pile10 | drop | 80 | **2.74** | 4.61 | 0.01 | - | - | 2.10 | 0.28 | 0.25 | - | - | - | 0.03 | 0.07 |
| cpu | pile10 | settled | 80 | **2.52** | 2.56 | 0.01 | 1.25 | 0.61 | - | 0.23 | 0.35 | - | - | - | 0.02 | 0.04 |
| gpu | pile10 | settled | 80 | **2.47** | 3.18 | 0.01 | - | - | 1.82 | 0.22 | 0.33 | - | - | - | 0.02 | 0.06 |
| cpu | pile25 | drop | 174 | **2.48** | 3.37 | 0.01 | 1.02 | 0.71 | - | 0.36 | 0.31 | - | - | - | 0.02 | 0.04 |
| gpu | pile25 | drop | 174 | **3.11** | 5.53 | 0.02 | - | - | 2.29 | 0.38 | 0.31 | - | - | - | 0.03 | 0.08 |
| cpu | pile25 | settled | 174 | **3.22** | 3.38 | 0.01 | 1.67 | 0.71 | - | 0.30 | 0.45 | - | - | - | 0.02 | 0.04 |
| gpu | pile25 | settled | 174 | **2.59** | 2.96 | 0.01 | - | - | 1.78 | 0.29 | 0.42 | - | - | - | 0.02 | 0.06 |
| cpu | pile100 | drop | 637 | **3.71** | 4.98 | 0.02 | 1.24 | 1.18 | - | 0.80 | 0.40 | - | - | - | 0.03 | 0.04 |
| gpu | pile100 | drop | 637 | **2.97** | 3.38 | 0.02 | - | - | 1.76 | 0.73 | 0.36 | - | - | - | 0.03 | 0.07 |
| cpu | pile100 | settled | 637 | **6.55** | 6.76 | 0.02 | 3.46 | 1.16 | - | 0.86 | 0.97 | - | - | - | 0.03 | 0.05 |
| gpu | pile100 | settled | 637 | **3.79** | 4.13 | 0.02 | - | - | 1.98 | 0.78 | 0.91 | - | - | - | 0.03 | 0.07 |
| cpu | pile2000 | drop | 12512 | **52.18** | 55.27 | 0.13 | 18.55 | 13.58 | - | 13.86 | 5.52 | - | - | - | 0.35 | 0.17 |
| gpu | pile2000 | drop | 12512 | **33.15** | 35.86 | 0.18 | - | - | 9.73 | 15.90 | 6.06 | - | - | - | 0.49 | 0.78 |
| cpu | pile2000 | settled | 12512 | **188.90** | 196.91 | 0.26 | 104.31 | 15.16 | - | 31.50 | 35.00 | - | - | - | 0.65 | 1.99 |
| gpu | pile2000 | settled | 12512 | **79.86** | 84.60 | 0.14 | - | - | 13.83 | 30.71 | 34.39 | - | - | - | 0.19 | 0.58 |

Reads:
- **Zero impact-zone fallbacks** in every run (all scenes, both backends, drop and settled).
- CPU small scenes: constraints are a flat ~0.55 ms (16 passes; 35 points cost the same as 330), the CCD broad
  phase a flat ~0.2-0.3 ms, contacts >= 0.25 ms: pure dispatch cost.
- CPU big settled piles: contacts dominate (pile1000: 51.8 of 93.6 ms = 55%), then CCD narrow+impulses (19%) and
  CCD broad (17%). With the GPU backend, PBD drops to 9.8 ms and **CCD is 33 of 43.5 ms (76%)**: CCD is the next
  wall for big scenes whichever way PBD is accelerated.
- Settling multiplies contact and CCD work (pile1000 contacts 10.5 -> 51.8 ms; CCD narrow 3.2 -> 17.8 ms, with
  CCD impulses + re-detection running every substep: 3 `_ccd_detect` calls/substep and ~35 hits/substep).
- newtons cradle: `_bounce` = 1.2 ms/frame (Python O(B^2) loop with `_outline_gap` per pair: 47 calls/frame, 0.9 ms),
  and 10 ropes x 16 passes = 320 `Rope.solve_constraints` calls/frame (1.1 ms in cProfile). The 124-point cradle costs
  more physics (4.6-6.1 ms) than the 1262-point pile200 drop (5.8 ms).

### F3. A/B repeats: CPU vs GPU crossover (interleaved, 3 runs each, settled states)

| scene (settled) | pts | CPU ms (3 runs) | GPU ms (3 runs) | GPU / CPU (median) |
|---|---|---|---|---|
| blank | 8 | 0.55 / 0.52 / 0.52 | 3.08 / 1.82 / 1.75 | **3.47** |
| looping | 35 | 1.24 / 1.23 / 1.26 | 2.66 / 2.32 / 2.07 | **1.88** |
| random walls | 81 | 2.43 / 2.47 / 2.49 | 3.82 / 3.46 / 2.59 | **1.40** |
| diamonds | 872 | 6.82 / 6.38 / 6.19 | 5.87 / 6.12 / 5.05 | **0.92** |
| pile50 | 330 | 4.24 / 4.29 / 4.25 | 3.96 / 4.82 / 5.03 | **1.13** |
| pile100 | 637 | 6.81 / 7.33 / 6.68 | 3.86 / 4.60 / 6.39 | **0.68** |
| pile200 | 1262 | 11.76 / 11.81 / 11.69 | 7.08 / 6.11 / 6.18 | **0.53** |
| pile500 | 3137 | 32.76 / 33.44 / 32.92 | 17.21 / 17.24 / 18.09 | **0.52** |

CPU repeats within ~2%; GPU varies up to +-30% run to run (p95 often 1.5-2x the mean: GPU clock ramping and/or
another process on the GPU). Crossover is ~600-900 points in contact-heavy scenes. `MLX_METAL_FAST_SYNCH=1`:
blank 2.72 -> 2.71 ms GPU PBD, pile200 5.63 -> 5.31 ms (within noise), so not a fix for the floor.

Where the GPU floor comes from (line profile, blank, 8 points): `mx.eval(x, npairs)` = 445 us per substep (85% of
`pbd_substep`); building the lazy graph (`_pairs`, 7 kernel calls, uploads via `_i32`/`_f32`) is ~50 us. The 445 us
is the whole dependent chain run at the sync: hash-grid pair build (3 kernels + cumsum) + 16 solver kernels + the
round trip itself. Topic 04 splits it: a bare MLX kernel + `mx.eval` round trip is ~150-190 us at normal QoS, ~3-8 us
per extra dependent kernel, and the pair build is the biggest GPU part (04 F1/F2: 13 bodies 499 us per synced
substep, matching this). So the floor is (pair build + chain latency + sync) x 4 substeps, not arithmetic. The split
design (PBD on GPU, CCD on CPU every substep) forces at least 4 round trips per frame; 04's fused single-dispatch PBD
roughly halves the per-substep latency for small scenes but, by 04's own measurement, still does not beat the CPU on
the small presets.

### F4. Top functions (cProfile, physics only, settled)

cProfile inflates small-call-heavy code ~25-35%; use it for ranking, the profiler-section tables for ms.

**cpu / default settled (78 pts)**: cProfile total 2.875 ms/frame (profiler-inflated); 3033 calls/frame (1186 to C functions); self time in C functions 0.459 ms, in Python functions 2.416 ms (numpy operators like `a + b` count as Python self time).

| # | by cumulative | calls/frame | cum ms | by self | calls/frame | self ms |
|---|---|---|---|---|---|---|
| 1 | `solver.py:313(_pbd_substep)` | 4.0 | 2.32 | `solver.py:450(_solve_contacts)` | 16.0 | 0.822 |
| 2 | `solver.py:450(_solve_contacts)` | 16.0 | 1.149 | `solver.py:152(_solve_body_constraints)` | 16.0 | 0.404 |
| 3 | `solver.py:400(_cell_pairs)` | 8.0 | 0.558 | `solver.py:388(_cells)` | 16.0 | 0.185 |
| 4 | `solver.py:627(_resolve_ccd)` | 4.0 | 0.521 | `<method 'sort' of 'numpy.ndarray' objects>` | 8.2 | 0.127 |
| 5 | `solver.py:152(_solve_body_constraints)` | 16.0 | 0.499 | `<method 'reduce' of 'numpy.ufunc' objects>` | 176.0 | 0.127 |
| 6 | `solver.py:440(_candidates)` | 4.0 | 0.4 | `entities.py:262(solve_constraints)` | 48.0 | 0.121 |
| 7 | `solver.py:539(_ccd_pairs)` | 4.0 | 0.305 | `solver.py:313(_pbd_substep)` | 4.0 | 0.092 |
| 8 | `solver.py:388(_cells)` | 16.0 | 0.254 | `solver.py:422(_filter_pairs)` | 8.0 | 0.083 |
| 9 | `entities.py:262(solve_constraints)` | 48.0 | 0.173 | `solver.py:400(_cell_pairs)` | 8.0 | 0.08 |
| 10 | `fromnumeric.py:2342(sum)` | 92.0 | 0.168 | `solver.py:557(_ccd_detect)` | 4.0 | 0.073 |
| 11 | `solver.py:557(_ccd_detect)` | 4.0 | 0.15 | `solver.py:525(_ccd_boxes)` | 8.0 | 0.062 |
| 12 | `fromnumeric.py:941(sort)` | 8.2 | 0.135 | `solver.py:483(<genexpr>)` | 80.0 | 0.05 |
| 13 | `<method 'sort' of 'numpy.ndarray' objects>` | 8.2 | 0.127 | `fromnumeric.py:66(_wrapreduction)` | 92.0 | 0.047 |
| 14 | `<method 'reduce' of 'numpy.ufunc' objects>` | 176.0 | 0.127 | `fromnumeric.py:2342(sum)` | 92.0 | 0.036 |
| 15 | `fromnumeric.py:48(_wrapfunc)` | 132.0 | 0.126 | `fromnumeric.py:48(_wrapfunc)` | 132.0 | 0.035 |
| 16 | `fromnumeric.py:66(_wrapreduction)` | 92.0 | 0.126 | `solver.py:440(_candidates)` | 4.0 | 0.033 |
| 17 | `solver.py:422(_filter_pairs)` | 8.0 | 0.104 | `_methods.py:115(_mean)` | 16.0 | 0.026 |
| 18 | `<method 'mean' of 'numpy.ndarray' objects>` | 16.0 | 0.066 | `<method 'repeat' of 'numpy.ndarray' objects>` | 48.0 | 0.025 |
| 19 | `solver.py:525(_ccd_boxes)` | 8.0 | 0.062 | `entities.py:250(end_pos)` | 96.0 | 0.023 |
| 20 | `_methods.py:115(_mean)` | 16.0 | 0.061 | `<method 'copy' of 'numpy.ndarray' objects>` | 108.2 | 0.022 |

**cpu / newtons cradle settled (124 pts)**: cProfile total 5.935 ms/frame (profiler-inflated); 8358 calls/frame (3638 to C functions); self time in C functions 1.124 ms, in Python functions 4.811 ms (numpy operators like `a + b` count as Python self time).

| # | by cumulative | calls/frame | cum ms | by self | calls/frame | self ms |
|---|---|---|---|---|---|---|
| 1 | `solver.py:313(_pbd_substep)` | 8.0 | 3.219 | `entities.py:262(solve_constraints)` | 320.0 | 0.805 |
| 2 | `solver.py:261(_bounce)` | 8.0 | 1.595 | `solver.py:24(_outline_gap)` | 47.0 | 0.58 |
| 3 | `entities.py:262(solve_constraints)` | 320.0 | 1.077 | `solver.py:261(_bounce)` | 8.0 | 0.443 |
| 4 | `solver.py:400(_cell_pairs)` | 16.0 | 1.066 | `solver.py:152(_solve_body_constraints)` | 32.0 | 0.419 |
| 5 | `solver.py:627(_resolve_ccd)` | 8.0 | 1.043 | `<method 'reduce' of 'numpy.ufunc' objects>` | 605.1 | 0.365 |
| 6 | `solver.py:24(_outline_gap)` | 47.0 | 0.894 | `solver.py:388(_cells)` | 32.0 | 0.347 |
| 7 | `solver.py:440(_candidates)` | 8.0 | 0.724 | `solver.py:450(_solve_contacts)` | 32.0 | 0.336 |
| 8 | `solver.py:450(_solve_contacts)` | 32.0 | 0.614 | `<method 'sort' of 'numpy.ndarray' objects>` | 17.6 | 0.203 |
| 9 | `solver.py:152(_solve_body_constraints)` | 32.0 | 0.613 | `solver.py:313(_pbd_substep)` | 8.0 | 0.175 |
| 10 | `solver.py:539(_ccd_pairs)` | 8.0 | 0.604 | `solver.py:400(_cell_pairs)` | 16.0 | 0.161 |
| 11 | `solver.py:388(_cells)` | 32.0 | 0.487 | `solver.py:557(_ccd_detect)` | 8.0 | 0.142 |
| 12 | `fromnumeric.py:2342(sum)` | 222.1 | 0.383 | `solver.py:422(_filter_pairs)` | 16.0 | 0.137 |

**cpu / pile1000 settled (6262 pts)**: cProfile total 96.136 ms/frame (profiler-inflated); 5150 calls/frame (2579 to C functions); self time in C functions 24.455 ms, in Python functions 71.681 ms (numpy operators like `a + b` count as Python self time).

| # | by cumulative | calls/frame | cum ms | by self | calls/frame | self ms |
|---|---|---|---|---|---|---|
| 1 | `solver.py:313(_pbd_substep)` | 4.0 | 61.058 | `solver.py:450(_solve_contacts)` | 16.0 | 29.686 |
| 2 | `solver.py:450(_solve_contacts)` | 16.0 | 41.267 | `<method 'reduce' of 'numpy.ufunc' objects>` | 292.8 | 10.728 |
| 3 | `solver.py:627(_resolve_ccd)` | 4.0 | 34.213 | `solver.py:557(_ccd_detect)` | 12.0 | 9.592 |
| 4 | `solver.py:400(_cell_pairs)` | 16.0 | 17.306 | `<method 'sort' of 'numpy.ndarray' objects>` | 16.2 | 8.696 |
| 5 | `solver.py:557(_ccd_detect)` | 12.0 | 14.721 | `solver.py:152(_solve_body_constraints)` | 16.0 | 7.43 |
| 6 | `solver.py:539(_ccd_pairs)` | 4.0 | 13.253 | `solver.py:422(_filter_pairs)` | 12.0 | 6.561 |
| 7 | `solver.py:440(_candidates)` | 4.0 | 10.95 | `solver.py:483(<genexpr>)` | 80.0 | 2.854 |
| 8 | `<method 'reduce' of 'numpy.ufunc' objects>` | 292.8 | 10.728 | `solver.py:525(_ccd_boxes)` | 17.7 | 2.659 |
| 9 | `fromnumeric.py:941(sort)` | 16.2 | 8.876 | `solver.py:400(_cell_pairs)` | 16.0 | 2.325 |
| 10 | `<method 'sort' of 'numpy.ndarray' objects>` | 16.2 | 8.696 | `solver.py:388(_cells)` | 20.2 | 2.082 |
| 11 | `fromnumeric.py:2342(sum)` | 166.7 | 8.451 | `<method 'repeat' of 'numpy.ndarray' objects>` | 60.6 | 1.981 |
| 12 | `fromnumeric.py:66(_wrapreduction)` | 166.7 | 8.356 | `solver.py:688(_ccd_impulses)` | 4.0 | 1.959 |
| 13 | `solver.py:152(_solve_body_constraints)` | 16.0 | 7.883 | `solver.py:313(_pbd_substep)` | 4.0 | 0.948 |
| 14 | `solver.py:422(_filter_pairs)` | 12.0 | 6.908 | `solver.py:677(touches)` | 8.0 | 0.88 |
| 15 | `fromnumeric.py:48(_wrapfunc)` | 190.4 | 4.299 | `solver.py:482(<genexpr>)` | 128.0 | 0.841 |
| 16 | `solver.py:672(_ccd_redetect)` | 4.0 | 3.698 | `solver.py:224(_step)` | 1.0 | 0.828 |
| 17 | `solver.py:388(_cells)` | 20.2 | 3.003 | `<method 'nonzero' of 'numpy.ndarray' objects>` | 36.0 | 0.769 |
| 18 | `solver.py:483(<genexpr>)` | 80.0 | 2.854 | `<method 'searchsorted' of 'numpy.ndarray' objects>` | 20.2 | 0.738 |
| 19 | `solver.py:525(_ccd_boxes)` | 17.7 | 2.659 | `solver.py:440(_candidates)` | 4.0 | 0.702 |
| 20 | `fromnumeric.py:437(repeat)` | 60.6 | 2.022 | `solver.py:535(_ccd_cells)` | 8.0 | 0.58 |

**gpu / pile1000 settled (6262 pts)**: cProfile total 45.273 ms/frame (profiler-inflated); 3343 calls/frame (1807 to C functions); self time in C functions 12.96 ms, in Python functions 32.314 ms (numpy operators like `a + b` count as Python self time).

| # | by cumulative | calls/frame | cum ms | by self | calls/frame | self ms |
|---|---|---|---|---|---|---|
| 1 | `solver.py:627(_resolve_ccd)` | 4.0 | 33.72 | `gpu_backend.py:363(pbd_substep)` | 4.0 | 10.943 |
| 2 | `solver.py:557(_ccd_detect)` | 12.0 | 14.849 | `solver.py:557(_ccd_detect)` | 12.0 | 9.663 |
| 3 | `solver.py:539(_ccd_pairs)` | 4.0 | 12.863 | `<method 'sort' of 'numpy.ndarray' objects>` | 13.4 | 6.985 |
| 4 | `solver.py:400(_cell_pairs)` | 12.0 | 11.542 | `<method 'reduce' of 'numpy.ufunc' objects>` | 148.2 | 3.593 |
| 5 | `gpu_backend.py:363(pbd_substep)` | 4.0 | 11.362 | `solver.py:525(_ccd_boxes)` | 18.2 | 2.616 |
| 6 | `fromnumeric.py:941(sort)` | 13.4 | 7.151 | `solver.py:422(_filter_pairs)` | 8.0 | 2.158 |
| 7 | `<method 'sort' of 'numpy.ndarray' objects>` | 13.4 | 6.985 | `solver.py:688(_ccd_impulses)` | 4.0 | 1.957 |
| 8 | `solver.py:672(_ccd_redetect)` | 4.0 | 3.65 | `solver.py:400(_cell_pairs)` | 12.0 | 1.406 |
| 9 | `<method 'reduce' of 'numpy.ufunc' objects>` | 148.2 | 3.593 | `solver.py:388(_cells)` | 13.4 | 1.043 |
| 10 | `solver.py:525(_ccd_boxes)` | 18.2 | 2.616 | `solver.py:677(touches)` | 8.0 | 0.925 |
| 11 | `solver.py:422(_filter_pairs)` | 8.0 | 2.32 | `<method 'repeat' of 'numpy.ndarray' objects>` | 40.2 | 0.87 |
| 12 | `solver.py:688(_ccd_impulses)` | 4.0 | 2.007 | `<method 'searchsorted' of 'numpy.ndarray' objects>` | 13.4 | 0.375 |

GPU pile1000 (top self, ms/frame): `pbd_substep` 10.9 (mostly the synced eval), `_ccd_detect` 9.7, `ndarray.sort`
7.0, `ufunc.reduce` 3.6, `_ccd_boxes` 2.6, `_filter_pairs` 2.2, `_ccd_impulses` 2.0 (Python loop, 35 hits/substep),
`_cell_pairs` 1.4, `_cells` 1.0, `touches` 0.9.

### F5. Line-level hotspots (line_profiler, us/frame, CPU)

pile1000 settled (6262 points; total 96.7 ms/frame under the line profiler):

| where | line | us/frame | share |
|---|---|---|---|
| `_pbd_substep` | `reach = self._solve_contacts(candidates, reach)` | 41,900 | 43% of frame |
| `_solve_contacts` | `p, a, b = P[:, pts], P[:, ia], P[:, ib]` (3 random gathers over all candidates) | 8,300 | 20% of contacts |
| `_solve_contacts` | `len2 = ...np.sum(e * e, axis=0)` / `t = np.clip(np.sum((p - a) * e, axis=0) / len2, 0, 1)` | 4,700 + 4,700 | 23% |
| `_solve_contacts` | `p, c, d, e = (x[:, near] for ...)` | 3,200 | 8% |
| `_pbd_substep` | `candidates = self._candidates()` | 10,800 | 11% of frame |
| `_cell_pairs` | dedupe `np.sort(i * n_edges + e)` (multi-cell points: CCD swept boxes) | 5,900 | 34% of `_cell_pairs` |
| `_cell_pairs` | `np.sort(edge_keys * n_edges + edge_owner)` | 3,700 | 21% |
| `_resolve_ccd` | `pairs = self._ccd_pairs(D, cells)` | 13,300 | 14% of frame |
| `_resolve_ccd` | `hits = self._ccd_detect(D, pairs)` | 11,500 | 12% of frame |
| `_ccd_detect` | swept-box `overlap = np.all((p_lo[:, pts] <= e_hi[:, eids]) & ...)` pre-reject | 7,500 | **50% of `_ccd_detect`** |
| `_resolve_ccd` | `_ccd_redetect` + `_ccd_impulses` (Python loop, 1443 hits per 10 frames) | 3,800 + 2,400 | 6% |
| `_solve_body_constraints` | spring gather `d = P[:, j] - P[:, i]` + 2 `bincount`s | 1,800 + 2,400 | 4% |

default settled (78 points, 2.3 ms): no line is above 10% of the frame; the cost is spread over ~200 numpy calls
per substep. Biggest: `_solve_contacts` 1.45 ms/frame cumulative over 16 calls (90 us per call for 141 candidates),
`_solve_body_constraints` 0.6 ms (37 us per call, of which `pts.mean` and the rigid-group loop ~40%), 3 ropes x 16 =
48 `Rope.solve_constraints` calls (0.3 ms, ~6 us each: `end_pos` copies, `np.hypot(*d)`, `float()`), `_cell_pairs`
0.07 ms per call.

### F6. Drawing hotspot and a validated prototype (`entities.Body.draw_filled` / `draw_border`)

Whole-frame cProfile, pile500: `draw_filled` 15.2 ms cumulative for 497 calls/frame, of which `draw_border` 12.5 ms;
inside it `np.roll` x2 (4.2 ms: Python-level function, 2.4 us per call even for 6 elements), `zeros_like`,
`normalize_axis_tuple`, `np.where`, `.tolist()`; the 6258 `pygame.draw.line` calls themselves are 1.5 ms.

Prototype (`/private/tmp/claude-501/research-09/drawproto.py`): round all of `World.P * zoom` once, then per body
build the loop and the border shifts with plain Python ints and call the same pygame primitives in the same order.
Result on pile500 (505 bodies): **World.draw(filled) 10.3 ms -> 3.0 ms (3.4x), byte-identical surface**. Expected
frame gains: pile500 -8.9 ms, pile200 ~-3.8 ms, diamonds ~-2.8 ms (12.3 -> ~9.5 ms/frame).

### F7. Python/dispatch overhead vs numpy kernel time

Three independent views agree:
- **Intercept of the scaling fit.** CPU frame time = ~1.9 ms (drop) to ~2.4 ms (settled) at ~80 points, nearly
  all of it fixed per-call cost; slope 4 us/point/frame (drop) to 15 us/point/frame (settled pile). So the fixed
  part is ~95% of the default demo, ~20% of pile200 settled, <3% of pile1000.
- **Call counts.** default: 3,033 Python + C function calls/frame (1,186 to numpy C functions) plus several thousand
  operator-level ufunc calls. Measured per-call cost on this M4 (P-core, `micro.py`): ufunc on a tiny array 0.18-0.3 us,
  fancy gather 0.5 us, `np.sum` (Python wrapper) 1.0 us, `np.clip` 0.9 us, `np.roll` 2.4 us.
- **cProfile self-time split** (numpy operators count as Python self time, so this overstates "Python"):
  C-function self time is 16% (default), 25% (pile200), 25% (pile1000) of profiled time.

Per-element throughput (`micro.py`, us per call):

| op | n=8 | 100 | 1k | 10k | 100k (us/call) |
|---|---|---|---|---|---|
| add (2,n) | 0.181 | 0.217 | 0.473 | 4.263 | 101.835 |
| fancy gather a[:, idx] | 0.5 | 0.891 | 4.349 | 42.656 | 482.296 |
| hypot | 0.292 | 0.436 | 1.839 | 15.463 | 151.115 |
| np.sum(axis=0) | 1.016 | 1.126 | 1.705 | 9.548 | 151.46 |
| bincount weights | 0.212 | 0.306 | 1.358 | 13.623 | 159.823 |
| sort int64 | 0.335 | 1.116 | 12.839 | 156.763 | 1857.565 |
| clip | 0.905 | 0.942 | 1.195 | 3.575 | 26.808 |
| boolean mask w[w<.5] | 0.456 | 0.616 | 2.331 | 18.74 | 273.944 |
| np.roll (2,n) axis=1 | 2.366 | 2.424 | 2.629 | 4.727 | 92.46 |

So per pair-evaluation the contact pass costs ~70-90 ns (pile200: 6.28 ms for 68k pair-evals/frame; pile1000:
51.8 ms for 625k), about 20-30 numpy passes over the pair arrays. A fused compiled kernel (numba/C/Metal) doing the
same arithmetic in one pass would be ~2-5 ns per pair: the 10-20x headroom topic 05 is after.

### F8. Work funnel: how loose the broad phases are (`counts.py`, per substep)

| scene | pts | contact candidates | reach set (dist < 2r) | touching (dist < r) | CCD broad pairs | `_ccd_detect` calls | CCD hits |
|---|---|---|---|---|---|---|---|
| default settled | 78 | 141 | 27 | 6 | 139 | 1.0 | 0.0 |
| diamonds settled | 872 | 3701 | 329 | 63 | 3372 | 1.93 | 0.7 |
| pile200 settled | 1262 | 10818 | 2062 | 619 | 7389 | 1.0 | 0.0 |
| pile1000 settled | 6262 | 100648 | 18572 | 6228 | 66262 | 3.0 | 35.1 |
| pile1000 drop | 6262 | 18067 | 40 | 0 | 12216 | 1.0 | 0.0 |
| pile2000 settled | 12512 | 165936 | 35631 | 13840 | 116582 | 3.0 | 42.2 |

Exact box tests on the same states (`micro4.py`):

| scene | contact candidates inside edge's padded AABB | CCD broad pairs whose swept boxes overlap |
|---|---|---|
| diamonds | 551 of 3,641 (15%) | 176 of 3,398 (5%) |
| pile200 | 2,726 of 10,888 (25%) | 610 of 7,547 (8%) |
| pile1000 | 23,984 of 100,757 (24%) | 4,094 of 66,079 (6%) |

The 24 px grid cells are coarse relative to the 6 px contact pad and the ~1 px per-substep motion, so most pairs
are generated, gathered, filtered and sorted only to be thrown away. Filtering after generation does not pay
(the AABB filter alone costs 2.5 ms at pile1000 because it gathers the same arrays); the win must come from not
generating them (tighter broad phase: topic 03). Cheap knob measured (`cellsweep.py`, settled): `cell_size` 12-16
instead of 24 gives pile1000 94.2 -> 81.5 ms (-13%), pile200 11.9 -> 10.8 (-9%), GPU pile1000 -> 39.7 (-8%),
diamonds 6.6 -> 6.2, but the default demo gets slower (2.31 -> 2.44 at 16, 2.92 at 8). The guarantee does not depend
on the cell size (edges and swept boxes are registered in every cell they touch).

### F10. Scaling exponents (log-log slope of ms/frame vs points, piles with >= 1000 points)

| backend / state | total | contacts (CPU) or GPU pbd | constraints | CCD broad | CCD narrow+impulses |
|---|---|---|---|---|---|
| cpu drop | 0.97 | 1.00 | 0.89 | 1.02 | 0.97 |
| cpu settled | **1.23** | 1.25 | 0.92 | 1.25 | **1.35** |
| gpu drop | 0.75 | 0.42 (GPU) | - | 1.06 | 0.98 |
| gpu settled | 0.93 | 0.37 (GPU) | - | 1.22 | 1.34 |

All per-pair costs are ~constant; the superlinear settled exponents come from pile physics, not algorithms: deeper
piles compress bodies, so contact candidates per point rise from 8.6 (pile200) to 16 (pile1000) and 13 (pile2000),
and CCD needs impulses + re-detection every substep (3 `_ccd_detect` calls per substep vs 1). On the GPU, PBD is
nearly flat (N^0.4) up to 12.5k points, so CPU CCD (N^1.2-1.35) sets the GPU backend's scaling. Fitted CPU model:
~2 ms + 4 us/point (falling bodies) to ~2.4 ms + 15 us/point (settled pile) per frame.

### F9. Allocation churn (tracemalloc, page faults, GC)

| backend | scene (settled) | pts | state arrays (P, PREV, V, W) | transient peak / frame (tracemalloc) | net growth / frame | minor faults / frame | gc gen0 / frame |
|---|---|---|---|---|---|---|---|
| cpu | default | 78 | 4.3 KB | 90 KB | 0.04 KB | 0.2 | 0.0 |
| cpu | newtons cradle | 124 | 6.8 KB | 76 KB | 0.07 KB | 0.4 | 0.0 |
| cpu | diamonds | 872 | 47.7 KB | 580 KB | 0.25 KB | 2.4 | 0.0 |
| gpu | diamonds | 872 | 47.7 KB | 376 KB | 0.2 KB | 0.5 | 0.02 |
| cpu | pile200 | 1262 | 69.0 KB | 1741 KB | 0.06 KB | 6.9 | 0.0 |
| cpu | pile1000 | 6262 | 342.5 KB | 16024 KB | 0.47 KB | 68.0 | 0.0 |
| gpu | pile1000 | 6262 | 342.5 KB | 4685 KB | 0.47 KB | 17.8 | 0.0 |
| gpu | blank | 8 | 0.4 KB | 39 KB | 0.04 KB | 0.3 | 0.0 |

- No leaks (net growth < 0.5 KB/frame), no GC pressure (gen0 collections ~0 per frame: numpy arrays are not
  GC-tracked containers).
- Transient peak scales with pairs: 16 MB/frame at pile1000 CPU (47x the state arrays), 4.7 MB with GPU PBD.
- On macOS, temporaries above ~1 MB are much more expensive (`micro2.py`, (2, n) float64 add):

  | n | array | `a + b` | `np.add(a, b, out=buf)` | ratio |
  |---|---|---|---|---|
  | 50k | 0.8 MB | 20.6 us | 20.6 us | 1.0 |
  | 100k | 1.6 MB | 110 us | 41.7 us | 2.6 |
  | 400k | 6.4 MB | 541 us | 170 us | 3.2 |

  (large allocations get fresh zero-filled pages each time). The contact pass at pile1000 works on ~100k-pair arrays
  in its first iteration, right in that regime. But chunking the real `_solve_contacts` into 12.5k-50k pair blocks
  (`micro3.py`) only gained 10-14% (5.15 -> 4.42 ms per first pass), because the random gathers dominate. Worth doing
  only as part of a rewrite, not on its own.

### F11. Validated prototype: CCD broad phase that applies the swept-box test before dedupe (B3)

`/private/tmp/claude-501/research-09/b3proto.py` (monkeypatch of `World._ccd_pairs` in scratch, no repo edit): run
the same cell join as `_cell_pairs`, then immediately keep only pairs whose float64 swept boxes overlap (or whose
edge collapses mid-substep, the same exception `_ccd_detect` keeps), then dedupe-sort and `_filter_pairs` the ~6%
that remain. `_ccd_detect`'s own pre-reject then runs on the small set. **Final velocities after every substep's CCD
are bit-identical to the stock code** (CPU backend; default 240 substeps, diamonds 120, pile200 80, pile1000 20).
Interleaved timing, median of 3 runs, physics ms/frame:

| scene (settled) | backend | stock | B3 prototype | change |
|---|---|---|---|---|
| diamonds | cpu | 6.18 | 6.03 | -2% |
| diamonds | gpu | 4.79 | 5.51 | +15% |
| pile200 | cpu | 11.62 | 10.87 | -7% |
| pile200 | gpu | 8.51 | 5.14 | -40% |
| pile500 | cpu | 33.37 | 30.21 | -9% |
| pile500 | gpu | 17.74 | 13.23 | -25% |
| pile1000 | cpu | 94.92 | 83.32 | -12% |
| pile1000 | gpu | 45.60 | 33.15 | -27% |

GPU-backend frames gain most because CCD is most of what is left there (the diamonds GPU +15% is GPU-side noise:
its CPU run moved -2%). The remaining CCD broad cost is the cell join itself (`_cells` + sort + searchsorted over all
point-cell and edge-cell entries); a sweep-and-prune or a compiled join would cut that too (topic 03 / 05), and 04's
GPU-built candidate set removes it from the CPU entirely.

## Applicability to this codebase (where the measured time goes, and what would change)

| hotspot (measured) | file / function | change | who covers it |
|---|---|---|---|
| GPU backend floor ~2 ms; loses below ~600-900 pts; silently CPU with ropes | `entities.EnvConfig.backend = "gpu"`, `solver._make_backend`, `gpu_backend.GpuBackend.pbd_substep` | `backend = "auto"`: CPU unless the world has >= ~800 points (re-checked in `_rebuild`, hysteresis); later, ropes on GPU | this file (B1), 04 |
| body drawing 20-25 us/body | `entities.Body.draw_filled`, `entities.draw_border`, `solver.World.draw` | numpy-free per-body prep (validated 3.4x, pixel-identical); further: one batched pass in `World.draw` | this file (B2) |
| contact pass ~80 ns/pair, 55% of big-pile CPU frames | `World._solve_contacts`, `_candidates`, `_cell_pairs`, `_filter_pairs` | fused compiled kernel (numba) for narrow phase + Jacobi accumulate; tighter broad phase | 05, 03 |
| CCD = 76% of frame once PBD is on GPU; pre-reject = 50% of `_ccd_detect`; 94% of broad pairs useless | `World._resolve_ccd`, `_ccd_pairs`, `_ccd_detect`, `_ccd_boxes`, `_ccd_impulses` | swept-AABB broad phase that emits only overlapping pairs (sort-and-sweep or fine grid), compiled float64 narrow phase | 03, 05, 02 |
| fixed per-frame dispatch ~2 ms (16 passes x dozens of numpy calls) | `_solve_body_constraints` (rigid group loop, `pts.mean`), `Rope.solve_constraints` (320 calls/frame in the cradle), `_bounce` | batch ropes into the spring/Jacobi pass or compile; compile `_bounce` pair loop | 05, 07 |
| sidebar redraw ~0.75 ms/frame | `hud_layer.HudLayer.draw`, `editor.draw_card`, `gravity_dial.draw_card`, `perf_hud.draw_card` | cache each card's surface, redraw only when its inputs change (the compute card at its own refresh rate) | this file (B4) |
| grid debug view 0.37-0.85 ms | `World.draw_grid` (Python loop over occupied cells) | vectorize the fills or draw the grid in the composite shader | minor |
| `ctx.finish()` 1-1.8 ms | `main_opengl.MEASURE_GPU` | measurement-only sync; leaving it on costs CPU/GPU overlap. Could sample it every Nth frame | minor |

## Expected gains and risks

| change | scene | measured now | expected | risk to zero-overlap guarantee |
|---|---|---|---|---|
| auto backend (B1) | blank / looping / random walls | 1.8-3.1 / 2.3 / 3.5 ms physics (GPU) | 0.52 / 1.24 / 2.47 ms | none: CPU path is the reference path; CCD unchanged |
| auto backend (B1) | pile200+ | GPU already | unchanged | none |
| numpy-free draw (B2) | diamonds / pile200 / pile500 frame | 12.3 / 15-18 / 29-38 ms | ~9.5 / ~11-14 / ~20-29 ms | none (drawing only); pixel-identical by test |
| card caching (B4) | every scene | 0.75 ms ui | ~0.15-0.3 ms | none |
| `cell_size` 16 (B3, knob only) | piles | pile1000 94 ms, pile200 11.9 | -9 to -13% | none in principle (pair superset unchanged by construction); default demo +5% slower |
| B3 CCD swept-box test before dedupe (prototype measured, F11) | pile1000 GPU / CPU | 45.6 / 94.9 ms | 33.2 / 83.3 ms | none if pairs passing the swept test are exactly today's post-pre-reject set (bit-identical in tests) |
| tighter broad phases + compiled contact/CCD (03/05) | pile1000 CPU / GPU | 93.6 / 43.5 ms | contacts and CCD 3-10x lower (not measured here) | CCD narrow phase must stay float64 and the broad phase must provably never drop a pair whose swept boxes overlap; test with the brute-force checker |

Risk notes: B1's only behaviour change is float32 vs float64 PBD on mid-size scenes (the GPU path proposes motion;
CCD still certifies it), so switching more scenes to CPU makes results *more* deterministic, not less. B2/B4 touch
presentation only. Any broad-phase change must be validated against the brute-force overlap checker
(`/private/tmp/claude-501/hud-layout-agent/suite/brute.py`, `ccd_test.py`) and the impact-zone count (0 today in
every measured run, a useful regression signal).

## Effort

- B1 auto backend: small (one function in `solver.py` + `EnvConfig` doc; ~30 lines incl. hysteresis and a test).
- B2 draw rewrite: small-medium (rewrite `draw_filled`/`draw_border` hot path, keep the generic path for custom
  bodies; pixel-diff test exists in `drawproto.py` and `hud-layout-agent/suite/test_render.py`).
- B3 broad-phase experiments: medium (cell-size knob trivial; swept-AABB emission medium; see 03).
- B4 card caching: medium (each card needs a dirty signal: hover state, editor state, profiler refresh tick).

## Cross-links to other topics

- 01 (XPBD/substeps): its cost model `frame = S * (F + I * c)` matches the data: per substep F (candidates + first
  full contact pass + CCD) is ~75% of a settled pile frame, so **more iterations are cheap and more substeps are
  expensive**: at pile1000 one extra contact iteration costs ~1.1 ms/substep (reach set) vs ~5.2 ms for the first pass
  plus ~2.7 ms candidates plus ~8 ms CCD per extra substep. On the GPU backend each substep also adds a ~0.45 ms sync.
- 03 (broad phase): F8 quantifies the slack (6% of CCD pairs and 24% of contact candidates are real) and the cell-size
  sweep; the CCD dedupe sort (`_cell_pairs` multi-cell path) is the single most expensive broad-phase line.
- 04 (Metal/MLX): agrees on both big points. (a) The synced substep (~0.45-0.5 ms for small scenes; 04 measures the
  bare round trip at 150-190 us and the pair build as the largest GPU part) is the floor. (b) 04 independently
  measured that only ~3% of CCD cell pairs survive the swept-box pre-reject (this file: 5-8% at the pair level after
  the body filter) and prototyped building that set on the GPU inside the PBD command buffer (CPU first-detect
  17 -> 2.5 ms at 1000 bodies). B3 below is the CPU-only version of the same cut (useful for CPU-backend scenes and as
  the fallback); the two compose. 04 also warns that background QoS inflates a GPU sync ~4x (733 vs ~186 us), matching
  the QoS caveat here. Earlier drafts of this file put the crossover "near 1200 points"; the interleaved A/B
  (F3) puts it at ~600-900. The per-substep `mx.eval` chain is the GPU floor; kernels themselves are cheap. Fewer
  syncs per frame matter more than faster kernels for every preset.
- 05 (compiled loops): the targets by measured weight are `_solve_contacts` (55% of big CPU frames), `_ccd_detect`
  + pre-reject, `_cell_pairs` sorts, then the Python loops `_ccd_impulses` (2 ms at pile1000), `_bounce` (1.2 ms in the
  cradle), `Rope.solve_constraints` (320 calls/frame in the cradle) and the draw path (B2 shows plain Python already
  gets 3.4x there).
- 07 (rigid stacking): rope-on-rigid generalized mass changes would touch `Rope.solve_constraints`, which is also a
  dispatch hotspot; batching ropes into the Jacobi pass could serve both.
- 08 (parallelism): measured QoS behaviour: background QoS is 10-19x slower than P-core under the current load,
  so any "run on E-cores" design must budget for that; physics and draw are single-threaded today.
- 10 (accuracy): impact-zone fallbacks were 0 in all runs here; use `zones` from `phys.py` as a cheap regression signal.

## Build plan (ranked; each fits one agent in its own git worktree)

Common measurement protocol for all four: P-core runs (`nice -n 10`, not `taskpolicy -b`, see setup), interleaved
baseline/new runs (ABAB, 3 repeats), scripts copied from `/private/tmp/claude-501/research-09/` (`phys.py` for
physics phases, `frame.py` for the whole frame, `scenes09.py` scenes and cached settled states), plus the
brute-force overlap checker (`/private/tmp/claude-501/hud-layout-agent/suite/brute.py`, `ccd_test.py`) and the
`zones` count for anything that touches physics.

### B1 (rank 1). Automatic backend choice: CPU below a point threshold

- Change: `EnvConfig.backend` accepts `"auto"` (make it the default). `solver._make_backend` returns `None` (CPU)
  when the world has fewer than ~800 points or has ropes / solo bodies that make `GpuBackend.pbd_substep` fall back
  anyway; `GpuBackend` otherwise. Re-decided in `World._rebuild` (already called on every add), with hysteresis
  (e.g. switch up at 900, down at 700) so the editor does not flip-flop. Show the active backend on the compute card.
- Success metric: for every preset and pile, physics ms/frame <= min(CPU, GPU) + 5%; blank 1.8-3.1 -> ~0.5 ms,
  looping 2.3 -> ~1.25, random walls 3.5 -> ~2.5; piles >= 200 unchanged.
- Measure: `phys.py` on all presets + pile50/100/200/500/1000, settled and drop, ABAB.
- Expected gain: 1-2.5 ms/frame on the small presets (up to 3.5x physics), none lost elsewhere.
  Risk: none to the guarantee (CCD path unchanged; CPU PBD is the reference). Threshold is machine dependent: keep it
  in `EnvConfig`. Effort: small (hours).

### B2 (rank 2). Numpy-free body drawing (then batched)

- Change: in `entities.Body.draw_filled` / `entities.draw_border` replace the per-body numpy work (`np.round`,
  `np.roll` x2, `np.where`, `zeros_like`, `.tolist()`) with plain-Python int math on a rounded point list; in
  `World.draw` round `self.P * zoom` once for all bodies and hand each stock body its slice (bodies that override
  draw keep the generic path). Prototype already written: `/private/tmp/claude-501/research-09/drawproto.py`.
  Optional second step: also cache the polygon outline lists for FixedBody (they never move) and skip their prep.
- Success metric: byte-identical render (`drawproto.py` comparison and `hud-layout-agent/suite/test_render.py`),
  `draw` section: pile500 11.9 -> <= 4 ms, diamonds 4.0 -> <= 1.5 ms.
- Measure: `frame.py cpu pile500|diamonds|default 120 --hidpi`, draw section, ABAB.
- Expected gain: 3.4x on body drawing (measured on pile500: 10.3 -> 3.0 ms); diamonds frame ~12.3 -> ~9.5 ms.
  Risk: none to physics; visual regressions caught by the byte comparison. Effort: small-medium (a day).

### B3 (rank 3). CCD broad phase that emits only swept-box-overlapping pairs

- Change: today `_ccd_pairs` emits ~16x more pairs than survive the swept-box test, then `_ccd_detect` spends 50% of
  its time on the pre-reject gather. Move the exact swept-AABB overlap test into pair generation (in `_cell_pairs`
  for CCD: compare the point's and edge's float boxes right after the cell join, before the dedupe sort, so the
  expensive `np.sort(i * n_edges + e)` runs on ~6% of the pairs), or replace the CCD grid with a 1-axis
  sort-and-sweep over swept boxes. Keep `_ccd_detect`'s collapsing-edge exception (pairs with a near-zero-length edge
  must be kept). Also try `cell_size` 16 for both broad phases (measured -9 to -13% on piles, +5% on the default demo).
  Coordinate with topic 03 (algorithm choice) and 05 (if the join is compiled).
- Success metric: identical CCD hit sets on recorded substeps (pairs that pass the swept-box test must be exactly
  the same set as today's post-pre-reject set); CCD broad + narrow: pile1000 settled 33.4 -> <= 15 ms/frame (GPU
  backend frame 43.5 -> <= 28 ms); 0 overlaps in the brute-force checker over the pile, diamonds and drag scenes.
- Measure: `phys.py` sections `ccd/broad`, `ccd/narrow+impulses`; `counts.py` for pair counts; brute-force checker.
- Expected gain (measured with the scratch prototype, F11, bit-identical results): physics pile1000 GPU 45.6 -> 33.2
  ms (-27%), pile500 GPU -25%, pile200 GPU -40%; CPU backend -7 to -12%. Further gain needs a cheaper join (03/05) or
  04's GPU-built set. Risk: this is the guarantee's broad phase; a dropped pair = a
  possible tunnel. Mitigation: property test comparing pair sets against a brute-force O(N*E) swept-box test on
  thousands of random substeps. Effort: medium (2-3 days with tests).

### B4 (rank 4). Cache sidebar cards; redraw only on change

- Change: `hud_layer.HudLayer.draw` keeps one surface per card; a card exposes a cheap `state_key()` (hover target,
  editor mode/selection, gravity setting, compute card refresh tick e.g. 10 Hz) and is redrawn only when the key
  changes; the HUD upload (`GLPresenter.upload_hud`) can also be skipped when nothing changed.
- Success metric: `ui` section 0.75 -> <= 0.25 ms/frame on all scenes, identical screenshots
  (`hud-layout-agent/suite/test_render.py` all modes).
- Measure: `frame.py` ui and present/upload/hud sections.
- Expected gain: ~0.5 ms/frame everywhere (8-15% of a light preset's frame). Risk: stale card if a key is missed;
  none to physics. Effort: medium (touches editor/perf_hud/gravity_dial card code).

Not ranked (measured but small or owned elsewhere): fused compiled contact kernel (largest single CPU win for big
scenes, topic 05: ~80 ns -> ~5 ns per pair-eval); `_bounce` compile (1.2 ms in the cradle); `draw_grid`
vectorization (0.4-0.85 ms with the grid view on); sampling `ctx.finish()` every Nth frame; chunking temporaries
under 1 MB (10-14%).

## Sources

- Measurements: this machine, scripts and raw JSONL in `/private/tmp/claude-501/research-09/`.
- Code read: `solver.py`, `gpu_backend.py`, `entities.py`, `main_opengl.py`, `profiler.py`, `hud_layer.py`, `scene.py`,
  `presets_basic.py`, `presets_cradle.py`, `looping.py`.
- NumPy small-call overhead (order of magnitude, cross-check for `micro.py`): [d2l.ai TVM book, "Function Call
  Overhead"](https://tvm.d2l.ai/chapter_cpu_schedules/call_overhead.html) (~0.6 us per small NumPy call) and
  [PyTorch forum, small-matrix overhead thread](https://discuss.pytorch.org/t/pytorch-tensor-slow-on-small-matrix-operations-large-overhead/67882)
  (~0.77 us for a 3x3 NumPy add).
- MLX `MLX_METAL_FAST_SYNCH` ("faster GPU-CPU synchronization"): [MLX Metal API notes](https://mintlify.wiki/ml-explore/mlx/api/metal);
  measured here to make no difference for this workload.
- line_profiler: [pyutils/line_profiler](https://github.com/pyutils/line_profiler); cProfile/tracemalloc: Python 3.13 stdlib docs
  ([profile](https://docs.python.org/3/library/profile.html), [tracemalloc](https://docs.python.org/3/library/tracemalloc.html)).
