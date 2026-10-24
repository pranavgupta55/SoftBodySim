# 10: Accuracy and validation benchmarks

Status: done for cycle 1. The harness is built and has been run on main (`1f34cd5`), on a substep/iteration sweep
and on four contact-solver variants.

- Harness: `/private/tmp/claude-501/research-10/bench/`. It has `run_bench.py` (the suite), `scenes.py`,
  `checker.py` (the independent overlap checker), `compare.py` (the PASS/WARN/FAIL gate) and `states/` (cached
  settled piles for timing).
- Baselines: `/private/tmp/claude-501/research-10/base_acc.json` (accuracy) and `base_perf.json`, `base_perf2.json`,
  `base_perf3.json` (timing).
- Sweeps and variants: `sweep_{8x2,16x1,4x1}.json` and `momfix{,B,C,C125,C200}_acc.json`, in the same folder.

## Summary

- **One command answers "faster without getting worse?"**
  `run_bench.py --code <worktree> --skip-perf --jobs 6 --out new.json` runs 14 accuracy tests:
  - energy (pendulum, Newton's cradle)
  - momentum in five two-body collisions
  - chain stretch at mass ratios 1 and 100
  - rigid and soft stacks, a 50-body pile and a resting soft body (area, sag, shape error)
  - free fall and terminal velocity, and the looping preset
  - CPU determinism and GPU-vs-CPU agreement
  - four CCD stress runs (random dragging, thin walls, diamonds, a 200-body pile)

  An independent brute-force overlap checker runs on **every substep of every test** (29,160 substeps per run). Then
  `compare.py base.json new.json` prints FAIL / WARN / better per metric and whether the end states are
  bit-identical. Runtime is 3 minutes with 6 jobs at background QoS while 9 other agents load the machine. The timing
  part (`--perf-only`) takes 10-25 s at `nice -n 10`.
- **Main's baseline:**
  - **0 overlaps** in all tests.
  - CPU runs are bit-reproducible.
  - The rigid stack sits dead still (0.0 px/s).
  - The pendulum period is within 0.08% of the exact elliptic-integral value, and free fall is within 0.44% of the
    analytic curve.
  - The cradle passes 99% of the swing height to the far ball.
  - **The rigid impact-zone fallback does fire under aggressive dragging:** 47 of 2,400 substeps in `ccd_demo`. 02's
    "never fired" holds only for gentler drags.
- **Three accuracy defects on main, all measured:**
  1. **The Jacobi contact pass doesn't conserve momentum.**
     - A soft body hitting a rigid square in zero gravity loses **31% of the pair's linear momentum**.
     - An off-centre rigid-rigid hit changes angular momentum by 16%.
     - CCD and impact zones change neither (exactly 0). The error is the per-point `count` averaging in
       `_solve_contacts`.
     - **Fix found and run through the full suite (variant C):** give each contact one shared scale,
       `1.5 / max(count over its 3 points)`.
       - Momentum error drops to 1e-13, angular to 0.006.
       - pile50 settles 7x lower in KE.
       - The worst soft body in the pile goes from 0.60x to 0.80x of its area.
       - Zone fallbacks drop from 47 to 0, overlaps stay at 0, and CPU speed is unchanged (within ±4%).
       - The soft box stack's jitter goes from 18.5 to 0.30 px/s at ω = 1.5, but it is **not monotonic in ω**
         (1.0: 21, 1.25: 6.7, 2.0: 36), so that part is fragile (F9).
       - The momentum and pile gains hold for every ω from 1.0 to 2.0.
  2. **Chains stretch in their rigid links, so the rope metric lies.** Ropes are projected last, so their strain
     always reads about 1e-16. With a 100:1 weight, the chain instead bounces like a bungee: 6.7 px mean and 13.9 px
     peak drop on 400 px, at 19 px/s RMS after 6 s. The stretch is in the shape-matched boxes, which matches research
     07's 8.9%.
  3. **PBD energy loss dominates anything that swings: 3.7% per period** on the rope pendulum.
     - It is set by substeps alone. 4x4 and 4x1 lose the same amount, and each doubling of substeps halves it
       (8x2: 1.9%, 16x1: 0.97%).
     - The same holds for chain stretch, rigid sink and free-fall error. This is the "Small Steps" result, reproduced
       on this sim.
- **Timing rules** (agreeing with research 04, 08 and 09):
  - At background QoS (`taskpolicy -b`) with the other agents on the E-cores, physics ran **~35x slower** (pendulum
    0.64 → 23 ms/frame, cradle 6.0 → 221).
  - Accuracy metrics don't depend on speed, so run them at background QoS. Time at `nice -n 10`, A/B interleaved, on
    cached settled states.
  - CPU medians repeat within ±3%. **GPU timings swing by up to ±40% between runs** (pile200 GPU 9.7 vs 5.9 ms), so a
    GPU speed claim needs at least 3 interleaved runs.
- **GPU backend:**
  - It agrees with the CPU to float32 rounding: 2e-5 px after 1 frame, 4e-3 px after 60 frames, then chaotic
    divergence in a tumbling pile.
  - It is **not reproducible**: float atomics make two GPU runs end 71-142 px apart after 4 s.
  - So GPU builds can be gated only on aggregate metrics, not hashes, until research 04's fixed-point atomics are in.

## Findings

### F1. How established engines and papers validate

| source | what they test | how they judge it | what we take |
|---|---|---|---|
| Box2D v3 CI determinism test ([Catto, "Determinism", 2024](https://box2d.org/posts/2024/08/determinism/); [box2d/test](https://github.com/erincatto/box2d/tree/main/test)) | The "falling hinges" scene (fast motion, collisions, sleeping, joint limits) on MSVC/Clang/GCC, x64/ARM, any thread count | Two numbers must match exactly: **steps until everything sleeps** and **a hash of all transforms** at that point. CI fails on any change; the author re-baselines on purpose | `state_hash` / `cpu_demo_hash` and `cpu_demo_repeat_identical`. Box2D's pitfalls apply here too: FMA contraction, libm `atan2`, fast-math, atomics. Research 05 found numpy's batched `matmul` uses FMA via Accelerate, which changes shape matching at 1e-9 px |
| Box2D solver comparison ([Catto, "Solver2D", 2024](https://box2d.org/posts/2024/02/solver2d/)) | Large pyramid, large mass ratios, large coordinates, initial overlap, ragdolls and joints, at reduced iterations (4 + 2 relax) | Mostly visual (jitter, tangential oscillation, floating, contacts pulling back); cost by counting loops | Stress scenes at **fixed, low** iteration counts, plus mass-ratio variants (`chain.m100`, `collisions.rr_mass`) |
| Box2D v2.4 sleep tolerances (`b2_linearSleepTolerance` 0.01 m/s, `b2_angularSleepTolerance` 2 deg/s, `b2_timeToSleep` 0.5 s, in `b2_common.h`) | n/a | A body under these limits for 0.5 s counts as at rest | Our "settled" threshold. At 500 px/s² ≈ 9.8 m/s², 1 m ≈ 51 px and 0.01 m/s ≈ 0.5 px/s. We use RMS < 1 px/s over 0.5 s (`settle_time_s`) |
| Jolt PerformanceTest ([docs](https://raw.githubusercontent.com/jrouwe/JoltPhysics/master/Docs/PerformanceTest.md)) | Ragdoll piles (3,680 bodies) and a 1,240-box pyramid | Record the state every step and validate a later run against it (`-rs` / `-vs`), plus a final hash | The GPU-vs-CPU per-frame divergence curve (`gpu_vs_cpu_maxdiff_px_f*`) |
| Macklin et al. 2019, "Small Steps in Physics Simulation" ([PDF](https://mmacklin.com/smallsteps.pdf)) | 20-particle hanging chain (stretch of the bottom particle), the same chain with a 1:100,000 end mass, residual vs iteration and substep count, constraint force vs the exact 190 N, cloth energy over time, FEM cantilever sag, a 1:1,000 rigid chain, a heavy box on a capsule stack | Quantitative: stretch, residual (log scale), force error, energy curve. Substeps beat iterations by about 2 orders of magnitude at equal cost | `chain.m1/m100` (end-to-end drop), pendulum energy and period, and the `--set substeps=S --set iterations=I` sweep (F8) |
| XPBD, Macklin, Müller, Chentanez 2016 ([PDF](https://matthias-research.github.io/pages/publications/XPBD.pdf)) | Stiffness vs iteration count and time step; constraint force estimates | Same material response whatever the iteration count | The sweep: once compliance is in (research 01 and 06), `soft_rest.sag_ratio` should stop moving with S and I |
| XPBD rigid-body survey ([arXiv 2311.09327](https://arxiv.org/pdf/2311.09327)) | Newton's cradle energy over time vs Havok and Unity | The energy curve, and where the momentum goes at the first impact | `cradle.transfer_height_ratio`, `leak_height_ratio`, `energy_retained_*` |
| Tonge et al. 2012, mass splitting ([doi](https://dx.doi.org/10.1145/2185520.2185601)); Macklin et al. 2014, "Unified Particle Physics" (averaging + SOR) | Parallel (Jacobi) constraint solves | Consistent averaging that keeps the solve physical | Why per-point averaging breaks momentum (F3). The SOR factor ω = 1.5 is in the range those papers recommend |

### F2. The overlap checker is independent of the solver and runs every substep, and it does catch failures

- **What it checks.** It wraps `World._resolve_ccd` and checks the substep's end state `PREV + V h`, which is exactly
  what `_step` copies into `P`. The idea comes from `debug-agent/merged/brute.py`, but this version is vectorized:
  - **Edge crossings:** a body-AABB prefilter (bounding boxes), then every edge pair of each overlapping body pair,
    then a strict segment-crossing test.
  - **Self-crossings:** self-collide bodies also get every non-adjacent pair of their own edges.
  - **Containment:** "body inside body" is tracked as a relation, and only a **change** counts. The default scene
    legitimately starts with a soft body inside the hollow box, which the old `brute()` would flag.
- **Independence.** It uses only topology from the World (`edges`, `offsets`, `fixed` / `self_collide`), never its
  broad phase, CCD or GPU pair lists.
- **`min_gap_px`.** It also reports the smallest point-to-other-body-edge distance: a margin, not just yes/no. Main
  keeps it ≥ 1.5 px in calm scenes and ≥ 0.0116 px (just above `ccd_gap` = 0.01) under dragging.
- **Negative controls** (`scratch/negctl.py`, default scene with random dragging, 300 frames):
  - CCD off: **379 of 1,200** substeps bad.
  - CCD with 1 pass and no impact zones: **641 of 1,200** bad, including 4 containment changes.
  - Impact zones only (`ccd_passes = 0`): 0 bad, as the guarantee says.
- **Cost.** The perf runs don't attach the checker.

### F3. The Jacobi contact pass doesn't conserve momentum

Setup: zero gravity, no damping, friction 0.5 (an internal force, so it must conserve both momenta), 1.5 s:

| pair | dP / (M v) | dL / (M v · 40 px) | KE kept |
|---|---|---|---|
| rigid vs rigid head-on (equal) | 5e-14 | 0 | 0.7% |
| rigid vs rigid, 3x mass | 0 | 0 | 26% |
| rigid vs rigid, off-centre | 2e-13 | **0.16** | 1% |
| soft 12-gon vs rigid square | **0.31** | 2e-17 | 8% |
| soft vs soft | 1e-13 | 0 | 19% |

- **Where it goes.** Instrumenting the phases (`scratch/mom.py`) shows the whole change happens in `_pbd_substep`.
  CCD impulses and impact zones change momentum by exactly 0.
- **Why.** `_solve_contacts` divides each contact's three corrections (the point and both edge ends) by *different*
  per-point `count`s, so the pushes on the two bodies are unequal and not opposite. Symmetric scenes cancel by luck.
- **Any per-contact scalar factor conserves both momenta**, since the three shares of a contact already sum to zero
  force and torque. Four variants of the factor went through the whole suite (`variant_momfix*`):

| variant: per-contact scale | sr dP | rr_offset dL | pile50 KE end | pile50 worst soft area | soft stack jitter (px/s) | ccd_demo zones | ccd_pile200 zones | pile50 min gap | pile200 / pile500 CPU ms |
|---|---|---|---|---|---|---|---|---|---|
| main: per point 1/count | 0.31 | 0.16 | 11,984 | 0.60 | 18.5 | 47 | 0 | 1.91 | 11.8 / 32.6 |
| A: 1/max(count) | 2e-13 | 0.004 | 1,235 | 0.87 | 21.0 | 0 | 15 | 0.02 | 13.4 / 35.4 (+13%, +8%: more CCD work) |
| B: 1/count[point] | 3e-14 | 0.004 | 292 | 0.76 | 18.8 | 18 | 0 | 0.25 | 11.7 / 32.7 (=) |
| **C: 1.5/max(count)** | **4e-14** | **0.006** | **1,640** | **0.80** | **0.30** | **0** | **0** | 0.51 | 11.8 / 34.0 (=, +4%) |

- **Why C.** A is the "safe" mass-splitting-style divisor, but it makes contacts softer: points sink deeper, CCD takes
  over (narrow phase +40-55%) and impact zones fire in the pile. Over-relaxing by 1.5 (as in Macklin 2014) restores
  the contact strength.
- **The soft stack's pumping oscillation goes away with C (ω = 1.5)**, but F9 shows the stack's response is not
  monotonic in ω. Treat that one as "contact-driven and sensitive", not as solved.
- **The one WARN on every variant** is a smaller `min_gap_px`, which is still legal (CCD keeps it > `ccd_gap`). The
  GPU contact kernel (`_CONTACT_ACC` / `_CONTACT_APPLY` in `gpu_backend.py`) needs the same change.

### F4. Ropes read ~0 strain; the stretch is in the rigid links

- **Why the rope metric reads zero.** Ropes are projected after `_solve_body_constraints` in each iteration, and
  nothing moves rope ends afterwards. So `Rope.length() - rest_length` is ~1e-16 at the end of every step, and it is
  **useless as a metric**.
- **Where the stretch shows.** With a 100:1 weight (`scenes.hanging_chain(100)`), the hook-to-weight distance swings
  between +1.4 and +10.4 px with a ~1.5 s period and never settles (damping 0). Box shape error peaks at 0.76 px on
  10 px boxes (`scratch/chain.py`).
- **What the suite measures instead:** the weight's drop (the chain starts straight and exactly taut, so any drop is
  stretch) and the box shape error.
- **Fix.** Research 07 fixes this by moving the whole rigid body with its generalized inverse mass at rope ends (box
  stretch 8.9% → 0%). `chain.m100.*` is the metric for that build.
- **Substeps also help a lot:** drop 6.7 → 3.5 → 1.8 px at 8x2 and 16x1, but 32.8 px at 4x1.

### F5. Energy and the free-fall check

- **Rope pendulum** (two ropes in a V, 16 px box, 300 px, 30 deg, no damping or friction). The period is −0.08% off
  the exact physical-pendulum value (complete elliptic integral via the AGM), so the dynamics are right. But 3.7% of
  the swing energy is lost per period. Energy never rises above the start value (`energy_gain_max_frac` = 0).
- **Cradle preset** (8 substeps, restitution 1):
  - 99.0% of the height goes to the far ball, and no motion leaks to the middle balls.
  - 97% of the energy is left after 8 s.
  - `min_gap` = exactly 3.0 px (= `contact_radius`): the restitution pass keeps the balls apart.
  - The survey above shows Havok and Unity losing energy in sudden drops at each impact; here the loss is smooth.
- **Free fall** (damping 0.2): the position after 1 s is +0.44% vs the analytic continuous solution, which is the
  O(h) offset of symplectic Euler; it halves with each doubling of substeps. Velocity is −0.05%.
- **Terminal velocity:** the cap is reached at 2.567 s (analytic 2.554 s) and never exceeded. The max speed reads
  999.17 = 1000 × (1 − 0.2 h) because damping runs after the clamp.
- **Looping preset:** 32 wraps in 8 s, 0 overlaps, and speed stays under the cap.

### F6. Stacking: rigid is solid; soft stacks pump; round soft bodies are a bad test

- **Rigid column of 6 squares:**
  - Settles in 0.97 s, then shows 0.0 px/s jitter and 0 creep.
  - Shape error is 0.027 px.
  - It sinks 0.24 px past ideal, so each contact ends ~0.04 px inside `contact_radius`.
  - Research 07 sees the same for towers of up to 20 boxes and a 21-box pyramid.
- **Soft column of 4 flat 8-point soft boxes:**
  - It stays standing, but on main it never settles. It pumps in bursts every ~4 s: RMS speed cycles 2 → 34 → 2 px/s,
    averaging 18.5 px/s over the last 6 s.
  - Variant C (F3) removes this (0.30 px/s, settled at 6 s).
- **Round soft bodies were dropped from this test.** A first version stacked 12-gons. Those toppled every time, even
  2-high and perfectly centred, under all four contact variants and all S x I settings. A ball balanced on a ball is
  an unstable equilibrium, so that test measured chaos, not the solver.
- **A single resting demo soft body** (stiffness 0.1, skip 2) sags to 0.80x its rest height and keeps 0.97x its area.
  Sag is set by S x I (F8), as research 01 found.
- **pile50 is still settling at 6 s** (RMS 15 px/s, KE 12k), so its jitter means "how fast it settles". The worst
  soft body is at 0.60x its area.

### F7. Determinism and GPU agreement

- **CPU:** two runs of the default scene with scripted random dragging give identical hashes.
- **GPU** (`pile50`; no ropes, so MLX really runs the PBD part) is **not** reproducible. Float atomics accumulate in
  scheduling order, so two runs differ from the first frame and end 71-142 px apart by 4 s.
- **GPU vs CPU:** 1.9e-5 px after 1 frame, 2.9e-4 after 10, 3.7e-3 after 60, then chaos (83 px at 4 s).
- **The right GPU check:**
  - (a) the early divergence, which should stay at float32 rounding level (< ~1e-3 px at frame 10)
  - (b) the accuracy metrics within tolerance
  - (c) zero overlaps.
- Research 04's fixed-point int32 atomics would make GPU runs hashable (+3-5% GPU time).

### F8. Substeps vs iterations on this sim (`--set substeps=S --set iterations=I`)

| metric | 4x4 (main) | 8x2 | 16x1 | 4x1 |
|---|---|---|---|---|
| pendulum energy loss / period (%) | 3.68 | 1.90 | 0.97 | 3.68 |
| pendulum period error (%) | −0.083 | −0.043 | −0.021 | −0.081 |
| chain m100 weight drop, end mean (px) | 6.7 | 3.5 | 1.8 | 32.8 |
| chain m100 weight drop, peak (px) | 13.9 | 7.0 | 3.5 | 56.5 |
| chain m100 weight RMS speed at end (px/s) | 19.2 | 13.5 | 9.2 | 64.6 |
| free fall 1 s position error (%) | 0.44 | 0.22 | 0.11 | 0.44 |
| rigid stack sink past ideal (px) | 0.24 | 0.12 | 0.06 | 0.97 |
| rigid stack settle time (s) | 0.97 | 0.88 | 0.75 | 2.33 |
| soft rest sag ratio (1 = none) | 0.80 | 0.86 | 0.92 | 0.59 |
| soft rest area ratio | 0.971 | 0.986 | 0.994 | 0.836 |
| soft box stack jitter (px/s) | 18.5 | 11.7 | 9.1 (toppled) | 9.7 |
| pile50 worst soft area | 0.60 | 0.80 | 0.86 | 0.19 |
| soft-vs-rigid momentum error dP | 0.31 | 0.35 | 0.31 | 0.19 |
| off-centre angular momentum error dL | 0.16 | 0.17 | 0.20 | 0.20 |
| ccd_demo impact-zone fallbacks | 47 | 0 | 0 | 17 |
| overlaps (all tests) | 0 | 0 | 0 | 0 |

What the sweep shows:

- **Energy, period, free fall:** only the number of substeps matters (4x4 = 4x1).
- **Chain stretch and rigid sink** halve with each doubling of substeps.
- **Momentum errors don't move:** they are structural (F3), not a convergence issue.
- Research 01 measured 8x2 at ~1.5x the cost of 4x4, and 16x1 at ~2x.
- **4x1** (0.56x cost) would fail the gate badly: the worst soft area in pile50 drops to 0.19, and the chain stretches
  33 px.

### F9. SOR factor ω for the symmetric contact scale (full suite at each ω)

| ω (scale = ω / max count) | sr dP | rr_offset dL | pile50 KE end | pile50 jitter | pile50 worst soft area | soft stack jitter | rigid sink (px) | pile50 min gap | zones demo / pile200 | overlaps |
|---|---|---|---|---|---|---|---|---|---|---|
| main (per point) | 0.31 | 0.16 | 11,984 | 15.1 | 0.60 | 18.5 | 0.24 | 1.91 | 47 / 0 | 0 |
| 1.0 (A) | 2e-13 | 0.004 | 1,235 | 8.5 | 0.87 | 21.0 | 0.24 | 0.02 | 0 / 15 | 0 |
| 1.25 | 2e-13 | 0.005 | **353** | **2.4** | 0.83 | 6.7 | 0.19 | 0.02 | 0 / 0 | 0 |
| 1.5 (C) | 4e-14 | 0.006 | 1,640 | 7.6 | 0.80 | **0.30** | 0.16 | 0.51 | 0 / 0 | 0 |
| 2.0 | 5e-14 | 0.005 | 1,261 | 10.0 | 0.80 | 36 | 0.12 | 1.10 | 0 / 0 | 0 |

What's robust and what isn't:

- **Robust for every ω:** exact momentum, a pile that settles 7-34x lower in KE, the worst soft area +0.2, impact
  zones gone from the whipped demo (all ω ≥ 1.25), and 0 overlaps.
- **Not robust:** the soft-stack pumping. It is chaotic in ω, so the build agent should check it over several stack
  heights and sizes before claiming it.
- **Tradeoff in ω:** larger ω keeps more contact gap and less sink, while ω = 1.25 settles the pile fastest.
- **Pick:** ω ∈ [1.25, 1.5] on the pile/stack evidence. Run `soft_rest.rms_speed_end` (WARN for every ω: 0.26 →
  0.6-0.8 px/s at 5 s) longer to tell slower settling from a small persistent wobble.

### F10. Cross-links and contradictions with the other research files

- **02 (CCD):**
  - 02 found the impact-zone fallback "never fired (0 of ~16,000 substeps)" on its drag scripts.
  - The `ccd_demo` test here (default scene at the real 1354x982 world, random targets up to 30 px outside the walls,
    regrab every 45 frames) fires it **47 times in 2,400 substeps**. 4x1 fires it 17 times. 8x2, 16x1 and every
    symmetric-contact variant fire it 0 times.
  - So zones are rare but real under aggressive dragging. Use `ccd_demo.zone_fallbacks` as the worst-case scene for
    any CCD change.
- **03 (broad phase):** a Verlet / fat-box pair cache can only be wrong by dropping pairs. `ccd_pile200`,
  `ccd_diamonds` and `ccd_walls` (thin crossing segments) plus the per-substep checker are the gate. Also run the
  negative control (`scratch/negctl.py`) with the cache to confirm the checker would see a dropped pair.
- **04 (MLX kernels):** independently confirms F7. Its fixed-point atomics are build item 3. Its warning that
  background QoS inflates GPU sync 4x more than numpy matches the ±40% GPU timing noise seen here.
- **05 (compiled loops) and 08 (parallelism):**
  - Both claim bit-identical results. `compare.py` prints `bit-identical end states: k/2` (the CPU demo with dragging,
    and pile50), which tests that directly.
  - 05's exception (FMA in the batched `matmul` of shape matching) will show as 0/2 with ~1e-9 px differences. Those
    grow chaotically, so judge such builds on the aggregate metrics.
- **06 (soft-body models):**
  - Mass splitting (Tonge 2012), cited in 06's F4, is the theory behind F3.
  - `soft_rest.sag_ratio`, `stacks.soft.area_ratio_*` and `pile50.soft_area_ratio_*` are 06's targets.
  - F8 reproduces 01/06's S x I dependence of sag: 0.59 at 4x1 to 0.92 at 16x1.
- **07 (rigid stacking):**
  - Uses F3's shared-scale idea in its rigid-aware contacts (A2).
  - **Correction for 07:** its "soft stack creep 0.5 → 23.6 px/s" regression came from this suite's first soft stack
    of round 12-gons. That stack topples under every solver variant (a ball on a ball is unstable), so its creep and
    top_dx were chaos, not signal. The suite now uses flat 8-point soft boxes (`stacks.soft.*`). With those, the
    shared scale at ω = 1.0 gives jitter 21.0 vs main's 18.5 px/s and keeps the stack standing.
- **09 (profiling):** the timing tables here (`nice -n 10`, settled cached states) agree with 09's P-core numbers
  within ~3% (pile200 CPU 11.8 vs 11.79 ms, pile500 32.4-34.1 vs 33.63 ms).

## Benchmark spec

Scenes are in `scenes.py`; dt = 1/60; CPU backend unless noted; the overlap checker runs on every substep everywhere.
Every test reports `overlap_substeps` (must be 0), `checked_substeps`, `min_gap_px` and `zone_fallbacks`.

| test | scene | sim time | metrics (compare.py rule) |
|---|---|---|---|
| pendulum | 16 px rigid box on two ropes in a V, 300 px, 30 deg, no damping or friction | 15 s | `energy_loss_frac_5s/end`, `energy_loss_per_period_pct`, `energy_gain_max_frac` (lower); `amplitude_ratio_last` (higher); `period_err_pct` vs the elliptic exact value (abs) |
| cradle | `newtons cradle` preset (its own env) | 8 s | `transfer_height_ratio`, `energy_retained_5s/end` (higher); `leak_height_ratio` (lower) |
| collisions | 5 two-body impacts in zero g: rr, rr_offset, rr_mass, sr, ss | 1.5 s each | `dP_rel`, `dL_rel` (lower); `ke_kept` (info) |
| chain | Chain of 5 ropes + 4 boxes from a hook to a diamond weight of 1x / 100x point mass, damping 0 | 6 s each | `weight_drop_end/peak_px` (abs); `box_shape_err_px`, `weight_speed_rms_end`, `rope_strain_*` (lower) |
| stacks | 6 rigid 30 px squares; 4 soft 8-point 30 px squares; on fixed ground | 12 s | `settle_time_s`; `jitter_rms/max` and `creep_px_s` (last 6 s); `top_dx_px`, `sink_excess_px` (abs); `standing` (must not go True → False); `shape_err_px` / `area_ratio_*` |
| soft_rest | one demo soft 12-gon on the ground | 5 s | `sag_ratio`, `area_ratio` (closer to 1); `rms_speed_end` |
| pile50 | 50 mixed bodies dropped into a box | 6 s | jitter (last 2 s), `ke_end`; `soft_area_ratio_min/mean`; `rigid_shape_err_end/peak`; `state_hash` |
| free_fall | one rigid square, no walls, default env | 4 s | `fall_1s_err_pct`, `v_1s_err_pct` (abs); `t_reach_cap_s` vs analytic; `vmax_overshoot` (≤ 0) |
| looping | `looping` preset | 8 s | `over_cap` (≤ 0), `wraps` |
| ccd_demo / ccd_walls / ccd_diamonds / ccd_pile200 | default scene with random dragging from 1 s; random walls with dragging; diamonds; pile200 with dragging | 10 / 5 / 5 / 4 s | `overlap_substeps` (must be 0), `min_gap_px`, `zone_fallbacks` |
| determinism | default scene with dragging, twice on CPU; pile50 twice on GPU and once on CPU | 4 s | `cpu_demo_repeat_identical` (must stay True); `cpu_demo_hash`; `gpu_*` (info) |
| perf (`nice -n 10`, never at background QoS) | demo (CPU); diamonds; pile200 + pile500 from cached settled states; CPU and GPU | 20 warm + 90 timed frames | `ms_frame` (median), `ms_frame_mean`, `ms_frame_p95`, ms per profiler section |

Usage for the cycle-2 build agents:

```
B=/private/tmp/claude-501/research-10/bench
PY=/Users/pranavgupta/PycharmProjects/SoftBodySim/.venv/bin/python3
# accuracy: background QoS is fine (deterministic); ~3 min with 6 jobs while the machine is busy
SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskpolicy -b $PY $B/run_bench.py \
    --code <worktree> --skip-perf --jobs 6 --out <scratch>/acc.json
$PY $B/compare.py /private/tmp/claude-501/research-10/base_acc.json <scratch>/acc.json
# timing: normal priority, A/B interleaved (main, candidate, main, candidate), ~15 s each
OMP_NUM_THREADS=1 nice -n 10 $PY $B/run_bench.py --code <tree> --perf-only --out <scratch>/perf_<tag>_<i>.json
$PY $B/compare.py <scratch>/perf_main_2.json <scratch>/perf_cand_2.json
# other settings, e.g. small steps: add --set substeps=8 --set iterations=2 (EnvConfig override for every test)
# a single test: --only pendulum,collisions
```

The gate:

- **No FAIL.** A FAIL means an overlap, CPU determinism lost, a stack that stood now falling, or a speed above the
  cap.
- **Every WARN needs a one-line reason** in the experiment report, e.g. "min gap 1.9 → 0.5 px: contacts are now
  momentum-conserving and act later; CCD still guarantees > 0.01".
- **Pure refactors** (numba, threads, reordering) should also show `bit-identical end states: 2/2`, or explain why not
  (FMA in shape matching per research 05).
- **Perf claims** compare medians from the same session, interleaved; GPU claims need 3 or more runs.
- The scripts write only to `--out`, which is made absolute before the chdir into `--code`; never point it into the
  repo.

## Baseline: main `1f34cd5`, 2026-10-24, numpy 2.5.4, Python 3.13.7

Accuracy (`base_acc.json`, CPU, background QoS; 29,160 checked substeps; **0 overlaps**):

| test | values |
|---|---|
| pendulum | energy loss 3.8% at 5 s, 11.0% at 15 s, **3.68% / period**; gain 0; amplitude 0.970 after 15 s; period 5.0165 s vs exact 5.0206 (**−0.083%**); rope strain 4e-16 |
| cradle | transfer **0.990**, leak 0.000, energy kept 0.982 (5 s) / **0.971** (8 s), min gap 3.0 |
| collisions | rr dP 5e-14; rr_mass 0; ss 1e-13; **sr dP 0.314**; **rr_offset dL 0.162**; KE kept rr 0.7%, rr_mass 26%, sr 8%, ss 19% |
| chain | m1: drop 0.28 px (peak 0.49), box shape err 0.03 px. **m100: drop 6.7 px (peak 13.9), box shape err 0.31 px, weight 19.2 px/s RMS at 6 s** |
| stacks | rigid: settles 0.97 s, jitter 0.0, creep 0.0, sink +0.24 px, shape err 0.027 px. **soft boxes: never settle, jitter 18.5 px/s RMS (max 97.5)**, sink +2.8 px, area min 0.972, standing |
| soft_rest | sag 0.804, area 0.971, 0.26 px/s at 5 s |
| pile50 | jitter 15.1 px/s RMS (still settling), KE 11,984, soft area min **0.599** / mean 0.929, rigid shape err 0.056 (peak 0.36) px, min gap 1.91 |
| free_fall | 1 s position +0.44%, velocity −0.05%, reaches cap at 2.567 s (analytic 2.554), overshoot −0.83 (never above) |
| looping | max speed 999.17, 32 wraps |
| ccd_demo | 0 overlaps, min gap 0.021 px, **47 zone fallbacks** / 2,400 substeps |
| ccd_walls | 0 overlaps, min gap 2.07 |
| ccd_diamonds | 0 overlaps, min gap 1.53 |
| ccd_pile200 | 0 overlaps, min gap 0.0116, 0 zone fallbacks |
| determinism | CPU repeat identical: **True**. GPU repeat identical: **False** (71-142 px apart at 4 s). GPU vs CPU: 1.9e-5 / 2.9e-4 / 3.7e-3 px at frames 1 / 10 / 60 |

Timing (`base_perf*.json`): `nice -n 10`, median ms/frame over 3 runs, load average ~25 from the other agents.
Agrees with research 09 within ~3%.

| scene | pts | CPU ms (3 runs) | CPU phases (median, ms) | GPU ms (3 runs) |
|---|---|---|---|---|
| demo (drop) | 78 | 2.35 / 2.31 / 2.29 | contacts 1.20, constraints 0.57, ccd broad 0.32, ccd narrow 0.13 | n/a (ropes: CPU PBD) |
| diamonds (drop) | 872 | 5.46 / 5.44 / 5.34 | contacts 2.37, constraints 0.98, ccd broad 1.13, ccd narrow 0.80 | 6.50 / 6.52 / 4.11 |
| pile200 settled | 1262 | 11.96 / 11.74 / 11.82 | contacts 6.48, constraints 1.78, ccd broad 1.81, ccd narrow 1.71 | 9.70 / 5.93 / 5.90 |
| pile500 settled | 3137 | 34.07 / 32.06 / 32.40 | contacts 17.6, constraints 3.86, ccd broad 5.64, ccd narrow 5.18 | 20.24 / 17.64 / 17.59 |

## Applicability to this codebase

- **Contact fix (F3):** `World._solve_contacts` in `solver.py`, about 5 lines (see `variant_momfixC/solver.py`).
  Replace `P[k] += acc / count` with a per-contact scale `1.5 / max(count[pts], count[ia], count[ib])` applied to
  `lam` before the bincount. Mirror it in `gpu_backend.py` `_CONTACT_ACC` / `_CONTACT_APPLY`: accumulate `cnt` first,
  then a second pass applies the scaled impulses, or use the previous pass's counts.
- **Chain stretch (F4):** this is research 07's rope-to-rigid generalized-inverse-mass build. `chain.m100` is its
  metric.
- **Energy (F5, F8):** research 01's small-steps schedule (8x2 or 16x1 with per-frame broad phase reuse). The sweep
  table above is its accuracy target; 4x1 must never ship.
- **Determinism (F7):** `GpuBackend` kernels. Research 04's fixed-point atomics make `gpu_pile50_repeat_identical`
  True and let GPU builds be hash-gated.
- **Overlap checker:** `checker.attach(world)` works on any World, including the app's (`main_opengl.main` test hooks)
  and the editor tests.

## Expected gains and risks

- **Contact fix C:**
  - Gains: exact momentum; piles settle 7-34x lower in KE; soft-body area in piles +0.2; zone fallbacks 47 → 0. Soft
    stacks may settle (18.5 → 0.3 px/s at ω = 1.5), but that is ω-sensitive.
  - Cost: none measured on CPU (pile500 +4%, within noise).
  - Risk: ω = 1.5 could overshoot in other configurations (dense piles, 16x1). The gate catches that as jitter or
    `min_gap` WARNs and zone fallbacks. **No risk to the zero-overlap guarantee:** CCD is untouched, and the suite
    measured 0 overlaps across all variants.
- **The suite itself:**
  - Risk: tolerance tuning. Chaotic scenes (pile50, ccd runs) change end states with any change in rounding. So
    their gate uses aggregate metrics (KE, area, jitter, overlaps), never trajectories.
  - The soft stack and pile metrics are settle-speed numbers and can swing ±25% from chaos alone. compare.py uses 25%
    relative tolerances there.

## Effort

- Contact fix C on CPU: about 1 hour plus a suite run. GPU kernel mirror: about half a day.
- Moving the suite into the repo (e.g. `tools/bench/`) with a 60 s smoke profile: about half a day.
- Fixed-point GPU atomics: research 04 estimates 1-2 days, including deterministic rigid sums.

## Build plan

Ranked. Each experiment runs in its own worktree and is judged with the suite above (`compare.py` against
`base_acc.json`, plus interleaved perf).

1. **Momentum-conserving contacts (variant C) on CPU and GPU.**
   - What changes: `solver.py` `_solve_contacts` (per-contact scale ω / max(count), ω = 1.5); `gpu_backend.py`
     `_CONTACT_ACC` / `_CONTACT_APPLY` to match.
   - Success:
     - `collisions.*.dP_rel` < 1e-9 and `dL_rel` < 0.01
     - `stacks.soft.jitter_rms_px_s` < 1
     - no FAIL and 0 overlaps
     - pile200/pile500 CPU ms within +5%
     - GPU `gpu_vs_cpu_maxdiff_px_f10` < 1e-3
   - Pick ω in [1.25, 1.5] (F9). Before claiming the soft-stack result, re-check it over 3-4 stack heights and box
     sizes, because it is chaotic in ω.
   - Expected: the accuracy gains in F3 at no cost. Risk: low. Effort: S (CPU), M (GPU).
2. **Suite into the repo as a gate.**
   - What changes: add the harness under `tools/bench/` with a `--smoke` profile (≤ 60 s at P-core: collisions,
     pendulum, chain, stacks, ccd_demo) and a pytest wrapper that runs the hard rules: 0 overlaps, CPU determinism,
     standing, the speed cap.
   - Success: the gate catches the negative controls (CCD off → FAIL) and passes main.
   - Expected: every later build is checked the same way. Risk: none to physics. Effort: S-M.
3. **Deterministic GPU accumulation.**
   - What changes: research 04's fixed-point int32 atomics in `_CON_ACC` / `_CONTACT_ACC`, plus deterministic rigid
     sums.
   - Success: `gpu_pile50_repeat_identical` True; GPU-vs-CPU early divergence unchanged; GPU perf within +5% (3
     interleaved runs).
   - Expected: GPU builds become hash-gateable, and GPU bugs reproducible. Risk: low (CCD untouched). Effort: M.
4. **Small-steps schedule under the gate**, together with research 01.
   - What changes: 8x2 (or 16x1 with per-frame broad phase reuse) as the default.
   - Success: F8's accuracy gains (energy loss/period ≤ 2%, chain m100 drop ≤ 3.5 px, soft sag ≥ 0.86) with perf
     ≤ main after 01's amortization and the speedups from 05 and 08.
   - Expected: halved energy and stretch errors. Risk: cost +50-100% unless amortized. Effort: M.
