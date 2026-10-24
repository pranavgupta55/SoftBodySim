# 01: XPBD, small steps, substeps vs iterations

Status: done for cycle 1. Scripts, logs and the prototype solver are in `/private/tmp/claude-501/research-01/`.
`xworld.py` there subclasses `World` with switchable XPBD springs, a frame-level contact broad phase and frame-level
CCD. There are no repo edits.

Terms used below:
- **PBD**: position-based dynamics.
- **XPBD**: extended PBD. Each constraint gets a *compliance* alpha (inverse stiffness; units s^2/mass here, with
  mass 1 per point) and a Lagrange multiplier lambda, so stiffness stops depending on the substep and iteration counts.
- **GS**: Gauss-Seidel, where constraints are solved one after another and each sees the previous ones' corrections.
- **Jacobi**: every constraint is computed from the same positions, and the corrections are then summed or averaged.
- **SOR**: over-relaxation, i.e. scaling corrections by omega.
- **CCD**: continuous collision detection.
- **S x I**: substeps per frame x solver iterations per substep. Stock is 4 x 4 at 60 FPS, so h = 1/240 s.
- **frame-level CCD / broad phase**: what this report proposes. Run the PBD substeps without CCD, then run one CCD
  over the frame's start -> end straight-line motion. Contact candidates are built once per frame.

Timing caveat: unless marked "P", numbers are thread CPU time at background QoS on loaded E-cores. That is 6-10x
slower than a P-core, so only ratios within one table are meaningful. "P" rows are short `nice -n 10` runs.

## Summary

1. **Today the material depends on the schedule.** Take a soft dodecagon (the default demo body). Its rest height is
   0.60 of rest at 4x1, 0.84 at 4x4 and 0.91 at 8x4. The 6 springs per point divide each correction, so topology
   changes stiffness too. Any per-world `substeps` override (the cradle uses 8) silently changes every soft body in
   that world.
   - With **XPBD springs solved Gauss-Seidel**, the same body stays at 0.85-0.87 over six schedules. With
     Jacobi + mass splitting it stays at 0.85-0.94 (it drifts only at 4x1).
   - Calibration: demo stiffness 0.1 (skip 2) ≈ alpha 3e-4; 0.05 ≈ alpha 1e-3.
   - XPBD Rayleigh damping beta ≈ 10 reproduces today's bounce and settling, and it is nearly schedule-independent.
2. **Substeps, not iterations, set accuracy. Here they are expensive only because CCD and the contact broad phase run
   inside the substep loop.** Energy loss, chain stretch, rigid shape error and free-fall error all fall with S and
   barely move with I (also research 10 F8: pendulum loss 3.68%/period at 4x4 = 4x1, 0.97% at 16x1).
   - But each substep pays a fixed cost F: contact broad phase, full first contact pass and CCD. A Macklin-style
     16x1 therefore costs 2.0x (demo) to 2.6x (200 bodies) of 4x4 in stock code.
3. **Moving the contact broad phase and CCD out of the substep loop makes small steps cheaper than today's 4x4, and
   keeps zero overlaps.** The zero-overlap guarantee only needs the frame's start -> end straight-line motion to be
   CCD-checked from an intersection-free start. Bridson et al. 2002 sec. 5 is written exactly this way, with internal
   dynamics allowed as "many small explicit steps".
   - Prototype (Jacobi XPBD with mass splitting, per-point swept candidate boxes, one CCD per frame), cost relative
     to stock 4x4:

     | scene | 8x1 | 16x1 |
     |---|---|---|
     | demo | **0.50x** (P: 1.66 -> 0.85 ms) | 0.87x (P: 1.43 ms) |
     | pile200 | 0.61x (P: 15.5 -> 10.7 ms) | 0.82x (P: 14.9 ms) |
     | diamonds | 0.73x | 0.98x |
     | 60-body 900 px/s impact | 0.62x | 0.90x |

   - Overlaps, from the brute-force checker: **0** in every run, including whip drags.
   - Impact-zone fallbacks:
     - 0 in all non-drag scenes;
     - demo whip 81 (stock) -> 0;
     - pile whip at 16x1: 1 -> 14, most likely the substep-coupled grab (F7); this needs a fix before 16x1 ships.
   - CCD-edited points per frame fell 2-8x, because PBD gets the whole frame to resolve contacts before anything is
     checked.
   - Accuracy at 16x1 vs stock 4x4: chain stretch 1.0% -> 0.3%, pendulum energy kept over 4 s 90.5% -> 97.3%,
     rigid-stack shape error 0.10 -> 0.027 px.

## Findings

### F1. Where a frame goes, and the cost model (code + measured)

Each frame, `World._step` (solver.py) runs S substeps. Each substep runs:
- predict;
- the contact broad phase (`_candidates`);
- I iterations of springs + shape matching + ropes, each followed by one Jacobi contact pass;
- the velocity update and terminal-velocity clamp;
- CCD (`_resolve_ccd`: swept broad phase + quadratic narrow phase + impulses).

That gives `frame = S * (F + I * c)`:
- F is the fixed cost per substep: contact broad phase, the first full contact pass and CCD.
- c is the cost per iteration: constraints, plus a contact pass over the cached `reach` set, which is reused while no
  point has moved more than contact_radius / 2.

Measured, stock code, relative to 4x4 (thread CPU, two interleaved repeats agree within ~5%):

| S x I | demo | 200-body pile | CCD share (demo / 200) |
|---|---|---|---|
| 4x4 (stock) | 1.00 | 1.00 | 20% / 32% |
| 4x1 | 0.55 | 0.87 | 31% / 46% |
| 8x2 | 1.45 | 1.53 | 33% / 40% |
| 8x1 | 1.04 | 1.35 | 40% / 45% |
| 16x1 | 1.8-2.1 | 2.6 | 35% / 46% |

- F/c is ~2 on the demo and >5 on piles. At 200 bodies, cutting iterations from 4 to 1 saves only 13%: later
  contact passes are cheap, and the per-substep work is not.
- This matches research 09 (P-core): CCD is 20-35% of CPU physics, and the GPU backend pays one sync per substep.
- It also matches research 03: both broad phases are rebuilt 8+ times per frame while points move < 1 px per substep.

### F2. Stock PBD stiffness depends on S, I and topology; XPBD removes that (measured)

`SoftBody` springs run as one Jacobi pass. Each correction is `stiffness * C`, divided by the point's spring count
(6 for skip_sets=2). So:
- Per pass, a spring loses about stiffness/count of its error.
- Per substep, it loses about 1 - (1 - s/count)^I.
- Gravity adds g h^2 per substep.

Rest sag therefore scales roughly like h^2 / f(I). This is the problem XPBD was written for ([Macklin, Müller,
Chentanez 2016](https://matthias-research.github.io/pages/publications/XPBD.pdf): constraints "become arbitrarily
stiff as the iteration count increases, or as the time step decreases").

Rest height / rest height of one resting soft dodecagon after 3 s (1.0 = rigid):

| body / solver | 4x1 | 2x8 | 4x4 | 8x2 | 16x1 | 8x4 | spread |
|---|---|---|---|---|---|---|---|
| stock, s=0.1, skip 2 (demo preset) | 0.60 | 0.69 | **0.84** | 0.86 | 0.90 | 0.91 | 0.31 |
| stock, s=0.05, skip 2 | 0.51 | 0.59 | **0.66** | 0.85 | 0.88 | 0.86 | 0.37 |
| stock, s=0.1, skip 1 | 0.40 | 0.47 | **0.47** | 0.58 | 0.64 | 0.65 | 0.25 |
| stock, s=0.3, skip 2 | 0.81 | 0.85 | 0.92 | 0.98 | 0.98 | 0.98 | 0.17 |
| XPBD GS, alpha 3e-4 | 0.85 | 0.86 | 0.85 | 0.85 | 0.87 | 0.85 | **0.02** |
| XPBD GS, alpha 1e-3 | 0.73 | 0.72 | 0.73 | 0.67 | 0.66 | 0.67 | 0.08 |
| XPBD GS, alpha 3e-3 | 0.59 | 0.58 | 0.59 | 0.61 | 0.60 | 0.61 | **0.03** |
| XPBD Jacobi, mass splitting, alpha 1e-4 | 0.85 | 0.92 | 0.94 | 0.94 | 0.94 | 0.98 | 0.13 |
| XPBD Jacobi, mass splitting, alpha 1e-3 | 0.66 | 0.69 | 0.66 | 0.64 | 0.63 | 0.64 | 0.06 |
| XPBD Jacobi, mass splitting, alpha 3e-3 | 0.58 | 0.58 | 0.59 | 0.61 | 0.60 | 0.61 | 0.03 |
| XPBD Jacobi, 1/max(count) scale, alpha 3e-4 | 0.60 | 0.81 | 0.75 | 0.64 | 0.59 | 0.75 | 0.22 |

Reads:
- **Calibration for migrating presets:** stock s=0.1 (skip 2) at 4x4 ≈ alpha 3e-4. s=0.05 ≈ alpha 1e-3. Research 06
  reuses this for the editor's strength slider.
- **GS XPBD is schedule-independent within a few percent.** Jacobi XPBD with plain averaging is not. When the
  multiplier only gets 1-2 updates per substep, the averaging divisor acts like extra compliance: a one-iteration
  steady state is off by about count/omega.
- **Mass splitting** ([Tonge et al. 2012](https://dx.doi.org/10.1145/2185520.2185601): n_i w_i in the denominator,
  no averaging) is the right Jacobi form. It matches GS within ~0.05 except at very stiff settings with 1 iteration
  (4x1, alpha 1e-4). Research 06 uses the same form.
- In the numpy prototype GS costs more: 7 colour batches for skip_sets=2 make the constraint phase 2x. Mass-splitting
  Jacobi costs the same as stock. Under numba (research 05) a sequential GS loop is cheaper than either.
- XPBD makes mass matter. Sag scales with m * alpha, as physics says. Stock PBD sag ignores mass. Calibrate per body
  if bodies get non-unit masses.

### F3. Substeps are what buy accuracy (measured, plus research 10)

Stock solver, CPU (rope and rigid scenes have no springs, so XPBD changes nothing in these rows):

| metric | 4x1 | 4x4 (stock) | 8x1 | 8x2 | 16x1 |
|---|---|---|---|---|---|
| 8-box chain hung from a hook, max stretch (last 1 s of 4 s) | 3.6% | 1.0% | 1.1% | 0.6% | **0.3%** |
| rope pendulum, energy kept after 4 s (damping 0) | 90.4% | 90.5% | 94.8% | 94.8% | **97.3%** |
| 6-box rigid stack, max shape error (px) | 0.37 | 0.10 | 0.11 | 0.054 | **0.027** |
| 6-box rigid stack, top-box drift (px) | 2.6 | 0.16 | 0.17 | 0.13 | 0.09 |
| overlaps / impact zones | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

- Energy and drift depend only on S. Stretch and shape error fall with both S and I, but more steeply with S.
- Research 10 F8 has the same picture on its suite:
  - pendulum loss 3.68%/period at both 4x4 and 4x1, 0.97% at 16x1;
  - the 100:1 chain drops 6.7 px at 4x4 and 1.8 px at 16x1;
  - the ccd_demo impact-zone fallbacks (47 at 4x4) go to 0 at 8x2 and 16x1.
- This is Macklin et al. 2019's result: positional error from external forces is O(h^2), so substeps win over
  iterations.
- **8x1 matches or beats 4x4 on every metric.** 16x1 is clearly better.
- **4x1 is not an option.** Research 10: pile50 worst soft area 0.19, chain drop 33 px.

### F4. Damping: what replaces PBD's numerical damping

More substeps remove numerical damping (Macklin 2019 fig. 6), so soft bodies get livelier. Test: a soft dodecagon
dropped 120 px.

| solver / schedule | bounce (px) | internal KE left 1 s after landing |
|---|---|---|
| stock 4x4 | 11.2 | 0.46% |
| stock 16x1 | **34.7** | **22.7%** |
| XPBD (mass splitting, alpha 3e-4) 4x4 | 14.7 | 0.46% |
| XPBD 16x1 | 20.0 | 1.9% |
| XPBD 16x1 + damping beta 3 / 10 / 30 | 17.7 / 14.3 / 7.6 | 1.3% / 0.8% / 0.04% |
| XPBD + beta 10 at 4x4 / 8x1 / 16x1 | 11.5 / 11.7 / 14.3 | 0.2% / 0.5% / 0.8% |

- Raising substeps on stock code makes soft bodies 3x bouncier (stock 16x1).
- XPBD Rayleigh damping (XPBD eq. 26; the gamma = alpha beta / h form also used in Macklin 2019 eq. 8) with beta ≈ 10
  restores today's look. It is roughly schedule-independent and damps only along springs, so free fall and rigid
  rotation are untouched.
- `EnvConfig.damping` (V *= 1 - damping h per substep) is already substep-invariant to first order.
- Research 06 F6 adds the PBD-paper alternative: per-body damping of deviation from rigid motion.

### F5. Frame-level contact broad phase + frame-level CCD (prototype, measured)

**Literature.**
- [Bridson, Fedkiw, Anderson 2002, sec. 5](https://www.cs.ubc.ca/~rbridson/docs/cloth2002.pdf) (the source of this
  CCD):
  - Internal dynamics advance x^n to a candidate x~^{n+1} "using any reasonable algorithm ... one large implicit time
    step ... or many small explicit Runge-Kutta steps".
  - Repulsions and CCD impulses then edit the *average* velocity (x~^{n+1} - x^n)/dt on linear trajectories from x^n.
  - Then x^{n+1} = x^n + dt v^{n+1/2}.
  - If nothing collided, v^{n+1} is the internal dynamics' end velocity.
  - They also halve the collision step on collision and double it after three clean steps.
- [Macklin et al. 2019](https://matthias-research.github.io/pages/publications/smallsteps.pdf) sec. 4.2 and fig. 1:
  "we have performed collision detection once per-frame", with contacts generated within a margin and reused for all
  substeps.
- [Müller et al. 2020](https://diglib.eg.org/handle/10.1111/cgf14105): broad phase once per step, narrow phase per
  substep.

**Prototype** (`xworld.py`; `frame_broad`, `swept_broad`, `frame_ccd`):

1. Frame start: X0 = P, which is intersection-free.
2. Contact candidates come from per-point predicted swept boxes, x .. x + v T + g T^2/2 grown by contact_radius, with
   edge boxes grown by 2r more.
   - They are valid while every point stays inside its box. An O(N) check runs each substep and triggers a rebuild if
     a point leaves.
   - About 1.0-1.7 builds per frame were measured; more under dragging (grab moves points up to 4 px per substep).
   - Same superset argument as research 03's fat-box list. The `reach` cache is kept across substeps.
3. S substeps of PBD with no CCD: predict, I iterations, V = (P - PREV)/h, terminal clamp, damping. **The contact
   normal's side comes from X0**, not from the substep's PREV (the stock code relies on PREV being CCD-valid; X0 is).
4. One CCD over the frame: PREV = X0, V_avg = (P - X0)/dt, `_resolve_ccd(dt)` (unchanged code, impact zones
   included), P = X0 + V dt.
5. Velocity: points the CCD did not touch keep the last substep's velocity. Touched points take the corrected average
   velocity, like stock. Bridson's "advance the midstep velocity" (or V_last + (V_avg' - V_avg)) is the more accurate
   alternative.

**Guarantee argument.** The end state is X0 + dt V, where V comes out of the stock `_resolve_ccd` on the straight
path from the intersection-free X0. That is exactly the stock per-substep argument with h = dt. The intermediate
substep states are only a proposal and may even overlap. Looping wraps stay between frames, as now.

**Measured** (zero overlaps everywhere: brute-force checker every 5th frame on stress scenes, every frame on the demo
and on `smoke.py`). Columns are stock 4x4 / frame-level 8x1 / frame-level 16x1:

| scene (points) | rel. cost | CCD-edited pts / frame | zones |
|---|---|---|---|
| demo (78) | 1 / 0.50 / 0.87 | 0 / 0 / 0 | 0 |
| pile200 (~2400) | 1 / 0.61 / 0.82 | 0.05 / 0.04 / 0.01 | 0 |
| diamonds (872, looping, ~700 px/s) | 1 / 0.73 / 0.98 | 0.25 / 0.11 / 0.10 | 0 |
| 60 bodies thrown down at 900 px/s | 1 / 0.62 / 0.90 | 1.69 / 0.57 / 0.46 | 0 |
| random walls preset, bodies at 800 px/s | 1 / 0.45 / 0.74 | 0 / 0 / 0 | 0 |
| P-core: demo | 1.66 / 0.85 / 1.43 ms | | |
| P-core: pile200 | 15.5 / 10.7 / 14.9 ms | | |

- The first, global-margin version (pad = 2 (vmax dt + g dt^2) + r) was *slower* on diamonds (1.10x / 1.39x). Every
  point falls at ~700 px/s, so a global pad of ~37 px swells the candidate list. Per-point swept boxes fixed it.
- At 4x4, frame-level CCD gave bit-identical results to per-substep CCD on the chain, pendulum and stack scenes.
  Those scenes have no CCD hits, so the two schedules do the same arithmetic.
- Research 02 F7 checked the chord independently: across 1,500 frames of whips, 0-4 chord hits where substeps had
  none. Corner cutting R(1 - cos(θ/2)) stays far below the 3 px skin below ~40 rad/s of spin. 02 notes that longer
  sweeps make each GS ping-pong (02 F4) freeze a whole frame instead of a substep, which makes 02's
  inelastic-projection fallback more valuable.
- Whip-drag results are in F7.

### F6. Ropes, chains, rigid bodies: getting them stiffer

- Ropes are already hard projections (stiffness 1 = XPBD with alpha 0). The stretch is in the shape-matched boxes,
  per research 07 and 10. Fixes, best first:
  1. **Generalized inverse mass for rope ends on rigid bodies** (research 07: chain box stretch 8.9% -> 0%).
  2. **More substeps** (F3: chain 1.0% -> 0.3% at 16x1). This is cheap only with F5.
  3. **Long range attachments** ([Kim, Chentanez, Müller-Fischer 2012](https://diglib.eg.org/handle/10.2312/SCA.SCA12.305-310)).
     For a chain hung from a Hook, add a unilateral max-distance tether from the hook to each box's attach point,
     with the length summed along the chain. It is O(n), vectorized, and removes cumulative stretch whatever the
     iteration count. It does nothing for chains between two moving bodies.
- Rope `stiffness < 1` (bungee) is PBD-iteration-dependent today. Give `Rope` a compliance instead.
- Ropes are a per-rope Python loop:
  - 60 us/rope/call at background QoS.
  - A vectorized batch of point-disjoint ropes is 3.5x faster for a 20-rope chain and bit-identical, because no two
    ropes share a point in Chain, the cradle or the demo.
  - Ropes also force the GPU backend onto the CPU path (`pbd_substep` returns False). Research 05 / 07 cover the batch
    (07's 2-colour version handles shared points).
- RigidBody shape matching with stiffness 1 is already exact each pass. Its error comes from contacts and ropes
  editing single points after it (research 07), and it falls with substeps (F3).

### F7. Other substep-coupled parameters found in the code

These change behaviour when S changes, so they need per-second forms before any schedule change ships.

- `grab_stiffness` (0.3 of the gap per substep) and `grab_max_step` (4 px per substep). At 16 substeps the hand pulls
  about 4x harder and up to 4x faster.
  - Per-second forms: a fraction 1 - exp(-k h) per substep, with k chosen so that 4 substeps give today's pull, and
    a max speed in px/s. Today's equivalent is 4 px x 240 substeps/s = 960 px/s.
- `_bounce` (restitution) runs per substep with its own reach look-ahead (`4 h vmax`); it is fine at any S.
- `ccd_push` (0.25 px target gap) is applied as dvn = push/h. With frame-level CCD, h = dt, so the push velocity is
  4-16x smaller per check. It still lands at the same gap, because positions are X0 + dt V.
- Rigid `stiffness` < 1 and SoftBody `stiffness` are per pass (F2).

Whip-drag stress (grab a random point every 45 frames and whip it on a 180 px circle at 2 Hz; brute-force check every
frame), 400 frames. Zone counts are per CCD call: one per substep for stock, one per frame for frame-level CCD, so a
frame-level zone freezes 4x more motion.

| scene | stock 4x4 | frame-level 8x1 | frame-level 16x1 |
|---|---|---|---|
| demo whip: overlaps / zone calls / CCD-edited pts per frame / rel. cost | 0 / **81** / 5.5 / 1 | 0 / **0** / 2.2 / 0.52 | 0 / **0** / 2.3 / 0.80 |
| 60-body pile whip: overlaps / zone calls / CCD-edited pts per frame / rel. cost | 0 / 1 / 8.1 / 1 | 0 / 1 / 6.4 / 0.83 | 0 / **14** / 7.2 / 1.07 |

- Zero overlaps in every whip run.
- On the demo, frame-level CCD removes all 81 zone fallbacks: PBD gets the whole frame to undo the hand's
  proposal. Research 10 saw the same 47 -> 0 effect from 8x2 / 16x1.
- **Pile whip at 16x1 is the one regression: 14 frame-level zone fallbacks.** The likely cause is F7's
  substep-coupled grab. At 16 substeps the hand pulls ~4x harder (the pile moved at 254 vs 104 px/s mean), and it
  crushes bodies into the GS ping-pong of 02 F4.
  - This was not separated out within budget. The fix order: per-second grab parameters first, then 02's projection
  fallback or an adaptive `ccd_stride` (redo the frame with per-substep CCD when the frame CCD needs a zone).

## Applicability to this codebase

The changes map onto these files and functions:

- **solver.py `World._step`.** Split into a substep loop (predict, `_pbd_substep`, velocity, clamp, damping) and a
  frame tail (one `_resolve_ccd(dt)` from `X0`).
  - Keep a `ccd_stride` setting: 1 = today's per-substep CCD, S = once per frame. This allows a gradual rollout and an
    A/B against research 10's harness.
  - Velocity rule: keep V_last where CCD did not edit V. Elsewhere use the corrected average, or V_last + dV.
  - Bridson-style adaptivity is possible: if the frame CCD needed more than k impulse passes or a zone, redo the frame
    with per-substep CCD.
- **`_solve_contacts`.** Take the side reference as an argument: `X0` in frame mode, `PREV` today. Friction still
  uses the substep's PREV.
- **`_pbd_substep` / `_candidates`.** Candidates from per-point swept boxes once per frame, plus an O(N) "still inside
  my box" check per substep. Research 03's fat-box list is the same mechanism, also covers CCD pairs and has
  validation; use it instead of building a second one.
- **`_solve_body_constraints`.**
  - XPBD springs: lambda per spring, reset each substep; alpha~ = alpha / h^2.
  - Mass-splitting Jacobi in numpy (one bincount pass, like today), or a sequential GS loop under numba (research 05).
  - Optional Rayleigh damping beta.
  - `SoftBody(stiffness=...)` keeps working through a calibrated stiffness -> compliance map: s=0.1 -> 3e-4,
    s=0.05 -> 1e-3, fitted on the F2 table. It can also take `compliance=` directly.
- **entities.py.** `SoftBody` / `Rope` get a `compliance` (and `damping`). EnvConfig gets per-second
  `grab_stiffness` / `grab_speed`. Defaults move to `substeps = 8, iterations = 1` once F5 lands, or 16x1 for
  quality.
- **gpu_backend.py.** With frame-level CCD, all substeps' PBD go into one MLX command buffer with one sync per frame
  (research 04 F7, an estimated 0.6 -> 0.15 ms sync floor). Research 04 F6: at 16 substeps g h^2 = 5e-4 px, so the
  GPU path needs per-body origins in float32. The CPU path is float64 and unaffected.
- **presets_cradle.py.** `ENV substeps: 8` no longer changes material stiffness once springs are XPBD; the cradle has
  only rigid balls and ropes anyway.

## Expected gains and risks

Gains:
- **Speed (frame-level broad phase + CCD, numpy):** 8x1 costs 0.45-0.73x of today's 4x4. It matches or beats 4x4 on
  chain stretch, energy, shape error and impact-zone count (research 10: 0 zones at 8x2/16x1 vs 47 at 4x4).
  16x1 costs 0.74-0.98x for a clearly more accurate sim (F3).
- **The gain stacks:**
  - with research 03's fat list, which shares the swept-box machinery;
  - with research 05's numba kernels: per-substep work becomes pure compute, so S x c dominates and S = 16 gets cheap;
  - with research 04's one-sync-per-frame GPU path.
- **Predictability:** XPBD makes stiffness a material property (spread 0.02-0.08 instead of 0.17-0.37 across
  schedules). Damping beta and compliance alpha become per-body physical knobs. Research 06's shape-matching and ARAP
  models plug into the same lambda machinery.

Risks:
- **Zero-overlap guarantee.** Unchanged in kind: it is the same `_resolve_ccd` on a straight path from a valid state,
  and only h differs. The new failure mode is longer sweeps:
  - more pairs (measured: fewer hits, not more);
  - when a GS ping-pong does happen, a zone freezes a whole frame of motion.
  - Mitigations: 02's projection fallback, or `ccd_stride` < S adaptively.
  - The brute-force checker found 0 overlaps in every run here.
- **Contact side from X0.** A point that legitimately passes around an edge's endpoint within one frame could get the
  "wrong" edge-normal side. The code only uses the edge normal when the point is behind the edge line *and* within r
  of the segment, and the CCD catches any real crossing. 0 overlaps measured, but rare visual pops are possible.
  Watch research 10's `ccd_*` tests.
- **Behaviour changes:**
  - livelier motion (less numerical damping, F4): needs beta tuning;
  - the grab feel changes with S (F7);
  - research 10's suite will report differences, not bit-identity.
  - Jacobi XPBD drifts at 1 iteration for very stiff springs (alpha ≤ 1e-4): use GS (numba) or I = 2 there.
- **Velocity after frame CCD.** Edited points take the frame-average velocity (an O(g dt / 2) error on impact). It is
  inelastic like today, so it is not a stability risk.

## Effort

| item | size | notes |
|---|---|---|
| XPBD springs (mass-splitting Jacobi, lambda, beta) + stiffness->compliance map | S (~80 lines) | prototype `xworld._xpbd_springs` |
| `ccd_stride` / frame-level CCD in `_step` + X0 side reference in `_solve_contacts` | S-M (~60 lines) | prototype `xworld._step`, `_solve_contacts_ref` |
| frame-level contact candidates (swept boxes + validity check) | M | or adopt research 03's fat list (preferred) |
| per-second grab parameters | XS | |
| GPU: all substeps in one command buffer | M | research 04 F7 |
| LRA tethers for hook-hung chains | S | optional, after 07's generalized-mass rope ends |

## Build plan

Ranked. Each item is independent enough for its own worktree. All are gated by research 10's
`run_bench.py --code <worktree>` (0 overlaps on every substep, `compare.py` vs `base_acc.json`) plus its
`--perf-only` timing at `nice -n 10`.

1. **Frame-level CCD + contact broad phase (`ccd_stride`), stock springs.** Highest value; it is the enabler.
   - Change: `World._step`, `_pbd_substep` and `_solve_contacts` (side reference) as in F5. The prototype is
     `/private/tmp/claude-501/research-01/xworld.py`. Reuse research 03's fat list for candidates if it has merged.
     Add `EnvConfig.ccd_stride` (default = substeps) and per-second grab parameters.
   - Success: at 4x4, cost ≤ 0.85x of main with all accuracy metrics within noise. Then switch the default to 8x1
     at ≤ 0.7x cost with chain, pendulum and zone metrics ≥ main. 0 overlaps, 0 new zone fallbacks in `ccd_demo`.
   - Measure: research 10 suite plus `stress.py` (diamonds, impact, whip) here.
   - Expected: −30-50% CPU physics. Bigger once 05's numba makes F the only numpy-overhead item left.
   - Risk: low-medium (F5 risks). Effort: S-M.
2. **XPBD springs with mass-splitting Jacobi + damping.**
   - Change: `_solve_body_constraints` / `_pack_constraints` (lambda per spring, alpha per body, beta). `SoftBody`
     gets `compliance` / `damping`. Map existing `stiffness` through the F2 calibration (s 0.1 -> 3e-4,
     0.05 -> 1e-3). Default beta ≈ 10.
   - Success: the soft rest sag spread across {4x4, 8x2, 8x1, 16x1} < 0.05 (stock 0.31), with the default demo body
     within ±0.03 of main's 4x4 sag. Bounce and settle within 20% of main. Research 10's `soft_rest` and pile50
     metrics not worse.
   - Expected: no speed change. Every later schedule change stops altering materials.
   - Risk: low. Effort: S. Pairs naturally with 06's shape-matching or ARAP constraints, which use the same lambda
     loop.
3. **Small-steps default (8x1 or 16x1) on top of 1 + 2, CPU numba or numpy.**
   - Change: `EnvConfig.substeps`/`iterations`; retune beta; the cradle's per-world 8 substeps becomes the global
     default.
   - Success: research 10 F8 metrics at 16x1 levels (pendulum ≤ 1%/period, chain m100 drop ≤ 2 px, 0 zone
     fallbacks) at ≤ 1.0x of main's cost.
   - Expected: F3 accuracy at today's cost (16x1) or half today's cost (8x1).
   - Risk: low for 8x1. For 16x1 it is medium until the pile-whip zone regression (F7) is fixed by per-second grab
     parameters plus an adaptive `ccd_stride` or 02's projection fallback. Effort: XS plus a tuning pass.
4. **GPU: all substeps in one command buffer (with research 04).**
   - Change: `GpuBackend.pbd_substep` -> `pbd_frame`: predict + I iterations x S on the GPU, one `mx.eval`; frame
     CCD on CPU float64. Needs per-body float32 origins (04 F6).
   - Success: GPU physics ≤ 0.5 ms/frame on the demo; crossover with CPU below 500 points; 0 overlaps.
   - Risk: medium (float32 at 16 substeps, MLX graph size). Effort: M. Lower priority, since research 05 shows
     numba CPU beats the GPU backend at every measured size.

Cross-links:
- 02 F7 (chord check, longer sweeps, projection fallback).
- 03 (fat-box list = the candidate cache F5 needs; validity argument identical).
- 04 F6/F7 (float32 at high S, one sync per frame).
- 05 (numba makes per-substep F the main cost, so frame-level CCD and broad phase matter more, and sequential GS
  XPBD becomes cheap).
- 06 (uses F2's calibration and mass splitting; its shape-matching/ARAP constraints get the same schedule
  independence).
- 07 (generalized-mass rope ends, rope batching, sleeping).
- 09 (per-substep fixed costs, GPU sync floor).
- 10 F8 (substeps, not iterations, set energy and stretch; the gate for all of the above).
