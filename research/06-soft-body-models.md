# 06: Soft-body models for accuracy and control

Status: complete for cycle 1 (cross-read with 01, 04, 05, 07, 09, 10 drafts). All numbers below are measured with prototypes that monkeypatch a built `World` (no repo
edits). Scripts, raw JSONL results: `/private/tmp/claude-501/research-06/` (`models.py` = the models, `bench.py` =
scenes + metrics, `calib.py` = iteration / slider sweep, `scale.py` = cost at 10 and 200 bodies, `settle.py` = soft
column + damping, `*.jsonl` = results).

Terms: **PBD** = position-based dynamics; **XPBD** = extended PBD (a compliance alpha = inverse stiffness, plus one
Lagrange multiplier lambda per constraint, so stiffness stops depending on substeps / iterations); **SM** = shape
matching; **FEM** = finite element method; **ARAP** = as-rigid-as-possible (co-rotational) strain, ||F - R||; **F** =
deformation gradient of a triangle (2x2, current edge matrix times inverse rest edge matrix); **Jacobi** = all
constraints computed from the same positions then combined; **GS** = Gauss-Seidel; **CCD** = continuous collision
detection.

## Summary

1. **The flattening is structural, not just "too soft".** A single resting demo dodecagon (stiffness 0.1, skip_sets 2)
   ends at **0.71x its rest height-to-width ratio**; at the bottom of a 3-body stack 0.41x with 4 folded corners. Even
   the strength slider's maximum (1.0) gives 0.96x alone and **0.69x in the stack**. A 40-point concave star (a
   freehand-like outline) **collapses to 46% of its area**, and a 200x20 bar on two posts collapses to 37%. The
   ring of short chords has no long-range or shape term; its softest mode (ovalising) barely changes any chord.
2. **A whole-polygon area (pressure) constraint does not fix it.** Area held at 1.000, but the rest aspect only went
   0.71 -> 0.76 and the stack 0.41 -> 0.57. Ovalising changes area only at second order. Pressure is right for
   balloons and tyres, not as the main fix.
3. **Shape memory fixes it, cheaply: soft shape matching on top of the springs.** Pulling each point toward the
   best-fit rotated rest shape (Müller et al. 2005, the same math `RigidBody` already uses) gives rest aspect 0.98-0.99,
   0 folds, star area 0.998, bar intact. As an **XPBD compliance** (`smx`) the rest shape error is the same at 2, 4 and
   8 iterations (0.008 / 0.007 / 0.006), where stock springs go 0.237 / 0.108 / 0.079. Cost: about 2x the constraint
   phase in this numpy prototype (+~0.5 ms/frame on the demo at P-core speed, est.). The GPU rigid shape-matching
   kernels already compute everything it needs.
4. **For physically meaningful, locally deformable bodies: XPBD ARAP triangles.** Two constraints per triangle,
   C_D = ||F - R|| (shear modulus mu) and C_H = det F - 1 (area), on a perimeter-only triangulation (ear clipping, no
   new points, so collision code is untouched). Rest error 0.006, 0 folds in a 200-body pile (stock: 321), bending
   like a real soft beam (bar sags but keeps its area), schedule-independent (mu = 1e3: 0.059 / 0.061 / 0.064 at
   2/4/8 iterations). Cost: 3-5x the constraint phase in numpy; needs numba / GPU (research 05 / 04) to be cheap.
5. **Macklin & Müller's Neo-Hookean constraints (as published) shrink at this sim's iteration budget:** area 0.53 at
   4x4, 0.79 at 4x16, 0.98 only at 4x64 (or 0.94 at 16x4). Its deviatoric target is 0, so rest stability needs a
   converged solve. The ARAP deviatoric term is zero at rest and has no such problem. Use ARAP + det.
6. **Zero-overlap guarantee:** every model here only changes the PBD proposal (`_solve_body_constraints`); CCD is the
   stock code. Overlap-checked whip runs (brute-force checker on every substep): 0 bad substeps and 0 impact-zone
   fallbacks for all models measured (table below). One visible side effect: bodies that keep their round shape roll
   like wheels instead of sitting flat, so small piles take longer to come to rest (F8). No new points or edges are added, so the broad phases,
   self-collision filters and GPU pair kernels are unchanged.

## Findings

### F1. Why today's soft bodies flatten, fold and change with the iteration count (code reading)

`SoftBody` (entities.py) builds springs from each point to its 1st .. (skip_sets+1)th neighbour.
`World._solve_body_constraints` (solver.py) solves them all as one Jacobi pass in which each spring's correction is
scaled by `stiffness` and then **divided by the point's spring count** (`spring_count`). Three consequences:

1. **The effective per-pass stiffness is tiny and topology dependent.** A demo dodecagon (`skip_sets=2`) has 6
   springs per point, so `stiffness=0.1` removes about 0.1/6 = 1.7% of each spring's error per pass. Adding skip sets
   (more springs) also raises the divisor, so "more springs" doesn't mean "stiffer" in a predictable way.
2. **Stiffness depends on substeps x iterations.** Plain PBD: after n passes the error left is (1-k)^n. Müller et al.
   2007 (PBD, sec. 3.3) give the patch k' = 1-(1-k)^(1/n) to make the error linear in k, and note the stiffness
   *still* depends on the time step ([PBD paper](https://matthias-research.github.io/pages/publications/posBasedDyn.pdf)).
   XPBD fixes both ([Macklin, Müller, Chentanez 2016](https://matthias-research.github.io/pages/publications/XPBD.pdf)).
   Research 01 measured the same body at 0.60x-0.91x rest height over six schedules (01 F2).
3. **No long-range or area term.** All springs are short chords around the ring (a "ring truss"). Its softest mode is
   the n=2 ovalisation, which stretches chords only slightly and changes the area only at second order. Under a
   point load a vertex can be pushed inward until the corner turns inside out (a "fold"), which self-collision then
   has to hold apart, and nothing pops it back. On concave outlines (freehand, up to `FREEHAND_MAX` = 64 points)
   chords are even less effective: the star collapses (F7).

### F2. Area / pressure constraints (2D "volume")

* Classic 2D pressure model: [Matyka & Ollila 2004](https://arxiv.org/abs/physics/0407003) (ideal gas, P = nRT / A,
  force P x edge length along each edge normal); 2D tutorial at
  [panoramx.ift.uni.wroc.pl/~maq/soft2d](https://panoramx.ift.uni.wroc.pl/~maq/soft2d/index.php). Acts like an air
  balloon, no non-uniform / semi-liquid behaviour ([arXiv 0907.4364](https://arxiv.org/pdf/0907.4364)).
* PBD/XPBD form: one scalar constraint per body, C = A(x) - A0, dA/dx_i = 0.5 perp(x_{i+1} - x_{i-1}) (signed
  shoelace area); dlambda = (-C - alpha~ lambda) / (sum w_i |grad_i|^2 + alpha~). Same as the PBD paper's balloon
  constraint and [OpenRNDR's 2D body-area constraint](https://orx.openrndr.org/orx-force-2d/org.openrndr.extra.force2d/body-area-constraint.html).
* JellyCar, the reference 2D soft-body game, uses pressure only for tyres and balloons, because it rounds every
  shape ([FitzRandolph, Game Developer deep dive](https://www.gamedeveloper.com/programming/deep-dive-the-soft-body-physics-of-jelly-car-explained)).
* Per-triangle area preservation is the hydrostatic half of the FEM model (F4); a closed-form optimal triangle area
  projector exists ([Arango Duque & Bartoli 2021](https://arxiv.org/pdf/2011.12074)).
* **Measured here:** holds area exactly but barely resists flattening (F7). Worth having as an opt-in "inflated"
  setting (balloons, tyres), and it costs only O(points) per pass.

### F3. Shape matching and region (cluster) shape matching

* [Müller, Heidelberger, Teschner, Gross 2005](https://cg.informatik.uni-freiburg.de/publications/2005_SIGGRAPH_deformationShapeMatching.pdf):
  goal g_i = R (x0_i - c0) + c, R = rotation part of A_pq = sum m (x - c)(x0 - c0)^T; points move by alpha (g - x).
  Linear mode: beta A + (1-beta) R with A = A_pq A_qq^-1 / sqrt(det A) (area-preserving in 2D); quadratic mode
  adds x^2, y^2, xy terms; clusters add local freedom. In 2D, R is one atan2, exactly as in `RigidBody`.
* JellyCar: a rigid "frame" fitted each step plus a spring from each point to its frame point, on top of
  perimeter + internal springs; several local frames for long thin shapes (same source as F2).
* Region / lattice shape matching ([Rivers & James 2007, FastLSM](https://dl.acm.org/doi/10.1145/1275808.1276480)):
  many overlapping regions, each point averages the goals of its regions; region size sets the stiffness.
* Drawbacks: geometric, not physical; doesn't conserve volume
  ([Müller, Macklin, Chentanez, Jeschke 2022, "Physically Based Shape Matching"](https://diglib.eg.org/handle/10.1111/cgf14618),
  which derives an SM-like method from continuum mechanics and reproduces FEM when the groups are elements); the
  plain per-pass alpha is iteration-dependent like PBD.
* **XPBD shape matching (used here, `smx`):** treat "point i at its goal g_i" as a zero-rest-length compliant
  constraint with its own lambda_i: dlambda_i = (-|x_i - g_i| - alpha~ lambda_i) / (w_i + alpha~). Measured to be
  iteration-independent (F7).
* **Measured here: clusters must be spatial, not index-consecutive.** Clusters of 5 consecutive outline points never
  span the body's thickness: the bar collapsed to 54% area (as bad as stock). Clusters = "all outline points within
  30 px of every 2nd point" kept the bar at 0.995 area while letting it bend (aspect 1.27), the star at 0.993.

### F4. 2D FEM-lite: XPBD continuum constraints on triangles

* [Macklin & Müller 2021, "A Constraint-based Formulation of Stable Neo-Hookean Materials"](https://matthias-research.github.io/pages/publications/neohookean.pdf)
  (MIG '21): per element C_H = det(F) - gamma and C_D = sqrt(tr(F^T F)), alpha_D = 1/(mu A), alpha_H = 1/(lambda A),
  gamma = 1 + mu/lambda for rest stability. First-order gradients only, no polar decomposition, recovers from
  inversion, incompressible (alpha_H = 0) without locking. They use 64 iterations or many substeps.
* **Measured caveat:** C_D's target is 0, so it is never satisfied at rest; the rest state only balances at
  convergence. At this sim's 4 substeps x 4 Jacobi iterations the body shrank to 53% area (4x16: 79%, 4x64: 98%,
  16x4: 94%). I found no published report of this; it follows from the formulation. A co-rotational deviatoric
  term C_D = ||F - R||_F, with R from the 2D atan2 of F, is zero at rest; with C_H = det F - 1 it holds area 0.997
  at 4x4. The derivative of R drops out of the gradient in 2D (F - R is orthogonal to dR), so the gradient is just
  (F - R)/C times D_m^-T.
* `sqrt(tr F^T F) - sqrt(2)` (a rest-stable shift of the Neo-Hookean term) was tried and is no better than stock (rest
  aspect 0.75): a Frobenius-norm target allows shear.
* Earlier continuum PBD: [Bender et al. 2014](https://doi.org/10.1016/j.cag.2014.07.004) (iteration-dependent,
  explicit inversion handling); [Müller et al. 2014, Strain Based Dynamics](https://diglib.eg.org/items/cc63570f-a032-410b-9d71-cebca83a936c/full).
* **Parallel (Jacobi) XPBD:** plain averaging breaks the lambda bookkeeping. Mass splitting
  ([Tonge, Benevolenski, Voroshilov 2012](https://dx.doi.org/10.1145/2185520.2185601)): n_i w_i in each constraint's
  denominator, dx_i = w_i sum grad C dlambda. The prototypes use it; research 01 found it close to GS XPBD but
  drifting at 1 iteration (01 F2). Alternative: averaging + SOR ([Macklin et al. 2014](https://mmacklin.com/uppfrta_preprint.pdf)).
* **Triangulation limit (measured):** perimeter-only triangulation of an n-gon always has ears with an angle of
  about 180/n degrees: 15 deg for the 12-gon, 6.5 deg for the 40-point star, **2.8 deg for a 64-point circle**.
  The 12-gon is fine; the star already shows it (ARAP err 0.043 vs SM 0.005). Bodies above ~16-20 points need
  interior (Steiner) points, or should use spatial-cluster SM instead.

### F5. Strain limiting

Provot-style strain limiting (project a spring only when its stretch leaves [1-s, 1+s]) is a cheap inequality
constraint. Strain Based Dynamics clamps principal stretches per element. For the triangles here: clamp det F to
[0.5, 2] and/or the 2x2 singular values (closed form) as hard inequality constraints after the compliant solve. Its
job is to stop a very soft body from collapsing to zero area or turning inside out under a heavy load, while
staying soft in normal use. Not prototyped (lower priority once shape memory exists).

### F6. Damping models

* Current: `EnvConfig.damping` scales every velocity by (1 - damping h), so it also slows free fall and spin.
* PBD paper sec. 3.5: per body, remove only the velocity deviation from rigid motion (v_cm + omega x r), times
  k_damping. Kills jiggle without touching falling or spinning
  ([PBD paper](https://matthias-research.github.io/pages/publications/posBasedDyn.pdf)).
* XPBD Rayleigh damping (eq. 26): dlambda = (-C - alpha~ lambda - gamma grad C (x - x_n)) /
  ((1+gamma) grad C M^-1 grad C^T + alpha~), gamma = alpha~ beta~ / h; damps only along each constraint
  ([XPBD paper](https://matthias-research.github.io/pages/publications/XPBD.pdf)).
* Measured: see F8.

### F7. Measurements (CPU backend, 4 substeps x 4 iterations unless stated, background QoS)

Scenes (`bench.py`): **rest** = one demo dodecagon (r 18, 12 points, skip 2) on the ground; **stack** = 3 dodecagons +
a 40 px rigid box on top, metrics of the bottom one (this stack sometimes topples, so treat it as qualitative);
**drop** = dodecagon dropped 200 px, metrics during and after impact; **star** = 40-point concave 5-point star (r 40 /
18), freehand-like; **bar** = 200x20 bar (42 points) on two posts 160 apart. Metrics: **aspect** = (height/width) /
rest value, 1 = no flattening; **err** = RMS distance to the best-fit rotated rest shape / RMS radius; **area** =
area / rest area; **folds** = corners whose turn direction flipped (meaningless on the star/bar, which have collinear
rest points). All after 5 s (300 frames) unless stated.

Models: `stock`; `area` = whole-polygon area, XPBD alpha 0 (PBD k=1 gives identical numbers); `sm b` = PBD shape
matching with per-pass factor b (springs kept); `smx a` = XPBD shape matching, compliance a (springs kept);
`sclust` = spatial-cluster SM (30 px, every 2nd point, b 0.05); `arap mu` = XPBD ARAP + det triangles, lambda = 10 mu,
springs **removed**; `arap_soft` = mu 3e3; `neo` = Macklin-Müller as published, mu 2e4, lambda 2e5.

| model | rest aspect | rest err | stack aspect | stack folds | drop min area | drop max err | star area / err | bar area / err |
|---|---|---|---|---|---|---|---|---|
| stock k 0.1 (demo) | 0.713 | 0.116 | 0.409 | 4 | 0.579 | 0.51 | **0.455 / 0.548** | **0.367 / 0.394** |
| stock k 0.05 | 0.553 | 0.225 | 0.384 | 4 | 0.443 | 0.61 | | |
| stock k 1.0 (slider max) | 0.964 | 0.012 | 0.694 | 2 | 0.872 | 0.28 | | |
| area (alpha 0) | 0.761 | 0.094 | 0.571 | 2 | 0.997 | 0.36 | | |
| sm 0.02 | 0.969 | 0.013 | 0.817 | 1 | 0.827 | 0.21 | | |
| sm 0.05 | 0.987 | 0.006 | 0.902 | 0 | 0.886 | 0.14 | 0.998 / 0.004 | 0.996 / 0.002 |
| sm 0.05, linear 0.5 | 0.982 | 0.008 | 0.893 | 0 | 0.892 | 0.19 | | |
| area + sm 0.02 | 0.969 | 0.013 | 0.810 | 1 | 0.995 | 0.20 | 1.000 / 0.008 | |
| smx 1e-4 | 0.984 | 0.007 | 0.987* | 0* | 0.877 | 0.16 | 0.998 / 0.005 | 0.996 / 0.002 |
| smx 1e-3 | 0.886 | 0.043 | | | 0.744 | 0.38 | | |
| index clusters (5 pts) | 0.984 | 0.006 | 1.005 | 0 | 0.868 | 0.20 | 0.934 / 0.082 | **0.537 / 0.186** |
| spatial clusters | 0.981 | 0.008 | | | | | 0.993 / 0.007 | 0.995 / 0.032 (bends) |
| arap 2e4 | 0.990 | 0.006 | 0.988* | 0* | 0.892 | 0.18 | 0.990 / 0.043 | 0.996 / 0.125 (sags) |
| arap 3e3 | 0.956 | 0.023 | 0.956* | 0* | 0.864 | 0.34 | 0.986 / 0.102 | 0.980 / 0.321 |
| neo (published) | 0.996 | 0.272 (area **0.531**) | | | 0.415 | 0.36 | | |
| XPBD springs only (alpha 1e-4) | 0.903 | 0.035 | | | 0.779 | 0.37 | | |

\* from the 7-second iteration sweep run (`iters.jsonl`), not the 5 s stack run.

**Iteration / schedule independence** (rest scene, 4 substeps, iterations 2 / 4 / 8; `calib.jsonl`):

| model | err @2 | err @4 | err @8 | aspect @2 / @8 |
|---|---|---|---|---|
| stock k 0.1 | 0.237 | 0.108 | 0.079 | 0.54 / 0.79 |
| stock k 0.3 | 0.091 | 0.051 | 0.023 | 0.77 / 0.94 |
| stock k 1.0 | 0.028 | 0.012 | 0.006 | 0.92 / 0.98 |
| smx 1e-5 | 0.001 | 0.001 | 0.001 | 1.00 / 1.00 |
| smx 1e-4 | 0.008 | 0.007 | 0.006 | 0.98 / 0.99 |
| smx 1e-3 | 0.046 | 0.040 | 0.032 | 0.88 / 0.91 |
| arap mu 1e3 | 0.059 | 0.061 | 0.064 | 0.90 / 0.89 |
| arap mu 1e4 | 0.011 | 0.008 | 0.006 | 0.98 / 0.99 |
| arap mu 1e5 | 0.007 | 0.004 | 0.003 | 0.99 / 1.00 |

Stock varies 3-4x across iteration counts; the XPBD models vary by 1.0-1.4x (still slightly softer at very low
iteration counts for stiff settings, as research 01 also saw for Jacobi XPBD). In the 3-body stack the stock bottom
body goes 0.48 -> 0.76 aspect from 1 to 8 iterations; smx 1e-4 stays at 0.984-0.987 for every schedule from 4x1 to
8x4; arap 2e4 0.978-0.993.

**Overlap-checked whip** (4 soft dodecagons + a soft hexagon + a rigid square, random drags every 45 frames, 600
frames, brute-force checker on every substep's end state, `whip.jsonl`):

| model | bad substeps | impact-zone fallbacks | mean err (5 bodies) | min area |
|---|---|---|---|---|
| stock | 0 / 2400 | 0 | 0.207 | 0.718 |
| sm 0.05 | 0 / 2400 | 0 | 0.044 | 0.994 |
| smx 1e-4 | 0 / 2400 | 0 | 0.056 | 0.987 |
| area + sm 0.02 | 0 / 2400 | 0 | 0.040 | 1.000 |
| index clusters | 0 / 2400 | 0 | 0.032 | 0.953 |
| arap 2e4 | 0 / 2400 | 0 | 0.049 | 0.995 |
| arap 3e3 | 0 / 2400 | 0 | 0.269 | 0.885 |
| neo (published) | 0 / 2400 | 0 | 0.246 | 0.529 (the shrink, F4) |

**Cost** (CPU, numpy prototypes, background QoS so only ratios; N soft dodecagons in a box, `scale.jsonl`):

| model | constraint phase, 10 bodies | frame, 10 bodies | constraint phase, 200 bodies | frame, 200 bodies | folds in the 200 pile | mean aspect, 200 |
|---|---|---|---|---|---|---|
| stock | 1.0x | 1.0x | 1.0x | 1.0x | 321 | 0.72 |
| sm 0.05 | 2.1x | 1.00x | 1.7x | 0.76x | 20 | 0.93 |
| smx 1e-4 | 2.2x | 1.01x | 1.9x | 0.80x | 30 | 0.93 |
| area + sm | 3.0x | 1.26x | 2.0x | 0.80x | 28 | 0.92 |
| arap 2e4 (no springs) | 5.5x | 1.54x | 3.3x | 1.09x | **0** | 0.94 |
| arap + ring springs | 6.4x | 1.66x | 4.0x | 1.20x | 0 | 0.97 |

At 200 bodies the shape-memory models make the **whole frame cheaper** (0.76-0.80x): bodies that keep their shape
have fewer, shallower contacts. In absolute P-core terms the stock constraint phase is ~0.55 ms of the demo's 2.1 ms
(research 09); SM adds roughly +0.5 ms there (the cost is numpy call overhead, ~20 extra calls per pass, not
arithmetic), which numba (research 05: springs + SM 8.6x faster) or fusing into the spring bincount removes.

### F8. Settling and damping

`settle.py`: a column of 4 soft dodecagons (like research 10's `stacks.soft`), 10 s, RMS point speed per second.
In every variant the column **toppled** within the first seconds (top body ends 130-290 px sideways), so this
measures how fast a small toppled pile calms down. Deviation-only damping k = 0.05 per substep, applied to V before
CCD (so CCD still checks the damped path; 0 bad substeps everywhere).

| model | damping | RMS px/s at 4 s | 6 s | 10 s | bottom body aspect |
|---|---|---|---|---|---|
| stock | none | 27.9 | 9.3 | 13.4 | 0.72 |
| stock | deviation 0.05 | 7.4 | 2.9 | **0.53** | 0.72 |
| smx 1e-4 | none | 37.0 | 41.5 | 21.0 | 0.985 |
| smx 1e-4 | deviation 0.05 | 34.2 | 21.7 | 8.5 | 0.984 |
| arap 2e4 | none | 23.2 | 11.8 | 5.7 | 0.990 |
| arap 2e4 | deviation 0.05 | 24.9 | 11.0 | 6.2 | 0.991 |

Reads: deviation damping is very effective on squishy bodies (stock: 25x calmer at 10 s) and leaves falling alone.
On shape-holding bodies most of the leftover motion is **rigid rolling**: a dodecagon that keeps its shape rolls like
a wheel, where today's flattened one just sits. Deviation damping deliberately keeps rigid motion, so it doesn't stop
that. If "soft bodies come to rest" is wanted, it needs rolling resistance (contact friction on the rigid part of the
motion, a research 07 topic) or a little global damping, not more internal damping. This is a visible behaviour
change of shape memory to call out to the user: round jelly bodies will roll more.

## Applicability to this codebase

Everything below touches only the PBD part of a substep. Points, edges, winding, `point_ring`, the contact and CCD
broad phases, the self-collision filters and the CCD itself are unchanged, so the zero-overlap guarantee rests on
exactly the same code as today.

1. **Soft shape matching ("shape memory") in `SoftBody` + `World._pack_constraints` / `_solve_body_constraints`.**
   - entities.py: `SoftBody.__init__(..., shape=0.0)` stores `rest = points - centroid` (like `RigidBody`).
   - solver.py: `_pack_constraints` gathers soft bodies with shape > 0 into one flat batch (point ids, body index,
     rest offsets, per-body compliance) instead of the same-size groups used for rigid bodies (soft bodies have
     arbitrary point counts; bincount segment sums handle that, as in `models.Batched`). `_solve_body_constraints`
     runs it after the springs each pass. lambda per point resets in `_pbd_substep` (one `fill(0)`).
   - gpu_backend.py: `_CON_ACC` already accumulates the 6 shape-matching sums per rigid body; soft bodies get a slot
     in the same `rsum` and an `rkind` flag so `_CON_APPLY` adds the goal pull **and** the spring average (today kind 2
     is exclusive). XPBD needs one float lambda per point carried between iterations (an extra in/out buffer).
   - Rigid bodies are the alpha = 0 limit of the same code. Research 07's generalized-inverse-mass work for rope ends
     on rigid bodies applies equally to soft bodies with shape memory.
2. **XPBD springs** (research 01 owns this) make the existing springs schedule-independent. On their own they don't
   fix flattening (rest aspect 0.90 at alpha 1e-4, F7), so do 1 and 2 together.
3. **Optional per-body area constraint** (`SoftBody(pressure=...)`): one per-body segment sum; useful for
   balloons / tyres. The GPU version is another per-body reduction like `rsum`.
4. **ARAP triangles (a new "jelly" soft-body model).** entities.py: a triangulation at construction (ear clipping
   with best-min-angle ear first, `models.ear_clip`; freehand outlines are simple polygons, which the editor
   validates already), rest D_m^-1 per triangle, rest area. solver.py: a batched triangle pass (2 constraints per
   triangle, Jacobi + mass splitting, lambda per constraint). For bodies above ~16-20 points, either add interior
   points (needs care: `Body.edges` must stay the outline loop, `point_ring` / `body_counts` used in the self-contact
   neighbour filter must count outline points only; CCD would then keep interior points inside the outline for free,
   since they are self-collision points vs the body's own edges) or fall back to spatial-cluster SM.
5. **Deviation-only damping** per soft body (`World` after `_limit_speed`, before CCD, or after the substep with the
   global damping): O(points) with bincount, gives a "wobble damping" setting that doesn't slow falling bodies.

### Keeping the user-facing settings meaningful

* **Strength slider (0.1-1.0)** today is a per-pass PBD factor, so it means different things at different substep
  counts (the Newton's cradle runs 8) and skip-set counts. Proposal: keep the slider, map it to a log-scale XPBD
  compliance so equal slider steps feel like equal stiffness steps and are schedule-independent. From the sweep:
  strength 1.0 -> smx alpha ~1e-5 (near rigid, err 0.001), 0.5 -> ~1e-4 (err 0.007, the look of today's demo
  dodecagons but without flattening), 0.1 -> ~1e-3 (err 0.04, visibly squishy). That is
  alpha = 1e-3 * 100^(-(s - 0.1)/0.9). For springs, research 01's calibration (demo s=0.1 ~ alpha 3e-4, s=0.05 ~
  1e-3) gives the matching spring compliance, so existing scenes can keep their look.
* **Spring sets** keep their meaning as "how much the outline resists local bending / denting" (skip springs), while
  shape memory controls "how strongly it returns to its drawn shape". Show it in the editor as a second slider
  ("shape memory", 0 = today's behaviour). For the ARAP model the sets control has no meaning; show the triangles in
  the preview instead (`editor.skip_springs` preview code becomes a triangle preview).
* **Iterations / substeps** stop changing the material, so research 01's schedule changes (fewer iterations, more
  substeps, or frame-level CCD) no longer alter how existing scenes look once springs and SM are XPBD.

## Expected gains and risks

Gains (measured, F7): demo dodecagon rest flattening 29% -> 1-2%; stack-bottom flattening 59% -> 1-10%; folds in a
200-body pile 321 -> 20-30 (SM) or 0 (ARAP); concave freehand-like star area 46% -> 99.8%; long bar from collapsed to
intact (SM) or naturally bending (spatial clusters, ARAP); shape stiffness constant across iteration counts within
1.0-1.4x instead of 3-4x. Frame cost: about neutral on the demo once fused/compiled (+~0.5 ms in plain numpy), and
**0.76-0.80x** in a 200-body pile because shaped bodies make fewer contacts.

Risks:
* **Zero-overlap guarantee: none found.** Internal constraints only change the PBD proposal; CCD then makes the
  straight-line motion collision-free as today. 0 bad substeps / 0 impact-zone fallbacks in every overlap-checked run.
  Residual risk: stiffer bodies fighting a squeeze (a body pushed into a gap smaller than its rest shape) could raise
  CCD impulse counts and impact-zone fallbacks (a frozen substep, not an overlap). Worth a dedicated "squeeze" scene
  in the build experiment.
* **Folded / inverted states with self-collision on.** If a body does fold (ARAP det constraint wants to un-invert),
  un-folding may require an outline point to cross an outline edge, which CCD forbids; the body then stays folded
  but stable. Shape memory makes folding much rarer (F7) rather than fixing an existing fold.
* **Behaviour change: round bodies roll.** A dodecagon that keeps its shape rolls like a wheel; today's flattened
  one sits still (F8). Piles take longer to come to rest unless rolling resistance is added (research 07).
* **Shape matching is non-physical**: global SM makes long bodies rigid-ish (bar err 0.002); use spatial clusters
  or ARAP for bendy shapes. Index-consecutive clusters must not be used (bar collapsed).
* **Neo-Hookean as published needs converged solves**; use ARAP + det at this iteration budget (F4).
* **Perimeter-only triangulations sliver** above ~16-20 points (F4).
* **Jacobi XPBD drifts at 1 iteration** (research 01); fine at 2+.
* **GPU backend**: new constraints must also exist as Metal kernels, otherwise scenes with them fall back to CPU PBD
  (`pbd_substep` returns False, as for ropes today), which is only a speed issue.

## Effort

* Soft shape matching, XPBD, CPU batched: ~0.5 day (about 60 lines in solver.py / entities.py, mirrors
  `models.Batched.shape_xpbd`). Editor slider + mapping: ~0.5 day. GPU: ~0.5 day (extend `_CON_ACC` / `_CON_APPLY`).
* Area constraint: ~2 hours CPU, ~2 hours GPU.
* Deviation damping: ~2 hours.
* ARAP triangles: ~1.5 days CPU (ear clipping, batched pass, editor preview), +1 day Metal kernel, +1-2 days for
  interior points if wanted. numba version via research 05's framework.
* Spatial clusters: ~0.5 day on top of SM.

## Build plan

Ranked; each can be prototyped in its own git worktree in parallel. Common harness: research 10's suite
(`/private/tmp/claude-501/research-10/bench/run_bench.py --code <worktree>`, gate = no FAIL; its `stacks.soft.*`,
`pile50.soft_area_ratio_*` and `ccd_*` overlap tests are the relevant ones), plus `bench.py` / `calib.py` here
(adapted to call the built-in model instead of `models.install`), plus the brute-force checker
(`/private/tmp/claude-501/debug-agent/merged/brute.py`, `ccd_test.py`).

1. **Shape memory for SoftBody (XPBD shape matching), CPU + GPU.**
   Change: `entities.SoftBody(shape=alpha)`, batched XPBD SM in `World._pack_constraints` /
   `_solve_body_constraints` (lambda reset per substep in `_pbd_substep`), the same in `gpu_backend._CON_ACC` /
   `_CON_APPLY`; demo presets set shape memory so today's look stays but without flattening.
   Success: rest aspect >= 0.97 and stack-bottom aspect >= 0.9 for the demo dodecagons at 4x4; rest err varies
   <= 1.5x across 4x2 / 4x4 / 4x8 / 8x4; star area >= 0.98; research 10 suite no FAIL (0 overlap substeps, 0 zone
   fallbacks); demo frame time within +0.3 ms (P-core, `nice -n 10`), pile200 not slower. Expect a WARN on
   `stacks.soft` settle time (shaped bodies roll, F8); report it with that reason, or pair with research 07's
   rolling resistance. Add a "squeeze" scene (a body pushed into a gap narrower than itself) and report CCD impulse
   counts / zone fallbacks vs stock.
   Expected: F7 numbers. Risk: low. Effort: 1-1.5 days.
2. **Schedule-independent strength slider (XPBD springs + SM compliance mapping) in the editor.**
   Depends on research 01's XPBD springs. Change: editor `STRENGTH_RANGE` maps to compliance
   alpha = 1e-3 * 100^(-(s-0.1)/0.9) for SM and research 01's spring calibration; a "shape memory" slider next to
   strength; presets converted so they look the same at 4x4. Success: the same body's rest err within 1.5x over all
   schedules in `calib.py`; Newton's cradle world (8 substeps) and default world give identical soft-body sag.
   Risk: low-medium (look of existing scenes). Effort: 1 day.
3. **ARAP triangle "jelly" model (perimeter-only triangulation) + per-body deviation damping.**
   Change: `SoftBody(model="fem", mu=..., area_compliance=...)` with ear clipping at construction, a batched XPBD
   triangle pass (numba kernel per research 05, Metal kernel per research 04), deviation-only damping per soft body;
   spatial-cluster SM as the fallback for > 20-point outlines. Success: 0 folds in the 200-body pile, rest err <= 0.01,
   bar sags but keeps area >= 0.98, err varies <= 1.5x over schedules, no FAIL in research 10, constraint phase
   <= 2x stock after compilation. Risk: medium (slivers on big outlines, cost before compilation). Effort: 3-4 days.
4. **Opt-in pressure (area) constraint for balloons / tyres.** Change: `SoftBody(pressure=compliance)`, one batched
   per-body pass, GPU reduction. Success: drop-test min area >= 0.99, no FAIL. Risk: low. Effort: 0.5 day. Lowest
   priority: it does not fix flattening.

Cross-links: research 01 (XPBD springs, schedules, frame-level CCD; its F2 calibration is the spring half of the
slider mapping), research 04 (Metal kernels: SM sums already exist; triangle pass = one thread per triangle with
atomics, or a CSR gather for static topology), research 05 (numba: springs + SM 8.6x faster; a triangle kernel fits
the same pattern), research 07 (generalized inverse mass for rope ends on shape-matched bodies applies to soft bodies
with shape memory too), research 10 (accuracy suite, soft stack settle-time and area metrics; its finding that
Jacobi contact averaging loses momentum is independent of the body model). Correction for research 10 F6, which
expected 06's area constraint to fix the soft stack: measured, the area constraint holds area but not shape
(stack-bottom aspect 0.41 -> 0.57); shape memory is what fixes `soft_rest.sag_ratio`, and the leftover settle time
is rolling (F8), which deviation damping by design doesn't remove.
