# 07: Rigid bodies, stacking, friction, restitution, sleeping

Status: cycle 1 complete. Everything below is measured with monkeypatch prototypes; no repo code was edited. Scripts and logs:
`/private/tmp/claude-501/research-07/` (`common.py` scenes, `baseline.py`, `stress.py`, `friction.py`, `patches.py`
prototypes, `sleep_patch.py`, `variant.py` runner, `pile_diag.py`, `pile_creep.py`, `ccd_diag.py`, `logs/`).
All runs: CPU backend, `taskpolicy -b` on a loaded machine, so ms numbers are only comparable within one table.

## Summary

- Rigid stacks of equal-mass boxes already work well: towers of 8-20 boxes and a 21-box pyramid settle with no
  jitter, tilt < 0.1 deg, shape error < 0.25 px, zero overlaps. Coulomb friction on an incline breaks away at the
  right angle (between 26 and 28 deg for mu = 0.5; analytic 26.6 deg).
- The real rigid-body problems show up where something acts on single points of a shape-matched body: ropes (chains,
  the hanging box) and big mass ratios. 49-segment chain between two hooks: boxes end 8.9% stretched, chain jitters
  at ~21 px/s. 100:1 mass-ratio stack: boxes 2 px out of shape, 1 px/s jitter that never stops.
- Root cause: shape matching runs first in each iteration, then rope / contact projections move single points using
  the point's inverse mass (1) rather than the body's generalized inverse mass (1/M + (r x n)^2 / I). Each substep
  ends with the box deformed by the last rope pull; the next substep's shape match snaps it back.
- Fix measured: a rope end on a RigidBody moves the whole body (translation + exact rotation) with the generalized
  inverse mass (Mueller et al. 2020). Hanging chain: box stretch 6.4% -> 0.0%, jitter 2.6 -> 0.0 px/s. Two-hook chain:
  8.9% -> 0.0%, 21 -> 5.7 px/s. Zero overlaps; CCD untouched. A batched, graph-coloured version (2 colours for a chain)
  removes the per-rope Python loop. On research 10's suite: 100:1 chain weight drop 6.7 -> 0.59 px, residual weight
  speed 19.2 -> 0.001 px/s; pendulum and cradle metrics unchanged.
- Per-body sleeping is safe here (a sleeper is just a temporary FixedBody that CCD still collides against) and works
  where Box2D-style island sleep never fires: 60/60 pile bodies asleep, settled-pile frame cost 136 -> 63 ms
  (background QoS) even with the broad phase still running over everything.
- Rigid-aware contacts (generalized mass + research 10's shared per-contact scale) conserve momentum, keep rigid
  bodies exactly rigid, cut pile50 jitter RMS 15.1 -> 2.7 px/s and residual KE 21x, and drop ccd_demo impact-zone
  fallbacks 47 -> 0. Downsides: soft stacks and 100:1 squeezes converge worse (skin penetrated down to the CCD gap), so
  this needs more work before shipping.
- Friction is already Coulomb-correct (UPP-style). Restitution exists only for the cradle; an XPBD-style velocity
  pass adds it for everything, velocity-only, so the guarantee is untouched.

## Findings

### F1. Stacking of equal-mass rigid boxes is already stable (measured)

10 s runs, 4 substeps x 4 iterations, mu = 0.5. "tail |v|" = mean point speed over the last 2 s.

| scene | bodies | max shape error (px) | tail mean |v| (px/s) | max tilt | overlaps | CCD zones |
|---|---|---|---|---|---|---|---|
| tower of 8 boxes 40x40 | 8 | 0.095 | 0.00 | 0.0 deg | 0 | 0 |
| tower of 12 | 12 | 0.143 | 0.00 | 0.0 deg | 0 | 0 |
| tower of 8, +-2 px lateral offsets | 8 | 0.189 | 0.01 | 0.1 deg | 0 | 0 |
| tower of 20 boxes 20x20 | 20 | 0.239 | 0.10 | 0.0 deg | 0 | 0 |
| pyramid, 6 rows | 21 | 0.062 | 0.00 | 0.1 deg | 0 | 0 |
| 10:1 alternating mass stack (4) | 4 | 0.272 | 0.00 | 0.0 deg | 0 | 0 |
| plank 10x heavier on 2 boxes | 3 | 0.325 | 0.00 | 0.0 deg | 0 | 0 |

The tower "compresses" by about 0.5 px per gap (skin of contact_radius = 3 px; gaps settle near 2.9 px). Stacks
need no shock propagation or mass scaling today. The XPBD paper itself warns that "updating constraint directions
after each projection might cause instabilities when simulating tall stacks or piles" ([Mueller et al. 2020][xpbd]),
so stacking is a regression risk to watch in any rigid rewrite, not a current defect.

### F2. Friction is already Coulomb-correct, with one coefficient (measured)

`World._solve_contacts` computes `fric = min(1, mu * depth / |tangential motion|)`: exactly the position-level
friction of Unified Particle Physics ([Macklin et al. 2014][upp], eq. 24) with mu_s = mu_k. Box (30x30) on a fixed
plank, slide over 3 s after a 0.5 s settle (`friction.py`):

| incline | 5-20 deg | 24 deg | 26 deg | 28 deg | 30 deg |
|---|---|---|---|---|---|
| slid (px) | 0.00 | 0.08 | 0.26 | 69.3 | 162.3 |
| analytic from rest (px) | 0 | 0 | 0 | 63.0 | 150.7 |

Break-away is between 26 and 28 deg (analytic tan^-1 0.5 = 26.6 deg), and kinetic sliding matches the analytic
a = g (sin - mu cos) within ~8% (part of which is speed gained during the settle). Gaps vs the literature:
- one global mu (EnvConfig.friction): no per-body / per-pair coefficients and no mu_s > mu_k (stick-slip). UPP eq. 24
  and XPBD sec. 3.5 both have separate static/kinetic coefficients; XPBD averages per-pair (mu1 + mu2) / 2.
- CCD impulses (`_ccd_impulses`) are frictionless: a fast sliding impact keeps its tangential speed for that substep.
  Minor; CCD rarely fires in resting scenes (F6: 0 impulses in a settled pile).

### F3. Ropes on shape-matched bodies: point mass instead of body mass (measured, root cause of chain jitter)

`Rope.solve_constraints` moves the attached point with that point's inverse mass (1 per point), so a rope "sees" a
6-point chain box as mass 1 at the attachment, and the box only follows after the next shape match averages the
correction over its points. Order per iteration is shape matching -> ropes -> contacts, so every substep ends with
each chain box stretched by the last rope pull; the next substep's shape match snaps it back, and the CCD then turns
that snap into velocity.

Prototype `patches.rope_solve_rigid` (and the batched `solve_ropes_batched`): for a rope end on a RigidBody, use the
generalized inverse mass w = 1/M + (r x n)^2 / I and apply the correction to the whole body: translate by dp/M,
rotate by (r x dp)/I about the centre of mass, exactly ([Mueller et al. 2020][xpbd], eqs. 2-6; 2D: I is a scalar,
r x n a scalar). Point ends on soft bodies are unchanged.

| scene (10 s) | variant | box stretch (attach-to-attach) | box shape error / size | rope stretch worst / total | tail mean |v| px/s |
|---|---|---|---|---|---|
| 49-seg chain hanging from one hook | baseline | 1.064x | 3.2% | 1.000 / 0.978 | 2.57 |
| | 4x16 iterations | 1.015x | 0.8% | 1.000 / 0.995 | 0.01 |
| | rigid ropes | **1.000x** | **0.0%** | 1.003 / 1.000 | **0.00** |
| 49-seg chain between two hooks (taut) | baseline | 1.089x | 4.5% | 1.000 / 1.000 | 20.94 |
| | 4x16 iterations | 1.027x | 1.4% | 1.000 / 1.000 | 11.25 |
| | 16 substeps x 1 iteration | 1.037x | 1.8% | 1.000 / 1.000 | 20.68 |
| | rigid ropes | **1.000x** | **0.0%** | 1.012 / 1.011 | **5.69** |
| | rigid ropes, 16x1 | 1.000x | 0.0% | 1.006 / 1.005 | 10.08 |
| same chain, 5% slack | baseline | 1.041x | 2.1% | 1.000 / 1.000 | 20.54 |
| | rigid ropes | 1.000x | 0.0% | 1.005 / 1.004 | 7.76 |

Overall chain elongation (boxes are half the chain length): baseline taut chain ~ +4.5%, rigid ropes ~ +0.55%. The
bottom box of the hanging chain sinks 8.0 px in the baseline vs 0.6 px with rigid ropes. Zero overlaps and zero
impact zones in every run. Note the two-hook "taut" chain is a near-singular configuration (an inextensible chain of
exactly the span length can't sag), so some residual motion there is expected from any solver; the slack variant and
the hanging chain are the fairer jitter tests. Substeps alone (16x1, the XPBD recommendation) do not fix it in the
baseline: the defect is the mass model, not convergence.

### F4. Large mass ratios deform shape-matched bodies (measured)

| scene | max shape error | mean shape error (2nd half) | tail mean |v| | min gap (contact skin 3 px) |
|---|---|---|---|---|
| alternating stack, ratio 10 | 0.27 px | 0.10 | 0.00 | 2.90 |
| alternating stack, ratio 100 | **2.11 px** | 0.96 | **1.04** | **2.02** |
| plank 10x on two boxes | 0.33 | 0.09 | 0.00 | 3.00 |
| plank 100x on two boxes | **2.00 px** | 0.87 | 0.04 | 3.00 |

Same mechanism as F3, through contacts: the contact projection splits each correction by point inverse masses, so a
light box's corner is pushed as if it weighed 1/4 of its body, and shape matching then redistributes it.

### F5. Mass and inertia come from the point count, not the shape (code reading)

Every point gets `mass` (default 1), so a body's mass is its point count: a 40x40 box weighs 4, a 10x10 chain box
weighs 6, a cradle ball (20 points) weighs 20. The rotational inertia is that of the point cloud: for a square of side s with
corner points I = M s^2 / 2, vs M s^2 / 6 for a solid square (3x too large); a ring of points on a
circle has I = M R^2 vs M R^2 / 2 for a disc (2x). Any generalized-mass solver (F3/F4 fixes) uses M and I explicitly,
so they can come from the polygon (density x area, polygon second moment) without touching the points. That changes
existing scenes' dynamics (cradle, default demo), so it should be opt-in (e.g. `RigidBody(density=...)`).

### F6. Restitution: only the cradle's ball pass (measured)

A 40x40 box dropped 300 px (impact 487 px/s) rebounds 0.09 px: PBD contacts and CCD impulses are perfectly
inelastic, and `World._bounce` only acts between two bodies that both have `restitution > 0`, treating them as balls
(no spin, impulse through the centres). The XPBD velocity pass ([Mueller et al. 2020][xpbd] sec. 3.6) generalizes it:
after V = (P - PREV) / h, for every contact set the relative normal velocity to -e * (its normal velocity at the start
of the substep), skipping contacts slower than 2 |g| h "to avoid jittering". Box2D v3 does the same with a
`restitutionThreshold` (1 m/s) and `restitutionIterations = 2` ([box2d types.c][b2types]). Prototype
`patches.apply_restitution` (velocity-only, before CCD, so the guarantee is untouched): the drop now bounces at
202 px/s (target e = 0.5 -> 243 px/s; the shortfall is the per-body Jacobi averaging over the two corner contacts,
fixable with a second sweep without averaging or a Gauss-Seidel loop).

### F7. Rigid piles never fully settle, but what's left is tiny and not CCD (measured)

60 rigid boxes / triangles / pentagons dropped into a U container (`pile_diag.py`, `pile_creep.py`, `ccd_diag.py`):
- Speeds read from V never drop below ~1-2 px/s even in calm piles. That is mostly a per-substep sawtooth: every
  substep adds g h = 500 / 240 = 2.1 px/s of gravity velocity, the contacts take it out again, and V records the
  difference. Its position amplitude is g h^2 ~ 0.009 px, invisible. **Any sleep or jitter test must use frame
  displacement (or V averaged over a frame), not V.** This is also exactly why XPBD's restitution threshold is 2|g|h.
- With frame displacement: after 15-20 s the median body moves 0.9-1.2 px/s, p90 2-3 px/s, max 3-8 px/s; occasional
  bursts reach 20-70 px/s in the first ~10 s. Over a 3 s window the net drift is only 31% of the path length (median
  0.75 px), so it is mostly rocking in place, not creep. The fastest bodies are pentagons and triangles resting on a
  single contact (balanced on a vertex) and a triangle wedged with 7 corner-to-corner contacts.
- CCD does nothing in a settled pile: 0 impulse passes, 0 hits over 720 settled frames. All residual motion comes
  from the PBD contact solve.
- Rigid contacts with generalized masses (A2) do not change this (median ~2 px/s by V at 6 s, same as baseline): it
  is not the shape-matching fight. Candidates: PBD's overlap recovery turning into velocity (the 3 px contact skin is
  re-entered every substep), which Box2D's "relax iterations" remove ([Catto, Solver2D][solver2d]: "This serves to
  remove the extra energy from the velocities"); prototype `patches.apply_relax` caps the separating speed of
  slow contacts at 2|g|h (results below).

### F8. Sleeping: island sleep can't trigger in piles here; per-body sleep can (measured)

Research 09 measured that **settled piles cost ~3x more than falling ones** (CPU pile200: 5.8 -> 11.8 ms, pile500:
13.7 -> 33.6 ms, pile1000: 29 -> 94 ms; contacts and CCD scale with touching pairs). Settled bodies are exactly
what sleeping skips. Box2D sleeps whole islands (a body below `sleepThreshold` = 0.05 m/s for `B2_TIME_TO_SLEEP` =
0.5 s; the island sleeps when its sleepiest-last body qualifies; waking propagates through touching bodies at once;
islands are kept incrementally with lazy splitting, ~10x cheaper than per-step DFS) ([Catto, islands][islands],
[box2d solver.c][b2solver], [types.c][b2types]). The island rule exists because putting one body of a stack to
sleep in a normal engine lets the others sink into it ("Sleeping must be done per island, not per body").

Here a sleeping body is simply a FixedBody for a while: the CCD still treats its edges as obstacles, so nothing can
ever sink into it. That makes **per-body sleep safe for the zero-overlap guarantee**, and it matters because a
60-body pile is one island and some body in it is always above threshold (F7): island sleep put 0/60 bodies to sleep
in 15-20 s, both with a 3 px/s threshold on V and with 4 px/s on frame displacement. 
Per-body prototype (`sleep_patch.py`, MODE=body: sleep after 0.5 s below 3 px/s frame speed; woken by a grab or by an
awake neighbour moving faster than 8 px/s): **60/60 bodies asleep**, frame cost over the last 2 s **136 -> 63 ms**
(same-load background QoS, pile60, CPU), 0 overlaps, 0 zones. That gain comes only from `_filter_pairs` dropping
sleeper pairs: broad phases, predict and CCD still run over all points. Building pairs only for awake points/edges,
and skipping the substep entirely when nothing is awake, should bring a fully settled scene near the blank-world cost
(0.5 ms on a P-core per research 09). Wake test (drag the top body out of a sleeping pile): 56/60 asleep at 15 s and
the grab works once grab wakes the body (`World.grab` ignores W = 0 points, so the real code must wake first). The
run was cut before the "force-wake and see if anything falls" check finished, so frozen-in-air risk is unmeasured.

### F9. Relaxation and mass scaling (measured, negative/inconclusive)

- Capping the separation speed of slow contacts at 2|g|h (`patches.apply_relax`) did not calm the pile: frame-speed
  median ~1 px/s, p90 2-3 px/s at 12-15 s, same as the baseline. The residual rocking isn't overlap recovery.
- 100:1 alternating stack with rigid contacts + rigid CCD projection + UPP mass scaling: the bodies stay rigid after
  PBD, but the skin is crushed to 0.1 px and CCD needs **28 impact-zone fallbacks** (baseline: 0 zones, min gap 2.0 px).
  Projecting CCD velocities onto rigid motion fights the per-point impulse loop. The 10:1 stack is fine (min gap 2.65).
  Extreme mass ratios need a real fix (shock propagation by contact level, [Guendelman et al. 2003][guendelman];
  [Chen et al. 2023][chen23]) and are not worth it for this app's scenes yet.

## Applicability to this codebase

Design principle: keep points as the collision representation (CCD, broad phase, impact zones all stay as they
are) and make every operation that touches a RigidBody's points move the body rigidly with its generalized inverse
mass. Then a rigid body is exactly rigid at every substep end without a separate pose state, and the zero-overlap
argument is unchanged: CCD still checks the straight-line point paths PREV -> PREV + V h from an intersection-free
start, whatever produced V. A full pose-authoritative rewrite (x, theta, v, omega per body) adds little on top of
this and the XPBD rigid literature itself reports weaker stacking than what this solver already has
([Mueller et al. 2020][xpbd] sec. 7; [Seabra et al. 2023][survey]: PBRBD "limited in its handling of stable stacks").

A1. Rigid-aware, batched rope solve (solver.py `_pbd_substep`, entities.py `Rope`)
- `World._pack_ropes()` (called from `_rebuild`, `add_rope`, `add_chain`): per rope and end a kind (anchor / point /
  rigid-body point), global point id, body id, rest length, stiffness; greedy graph colouring so no colour touches a
  point or a rigid body twice (a chain needs 2 colours). Prototype: `patches._rope_setup`.
- `World._solve_ropes()` replaces `for r in self.ropes: r.solve_constraints()`: per colour, ~25 numpy calls compute
  all taut ropes, w = 1/M + (r x n)^2 / I for rigid ends (W for points, 0 for anchors), lambda, then point ends get
  `w * dp` and rigid ends translate dp/M and rotate (r x dp)/I about the com. Prototype: `patches.solve_ropes_batched`.
- Ropes whose class overrides `solve_constraints` stay on the per-rope path (same rule as `solo_bodies`).
  Hooks move by editing `Hook.pos` in place, so anchors are read every call (cheap).
- GPU backend: scenes with ropes already fall back to CPU PBD (`pbd_substep` returns False), so nothing breaks;
  a rope kernel is a later step. Looping: roped groups already wrap together; unaffected.

A2. Rigid-aware contacts (solver.py `_solve_contacts`, `_pack_constraints`; gpu_backend.py `_CONTACT_ACC`)
- `_pack_constraints` adds per-body rigid mask, M and I (from the rest shape, or from density / polygon if F5 is
  adopted). In `_solve_contacts`, for each touching pair: w_point = 1/M + (r_p x n)^2/I if the point's body is
  rigid, else W[p]; w_edge = 1/M + (r_c x n)^2/I if the edge's body is rigid (r_c = contact point - com), else the
  barycentric W[a](1-t)^2 + W[b]t^2. Rigid participants accumulate linear and angular positional impulses per body
  (bincount), then translate + rotate the body once per pass. Prototype `patches.solve_contacts_rigid`.
- Use one shared scale per contact, 1 / max(contact count over its participants), for both sides (research 10 F3):
  with it the prototype conserves momentum (soft-vs-rigid dP 0.31 -> 1.3e-13, off-centre rigid dL 0.16 -> 0.0018).
- Shape matching then becomes a final exact snap per substep (or stays for RigidBody(stiffness < 1)).
- Mass ratios: generalized-mass Jacobi contacts converge more slowly for a light body squeezed by heavy ones (F4
  table below: 100:1 alternating stack reaches the CCD barrier), so A2 needs either more iterations for those
  contacts, contact-level ordering (shock propagation, [Guendelman et al. 2003][guendelman]) or UPP mass scaling.

A3. Rigid-aware CCD (solver.py `_ccd_impulses`, `_impact_zones`)
- After each sequential impulse pass, project each rigid body's point velocities to its momentum-preserving rigid
  field, using the exact chord of the rotation (V = v + ((R(omega h) - I) r) / h) so the end pose is rigid. The
  existing redetect / fresh broad phase loop then confirms the projected paths, so the guarantee holds.
- In `_impact_zones`, pre-merge all points of each rigid body into one zone (initial union-find parents) and use
  the exact chord instead of v + omega x r. Linear interpolation between two congruent poses is a similarity
  transform ((1 - t) I + t R is a scaled rotation unless the body turns 180 deg in one substep), so members of a
  zone still cannot collide with each other and the fallback still terminates. Prototype `patches.apply_rigid_ccd`.

A4. General restitution (solver.py, new `World._restitution(h)` between `self.V[:] = ...` and `_limit_speed`)
- Save V at substep start (before predict), then for contacts touching at the end of PBD: target normal velocity
  -e * v_n(start) if the contact approached faster than 2 |g| h, else leave it. Rigid participants get body impulses
  (v += J/M, omega += (r x J)/I written back as point velocities). 2+ sweeps; e per pair = max (or min) of the
  bodies' `restitution`, FixedBody default 0. Velocity-only before CCD: guarantee untouched. Keep `_bounce` for the
  cradle until the general pass matches research 10's cradle transfer metric (99%). Prototype `patches.apply_restitution`.

A5. Friction coefficients (solver.py `_solve_contacts`, gpu_backend.py `_CONTACT_ACC`)
- `Body.friction` (None -> EnvConfig.friction) and optional `static_friction`; pair mu = average (XPBD 3.5).
  `fric = 1 where |tang| < mu_s depth else mu_k depth / |tang|` (UPP eq. 24). A few lines; same in the Metal kernel.

A6. Sleeping islands (solver.py `_step`, `_filter_pairs`, broad phase; entities.py)
- Sleeping body = temporarily fixed: W = 0, V = 0, `body_fixed` True, so `_filter_pairs` already drops
  sleeping-sleeping and sleeping-fixed pairs for contacts and CCD. CCD still sees its edges as static obstacles, so a
  wake that comes a frame late costs physics accuracy (an awake body treats it as a wall for <= 1 frame), never an
  overlap. Prototype `sleep_patch.py`.
- Needed changes beyond the prototype: guard the batched spring denominator (w_i + w_j = 0 -> skip), skip rigid
  groups that are asleep, wake on grab / editor edits / `World.add` / looping wraps / gravity change (gravity dial),
  ropes link islands, measure speed from frame displacement (not V, see F8). For real speedups the broad phase should
  build pairs only for awake points and awake edges (like `_ccd_new_pairs` does for grown cells), not filter after.

## Expected gains and risks

| change | gain (measured) | risk to zero-overlap | other risks |
|---|---|---|---|
| A1 rigid batched ropes | chain box stretch 6-9% -> 0; hanging chain jitter 2.6 -> 0 px/s; slack chain high-frequency motion 2.7x lower; 100:1 chain weight drop 6.7 -> 0.6 px | none (PBD only) | rope stretch moves into the ropes (0.5-1.2%); per-rope Python version is ~3x slower, so ship the batched one |
| A6 per-body sleep | settled pile60 136 -> 63 ms (2.2x) with filter-only; more with awake-only pair building | none: sleeper = fixed obstacle | frozen bodies in unstable poses; wake latency up to 1 frame; must wake on grab, edit, gravity change, looping |
| A2 rigid contacts (shared scale) | momentum exact; pile50 jitter 15 -> 2.7 px/s; ccd_demo zones 47 -> 0; 100:1 plank shape error 2.0 -> 0 | none (PBD only), but more CCD work when the skin is crushed | soft stack creep 0.5 -> 23.6 px/s; 100:1 squeeze crushed to CCD gap; needs the GPU kernel ported |
| A4 restitution pass | bouncing for any body (drop: 0 -> 202 px/s for a 243 px/s target) | none: V edit before CCD | energy gain if the threshold is wrong; cradle must keep 99% transfer |
| A5 friction coefficients | static/kinetic + per-body mu | none | none significant |
| F5 density/inertia | physically correct mass (chain box 6 vs 40x40 box 4 today) and 2-3x lower inertia | none | changes existing scenes; must be opt-in |

## Effort

A1 ~1 day (prototype exists: `patches.solve_ropes_batched`). A5 ~0.5 day. A4 ~1 day. A6 ~1.5-2 days (wake hooks,
awake-only broad phase). A2+A3 ~3 days incl. the Metal kernel, plus mass-ratio work. F5 ~0.5 day once A1/A2 exist.

## Build plan

Ranked; each can be prototyped in its own worktree. Measure with research 10's suite (`run_bench.py --code <worktree>`,
per-substep brute overlap checker; or `bench_patched.py` for monkeypatches), the scenes in
`/private/tmp/claude-501/research-07/common.py`, and `/private/tmp/claude-501/debug-agent/merged/chain_phys.py`.

1. **Rigid-aware batched rope solver** (A1). Change: `World._pack_ropes` + `World._solve_ropes` in solver.py replacing
   the per-rope loop in `_pbd_substep`; keep `Rope.solve_constraints` for subclasses and for `Rope` used standalone.
   Success: chain box stretch < 0.1% and hanging chain tail speed < 0.05 px/s; research 10 `chain.m100.weight_drop_end_px`
   < 1 px; pendulum/cradle unchanged; ms/frame for the 49-segment chain no worse than baseline (P-core, `nice -n 10`);
   0 overlaps. Risk low. Effort ~1 day.
2. **Per-body sleeping** (A6). Change: sleep state + timers in `World._step`, frame-displacement speed, wake on
   grab / `add` / editor / gravity / looping / fast awake neighbour; guard spring denominators; awake-only pair
   generation in `_candidates` and `_ccd_pairs`; skip the substep when nothing is awake. Success: settled pile200/500
   (research 09 scenes) at least 3x cheaper than today's settled cost; drag-out and drop-onto-pile tests wake
   correctly; force-wake after settling moves no body more than 1 px (no frozen-in-air bodies); 0 overlaps.
   Risk medium (behaviour). Effort ~2 days.
3. **Velocity pass: restitution + friction coefficients** (A4 + A5). Change: `World._restitution(h)` between the V
   update and `_limit_speed`, `Body.friction` / `static_friction`, per-pair combine rules in `_solve_contacts` and the
   Metal contact kernel. Success: drop rebound height within 10% of e^2 h for e in {0.3, 0.5, 0.8}; towers and pyramid
   still 0.0 px/s; cradle transfer >= 0.98; incline break-away at atan(mu_s) +- 1 deg and sliding at mu_k. Risk low.
   Effort ~1.5 days.
4. **Rigid-aware contacts with shared scale** (A2, optionally A3). Change: `_solve_contacts` generalized masses + per-body
   rigid accumulation + 1/max(count) scale; port to `_CONTACT_ACC`. Success: research 10 momentum metrics at 1e-12;
   pile50 jitter RMS < 5 px/s; rigid shape error 0; soft stack and 100:1 cases no worse than baseline (the prototype
   fails these today, so try a separate shared scale for soft points or more iterations). Risk medium-high.
   Effort ~3 days.

[survey]: https://arxiv.org/abs/2311.09327
[guendelman]: https://graphics.stanford.edu/papers/rigid_bodies-sig03/
[chen23]: https://research-explorer.ista.ac.at/record/15292
[islands]: https://box2d.org/posts/2023/10/simulation-islands/
[solver2d]: https://box2d.org/posts/2024/02/solver2d/
[b2solver]: https://github.com/erincatto/box2d/blob/main/src/solver.c
[tonge]: https://dx.doi.org/10.1145/2185520.2185601

[xpbd]: https://matthias-research.github.io/pages/publications/PBDBodies.pdf
[upp]: https://mmacklin.com/uppfrta_preprint.pdf
[b2types]: https://github.com/erincatto/box2d/blob/main/src/types.c
