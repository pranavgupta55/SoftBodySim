# 02 - CCD and contact accuracy

Scratch dir (all scripts, logs): `/private/tmp/claude-501/research-02/`. Repo untouched apart from this file.

Terms: **CCD** = continuous collision detection (checks the whole straight-line motion of a substep, not just its
end). **TOI** = time of impact. **GS** = Gauss-Seidel (apply corrections one at a time, each seeing the ones before).
**RIZ** = rigid impact zone (Provot 1997 / Bridson 2002 fail-safe: a cluster of colliding points moves as one rigid
body for the substep). **LCP** = linear complementarity problem (impulses >= 0, gaps >= 0, and for each contact one of
the two is zero). **PGS** = projected Gauss-Seidel (GS on an LCP, clamping each impulse at 0). **FN / FP** = false
negative (a real touch the test misses: the dangerous kind) / false positive (a reported hit that isn't one: costs
time, harmless). **E-core numbers**: every timing here was taken at background QoS on the efficiency cores while 9
other agents loaded the machine. They are ~15x slower than P-core numbers, so read them as ratios only.

## Summary

1. **CCD time goes into proving there are no hits, not into resolving them.** Across demo, cradle, diamonds,
   pile200 (rest and drag) and the 7 `ccd_test.py` stress drags, 85-100% of substeps have zero CCD hits. Impulses
   are <0.5% of CCD time, and impact zones never fired in my gentler scenes (0 in ~16,000 substeps). CCD itself is
   20-30% of a CPU frame, split about evenly between broad phase and narrow phase. The speed lever is a cheaper
   no-hit proof: 03's fat-box pair list, 04's GPU prefilter and 05's numba narrow phase. I confirmed the
   pair-list idea independently with a simpler distance certificate. Results were bit-identical, and CCD time fell
   2.7x (demo) and 3.9x (pile200) at rest. It needs 03's motion look-ahead to help while dragging.
2. **The float64 quadratic test in `_ccd_detect` has two real defects, and one ~40-line change fixes both and is 17%
   faster.**
   - (a) *Endpoint false negatives.* The roots come from the naive formula `(-c1 ± sqrt(disc)) / 2 c2`, then the
     test is an exact `0 <= s <= 1`. So a path that crosses an edge near its endpoint can be missed. Rates against a
     50-digit ground truth:
     - adversarial "nearly translating edge, path through an endpoint" cases: 317 of 749 real touches missed
       (463/999 in a rerun);
     - the same at world coordinates ~1000 px: 373/1042 missed;
     - plain endpoint cases: 7-14 per ~800 missed.

     On real scenes the naive and stable roots differ by at most 1.5e-10 px. So a miss needs a crossing within about
     1e-10 px of a vertex: rare, but it is a hole in a guarantee that should have none.
   - (b) *Collapsing-edge false positives.* When an edge collapses to near zero length mid-substep, far points
     produce spurious hits (56-78 per 1,500-2,000 such cases). The pre-reject keeps those far pairs on purpose
     ("so the results stay the same"). A spurious hit the impulses can't clear ends in an impact-zone fallback.
   - *The fix.* Compute the roots with the cancellation-free formula. Then accept a root only if the point is within
     `TAU = 1e-6 px` of the *clamped* segment at that time, instead of testing `0 <= s <= 1` exactly.
     - Results: 0 false negatives in 16,500 adversarial cases, 0 far false positives. All remaining FPs are within
       1e-6 px of an edge, so they are harmless.
     - It gives the same hit sets as the current code on 127 real snapshots (96 hits) and at every call of a whipped
       pile200 run. It is not bit-identical: the roots differ in the last bits.
     - It is 17% faster, because the collapsing-edge exception and its per-edge pass go away.
3. **The impact-zone fallbacks that do fire (47/2,400 substeps in research-10's aggressive `ccd_demo`) are one
   chronic GS ping-pong, not missed physics.**
   - Logging every pass shows one point (point 8 of a crushed soft 12-gon) alternating between the two edges that
     meet at vertex 10, (9,10) and (10,11). It sits 0.53 px from vertex 10, in a ~2 degree wedge.
   - All 8 passes show 1 hit each, alternating pairs, for ~47 consecutive substeps.
   - Sequential impulses converge only geometrically here: 16 passes leave 4 zones, 32 passes leave 2.
   - This is Bridson's own "node sandwiched between two approaching triangles" case.
   - A Harmon et al. (2008)-style **inelastic projection fallback** resolves 41 of the 47 with 0 overlaps:
     - per impact zone, one small LCP over all of the zone's accumulated constraints, solved by PGS;
     - it changes only normal velocities, so sliding is kept;
     - it removes 0 kinetic energy instead of the RIZ's 734k units;
     - RIZ stays as the last resort.
   - 07 (rigid-aware contacts) and 10 (momentum-conserving contact scale) remove the same 47 zones from the PBD side,
     by not crushing the body in the first place. The two fixes stack.
4. The rest is low priority, and the reasons are below:
   - friction and split impulses in CCD (hits are rare, so the effect is small);
   - IPC-style barriers (Incremental Potential Contact) and global conservative advancement (they need a global
     step size or a Newton solve, so they don't fit a real-time Python loop);
   - exact root-parity CCD (Brochu 2012) and Bernstein sign classification (Tang 2014) (in 2D the point-edge test is
     only a quadratic, and the TAU-tolerance version already has no false negatives in testing);
   - edge-edge CCD (unnecessary in 2D).

## How the CCD works today (solver.py)

Per substep after PBD: `V = (P - PREV)/h`, terminal clamp, then `_resolve_ccd(h)`:

1. `_ccd_pairs`: swept-box grid broad phase (24 px cells; edge boxes grown 1 px), `_filter_pairs(contact=False)`.
2. `_ccd_detect`: swept-box pre-reject, which keeps pairs whose edge nearly collapses. Then the 2D point vs moving
   segment test: `cross(e(t), q(t)) = c2 t^2 + c1 t + c0 = 0` for t in [0, 1]. It is a hit if the projection
   parameter `s` at the root is in [0, 1]. There is also an end-of-substep proximity test (`< ccd_gap = 0.01 px`).
3. Up to `ccd_passes = 8` passes of sequential impulses (`_ccd_impulses`).
   - Earliest hit first; the normal is the start-of-substep closest-point direction; the target end gap is
     `ccd_push = 0.25 px`.
   - Each pass is followed by an incremental re-detect (`_ccd_redetect`).
   - A clean result is confirmed by checking only the pairs a fresh broad phase could add (`_ccd_new_pairs`).
4. If still colliding: `_impact_zones`.
   - Zones move by v_com + omega x r. This linearized rotation is a similarity transform, so it can't make a zone's
     own members collide.
   - A zone with a fixed point gets V = 0.
   - Last resort: every movable point stops, which is the valid start state.

**2D needs only point-edge CCD.** Two closed polygons moving continuously from a disjoint start can only begin to
intersect when a vertex touches an edge (or a vertex). So the 3D edge-edge test is unnecessary. `_ccd_detect` covers
vertex-vertex contact as a point hitting an edge endpoint. That endpoint case is exactly where the rounding problem
of F3 lives.

## Findings

### F1. Where CCD time goes (measured, CPU backend)

`stats.py` and `zones.py` count hits per pass, passes per substep, zone fallbacks and the time split.

| scene | substeps | substeps with hits | hits | passes > 1 | zone fallbacks | CCD share of frame | impulses share of CCD |
|---|---|---|---|---|---|---|---|
| demo (rest + drop, 300 fr) | 1200 | 0 | 0 | 0 | 0 | 20% | 0 |
| demo drag (240 fr) | 960 | 0 | 0 | 0 | 0 | 19% | 0 |
| cradle (300 fr) | 2400 | 0 | 0 | 0 | 0 | 18% | 0 |
| diamonds (300 fr) | 1200 | 30 | 30 | 0 | 0 | 34% | 0.06% |
| pile200 drop (300 fr) | 1200 | 176 | 669 | 0 | 0 | 31% | 0.4% |
| pile200 drag (240 fr) | 960 | 53 | 62 | 0 | 0 | 28% | 0.05% |
| `ccd_test.py` 5 targeted drags + 2 random whips (640x360) | 12,960 | - | - | - | **0** | - | - |
| research-10 `ccd_demo` (1354x982, regrab every 45 frames, 10 s) | 2400 | 145 | - | 69 | **47** | - | - |

- Detection dominates. The broad phase is about half of CCD time (`_ccd_cells` + `_cell_pairs` + `_filter_pairs`)
  and `_ccd_detect` the other half. 09 and 03 measured the same: only 5-8% of CCD grid pairs survive the swept-box
  pre-reject, and about 0 of those hit.
- The time per impulse doesn't matter (05: "not hot in any benchmark scene"). What can matter is the *number of
  passes*, through re-detection, and only in the ping-pong case of F4.

### F2. Pair-list reuse with a displacement certificate (confirms 03 F3 from the CCD side)

Prototype: `verlet_ccd.py`, which replaces `_resolve_ccd`.

- **The list.** At a start-of-substep state REF (intersection-free), store every (point, edge) pair whose point lies
  in the edge's AABB grown by SKIN = 6 px.
- **The certificate.** A pair not in the list is > SKIN apart at REF. Say every point stays within R of its REF
  position along the whole straight-line substep. A ball is convex, so checking PREV and PREV + D is enough, and
  every point of a segment moves at most as far as its farther endpoint. Then the pair's distance stays
  > SKIN - 2R. With `2R < SKIN - ccd_gap`, the pair can neither touch nor end within `ccd_gap`.
- **Consequence.** `_ccd_detect` over the list gives the same hits as over the full broad phase, minus the spurious
  collapsing-edge hits. The certificate is re-checked after impulses, which replaces the fresh-broad-phase
  confirmation.

Result (E-core, from settled states):

| scene | stock CCD ms/frame | list CCD ms/frame | rebuilds / substeps | end-state hash |
|---|---|---|---|---|
| demo rest | 13.5 | 4.9 (2.7x) | 4 / 800 | identical |
| pile200 rest | 225.7 | 58.2 (3.9x) | 57 / 480 | identical |
| pile200 drag | 140.9 | 126.8 | 348 / 480 (31 fall back to stock) | identical |

- Without a motion look-ahead, the list is rebuilt almost every substep while dragging.
- 03's version fixes that and shares the list with the contact broad phase. It stretches fat boxes 4 substeps
  along D (Box2D's `b2_aabbMultiplier` idea) and measures -10 to -43% of whole-frame physics.
- **03 owns this build item.** My data adds two things: the bit-identical end states, and a second, distance-based
  completeness argument a reviewer can check.
- Literature: classic Verlet / skin neighbour lists ([LAMMPS neigh_modify](https://docs.lammps.org/neigh_modify.html)).
  GPU cloth codes skip CCD altogether by bounding per-vertex displacement against proximity distances: Wu et al.
  2020, ["A Safe and Fast Repulsion Method for GPU-based Cloth Self Collisions"](https://doi.org/10.1145/3430025)
  ("vertex displacement constraints"), and the CCD-free step size of Wang et al. 2022,
  [arXiv 2211.04045](https://arxiv.org/abs/2211.04045).

### F3. Float64 robustness of the quadratic root test (measured against a 50-digit ground truth)

`robust.py` builds adversarial point/edge trajectory families as float64 inputs. It classifies each case with mpmath
at 50 digits (`touch` = distance < 1e-30 at a root in [0, 1]). It then runs `World._ccd_detect` itself, through a
stub World, and the candidate `variants.robust`.

| family (1,500-2,000 cases each) | real touches | current FN | current far FP | robust FN | robust far FP |
|---|---|---|---|---|---|
| random | 51-67 | 0 | 0 | 0 | 0 |
| near-linear (edge nearly translating) | 52-81 | 0 | 0 | 0 | 0 |
| path through an endpoint (±1e-15..1e-9 px) | 800-1110 | 3-9 | 0 | 0 | 0 |
| same, resting-scale motion (0.05 px) | 749-984 | 10-14 | 0 | 0 | 0 |
| near-linear + through an endpoint | 749-999 | **317-463** | 0 | **0** | 0 |
| same at world coords ~1000 px | 1042 | **373** | 0 | **0** | 0 |
| grazing (tangential touch) | 1351-1795 | 0 | 0 | 0 | 0 |
| edge collapses mid-substep, far point | 17-27 | 0 | **56-78** | 0 | **0** |
| resting (point 0.005-1 px from edge, tiny motion) | 67-83 | 0 | 0 | 0 | 0 |

"Far FP" means a reported hit with the true minimum distance > 1e-3 px. The robust variant's other FPs are all within
`TAU = 1e-6 px` of the segment, by construction, and get a harmless push.

Why the current code misses:

- **Cancellation.**
  - `(-c1 + sqrt(c1^2 - 4 c2 c0)) / (2 c2)` loses most of its digits when `|c2 c0| << c1^2`. That is exactly an edge
    that barely rotates or stretches: rigid bodies, resting bodies, translation.
  - The root error then makes `s` land just outside [0, 1] for a crossing near an endpoint.
  - Fix: `q = -(c1 + sign(c1) sqrt(disc)) / 2`, roots `q / c2` and `c0 / q`. This is the standard stable form; see
    Goldberg's analysis revisited in [arXiv 2012.02492](https://arxiv.org/pdf/2012.02492).
- **Exact interval test at the root.**
  - The code tests `0 <= s <= 1` exactly. Even with perfect roots, a crossing exactly through a shared vertex can get
    `s = 1 + 1e-16` on one edge and `-1e-16` on the next, and be missed by both.
  - Bridson et al. 2002 avoid this by a proximity test at the root time: "if the elements are closer than a small
    rounding error tolerance (1e-6 m ...) we register a collision"
    ([paper](https://www.cs.ubc.ca/~rbridson/docs/cloth2002.pdf), sec. 6).
  - Fix: clamp `s`, then accept if `|q(r) - s e(r)| <= TAU`. This also rejects the collapsing-edge spurious roots,
    where the far point is tens of px from the degenerate segment. So the pre-reject no longer needs its
    `collapses` exception: one fewer pass over all edges (`t_min`, `shortest`), and that is where the 17% comes from.
- **How big is the hole in practice?** On real pair sets (`root_err.py`), the naive and stable roots put the contact
  point at most 1.5e-10 px apart (pile200: 79k roots, p99.99 2.7e-11 px; demo 1.2e-12 px). A miss therefore needs a
  crossing within about 1e-10 px of a vertex.
  - Not observed in any run.
  - More likely than "uniformly random" suggests, because axis-aligned boxes put vertices exactly on each other's
    edge lines.
- **Literature.** Wang et al. 2021 ([arXiv 2009.13349](https://arxiv.org/abs/2009.13349)) benchmark 3D CCD
  methods against a Mathematica ground truth:
  - The classic floating-point root finder (Provot-style, the one Bridson uses) misses 0.6-1% of hand-crafted
    degenerate queries and 0.006-0.03% of queries taken from simulations.
  - Only interval-based methods (their Tight Inclusion, interval root finding, TightCCD) have zero misses.
  - Exact alternatives: root parity, [Brochu, Edwards, Bridson 2012](https://www.cs.ubc.ca/~rbridson/docs/brochu-siggraph2012-ccd.pdf),
    which needs exact predicates and still has rare parity misses; Bernstein sign classification (Tang et al. 2014,
    DOI 10.1145/2661229.2661237), which gives many false positives.
  - In 2D the problem is one quadratic per pair, so the cheap TAU-tolerance version is enough. Tight Inclusion's
    "minimum separation" CCD is the 3D analogue of this repo's `ccd_gap`; its authors find that separations of up
    to 1e-3 have negligible effect, and that larger ones start to cause trouble.

Performance and equality on real data (`detect_bench.py`): 127 snapshots of `_ccd_detect` inputs from a pile200
drop. Current 9.87 ms/call, robust 8.22 ms/call (E-core). The hit sets and times are identical in all 127 (96 hits).
End-to-end checker runs: see F5.

### F4. Why the impact-zone fallback fires, and a fallback that keeps sliding

Diagnosis (`zones.py diag`, `pingpong.py` on research-10's `ccd_demo`: default preset at 1354x982, random whip from
1 s, 10 s, CPU):

- All 47 fallbacks are the **same configuration** over ~47 consecutive substeps.
  - Point 8 of a soft 12-gon (body 9) sits 0.019 px from edge (10,11), near its start (s = 0.013), and 0.53 px from
    vertex 10.
  - The body is badly crushed: edge (10,11) is stretched to 37.8 px against a rest length of ~9.3 px.
  - Every pass has exactly one hit, alternating between edges (9,10) and (10,11). The hit time creeps later each
    pass (0.009, 0.275, 0.06, 0.405, ... 0.875).
- **Cause.**
  - The point is wedged in a ~2 degree angle between the two edges meeting at vertex 10, so their normals are
    nearly opposite.
  - Each sequential impulse pushes it into the other edge. The convergence ratio is ~cos^2 of the wedge angle.
  - The code's corner rule (use the corner -> point direction near a corner) doesn't trigger, because the closest
    point is interior to the edge (s = 0.013), not the vertex.
  - Bridson 2002 names this case: "a node sandwiched between two approaching triangles ... resolved immediately if we
    apply impulses in parallel (but can iterate for a long time if they are applied sequentially)".
- **Why PBD contacts don't help.** `_filter_pairs(contact=True)` drops a point's neighbours' edges for self-colliding
  bodies, so point 8 vs edge (9,10) is seen *only* by CCD. The PBD springs keep pushing the fold.
- **Consequence today.** The 3 points are rigidified every substep for ~0.8 s (RIZ removes 734k units of kinetic
  energy over the run). It is visible as a stuck crease while dragging, never as an overlap.

Fallback variants, all with the research-10 checker on every substep (`zone_variants.py`):

| variant | overlaps | zone fallbacks | substeps with hits | KE removed by fallback |
|---|---|---|---|---|
| base (8 passes, RIZ) | 0 | 47 | 145 | 734,457 |
| `ccd_passes = 16` | 0 | 4 | 125 | 93,046 |
| `ccd_passes = 32` | 0 | 2 | 116 | 29,840 |
| self-verifying impulses (double the impulse while its own pair still hits) | 0 | 47 | 145 | 734,457 (no effect: each impulse does clear its own pair, the *other* edge then hits) |
| **Harmon-style inelastic projection, then RIZ** | 0 | **6** | 115 | **-34,069** (no loss) |

How the inelastic projection works (Harmon, Vouga, Tamstorf, Grinspun 2008,
[RTSC](https://www.cs.columbia.edu/cg/pdfs/RTSC.pdf)):

- Each hit k becomes a constraint row `G_k V >= b_k`.
  - `G_k` is +n on the point and -(1-s) n, -s n on the edge's endpoints.
  - `b_k = (ccd_push - gap0_k) / h`, i.e. the same end-gap target as `_ccd_impulses`.
- Constraints accumulate over rounds. Zones = connected components of the constraint stencils.
- Each zone is reset to its pre-CCD velocities (Harmon's Algorithm 1, line 7). Then it is projected onto
  `min ||V' - V||_M` subject to all its rows, by PGS on the dual: the Delassus matrix `G M^-1 G^T` is k x k with k
  small.
- Then a fresh full `_ccd_detect`. New hits are added and the round repeats. After 8 rounds the old RIZ runs, so
  termination and the guarantee are unchanged.
- **Why it keeps sliding.** The projection changes velocity only along the constraint normals. Tangential sliding
  survives, unlike RIZ, which "eliminates all relative tangential velocity, unphysically dissipating energy"
  (Harmon et al.).
- **Departures from the paper.**
  - Harmon solves the equality-constrained version with a linear solve and warns about near-singular systems. PGS on
    the LCP has neither the pulling (negative impulse) problem nor the singularity problem.
  - The paper's Coulomb friction approximation (`dv = min(mu |dv_n|, |v_slide|)` per zone) can be added the same
    way.
  - Harmon reports at most 15% extra cost over RIZ in cloth. Here the fallback runs about 2% of substeps, and only in
    `ccd_demo`.
- **The 6 leftover zones** are the same wedge in substeps where 8 rounds aren't enough. A 16-round variant and one
  with a smaller projection target (0.2 x `ccd_push`) are in F5.
- **Combination with 07 A3** ("rigid-aware CCD"). 07 found that projecting CCD velocities onto rigid motion *after*
  each impulse pass fights the per-point loop: 28 zones in a 100:1 stack. The projection is the natural place to do
  it properly. Use rigid bodies' 3 degrees of freedom (vx, vy, omega) as the unknowns, i.e. `V = J u`; then the LCP
  is in u and rigid bodies stay rigid by construction.

### F5. End-to-end checker runs of the candidates

All runs: CPU, research-10's independent checker on every substep, random whip from 1 s (`zones.py`).

| variant | scene | overlaps / checks | min gap px | zones | substeps with hits | notes |
|---|---|---|---|---|---|---|
| base | ccd_demo (10 s) | 0 / 2400 | 0.0213 | 47 | 145 | |
| robust detect | ccd_demo | 0 / 2400 | 0.0213 | 47 | 145 | every statistic identical to base |
| robust + projection | ccd_demo | 0 / 2400 | 0.0213 | **6** | 115 | |
| projection, 16 rounds | ccd_demo | 0 / 2400 | 0.0213 | 6 | 115 | the round count is not the limit |
| projection, target 0.2 x push | ccd_demo | 0 / 2400 | 0.0213 | 8 | 99 | a smaller target doesn't help |
| base | random walls (10 s) | 0 / 2400 | 2.07 | 0 | 48 | |
| robust / robust + projection | random walls | 0 / 2400 | 2.07 | 0 | 48 | identical to base |
| base | pile200 (4 s) | 0 / 960 | 0.0116 | 0 | 641 | |
| robust + projection | pile200 | 0 / 960 | 0.0248 | 0 | 641 | trajectory diverges (see below) |

- **Why pile200 diverges.** A side-by-side check over the same run (`diverge.py`) found the robust and current hit
  sets identical at every call. The divergence comes from the stable roots differing from the naive ones in the last
  bits, which feeds through t and s into the impulses, and a whipped pile is chaotic.
  - So the robust detect makes the same *decisions*, but it is not bit-identical.
  - Judge it on research-10's aggregate metrics, as 10 already prescribes for 05's FMA case.
- **The 6 leftover projection failures** don't respond to more rounds or a smaller target. Likely causes:
  - the fixed (s, n) linearization of a constraint whose edge rotates fast inside the wedge;
  - resetting only zone members to their pre-CCD velocities while neighbours keep their impulse-pass velocities.

  Worth a day of tuning in the build, e.g. re-linearizing s and n each round, or evaluating normals at the TOI as
  Harmon does ("the gradients we use are evaluated at the time of collision"). RIZ still catches them.

### F7. What frame-level CCD (01, 04 F7) would face, measured from the CCD side

01 and 04 propose running CCD once per frame on the start -> end chord instead of per substep. The CCD-side risk:
the chord can cut corners that the 4 substep paths went around, which would add hits. `chord.py` runs the normal sim
and, after every frame, tests the chord of the already-valid frame motion with the same `_ccd_detect` (fresh broad
phase):

| scene | frames | per-substep CCD hits (PBD proposals) | chord hits of the valid frame motion | p99 max point displacement / frame |
|---|---|---|---|---|
| default, no drag | 600 | 0 | 0 | 11.7 px |
| default, whip | 600 | 157 | 1 (1 frame) | 17.1 px |
| diamonds | 92 (looping frames skipped) | 79 | 0 | 16.7 px |
| pile200, whip | 240 | 1479 | 4 (3 frames) | 16.8 px |

- Chord corner-cutting is negligible: 0-4 hits per run. The chord's deviation from the substep path, R(1 - cos(θ/2))
  for spin θ per frame, stays far below the 3 px contact skin unless bodies spin faster than ~40 rad/s.
- So frame-level CCD should face about the per-substep hit count, or fewer, since PBD gets 4 substeps to undo a
  proposal before anything is checked.
- With 4x longer sweeps, though, each hit is resolved with a 4x longer h. GS ping-pong cases (F4) would then freeze
  4 substeps' worth of motion at once. That makes the projection fallback (and the robust endpoint test, with 4x
  longer paths) more valuable once 01 lands.

### F6. Smaller items (literature + reasoning; not prototyped)

- **Friction in CCD impulses.**
  - `_ccd_impulses` is frictionless (07 noted this too). Bridson 2002 sec. 7.3 applies Coulomb friction with each
    normal impulse: `v_T' = max(1 - mu * dv_N / |v_T|, 0) v_T`. Adding it to `_ccd_impulses`, weighted like the
    normal impulse (1, -(1-s), -s), is about 6 lines.
  - It is safe for the guarantee: velocity edits are re-verified by the same redetect loop, and the fallback still
    terminates.
  - Effect: only on the ~0.1-0.6 hits per substep in busy scenes, i.e. fast sliding impacts that PBD missed. Low
    priority.
- **Split impulses for `ccd_push`.**
  - The impulse aims for a 0.25 px end gap within one substep. That can leave up to `0.25 px / h = 60 px/s` of
    separating velocity, which carries into the next substep: a small artificial restitution.
  - Split it: keep the inelastic part `max(-v_rel.n, 0)` in V; apply the remaining position-correction part only to
    this substep's displacement (`PREV + V h`), then remove it from V afterwards. This is Catto's / Bullet's split
    impulse.
  - Guarantee untouched: only the next substep's starting velocity changes.
  - 07 measured a 300 px drop rebounding only 0.09 px, so the effect is small today. It would matter more with
    frame-level CCD (01's proposal), where push / h is 4x smaller but hits are more frequent.
- **Reducing GS passes.**
  - Passes are not a cost problem (F1). Most substeps with hits finish in 1 pass (76 of 145 in `ccd_demo`).
  - Failures are ping-pong (F4), which the projection fixes better than more passes or Jacobi averaging.
  - Raising `ccd_passes` to 16 is a cheap stopgap: 47 to 4 zones, a few extra redetects only in bad substeps.
- **IPC-lite / barriers** ([Li et al. 2020, IPC](https://cims.nyu.edu/gcl/papers/2020-IPC-A-Technical.pdf);
  [Codimensional IPC](https://arxiv.org/abs/2012.04457)).
  - IPC guarantees intersection-freedom with log barriers inside a Newton solve, using a CCD-filtered line search.
  - The line search gives *one global step size*, so one tight contact slows motion everywhere. Even 2026 work is
    still fixing this with per-subdomain step sizes ([arXiv 2604.19892](https://arxiv.org/abs/2604.19892)).
  - It needs Hessian assembly and linear solves per substep. That doesn't fit a real-time numpy/PBD loop, and this
    CCD already guarantees intersection-freedom more cheaply.
  - Not recommended.
- **Conservative advancement / TOI stepping** (Mirtich; Catto,
  [GDC 2013](https://box2d.org/files/ErinCatto_ContinuousCollision_GDC2013.pdf); Box2D v3 uses
  "speculative + TOI", [release notes](https://box2d.org/posts/2024/08/releasing-box2d-3.0)).
  - Advancing everything to the first TOI is a *global* freeze, i.e. a worse version of the current last resort.
  - Speculative contacts are what the 3 px PBD contact radius already does.
  - Not recommended.
- **Thickness.**
  - Today: `contact_radius = 3 px` (PBD, soft), `ccd_gap = 0.01 px` (hard floor), `ccd_push = 0.25 px` (impulse
    target).
  - 10 measured a min gap of 0.0116-0.02 px under dragging, so the hard floor is reached but never broken.
  - Keep the 3-tier scheme. Bridson's ratio (rounding tolerance 1000x below thickness) holds here (0.01 vs 3 px).
  - With the TAU change, the rounding tolerance (1e-6 px) sits 4 orders below `ccd_gap`, as Bridson recommends.

## Applicability to this codebase (what would change)

1. `solver.py` `_ccd_detect` (F3):
   - In the pre-reject, delete `t_min`, `shortest`, `collapses` and `| collapses[eids]`.
   - Replace `r1, r2` with the stable pair `q / c2, c0 / q`, where `q = -0.5 (c1 + copysign(sqrt(disc), c1))`. Keep
     the linear fallback when `c2 == 0`.
   - In the root loop, replace `(s >= 0) & (s <= 1)` with "clamp s, accept if `|q - s e|^2 <= TAU^2`".
   - Add `EnvConfig.ccd_tau = 1e-6`.
   - Update the docstring, which currently explains the collapsing-edge exception.
   - 05's numba port of the narrow phase must port the same math. Reference code: `variants.robust` in the scratch
     dir.
2. `solver.py` `_resolve_ccd` / `_impact_zones` (F4):
   - Save the pre-CCD velocities at the top of `_resolve_ccd`.
   - Add an `_inelastic_zones(h)` step between the impulse passes and `_impact_zones`: accumulate constraints,
     union-find the zones, PGS per zone, full `_ccd_detect`, up to N rounds.
   - Keep `_impact_zones` unchanged as the last resort.
   - `debug_overlay.StepRecorder` gets a counter for projection rounds.
   - Reference: `make_harmon` in `zone_variants.py`, ~90 lines.
3. Optional (F6): friction in `_ccd_impulses`; split impulse; `ccd_passes` 8 -> 16 as an interim knob.

## Expected gains and risks

- **Robust detect.**
  - Gains:
    - closes the endpoint-rounding hole (0 FN vs up to 46% on the adversarial family);
    - removes spurious collapsing-edge hits, a source of pointless impulse passes and zone fallbacks;
    - -17% `_ccd_detect` time;
    - identical results on real snapshots.
  - Risk: TAU too small. Real root errors are <= 1.5e-10 px, so 1e-6 has 4 orders of margin. FPs within 1e-6 px get
    a 0.25 px push, which is harmless and very rare.
  - Determinism: the hit sets matched the old test at every call in every run. The stable roots still differ in the
    last bits, so chaotic scenes (whipped pile200) end in different but equally valid states (F5). Not
    bit-identical.
- **Inelastic projection fallback.**
  - Gains:
    - `ccd_demo` zones 47 -> 6, with no fallback energy loss;
    - no locked creases while dragging;
    - it can host 07's rigid-aware CCD in reduced coordinates.
  - Risk to the guarantee: none. It only proposes velocities, `_ccd_detect` re-verifies them, and RIZ remains the
    terminating fallback.
  - Risks to behaviour: PGS can add a little energy (the measured -34k is a gain). The cost is a few ms in a bad
    substep, in Python loops; port to numba later (05) if it ever shows up.
- **Pair-list reuse** (03's item): 2.7-3.9x less CCD time at rest, same guarantee argument (F2).

## Effort

| item | lines | days |
|---|---|---|
| robust `_ccd_detect` + EnvConfig knob + docstring + tests | ~40 | 0.5 |
| inelastic projection fallback | ~100 | 1-2 (incl. tuning rounds / target, debug overlay hook) |
| CCD friction + split impulse | ~25 | 0.5 + accuracy-suite tuning |
| rigid-aware projection (with 07) | ~60 more | 1-2 |

## Cross-links and contradictions

- **10.** My earlier "zones never fire" was true only for gentler drags. 10's `ccd_demo` fires 47. F4 shows all 47
  are one chronic wedge configuration in a crushed soft 12-gon, which explains why 10's variant C and 07's A2 (both
  PBD-side changes that crush bodies less) take them to 0.
- **03 / 04.** Pair-list reuse (03) and the GPU-verified box set (04 F3) are the speed items for CCD. F2
  independently confirms 03, with bit-identical end states. 04 notes that a global max-displacement rule is nearly
  useless for *contacts* in moving piles. That matches my drag result (348 rebuilds / 480) and is why 03 adds motion
  look-ahead.
- **05.** The robust detect must be ported into 05's numba narrow phase together, or the two will disagree.
  04 measured a compressed 1000-body pile with 66-127 hits per substep, where `_ccd_impulses` and the redetects
  dominate (25.6 of 44 ms). There 05's numba impulses are the fix. The projection fallback is also a Python loop and
  would need the same port if such states became common.
- **06.** 06 notes that un-folding a soft body may require an outline point to cross an outline edge, which CCD
  forbids. F4 is that situation in the wild. Its cause is that `_filter_pairs(contact=True)` leaves neighbour edges
  to CCD alone. A soft-body-side fix would let PBD see point i vs edges (i+1, i+2) / (i-2, i-1) when the angle at the
  shared vertex is sharp. That is 06's area.
- **07.** A3's rigid projection fights per-point impulses (28 zones in a 100:1 stack). Doing the rigid projection
  inside the inelastic-projection LCP in reduced coordinates is the clean version (F4).
- **01.** F7: chord corner-cutting is negligible, and the projection fallback matters more under frame-level CCD.

## Build plan

Ranked. All four can be prototyped in parallel worktrees. Gate every one with research-10's suite:

```
taskpolicy -b .venv/bin/python3 /private/tmp/claude-501/research-10/bench/run_bench.py --code <worktree> --skip-perf --jobs 6
compare.py base_acc.json new.json
```

Rules: no FAIL; `overlap_substeps` = 0 everywhere. My scripts in `/private/tmp/claude-501/research-02/` (`zones.py`,
`robust.py`, `detect_bench.py`) take `REPO=<worktree>` (or edit the path).

1. **Robust float64 point-edge test** (highest value per line of code).
   - *Change:* `solver.py` `_ccd_detect` as in "Applicability" item 1, plus `EnvConfig.ccd_tau = 1e-6`. Reference:
     `variants.robust`.
   - *Success:*
     - `robust.py` shows 0 FN across all families (seeds 0, 1, 7; N >= 2000) and 0 far FPs;
     - research-10 suite: 0 overlaps, aggregate metrics within noise. Bit-identical end states are *not* expected
       (last-bit root differences, F5);
     - `_ccd_detect` >= 10% faster (`detect_bench.py`).
   - *Expected gain:* closes a real (if tiny) hole in the guarantee; removes spurious collapsing-edge hits; -17%
     narrow phase.
   - *Risk:* very low. *Effort:* 0.5 day. Coordinate with whoever builds 05's numba narrow phase.
2. **Inelastic-projection fallback before RIZ** (Harmon 2008).
   - *Change:* new `World._inelastic_zones(h)` between the impulse passes and `_impact_zones`; save pre-CCD
     velocities in `_resolve_ccd`. Reference: `make_harmon` in `zone_variants.py`. Then try TOI-time normals and
     re-linearizing s and n each round, to close the last 6 failures.
   - *Success:*
     - `ccd_demo` zone fallbacks 47 -> <= 6 (stretch: 0) with no overlaps;
     - fallback KE removal ~0;
     - `ccd_demo` wall time not worse;
     - accuracy suite unchanged, since the fallback is inactive elsewhere.
   - *Expected gain:* no locked creases or frozen sliding when the fallback triggers. It also becomes the host for
     07's rigid-aware CCD.
   - *Risk:* low (RIZ remains the terminating last resort). *Effort:* 1-2 days.
3. **Friction + split impulse in `_ccd_impulses`** (Bridson 2002 sec. 7.3; split impulse).
   - *Change:* about 25 lines in `_ccd_impulses`; the split part also touches the end of the substep loop in
     `_step`.
   - *Success:* research-10 `collisions` (dP, dL), `stacks` jitter and `cradle` transfer not worse; a new
     "fast slide into a wall" check shows tangential speed reduced by mu x dv_n; 0 overlaps.
   - *Expected gain:* small today (rare hits), larger under frame-level CCD.
   - *Risk:* low (velocity-only, re-verified). *Effort:* 0.5-1 day.
4. **Pair-list reuse for CCD.** Build 03's fat-box list rather than my CCD-only certificate (F2).
   - *Success:* CCD ms/frame down >= 2.5x at rest (demo, pile200); 0 overlaps; bit-identical end states in rest
     scenes.
   - Listed so the CCD owner knows it is the cost lever, not something to build twice. Effort is in 03.

Not recommended: IPC barriers, global conservative advancement, exact root-parity CCD, edge-edge CCD (F6).

## Sources

- Bridson, Fedkiw, Anderson 2002, Robust treatment of collisions, contact and friction for cloth animation: https://www.cs.ubc.ca/~rbridson/docs/cloth2002.pdf
- Harmon, Vouga, Tamstorf, Grinspun 2008, Robust treatment of simultaneous collisions: https://www.cs.columbia.edu/cg/pdfs/RTSC.pdf
- Wang, Ferguson, Schneider, Jiang, Attene, Panozzo 2021, A large scale benchmark and an inclusion-based algorithm for CCD: https://arxiv.org/abs/2009.13349
- Brochu, Edwards, Bridson 2012, Efficient geometrically exact CCD: https://www.cs.ubc.ca/~rbridson/docs/brochu-siggraph2012-ccd.pdf
- Tang, Tong, Wang, Manocha 2014, Fast and exact CCD with Bernstein sign classification, TOG 33(6) 186, DOI 10.1145/2661229.2661237; TightCCD 2015: https://diglib.eg.org/handle/10.1111/cgf12767
- Stable quadratic roots (Goldberg revisited): https://arxiv.org/pdf/2012.02492
- Li et al. 2020, Incremental Potential Contact: https://cims.nyu.edu/gcl/papers/2020-IPC-A-Technical.pdf ; Codimensional IPC: https://arxiv.org/abs/2012.04457 ; per-subdomain CCD step sizes: https://arxiv.org/abs/2604.19892
- Wu, Wu, Yang, Wang 2020, A safe and fast repulsion method for GPU-based cloth self collisions: https://doi.org/10.1145/3430025 ; Wang et al. 2022, Fast GPU-based two-way continuous collision handling: https://arxiv.org/abs/2211.04045
- Catto, Continuous collision (GDC 2013): https://box2d.org/files/ErinCatto_ContinuousCollision_GDC2013.pdf ; Box2D 3.0 release notes: https://box2d.org/posts/2024/08/releasing-box2d-3.0
- LAMMPS neighbor lists (skin): https://docs.lammps.org/neigh_modify.html
