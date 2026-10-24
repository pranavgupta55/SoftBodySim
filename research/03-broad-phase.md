# 03: Broad phase (contacts + swept CCD) on Apple Silicon

Status: cycle 1 complete. Scratch code and logs:
`/private/tmp/claude-501/research-03/` (`micro.py` work funnel + cell sweep, `verlet.py` fat-list prototype,
`nb_grid.py` numba exact grid, `exact.py` per-call exact broad phase, `gpufat.py` GPU-backend hookup, `bench_v.py`
A/B harness with the research-10 overlap checker and a direct pair-superset check; results in `*.jsonl` / `pcore*.txt`).
The repo is untouched apart from this file.

Terms: **broad phase** = cheap step that lists the (point, edge) pairs the exact tests ("narrow phase") must look at;
**AABB** = axis-aligned bounding box; **fat AABB / skin** = a box grown by a margin so it stays valid while things move a
little; **Verlet list** = molecular-dynamics name for a pair list built with cutoff + skin and reused until something
moves more than skin / 2; **SAP** = sweep and prune (sort boxes along an axis); **BVH / LBVH** = (linear) bounding volume
hierarchy; **CCD** = continuous collision detection.

## Summary

1. **The 24 px grid is 4-13x looser than what the narrow phases need, and its result barely changes between
   substeps.** Per substep at a settled 500-body pile: 32.3k contact candidates, of which 26% have the point inside
   the edge's padded box and 6.7% touch; 22.3k CCD pairs, of which 7.9% pass the swept-box pre-reject and ~0 hit.
   Meanwhile points move p99 0.3 px / max 0.4 px per substep (demo max 0.9 px) against 24 px cells. Both broad
   phases are recomputed from scratch 8+ times per frame (4 contact + 4-12 CCD calls).
2. **A persistent fat-box pair list (Verlet skin), shared by contacts and CCD, cuts CPU physics 10-32% in pure numpy
   and 28-41% with a numba rebuild, and halves GPU-backend frames at 200-500 bodies.** Each point gets a fat box
   (its motion box stretched 4 substeps ahead, plus skin / 2); the list holds every (point, edge) pair whose fat boxes
   overlap (edge box grown by 2r). Before each use an O(N) check confirms every point's current motion box is inside
   its fat box; if not, the list is rebuilt. Since a pair whose swept boxes overlap must have overlapping fat boxes,
   the list is a superset of what CCD can hit, exactly as the grid is: **the zero-overlap guarantee keeps the same
   argument.** Measured with the research-10 overlap checker on every substep plus a direct check that the list
   contains every stock-grid pair that passes the swept-box test: 0 violations in 49,800 checked substeps over 8 scenes,
   CPU and GPU backends (validation table below).
   P-core physics ms/frame, stock -> fat list: demo 2.25 -> 1.63, pile200 11.7 -> 7.3, pile500 32.7 -> 19.8
   (numba rebuild); GPU backend pile500 17.7 -> 8.4, pile200 ~7 -> 3.4.
3. **Per-call exact broad phase in numba is the other good option (about the same gain, no cached state).** A
   counting-sort grid in numba that emits exactly the box-overlapping pairs reproduces "grid pairs that pass the
   swept-box test" bit for bit (set equality on pile200/pile500) at ~1.2 ms vs 6.9 ms for the numpy grid (pile500,
   E-core thread time), with 12.7x fewer pairs. End to end it matches the fat list on CPU (pile200 7.1 ms, diamonds
   3.5 ms). It does not help the GPU backend's per-substep pair build the way the cached list does.
4. Not worth it here: BVH/LBVH (scene is uniform-size, 3k-12k primitives, sync cost dominates on GPU), numpy SAP
   (1-D sort on a dense 2-D pile gives ~10x more x-overlaps than grid cells), body-level AABB expansion (pile500:
   1,876 overlapping body pairs expand to 134k point-edge pairs, 4x the grid's 32k). Cell size is a minor knob
   (12-16 px: -9 to -13% on big piles, worse on the demo; topic 09 F8) and stops mattering once the broad phase runs
   rarely or in compiled code.
5. Latent bug found: `World._cell_pairs` crashes (`IndexError`) when point boxes span several cells and the join is
   empty (`code[np.r_[True, code[1:] != code[:-1]]]` on an empty array). Unreachable with today's callers (a point
   always shares a cell with its own edges before filtering), but any new subset query (fat list, incremental
   update) hits it. Repro: `/private/tmp/claude-501/research-03/bug_cellpairs.py`.

## Measurement setup

- Scenes: `demo` = `scene.build_scene(1512, 982)` (13 bodies, 78 points, 3 ropes); `pileN` = the
  `perf-engine-agent/scenes.py` pile (N mixed bodies in a box, ~6.3 points/body), "settled" = after 300 frames;
  `fall200` = pile200 from frame 0 (everything falling); `drag` / `dragpile200` = random grab-and-drag to random targets
  incl. outside the box every 30 frames (perf-engine-agent bench); `diamonds`, `newtons cradle`, `random walls`,
  `looping` = `scene.PRESETS` at 1512x982.
- Counts and the cell sweep: `micro.py` (background QoS). **Timings in the tables are short P-core runs
  (`nice -n 10`, seconds each)**, because background QoS on the shared E-cores inflated times 5-15x and drifted
  (same finding as topic 09's setup section). Sections are `time.thread_time` deltas; CPU repeats agree within ~3%,
  GPU-backend runs vary up to +-30% (two interleaved repeats shown).
- Correctness: `checker10.py` (copy of `research-10/bench/checker.py`, independent of the solver, every substep)
  plus `verlet.install(..., invariant=True)`: at every hand-out the stock grid broad phase is also run and every stock
  CCD pair whose swept boxes overlap, and every stock contact candidate within 2r, must be in the list.

## Findings

### F1. Work funnel per substep (stock CPU, `micro.py`)

| scene | pts / edges | contact grid candidates | point in edge box + 2r | dist < 2r (reach) | dist < r (touch) | CCD grid pairs | swept boxes overlap | CCD hits |
|---|---|---|---|---|---|---|---|---|
| demo | 78 / 74 | 149 | 22 | 20 | 10 | 125 | 0 | 0 |
| pile200 settled | 1262 / 1258 | 10,888 | 2,726 (25%) | 2,026 | 652 (6.0%) | 7,547 | 609 (8.1%) | 0 |
| pile500 settled | 3137 / 3133 | 32,298 | 8,509 (26%) | 6,572 | 2,165 (6.7%) | 22,328 | 1,759 (7.9%) | 2 |

Per-substep displacement |PREV -> PREV + D| (p50 / p99 / max, px): demo 0.09 / 0.91 / 0.91, pile200 0.05 / 0.32 /
0.50, pile500 0.07 / 0.27 / 0.37. Edge length p50 / p99: 18 / 36 px, plus the world-box walls (986-4526 px).
Topic 09 F8 measured the same funnel up to pile2000 (6% of CCD pairs survive at pile1000).

Where the stock CCD broad phase time goes (pile500, E-core thread time per call): `_ccd_cells` 3.5 ms, cell
expansion (`_cells`) 4.6, edge-key sort 2.8, searchsorted 1.4, dedupe + `_filter_pairs` 6.2 (the swept point boxes
span several cells, so pairs repeat and need a sort). The narrow phase then spends ~60% of `_ccd_detect` on the
pre-reject for the 92% of pairs that fail it (detect 32.5 ms on grid pairs vs 13.5 ms on pre-filtered pairs).

### F2. Cell size sweep (counts exact; times noisy E-core)

| cell px | demo cand / CCD | pile200 cand / CCD | pile500 cand / CCD |
|---|---|---|---|
| 8 | 38 / 15 | 4,375 / 2,283 | 13,946 / 7,109 |
| 12 | 48 / 60 | 5,609 / 3,431 | 17,757 / 10,253 |
| 16 | 85 / 70 | 7,021 / 4,567 | 22,069 / 13,968 |
| 24 (default) | 149 / 125 | 10,888 / 7,547 | 32,298 / 22,328 |
| 32 | 184 / 193 | 14,912 / 11,136 | 45,276 / 33,563 |
| 48 | 354 / 353 | 25,397 / 20,108 | 75,180 / 60,304 |

Smaller cells cut pairs roughly linearly but cost more cell entries per edge (the walls alone register hundreds of
cells), so in numpy the total is flat-to-slightly-better at 12-16 for piles and worse for the demo (09: pile1000
-13%, demo +6% at 16). Classic guidance (Teschner et al. 2003; Bentley 1977 via the Freiburg course notes) is a cell
about the size of a primitive's box, which here is 18-36 px edges + 6 px pad: 24 is already in that range. The
guarantee does not depend on cell size. Conclusion: not the lever; a tighter test is.

### F3. Fat-box pair list (Verlet skin), the main proposal

Literature: molecular dynamics has used "cutoff + skin, rebuild when any atom moved skin / 2" for decades (LAMMPS
`neigh_modify check yes` rebuilds only "if at least one atom has moved more than half the neighbor skin distance";
[LAMMPS docs](https://docs.lammps.org/neigh_modify.html), [LAMMPS developer notes](https://docs.lammps.org/Developer_par_neigh.html)).
Game physics does the same with fat AABBs: Box2D 2.x pads each proxy by `b2_aabbExtension` and stretches it along
the displacement by `b2_aabbMultiplier`; `MoveProxy` returns early when the tight box still fits the fat box
([b2DynamicTree.cpp](https://docs.ros.org/en/lunar/api/mvsim/html/b2DynamicTree_8cpp_source.html)). Box2D v3 keeps
fat AABBs but notes in `constants.h` that "keeping the margin small improves performance by reducing the number of
contacts" (`B2_MAX_AABB_MARGIN` 5 cm, `B2_AABB_MARGIN_FRACTION` 0.125 of the shape extent;
[box2d constants.h](https://github.com/erincatto/box2d/blob/main/include/box2d/constants.h)), and it persists pairs
in a hash set, re-querying only moved proxies ([broad_phase.c](https://github.com/erincatto/box2d/blob/main/src/broad_phase.c)).
Bridson et al. 2002 and Müller et al. 2020 run collision broad phases once per step and reuse them over substeps
(topic 01 F-section).

Prototype (`verlet.py`, monkeypatches `World._candidates`, `_ccd_cells`, `_ccd_pairs`, `_ccd_new_pairs`):

- State: per-point fat box `F_i = [lo_i, hi_i]` (float64), pair list `L` (CCD body rules from `_filter_pairs`,
  i.e. everything except a point's own edges and fixed-fixed), and a boolean mask of which pairs also pass the contact
  rules (no neighbour edges of self-colliding bodies).
- Edge box `U_e = (F_a ∪ F_b) ± m`, `m = max(2 r, 1.0) = 6 px`. `L` = all pairs with `F_i ∩ U_e ≠ ∅`.
- `update(A, B)`: a point is valid when the box of its segment A..B lies in `F_i` (contacts: A = PREV, B = P after
  predict; CCD: A = PREV, B = PREV + D, called again after impulses). If any point is invalid, every point gets
  `F_i = box(A_i, B_i, B_i + k D_i) ± skin / 2` (k = `ahead`, Box2D-style motion stretch) and `L` is rebuilt.
- `_ccd_new_pairs` returns nothing when the post-impulse motion is still inside the fat boxes (the stock code then
  returns "no hits", because every pair in `L` was just re-checked by `_ccd_redetect`), otherwise the rebuilt list.

Why the guarantee is unchanged: the stock argument is "every pair whose swept boxes overlap shares a grid cell, so
`_ccd_detect` sees it". With the list: point swept box ⊆ `F_i` and edge swept box ⊆ box of `F_a ∪ F_b` ⊆ `U_e`
(the + 1.0 growth is inside `m`), so overlapping swept boxes imply overlapping fat boxes, i.e. the pair is in `L`.
The containment check is exact float64 comparisons of the same numbers CCD uses. One behaviour change: `_ccd_detect`
keeps far pairs of an edge that collapses to ~zero length mid-substep (they produce spurious quadratic "hits"); with
the list those far pairs are no longer present. A real crossing still needs overlapping swept boxes, so it is still
caught; the checker runs confirm no overlaps (and the collapse case is also absent from the pile scenes).

Rebuild policy (P-core, ms/frame physics):

| scene | stock | partial updates, skin 4 | global rebuild, skin 2 | skin 4 | skin 8 |
|---|---|---|---|---|---|
| pile200 settled | 11.8 | 14.1-14.9 | 9.0 | 9.0 | 8.1 |
| pile500 settled | 32.8 | 28.7-29.6 | 24.0 | 26.1 | 23.2 |

Per-point partial re-queries (Box2D-style) cost ~0.4 ms each in numpy (fixed overhead of ~20 numpy calls, plus a
Python loop for incident edges) and fire nearly every check in a jittering pile, so the global rebuild ("classic
Verlet") wins in numpy. In compiled code partial updates would be cheap; not needed at these sizes.

Skin / look-ahead sweep, global rebuild, numpy rebuild (P-core, ms/frame physics; list = pairs in L at the end):

| scene | stock | 4 / 2 | 8 / 0 | 8 / 2 | **8 / 4** | 12 / 2 | 16 / 2 |
|---|---|---|---|---|---|---|---|
| demo | 2.26 | 1.76 | 1.68 | 1.67 | **1.63** | 1.67 | 1.70 |
| drag (demo) | 2.62 | 2.17 | 2.48 | 2.12 | **2.07** | 2.18 | 2.13 |
| fall200 (drop) | 9.57 | 8.11 | 8.78 | 8.54 | **7.99** | 8.98 | 9.45 |
| diamonds | 5.41 | 5.15 | 7.32 | 5.68 | **4.90** | 5.04 | 5.40 |
| pile500 settled | 33.1 | 26.0 | 30.9 | 23.1 | **22.5** | 26.2 | 29.8 |
| pile500 list size | (32k + 22k) | 16.9k | 25.0k | 25.3k | 25.7k | 37.0k | 50.4k |

Look-ahead matters a lot for falling scenes (diamonds: 7.3 ms without it vs 4.9 with k = 4): a falling point's box
then lasts ~4 substeps. Large skins bloat the list (narrow phase grows) as Box2D v3's comment predicts.

With a numba rebuild (`nb_grid.pairs_nb` on the fat boxes, ~0.2-0.3 ms at pile500 on a P-core) a smaller skin
is better, because rebuilds are cheap and the list is tighter:

| scene | stock CPU | fat, numpy rebuild (8 / 4) | **fat, numba rebuild (4 / 4)** | fat, numba (2 / 4) | exact per call, numba |
|---|---|---|---|---|---|
| demo | 2.25 | 1.61 | **1.63** | 1.65 | 1.80 |
| fall200 | 9.45 | 7.91 | **6.00** | 5.88 | 5.93 |
| diamonds | 5.43 | 4.88 | **3.81** | 3.70 | 3.52 |
| pile200 | 11.7 | 8.11 | **7.33** | 7.19 | 7.13 |
| pile500 | 32.7 | 23.4 | **19.8** | 19.2 | 22.1 (p95 40, noisy) |

Rebuild frequency in the timed window (settled piles, P-core) and the pile1000 point:

| scene | stock ms | fat numba 4 / 4: ms, rebuilds per substep | fat numba 4 / 0 | fat numpy 8 / 4 |
|---|---|---|---|---|
| pile200 | 11.7 | 7.2, 25 / 240 (1 per ~10) | 7.5, 38 / 240 | 8.0, 15 / 240 |
| pile500 | 33.0 | 19.7, 24 / ~217 | 19.9, 36 / ~225 | 23.3, 14 / ~232 |
| pile1000 | 93.1 | **53.0** (-43%), 15 / 120 | 56.4, 30 / 120 | 87.3 (-6%), 11 / 120 |

The numpy rebuild stops paying around 1000 bodies (list 76k pairs at skin 8, and each rebuild is a full numpy grid);
the numba rebuild keeps scaling. Topic 04 (cross-links) measured a global max-displacement rule rebuilding in
111/240 substeps at 200 bodies and 239/240 at 1000 in its own (more compressed, 12-point-body) piles and concluded
per-item verification is needed. Here the per-point box with look-ahead rebuilds every ~8-10 substeps in settled
piles; the difference is scene-dependent (jitter amplitude vs skin). Either way the design holds up: even at one
rebuild per substep the cost is one numba rebuild (~0.3 ms at pile500) or one stock-like numpy join, and the list
stays 2-4x tighter than the grid. 04's per-item `_box_patch` and this file's global rebuild are interchangeable
policies on the same invariant; per-item patching is the better choice in compiled code at large N.

What remains at pile500 (fat, numba): contacts pass ~10 ms, CCD detect ~4.2 ms, broad phase ~0.9 ms. `_ccd_detect`
is now dominated by its O(points + edges) setup per call (`_ccd_boxes`, the collapse test), not by pairs: the exact
1.7k-pair set and the 17k-pair list cost nearly the same there. That fixed cost is topic 05's (numba detect 9x).

### F4. GPU backend: the fat list removes the per-substep pair build from the command buffer

Topic 04 F2 measured that the GPU hash-grid pair build is the biggest part of a synced GPU substep (200 bodies:
~0.8 of 1.39 ms). `gpufat.py` feeds the CPU fat list's contact pairs to the existing contact kernels as
`(npairs, cpt, ce)`, re-uploading only when the list was rebuilt; the CCD side uses the same list on the CPU.
Interleaved P-core runs (GPU numbers vary +-30%):

| scene (GPU backend) | stock (2 runs) | fat list + numba rebuild, skin 4 / 4 (2 runs) | change |
|---|---|---|---|
| diamonds | 4.42 / 4.23 | 3.05 / 3.00 | -30% |
| pile200 | 8.59 / 5.94 | 3.40 / 3.47 | -42 to -60% |
| pile500 | 17.3 / 18.1 | 8.39 / 8.50 | -52% |

This moves the CPU/GPU crossover down (topic 09 F3: GPU was break-even at diamonds); the small presets still pay the
~4 syncs/frame floor, so `backend = "auto"` (09 B1) still applies. The contact list is checked against `P` after the
float64 predict on the CPU, so the GPU float32 math does not enter the validity test. Topic 04's GPU-built exact CCD
set is an alternative for CCD; the fat list makes CCD's broad phase nearly free on the CPU instead (0.03-0.1 ms/frame),
so building CCD pairs on the GPU is no longer needed for speed.

### F5. Exact compiled grid (numba), measured (`nb_grid.py`)

Counting sort of point boxes into a dense cell array, one pass per edge over its cells, inline exact box test and
`_filter_pairs` rules, each pair reported once at the low corner of the cell-range intersection (same trick as the
existing Metal `_PAIRS` kernel). E-core thread time per call:

| scene | numba exact (cell 8 / 16 / 24 / 32) | pairs | numpy grid `_ccd_cells` + `_ccd_pairs` | numpy pairs |
|---|---|---|---|---|
| pile500 | 1.27 / 1.21 / 1.20 / 1.34 ms | 1,759 | 6.87 ms | 22,328 |
| pile200 | 0.51 / 0.30 / 0.28 / 0.55 ms | 609 | 2.64 ms | 7,547 |
| demo | 1.57 / 0.19 / 0.13 / 0.08 ms | 0 | 0.58 ms | 125 |

Set equality with "stock grid pairs whose swept boxes overlap" holds on all three. Long walls make small cells
expensive (demo at 8 px). Contact form (point vs edge box + 2r) at pile500: 8,705 pairs in 2.0 ms. Topic 05 ports
the *stock* grid to numba bit-identically (6x); emitting only overlapping pairs is the extra step that shrinks every
downstream array, and it is what the fat list's rebuild uses.

### F6. Options considered and rejected (with numbers)

- **Body-level AABB culling as the pair source**: pile500 has 1,876 overlapping body pairs (boxes + 2r) that expand to
  134k point-edge pairs (pile200: 670 -> 45k), 4x the grid. Useful only as a first stage (e.g. skipping sleeping
  bodies, topic 07) or to treat the long walls separately; with the fat list or numba the walls cost nothing extra.
- **Sweep and prune**: good when boxes move coherently and the scene is sparse along the sweep axis (insertion sort on
  nearly sorted lists; [gSaP, Liu et al. 2010](https://graphics.ewha.ac.kr/gSaP/) picks the axis by PCA and adds
  spatial subdivision to cut false positives). In a dense 2-D pile, a 48 px x-window holds ~3% of all edges, so by
  estimate (not measured) ~100 x-overlap candidates per point (~300k at pile500, vs 32k grid). In numpy it also needs per-point ranges expanded with
  `np.repeat`, the same cost structure as the grid. Practitioners report grids beat SAP for dense uniform objects
  ([gamedev.net thread](https://gamedev.net/forums/topic/557805-my-broadphase-is-hideous-can-i-do-better/557805/)).
- **BVH / LBVH (Karras 2012)**: fully parallel radix-tree build over Morton codes
  ([Karras, HPG 2012](https://research.nvidia.com/publication/2012-06_maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees)).
  It shines for non-uniform sizes and big scenes. Here primitives are near-uniform (18-36 px) except a handful of
  walls, N is 1-12k, and on the GPU a CPU<->GPU round trip is ~150-190 us at normal QoS (topic 04 F1), more than a
  numba grid takes for the whole pile. MLX does have GPU `sort` / `argsort`, but the tree traversal would be a custom
  kernel and the CCD pairs have to come back to the CPU. Not worth it at this scale.
- **Hash collisions in the Metal kernel**: `gpu_backend._PAIRS` checks `slot_cx/cy` so collisions only cost time; the
  table is sized 2x entries. Fine as is.

## Applicability to this codebase

| change | where | what changes |
|---|---|---|
| fat-box pair list shared by contacts and CCD | new `World` state + `_candidates`, `_ccd_cells`, `_ccd_pairs`, `_ccd_new_pairs` in `solver.py`; invalidated in `_rebuild` (topology) | O(N) validity check per use; rebuild with the existing `_cells`/sort join on fat boxes + exact box filter (or numba); `_ccd_cells` disappears from the hot path |
| `EnvConfig.broad_skin` (4 px with numba / 8 px numpy), `broad_ahead` (4 substeps) | `entities.EnvConfig` | per-world overridable via `World.env` like other knobs |
| GPU backend reads the list | `gpu_backend.GpuBackend.pbd_substep` / `_pairs` | upload `(npairs, cpt, ce)` only when the list version changes; drop the per-substep hash-grid kernels for contacts; keep the kernel for `gpu_ccd_broadphase` or delete it |
| looping wraps | `looping.loop_step` | nothing: wrapped bodies leave their fat boxes and trigger a rebuild |
| debug grid view | `World.draw_grid` | unchanged (draws the cell grid); optionally draw fat boxes instead |
| margin depends on `contact_radius` | list state | store the margin used at build; force a rebuild when `contact_radius` (or `World.env` override) changes, else contacts (not CCD, which needs only 1.0) could be missed |
| `_cell_pairs` empty-join crash | `solver.World._cell_pairs` | guard `if i.size == 0: return i, e` before the dedupe (needed by any subset query) |

Determinism: the list order is a deterministic function of the state, so runs stay reproducible, but pair order (and
so Jacobi summation order and `argsort(ts)` tie-breaks) differs from stock. Results are not bit-identical to the
current code; compare with metrics (topic 10) rather than bitwise. If bitwise identity with the stock grid is a
requirement, the list can be sorted by `(point, edge)` code (one `np.sort` per rebuild, not per substep).

## Expected gains and risks

- CPU physics: -10% (diamonds) to -32% (pile500) in pure numpy; -28% to -41% with a numba rebuild. GPU backend:
  -30% to -55% at diamonds..pile500. Small presets: demo -28% (2.25 -> 1.63 ms) because the 8 numpy-heavy grid calls
  per frame become a rebuild every ~10 substeps plus a cheap O(N) check.
- Spikes: a rebuild costs about one stock `_candidates` call (numpy) or ~0.2-0.3 ms (numba, pile500). Worst case is
  everything moving fast (falling pile, drag whips): rebuild every substep, i.e. about stock cost. fall200 still won
  (-16% numpy, -37% numba) thanks to the look-ahead stretch.
- Zero-overlap risk: only if the containment check is skipped or wrong. Mitigations: keep the check in the same
  function that returns the pairs; property test (`invariant=True` mode) in the test suite; the research-10 checker on
  every substep. The spurious collapsing-edge hits disappear (behaviour change, not a safety change).
- Contact behaviour: candidates become "point within 2r + skin of the edge's box" instead of "shares a 24 px cell".
  Every pair the contact solver can act on (dist < r, and the reach set dist < 2r) is included whenever the list is
  valid, so contact forces are the same up to summation order. Checked: `contact_missing = 0` in all runs.

## Validation (research-10 checker every substep + direct superset check)

Background QoS, CPU backend unless noted. `ccd_missing` / `contact_missing` = stock pairs that mattered but were not
in the list (must be 0).

| config | scenes (frames) | substeps checked | overlap violations | containment changes | ccd_missing | contact_missing | impact zones |
|---|---|---|---|---|---|---|---|
| fat numpy, skin 8 / ahead 4 | drag (600), fall200 (300), dragpile200 (450), diamonds (300), newtons cradle (300), random walls (300), looping (300), fall500 (200) | 12,200 | 0 | 0 | 0 | 0 | 0 |
| fat numpy, 4 / 2 | same minus fall500 | 11,400 | 0 | 0 | 0 | 0 | 0 |
| fat + numba rebuild, 4 / 4, CPU | drag, dragpile200, fall200, diamonds, newtons cradle | 9,000 | 0 | 0 | 0 | 0 | 0 |
| fat + numba, 4 / 4, GPU backend | same 5 scenes (the cradle has ropes, so its PBD ran on the CPU) | 9,000 | 0 | 0 | 0 | 0 | 0 |
| exact per call (numba), CPU | same 5 scenes | 9,000 | 0 | 0 | n/a | n/a | 0 |
| stock (reference) | drag, dragpile200 | 4,200 | 0 | 0 | n/a | n/a | 0 |

Min point-to-edge gap (checker's margin metric) matched stock in every CPU run where both were measured (drag 0.437 px,
dragpile200 0.014 px stock vs 0.010-0.014 px with the list; the drag whip into the pile is the tight case in all
builds). GPU-backend gaps (fall200 0.19 px, diamonds 0.51 px) were not compared against a stock GPU run.

## Build plan

Ranked; each can be prototyped independently in a git worktree. Shared harnesses: `bench_v.py` (A/B, `BACKEND=gpu`
env var, `check=1` attaches the research-10 checker and the superset invariant), `/private/tmp/claude-501/research-10/bench/`
(accuracy + overlap suite, `compare.py`), `/private/tmp/claude-501/debug-agent/merged/ccd_test.py` (drag stress).
Time on a P-core (`nice -n 10`, interleaved ABAB); validate at background QoS.

### B1 (rank 1). Fat-box pair list in `solver.py` (pure numpy)

- Change: add `World._broad` state (fat boxes, list, contact mask, version) built lazily and reset in `_rebuild`;
  rewrite `_candidates`, `_ccd_pairs`, `_ccd_new_pairs` per `verlet.py` (global rebuild, `EnvConfig.broad_skin = 8`,
  `broad_ahead = 4`); `_ccd_cells` becomes the O(N) motion box; fix the `_cell_pairs` empty-join guard; rebuild via
  `_cell_pairs` on fat-box cells followed by the exact box test before dedupe (topic 09 B3's ordering).
- Success: 0 violations / 0 containment changes in the research-10 suite and `ccd_test.py`; an `invariant` unit test
  (list ⊇ stock pairs with overlapping swept boxes, and ⊇ contacts within 2r) on every substep of drag, fall, pile and
  looping scenes; physics ms/frame (P-core) demo <= 1.7, pile200 <= 8.5, pile500 <= 24, no scene slower than stock
  by > 3% (p95 included); accuracy metrics within topic 10's tolerances.
- Expected: -10..-32% CPU physics. Risk: low (guarantee argument is one line, check is exact); behaviour differs
  only in summation order and in dropping spurious collapsing-edge hits. Effort: ~1 day.

### B2 (rank 2). GPU backend consumes the list

- Change: in `gpu_backend.GpuBackend.pbd_substep`, take contact pairs from B1's list (`gpufat.py`), upload on version
  change only, drop `grid_count` / `grid_fill` / `pairs` dispatches for contacts. Depends on B1 (or prototype on the
  monkeypatch).
- Success: same checks as B1 with `EnvConfig.backend = "gpu"`; GPU physics pile200 <= 4 ms, pile500 <= 10 ms,
  diamonds <= 3.3 ms (P-core, median of 3 interleaved runs); re-measure the CPU/GPU crossover for `backend = "auto"`.
- Expected: -30..-55% GPU-backend physics at 870-3100 points. Risk: low (GPU is still only proposing motion; CCD
  unchanged). Effort: ~0.5 day after B1.

### B3 (rank 3). Compiled rebuild / exact broad phase (numba)

- Change: `nb_grid.pairs_nb` (counting-sort grid, inline exact box test + body rules) as the list rebuild, with
  `broad_skin = 4`; optionally also use it per call without the list (`exact.py`) to compare. Coordinate with topic 05
  (same numba dependency and JIT warm-up handling; cache=True).
- Success: pile500 CPU physics <= 20.5 ms, fall200 <= 6.5, diamonds <= 4.0; set equality vs "stock pairs passing
  the swept-box test" as a unit test; first-frame JIT cost hidden (warm-up at import or AOT).
- Expected: a further -10..-20% over B1 on big/falling scenes. Risk: numba dependency, JIT stalls. Effort: ~1 day.

### B4 (rank 4, cheap fallback). Exact box test before dedupe + cell size 16

- If B1 is not taken: topic 09 B3 (swept-box test right after the cell join, before the dedupe sort; bit-identical)
  plus the contact-side equivalent and `cell_size = 16` for scenes above ~500 points via `World.env`.
- Expected -7..-13% CPU, -25..-40% GPU backend on big piles. Risk: none to the guarantee (bit-identical CCD). Effort:
  hours.

## Sources

- LAMMPS `neigh_modify` (skin, check yes / half-skin rule): https://docs.lammps.org/neigh_modify.html ;
  developer notes: https://docs.lammps.org/Developer_par_neigh.html
- Box2D 2.x fat AABBs and motion stretch: https://docs.ros.org/en/lunar/api/mvsim/html/b2DynamicTree_8cpp_source.html ;
  class doc: https://box2d.org/documentation/classb2_dynamic_tree.html
- Box2D v3 margin comment and pair set: https://github.com/erincatto/box2d/blob/main/include/box2d/constants.h ,
  https://github.com/erincatto/box2d/blob/main/src/broad_phase.c
- Teschner et al. 2003, Optimized Spatial Hashing for Collision Detection of Deformable Objects:
  https://cgweb.informatik.uni-freiburg.de/publications/2003_VMV_collisionDetectionHashing.pdf ; cell-size notes:
  https://cgweb.informatik.uni-freiburg.de/course_notes/sim_07_spaceSubdivision.pdf
- Liu, Harada, Lee, Kim 2010, Real-time Collision Culling of a Million Bodies on GPUs (gSaP):
  https://graphics.ewha.ac.kr/gSaP/
- Karras 2012, Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees:
  https://research.nvidia.com/publication/2012-06_maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees
- Highway vqsort (NEON sort used by numpy 2.x): https://arxiv.org/pdf/2205.05982 ;
  https://www.phoronix.com/news/NumPy-2.0-Released
- Bridson, Fedkiw, Anderson 2002 (collision step over many internal steps): https://www.cs.ubc.ca/~rbridson/docs/cloth2002.pdf
- Cross-links: topic 01 (hoisting broad phase/CCD out of the substep loop), 04 F1/F2 (sync and GPU pair-build
  cost, GPU-built CCD set), 05 (numba ports, bit-identical stock grid), 09 F8/F11 (funnel, cell sweep, B3 prototype),
  10 F2 (checker).
