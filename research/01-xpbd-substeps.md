# 01: XPBD, small steps, substeps vs iterations

Status: in progress. Code reading done, literature done, calibration measured; cost and accuracy sweeps running
(scripts and logs in `/private/tmp/claude-501/research-01/`; prototype solver `xworld.py` there subclasses `World`,
no repo edits).

Terms used below: **PBD** = position-based dynamics; **XPBD** = extended PBD, where each constraint has a
*compliance* alpha (inverse stiffness, units s^2/mass here) and a Lagrange multiplier lambda, so stiffness stops
depending on the substep and iteration counts; **GS** = Gauss-Seidel (constraints solved one after another, each
seeing the previous ones' corrections); **Jacobi** = all constraints computed from the same positions, then summed
or averaged; **SOR** = successive over-relaxation (scale corrections by omega); **CCD** = continuous collision
detection; **S x I** = substeps per frame x solver iterations per substep (stock: 4 x 4 at 60 FPS, so h = 1/240 s).

## Summary
(draft, being filled as runs land)

1. **Stock spring stiffness is a function of the schedule, not of the material.** Measured rest sag of a soft
   dodecagon (preset stiffness 0.1, skip_sets 2): the same body ends at 60% of its rest height at 4x1, 69% at 2x8,
   84% at 4x4 (default), 90% at 16x1 and 91% at 8x4. Changing `EnvConfig.substeps` (e.g. the Newton's cradle's
   per-world 8 substeps) silently changes every soft body's material. With XPBD springs solved Gauss-Seidel
   (compliance 3e-4) the same test spans 85-87% over all six schedules.
2. **Per-substep fixed cost dominates, so "small steps" (Macklin et al. 2019) is not free here.** Each substep pays
   a contact broad phase + a full first contact pass + a full CCD; extra iterations are cheap because later contact
   passes only touch the cached `reach` set. Many 1-iteration substeps therefore cost about S x F, not S x I x c.
3. **The fix for that is structural: hoist the contact broad phase and CCD out of the substep loop.** Bridson et al.
   2002 (the paper this CCD comes from) already allows "many small explicit steps" of internal dynamics per
   collision step; the zero-overlap guarantee only needs the *frame's* start -> end straight-line motion to be CCD
   checked, from an intersection-free start. Macklin et al. 2019 and Müller et al. 2020 likewise do collision
   detection once per frame and reuse it over the substeps.

## Findings

### F1. How the current solver spends a frame (code reading + measured)

Per frame (`World._step`, solver.py): `substeps` (4) x [predict, contact broad phase (`_candidates`),
`iterations` (4) x (springs + shape matching + ropes, then one Jacobi contact pass), velocity update, terminal
velocity clamp, CCD (`_resolve_ccd`: swept broad phase + quadratic narrow phase + impulses)].

Cost model: `frame = S * (F + I * c)`, F = per-substep fixed cost (contact broad phase + first, full contact pass +
CCD), c = per-iteration cost (springs, shape matching, ropes, a contact pass over the cached `reach` set, which is
reused while no point has moved more than contact_radius / 2).

Measured shares (CPU backend, demo scene, thread CPU time, background QoS, so only ratios are meaningful):

| S x I | rel. cost | CCD share | constraints | contacts |
|---|---|---|---|---|
| 4x4 (stock) | 1.00 | 20% | 21% | 58% |
| 4x1 | 0.56 | 32% | 15% | 52% |
| 8x2 | 1.48 | 33% | 17% | 48% |
| 8x1 | 1.05 | 39% | 12% | 47% |

(16x1 and 200-body rows pending.) Agent 09's P-core table (research/09-profiling-hotspots.md) agrees: in the default
demo, contacts 1.08 ms + CCD 0.42 ms + constraints 0.55 ms of 2.13 ms; in a settled 1000-body pile CCD is 33 of
94 ms on CPU and 76% of the frame on the GPU backend. For the 200-body pile at background QoS, 4x1 cost about the
same as 4x4: the first contact pass + broad phase per substep dominates, the extra 3 iterations are nearly free.
So: F/c is ~2 for the demo and >5 for big piles.

Consequence: 8x1 costs about what 4x4 costs today; 16x1 costs ~2x. The full Macklin schedule (e.g. 20-30 x 1)
is out of reach unless F leaves the substep loop (F5).

### F2. Stock PBD stiffness depends on S, I and topology (measured)

`SoftBody` springs are solved as one Jacobi pass where each correction is `stiffness * C` and then divided by the
point's spring count (6 for skip_sets=2). Per pass a spring loses roughly stiffness/count of its error, per substep
1 - (1 - s/count)^I, and gravity adds g h^2 per substep. Rest sag therefore scales roughly like h^2 / f(I): more
substeps (quadratically) and more iterations (sub-linearly) make every soft body stiffer, and adding skip springs
raises the divisor. This is the classic PBD problem XPBD was written to fix ([Macklin, Müller, Chentanez 2016,
XPBD](https://matthias-research.github.io/pages/publications/XPBD.pdf): "constraints become arbitrarily stiff as
the iteration count increases, or as the time step decreases"). Agent 06 (research/06-soft-body-models.md F1)
reached the same conclusion from code reading.

Measured rest sag (bbox height / rest height of one resting soft dodecagon, 3 s; 1.0 = rigid):

| body | 4x1 | 2x8 | 4x4 (stock) | 8x2 | 16x1 | 8x4 | spread |
|---|---|---|---|---|---|---|---|
| stock, s=0.1, skip 2 (demo preset) | 0.60 | 0.69 | **0.84** | 0.86 | 0.90 | 0.91 | 0.31 |
| stock, s=0.05, skip 2 | 0.51 | 0.59 | **0.66** | 0.85 | 0.88 | 0.86 | 0.37 |
| stock, s=0.1, skip 1 | 0.40 | 0.47 | **0.47** | 0.58 | 0.64 | 0.65 | 0.25 |
| XPBD GS, alpha 3e-4 | 0.85 | 0.86 | 0.85 | 0.85 | 0.87 | 0.85 | **0.02** |
| XPBD GS, alpha 1e-3 | 0.73 | 0.72 | 0.73 | 0.67 | 0.66 | 0.67 | 0.08 |
| XPBD GS, alpha 3e-3 | 0.59 | 0.58 | 0.59 | 0.61 | 0.60 | 0.61 | **0.03** |
| XPBD Jacobi + mass splitting, alpha 1e-3 | 0.66 | 0.69 | 0.66 | 0.64 | 0.63 | 0.64 | 0.06 |
| XPBD Jacobi, 1/max(count) scaling, alpha 3e-4 | 0.60 | 0.81 | 0.75 | 0.64 | 0.59 | 0.75 | 0.22 |

Reads:
- The default soft body (s=0.1, skip 2, 4x4) ~ XPBD alpha 3e-4; the "50% weaker" one (s=0.05) ~ alpha 1e-3.
- GS XPBD is schedule-independent to a few percent. Jacobi XPBD with plain averaging is *not*: with few
  iterations per substep the multiplier never accumulates and the averaging divisor acts like extra compliance (a
  one-iteration steady state is off by about count / omega). Mass splitting ([Tonge et al. 2012](https://dx.doi.org/10.1145/2185520.2185601),
  denominators n_i w_i) is the right Jacobi form (agent 06 uses it too) and is close, but still drifts at 4x1.

### F3. Literature: small steps, XPBD, damping

- **Small steps** ([Macklin, Storey, Lu, Terdiman, Chentanez, Jeschke, Müller, SCA 2019](https://matthias-research.github.io/pages/publications/smallsteps.pdf)):
  n substeps x 1 XPBD iteration beats 1 step x n iterations for the same work. Positional error from external
  forces is O(h^2), so error drops quadratically with substeps; hanging chain: two orders of magnitude lower residual
  for 100 substeps vs 100 iterations; cloth: 30x1 13.5 ms vs 1x30 12.4 ms with "significantly less stretching".
  Numerical damping also drops (more lively motion), so explicit damping must be modelled (their eq. 8, XPBD
  Rayleigh damping). Key implementation detail: **"we have performed collision detection once per-frame"**, with
  contacts generated within a margin and reused for all substeps (sec. 4.2). They note single-iteration Jacobi works
  on GPU and that GS order bias can be fixed with symmetric SOR; float32 limits appear at high substep counts
  (h^2 terms) — irrelevant here on the float64 CPU path, relevant for the MLX path (agent 04).
- **XPBD** (2016): dlambda = (-C - alpha~ lambda) / (grad C M^-1 grad C^T + alpha~), alpha~ = alpha / h^2;
  dx = M^-1 grad C^T dlambda. With alpha = 0 it is exactly PBD with stiffness 1. Damping: eq. 26,
  dlambda = (-C - alpha~ lambda - gamma grad C (x - x_n)) / ((1 + gamma) grad C M^-1 grad C^T + alpha~),
  gamma = alpha~ beta~ / h, beta~ = h^2 beta.
- **Jacobi convergence**: plain Jacobi needs averaging/under-relaxation to be stable ([Macklin et al. 2014,
  Unified Particle Physics](https://mmacklin.com/uppfrta_preprint.pdf), constraint averaging + SOR omega in [1, 2]);
  graph-coloured GS keeps GS convergence with parallel batches ([Fratarcangeli, Tibaldo, Pellacini 2016, Vivace](https://research.chalmers.se/en/publication/507089));
  Chebyshev semi-iterative acceleration speeds Jacobi PBD/PD up by ~an order of magnitude ([Wang 2015](https://sa2015.siggraph.org/en/attendees/technical-papers/event/techpapers/283.html)).
- **Collision step vs internal dynamics** ([Bridson, Fedkiw, Anderson 2002, sec. 5](https://www.cs.ubc.ca/~rbridson/docs/cloth2002.pdf)):
  the internal dynamics advances from x^n to candidate x~^{n+1} by any method ("one large implicit time step ... or
  many small explicit Runge-Kutta steps"); then the *average* velocity (x~^{n+1} - x^n)/dt gets repulsions, then CCD
  impulses on the linear trajectories from x^n; x^{n+1} = x^n + dt v^{n+1/2}; if nothing collided, v^{n+1} is the
  internal dynamics' end velocity, else the corrected midstep velocity is advanced. They also adapt the collision
  step (halve on collision, double after three clean steps).
- **Rigid bodies with XPBD substeps** ([Müller, Macklin, Chentanez, Jeschke, Kim 2020](https://diglib.eg.org/handle/10.1111/cgf14105);
  survey [arXiv 2311.09327](https://arxiv.org/pdf/2311.09327)): broad phase once per step, narrow phase per substep.
- **Long range attachments** ([Kim, Chentanez, Müller-Fischer 2012](https://diglib.eg.org/handle/10.2312/SCA.SCA12.305-310)):
  unilateral max-distance tethers from a fixed anchor to every downstream particle remove the stretch of hanging
  chains/cloth at O(n) cost, regardless of iteration count.

(F4-F7 pending: accuracy sweep, frame-level CCD prototype, ropes.)

## Applicability to this codebase
(pending)

## Expected gains and risks
(pending)

## Effort
(pending)

## Build plan
(pending)
