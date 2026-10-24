# SoftBodySim research

Deep-research program on simulation speed and accuracy for this machine (started 2026-10-24 06:30).

**Machine:** MacBook Pro, Apple M4 (4 performance + 6 efficiency cores, 10-core GPU), 16 GB unified memory, macOS, Python 3.13 venv (`.venv`: numpy, mlx, moderngl; torch, taichi, wgpu, numba also installed).

## Cycle 1: topics

| # | Topic | File |
|---|---|---|
| 01 | XPBD, small steps, substeps vs iterations | [01-xpbd-substeps.md](01-xpbd-substeps.md) |
| 02 | CCD and contact accuracy | [02-ccd-contact-accuracy.md](02-ccd-contact-accuracy.md) |
| 03 | Broad phase on Apple Silicon | [03-broad-phase.md](03-broad-phase.md) |
| 04 | Metal / MLX custom kernels and float32 robustness | [04-metal-mlx-kernels.md](04-metal-mlx-kernels.md) |
| 05 | numba / Cython / C extensions for the remaining Python loops | [05-compiled-loops.md](05-compiled-loops.md) |
| 06 | Soft-body models for accuracy | [06-soft-body-models.md](06-soft-body-models.md) |
| 07 | Rigid bodies, stacking, friction, restitution, sleeping | [07-rigid-stacking.md](07-rigid-stacking.md) |
| 08 | Parallelism on M4 P/E cores | [08-parallelism-m4.md](08-parallelism-m4.md) |
| 09 | Profiling-driven hotspot analysis (measured) | [09-profiling-hotspots.md](09-profiling-hotspots.md) |
| 10 | Numerical accuracy, energy, validation benchmarks | [10-accuracy-benchmarks.md](10-accuracy-benchmarks.md) |

## Ranked build plans

Ranked by expected gain for the risk, across all ten cycle 1 reports. "Exact" means results stay bit-identical to today.

| Rank | Build | Source | Expected gain | Risk |
|---|---|---|---|---|
| 1 | numba port of the CPU solver loops (exact) | 05 | 4.5-13x CPU physics; beats the GPU backend at every measured size | low |
| 2 | Default backend: CPU (numba) instead of GPU, or auto by scene size | 09, 05 | removes the ~2 ms/frame fixed GPU cost on every shipped preset | low |
| 3 | Frame-level CCD and contact broad phase (`ccd_stride`), per-second grab settings | 01, 02 F7 | -30-50% CPU physics; makes small steps cheap | low-medium |
| 4 | Fat-box cached pair list, and fix the latent `_cell_pairs` IndexError | 03 | 25-35% CPU, ~2x GPU broad phase | low |
| 5 | Robust float64 CCD root (closes the endpoint-miss hole) and an inelastic fallback | 02 | correctness | low |
| 6 | Small-steps default (8x1, then 16x1) | 01, 10 F8 | better energy, stretch and stacking accuracy | medium (pile-drag fallbacks) |
| 7 | XPBD springs (mass-splitting Jacobi, Rayleigh damping) and shape-matching shape memory | 01, 06 | soft materials stop depending on the schedule; no flattening | low |
| 8 | Shared contact scale (momentum fix), rigid-aware ropes, per-body sleeping | 10, 07 | up to 31% momentum loss fixed; chain stretch and jitter fixed | low-medium |
| 9 | Draw-path optimisation | 09 | drawing is a top hotspot | low |
| 10 | Pipelined physics worker process | 08 | ~1.3x end to end | medium |
| 11 | GPU: CCD pair set built on the GPU, all substeps in one command buffer | 04, 01 | 1.8-3x on piles (GPU path only) | medium |

Every build is gated on research 10's benchmark (`compare.py`) and the brute-force overlap checker staying at 0.
