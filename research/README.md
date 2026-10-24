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

(Filled in after cycle 1.)
