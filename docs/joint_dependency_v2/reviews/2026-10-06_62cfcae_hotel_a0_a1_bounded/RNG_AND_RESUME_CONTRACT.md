# RNG and resume contract

The training DataLoader has a dedicated saved `torch.Generator`. Python, NumPy, Torch
CPU, all CUDA RNG states, loader generator state, optimizer, scheduler, GradScaler,
successful/attempt/skipped/failed counters, planned steps, curriculum progress, cache
identity, parent identity, frozen-family hash, best-selection tables, and ObjectiveRNG
state are saved at complete epoch boundaries.

A1 neighbour IDs use the independent CPU `ObjectiveRNG`; only detached endpoint `q`
crosses to CPU. Candidate tensors, unary, costs, energies and gradients remain on the
compute device. Validation uses an isolated seed context and restores training RNG.

Numbered and moving-last formal checkpoints are atomic and resumable. Best checkpoints
are explicitly weights-only. The step cap, arm deadline and queue deadline are checked
before fetching the next window, so they cannot consume that window's candidate,
diffusion or MC draw. A post-draw failure remains pending and cannot be certified as a
resume boundary.

Formal test seeds are exactly 2035–2039. Evaluation regenerates the canonical 21 goals
inside each isolated seed stream; training continues to use the single pre-materialized
candidate bank. The exact persistent sampler derives its own stable per-window streams,
so objective draws and assignment draws do not perturb the shared base diffusion stream.

This is a state-correct epoch-boundary resume contract. CUDA kernels may still have
documented small non-determinism; it is not a claim of bitwise identical re-execution.
