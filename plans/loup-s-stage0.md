# Loup-S Stage 0 — working record

Base `mh1-two-half-adjoint` at `0b1107b`; design imported unchanged in `0c600aa`. Contract and exit criteria: `LOUP_S_DESIGN.md` §§4, 7–10. No kernel adoption in this stage.

| Block | Work and evidence required | State |
| --- | --- | --- |
| 1. Kept workflow | Tests first. Generate labelled periodic SrTiO3 (320 atoms, MH-1 `omat_pbe`); stress-trained MACE, export and independent 3× student gate; report student-visited E/F/species/tail/S. Keep training, validation, teacher test, harvest and fixed student test distinct. | Started. List 15/16 inputs inspected: 300/60 configurations from ASE/torch; no stress. Not Stage 0 labels. |
| 2. Baselines | s64l0 / s64r8, seeds 1 and 2 on identical splits; store radial basis/input/head and pre-phase-two EMA checkpoint. Paired full-schedule float32 s64l0, subtract energy offsets in float64 first. Accept float32 only within float64 seed spread. | Pending |
| 3. Calibrate | Two independent teacher trajectories: RDF, species RMS displacements, NPT lattice/density and diffusion where relevant, with uncertainties. Fix tail multiplier, relaxed lattice and phonon tolerances and conserving NVE step; compare drift and RMS at both steps. | Pending |
| 4. Close loop | Harvest worst student-visited frames, resume pre-phase-two EMA with augmented training; test on untouched relabelled student frames after each round. Falsifier 1: compare student-visited to teacher-test RMSE after two rounds; third/hotter round then stop if still open. | Pending |
| 5. Ada census | Both students at 320/5,000/40,000, 300 K: unprofiled sustained steps, separate graph-node nsys census, NVML memory slope vs §6.4, clocks/power. Answer falsifier 6 and replace §6.5 columns **with card identified**. No local timing absent explicit approval. | VPN approved; sm_89 build ready. Awaiting retrained students. |
| 6. Other systems | Owner selects water teacher and four-species system before starting; try molecular force/torque loss on water; answer falsifier 2. | Pending choices |
| 7. Variants and evaluation | 4.5/6 Å; 32/96 channels. Fresh starts at 300/1,000/3,000/10,000 frames with independent thermodynamic/strain samples and shorter schedules. Diffusive system: trajectory vs 50% relabelled Gaussian rattles, transport. Freeze student-visited + hard sets on all three systems; score §8 candidates by precision/recall against per-atom teacher error, thresholds on held-out data. | Pending |

## Operations

- Local 5060 Ti: WSL side build, reservation `status` first; every GPU operation under foreground `local_gpu.py run`, one short job at a time. No timing, A/B or memory measurements. Do not modify the source or the other development checkout.
- Ada: Windows OpenSSH only after go; git archive to `/mnt/homes/zwingel`, NFS nvcc/static libstdc++, detached jobs under `gpu_guard.py` on a card with no GPU processes. CPU suite on leto. Sustained runs ≥3 s back to back, second half reported; random operands for kernel benches; clocks and power recorded.
- Colab: T4 smoke, L4 training only. Named sessions, create destination, ≤16 MB chunks, SHA256, retry lost connections, detach/poll, fetch/check artifacts then stop. Owner logs account 2 in when needed. Stage estimate **75–95 CU**, not a measurement.
- Evidence goes in `OPTIMISATION_LEDGER.md` and `STATUS_AND_PROFILE.md` per measurement; §6.5 and design status header together. Keep cross-card estimates separate. Ask which adoption GPUs before an adoption decision. Never push; no `.internal/` work. Clean only this checkout's scratch after preserving results.

## Runs and spend

- Local `make -j4 CUDA_ARCH=sm_120 build/sm_120/libloupmd.so`: success. Free reservation/empty queue before first local job. Historical list 15 s64l0 gate: 2/2 passed (STATUS §6.70); **does not count for a retrained student**.
- VPN go received. First preflight: leto had a GPU compute process; leeloo had none. Guarded detached leeloo build: NFS nvcc, static libstdc++ flags, SHA256 `e6cacc42b510dc648287c190bac18c16817ed3ceff1bbfdd398ac78b92e69eb1`, `/mnt/homes/zwingel/loup-s-stage0/tree/build/sm_89/libloupmd.so`.
- No Stage 0 Colab allocations yet. Account 1: **0 CU**, account 2: **0 CU**. Account 1 balance at preflight: 125.95 CU **including unrelated usage** (`l37a` active; do not touch). Do not charge balance differences to this stage while assignments overlap.
