# Handoff: Loup-S training engine (`libloupstrain`), 25 November 2026, afternoon

The worktree is `D:\Margraf\LoupMD-loup-s-train`. Everything below is **uncommitted**: no commit has been asked for. The goal is open-ended:

> "Keep optimizing the training engine for loup-s in all ways (start with the kernels for now and then later keep working on end to end time and then on better recipe to save epochs)"

The latest focus was "Maybe start working on optimizing the kernel again". The owner then asked to stop after the current runs, update the docs and write this file. No GPU job is running.

## Rules that bind this work (owner's instructions and memory)

**The local GPU (RTX 5060 Ti, sm_120)**
- It is this work's GPU. The stage-0 session must not use it, and the LoupMD release session also queues there.
- Every GPU job, and every CPU-heavy diagnostic, runs through the queue. Reserve one short job (about 15 minutes or less) at a time:
  `python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --quiet --wait-minutes N --purpose "..." -- bash <script>`
- Run `check --cpu-only` before CPU-heavy builds.
- Never run CPU pytest locally. The full `tests/cpu` runs on leto.
- Workstations may be used when their GPU has no process; ncu does not work there.

**Files and builds**
- Never edit `D:\Margraf\LoupMD` or `D:\Margraf\LoupMD-loup-s-stage0`.
- Do not touch `.internal/`.
- Never rebuild a library in use. Build into side directories (`make -j2 CUDA_ARCH=sm_120 OBJDIR=build/sm_120-<tag> build/sm_120-<tag>/libloupstrain.so`) and select one with `LOUP_STRAIN_LIB`.
- A/B runs execute from the checkout, so never edit files a running job uses.

**Shell**
- Run Python only through WSL: `MSYS_NO_PATHCONV=1 wsl -e bash -lc '...'`. In the Git Bash tool, `python3` hangs.
- Run git from Git Bash; git inside WSL fails in worktrees.
- Write scripts that contain quotes with the Write tool, then run `sed -i 's/\r$//'` on `.sh` files.
- `cuobjdump` needs its full path: `$(ls -d /usr/local/cuda*/bin | head -1)/cuobjdump`.

**Behaviour**
- Interrupt or stop nothing without approval. Do not touch Colab session `ls0-4295189447f4`.
- Commit only when asked. The message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Ask questions in chat text, not as cards. Give short status updates. Do not use the beads or superpowers skills.

**Records**
- Every attempt gets an index row and an appendix entry in `OPTIMISATION_LEDGER.md`, plus `STATUS_AND_PROFILE.md` §7. Never delete rejected entries. Note "5060 Ti only".
- Auto mode refuses scratch deletes: hand the owner the commands instead.

## State of the kernel work (batch 1 blocked SOAP every 10, `LST_SOAP_SWEEPS=2 LST_SOAP_BLOCK=1`)

The default library is about 1.94 ms per step; the named soap40 recipe takes 33.6 s wall and 23.4 s of steps.

The ledger rows are `ft-kern-*` and `ft-e2e-cuda-eval`.

| Change | Switch | Result | State |
|---|---|---|---|
| Library evaluator (`lst_evaluate`, `CudaEvaluator`) | `LOUP_FT_CUDA_EVAL=0` turns it off | soap40 wall −1.1 s | ADOPTED, recorded |
| `k_slot_pp` (pairs per owner) | `LST_SLOT_PP=1` | tie | REJECTED, code kept, off |
| **`k_mlp_bwdp`**: gh3 first, `cp.async` stages (`src/train/mlp_bwdp.cuh`) | `LST_MLP_PIPE=1` | b1 1.95 → 1.90 ms, b4 6.75 → 6.52; bitwise the same sums | **WIN, not yet the default** |
| MLP backward grid balance | `LST_MLP_GRIDBAL=1` | tie | REJECTED, off |
| L2 prefetch of the next tile's gout | `LST_MLP_L2PF=1` (kernel argument `pf`) | tie | REJECTED, off |
| **k_force_node in each frame's own atom order** | experiment lib an30; production form `lst_set_node_order` | b1 1.88 → 1.86 ms; k_force_node 141 → 122 µs | **WIN as an experiment; production form built (lib ao31), NOT YET RUN** |

### Source on disk now = lib `build/sm_120-ao31`

Lib `ao31` contains:
- `k_mlp_bwdp`, with W2ᵀ loaded by `cp.async` (the change from lib `al28`);
- the `LST_MLP_GRIDBAL` and `LST_MLP_L2PF` switches;
- the production node order. It replaced the `LST_FORCE_ORDER_FILE` experiment, which no longer exists in the source.

The production node order consists of:
- `include/loupstrain.h`: `int lst_set_node_order(LstEngine*, const int32_t* order)`. The order is device int32 [nodes], frame-local, in the frames layout. `lst_set_frames` clears it. The ABI number stays 6: one new symbol, no struct change.
- `src/train/strain.cu`:
  - `Topo::norder`
  - `BatchT::norder` [N], allocated after `pptr`
  - `k_gather` writes `b.norder[i] = tp.norder[node_off[f] + local] + nn0`, or `i` for dummy nodes and when no order is set
  - `k_force_node` gets the extra argument `norder`, used for slots `nn` → node `n`
  - `force_order` is on by default; `LST_FORCE_ORDER=0` turns it off
  - `set_node_order` in the engine and the C wrapper
- `loupmd/fasttrain/data.py`:
  - `Frame.node_order`: the inverse of the species sort, filled in both conversion paths
  - the helper `_inverse`
- `loupmd/fasttrain/cuda.py`:
  - `CudaTrainer` uploads the order and calls `lst_set_node_order` when the library has the symbol and every frame has an order
  - `LOUP_FT_NODE_ORDER=0` skips it
  - argtypes are set when the symbol exists

  This is **active in the checkout for any library that has the symbol (only ao31)**.

Backups before the node-order patch are in `.tmp_o/bak_prenorder/`. Earlier per-library backups are in `.tmp_o/bak_preslotpp/`: `strain.cu.pre_mlppipe`, `.ak27`, `.al28`, `.am29`, `mlp_bwdp.cuh.ak27`, `.al28`.

## Next steps, in order

1. **Test ao31**, in one short job; a template is in `.tmp_o/r144.sh`.
   - Parity through `--lib-ab LST_FORCE_ORDER=0`. The order is now on by default when the trainer passes it, so side A is ordered and side B is index order. Also run the FP64 check against `muon.py`.
   - Step A/B of `LST_FORCE_ORDER=0` against 1 at batch 1 and 4, with `LST_MLP_PIPE=1`. Expect about −20 µs of k_force_node.
   - Check `CudaEvaluator` parity: `.tmp_o/t_cuda_eval_parity.sh`, after switching its `LOUP_STRAIN_LIB` to ao31. The evaluator uses `CudaTrainer(by_name=True)`, so it gets the order too.
2. **Make `LST_MLP_PIPE` the default** (and the node order, if step 1 holds).
   - In `strain.cu`, `mlp_pipe = !(ep && ep[0] == '0')`; keep `=0` as the off switch.
   - Then run a named soap40 A/B: `.tmp_o/r137.sh` is the template; give each side its own libs or switches.
   - Update the ledger rows `ft-kern-mlp-pipe` and `ft-kern-force-order` to ADOPTED, plus STATUS §7's switch table and defaults sentence.
3. Further kernel targets (nsys under-reads, so use real step times). The per-step timeline is in `.tmp_o/prof27/pipe-b1.timeline.txt`, from `timeline.py`.
   - **`k_mlp_bwdp`**: about 150 µs per launch, ×2, on the critical path. Its FMA pipe runs at 43% with 255 registers and 8 warps per SM. One idea: drop `z1` from registers and recompute `z1 = x W1` (K = 8) at the end, which frees 32 registers for ILP.
   - **`k_force_node`**: 122–139 µs and DRAM-bound. Doing both directions of a pair in one warp would make the R/T read compulsory only, about −16 MB more DRAM; costly.
   - **`k_slot`** ×2: about 139 µs each, latency-bound on the sender G/A gathers. `SLOT_ES` and `SLOT_PP` tied.
   - **`k_mlp_fwd`** ×2: about 100 µs each, at the step start.
   - The k_conv kernels, and the small cutlass dW GEMMs with split-K reduces on the side stream.

   Do not repeat: `k_mlp_bwd2`, `LST_MLP_SPLIT`, `LST_MLP_DUAL`, `SLOT_ES`, `SLOT_PP`, `GEMM_TN`, `CB_WS=0`, `MLP_GRIDBAL`, `MLP_L2PF`.
4. After the kernels, continue with end-to-end time, then recipes that save epochs (per the goal).

## Before any commit (only if the owner asks)

- Run the full `tests/cpu` on leto against the merged tip (recipe: memory `loupmd-workstation-recipe`, a git-archive tree).
- Tell the owner about:
  - `oxide_loop.py`'s `test_vendor` failure, which was already there before this work;
  - the `~/.cache/loupmd/train/cg` directory, which this work created (the Clebsch–Gordan cache);
  - the 215 recipes moved to `/home/fzwin/loup-train/recipes_from_checkout`.
- Hand the owner deletion commands for the scratch:
  - `.tmp_o/` scratch: `tree_pathA/B`, tarballs, the `bak*` directories, `startup-prof*`, `parent-cost-*`;
  - `/home/fzwin/loup-train/tree_pathA`;
  - the side libraries `build/sm_120-a*`, which are no longer needed once one is adopted.
- Do not leave `.tmp_o` scripts in a commit.

## Useful scripts (`.tmp_o/`)

- **Round runners:** `r139.sh` (slot_pp), `r140.sh` (mlp pipe), `r141.sh` (three-way lib A/B), `r143.sh`, `r144.sh`. Outputs are in `round1NN.out`.
- **Profiles:**
  - `prof27/p9_slotpp.sh`, `prof27/p10_pipe.sh`: nsys kernel tables plus `mlp_overlap.py`.
  - `prof27/timeline.py`: one step's kernels by stream.
  - `r142.sh` with `prof27/ncu_stalls.sh` / `ncu_stalls.py`: ncu stall reasons.
- **Analysis:** `sim_force_l2.py`, the LRU estimate of k_force_node's R/T reuse. It wrote the STO-320 order to `/home/fzwin/loup-train/sto320_force_order.bin`.
- **Patches:** `patch_*.py`, applied in this order: `patch_slot_pp`, `patch_mlp_pipe`, `patch_mlp_gridbal`, `patch_mlp_l2pf`, `patch_force_order` (the experiment, since replaced), `patch_node_order`. Ledger patches: `patch_ledger_ceval_slotpp.py`, `patch_ledger_r140.py`.
- **Build:** `build_<tag>.sh` builds a side library and prints the registers.
- **Setup:** `prof27/common.sh` sets `$B` (strain_bench), `$NSYS`, `$NCU`, `$O` and `$PY`.
