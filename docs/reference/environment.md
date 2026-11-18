# Environment and cache

Run options are arguments of `lm.md`, `lm.Simulation` and `loupmd md`. The few process-wide settings on this page are
environment variables. The variables in the first table are for users. The last table lists switches for testing and
development; they are not for production runs. Installation is in [Install](../install.md); common problems are in
the [FAQ](../faq.md).

## Variables for users

| Variable | Value | Default | What it does | Read by |
| --- | --- | --- | --- | --- |
| `LOUPMD_MODELS` | directory | unset | searched for a label's checkpoint file before `<cache root>/models`; an empty file there is skipped | `loupmd/models.py` |
| `LOUPMD_CACHE` | directory | `~/.cache/loupmd` | the cache root (below) | `loupmd/cache.py` |
| `LOUP_CACHE_DIR` | directory | unset | an older name of the cache root, used when `LOUPMD_CACHE` is unset; also the directory where the engine keeps its GEMM plan files (below); set from `LOUPMD_CACHE` when only that is set | `loupmd/cache.py`, `src/engine.cu` |
| `LOUPMD_BENCH_STRUCTURES` | directories separated like `PATH` | unset | folders searched for the structure files of `loupmd bench` that are not shipped with the package | `loupmd/bench/suites.py` |
| `LOUP_LIB` | path to a `libloupmd.so` | unset | the library this process loads, instead of the one shipped in the package (`loupmd/_lib/libloupmd.so`) or, in a source checkout, the one `make` installs at its root | `loupmd/_core.py` |
| `LOUP_KERNEL_PATH` | directories separated like `PATH` | `<cache root>/lib` is appended | folders the engine searches for kernel plugins `loupmd-kernels-<signature>.so` when a bundle needs kernels the library was not built with | `loupmd/kernels.py`, `src/generic.cu` |
| `LOUP_GATE_CACHE` | directory | `<cache root>/gate_refs` | where the accuracy gate keeps its cached MACE references | `loupmd/validate/references.py` |
| `LOUP_RETUNE` | any value | unset | the engine times every GEMM again instead of reading its plan file; the new plans still overwrite the file | `src/engine.cu` |

`LOUP_LIB` must name a library with the package's C interface version. Another version raises `BackendError` with
the advice to rebuild or unset `LOUP_LIB` ([Errors](errors.md)).

The search order for a label's file is in
[Models](models.md#finding-and-trusting-the-file).

## Variables LoupMD sets

| Variable | Set to | When |
| --- | --- | --- |
| `LOUP_KERNEL_PATH` | what the caller set, followed by `<cache root>/lib` | in the process's own environment, each time an engine is made (`LoupEngine`) |
| `PYTHONPATH` | the package's parent directory first | in the export process that writes a bundle, so it imports the same package |
| `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` | `1`, where unset | by the modules that unpickle checkpoints: the exporter, the gate's references, the model probe and `loupmd bench --compare` |
| `CUDA_VISIBLE_DEVICES` | empty, where unset | in the export process, which then sees no GPU |

## Cache

The cache root is `$LOUPMD_CACHE`, else `$LOUP_CACHE_DIR`, else `~/.cache/loupmd`. It must lie outside the Python
installation (`sys.prefix`); a root inside it raises `CacheLocationError`, since a cache there is deleted and copied
with the environment.

| Kind | Directory | Content | Removed by `loupmd cache prune` |
| --- | --- | --- | --- |
| `models` | `<root>/models/` | downloaded checkpoints (`--download`) | only with `--kind models` |
| `bundles` | `<root>/bundles/` | exported `.loup` bundles, one per model contents and path, element set, head, charge and spin, and exporter sources | yes |
| `plans` | `<root>/plans/`, and `plans_*.bin` files in the root | GEMM plans: the timed choice of GEMM kernel per shape and GPU | yes |
| `gate_refs` | `<root>/gate_refs/` | MACE float64 and float32 references of the accuracy gate | yes |
| `lib` | `<root>/lib/` | kernel plugins built by `loupmd build-kernels` | only with `--kind lib` |

`loupmd cache list` shows every cached file with its size and last use, `loupmd cache size` the bytes per kind, and
`loupmd cache prune --older-than DAYS` or `--max-size SIZE` removes files ([Command-line reference](cli.md)). Prune
leaves files another process holds a lock on. Bundles and plans are made
again when they are missing; models are downloaded again only with `--download`.

`.loup` bundles and `plans_*.bin` files in the root itself are files of the layout before the sub-directories;
`loupmd cache` counts them as `bundles` and `plans`.

### GEMM plan files

The engine times the GEMM kernels for each matrix shape once and stores the choice in a plan file,
`plans_<gpu>_sm<major><minor>_lt<cublasLt version>_hyb<0|1>_<library build>[_c<cubin hash>].bin`. The engine reads
the directory from `LOUP_CACHE_DIR`, else `$HOME/.cache/loupmd`; when only `LOUPMD_CACHE` is set, the package sets
`LOUP_CACHE_DIR` to the cache root before it makes an engine, so the plan files lie in the root, where `loupmd cache`
counts them. A
plan file is keyed by the GPU, the library build and the loaded tensor-core cubin, so a new library times again.
Different plan choices can make two runs of the same build differ in speed by a few percent; a fresh
`LOUP_CACHE_DIR` per run gives each its own plans.

## Testing switches

These switches change which kernels or code paths run. They exist for the accuracy gate, for comparisons and for
debugging. They are not for production runs: a result obtained with them has not been through the release's
gates in that form. Most are read when an engine is made; the MD-loop switches are read at each call.

| Variable | Values | Effect | Read in |
| --- | --- | --- | --- |
| `LOUP_GENERIC` | `0`, `1` | `0`: MACE-OMAT-0 medium runs its specialised kernels instead of the generic ones | `loupmd/bundles.py` |
| `LOUP_SPLINE_VARIANT` | `R1`, `R1p`, `R4`, `R4p`, `R5` | forces the specialised path's spline kernel variant | `src/engine.cu` |
| `LOUP_WPRIME_THREADS` | `256`, `512` | forces the w' backward kernel's block size | `src/engine.cu` |
| `LOUP_WPRIME_RCV` | `0`, `1` | forces the receiver-owned w' kernel off or on (default: on from 512 atoms) | `src/engine.cu` |
| `LOUP_REORDER` | `0`, `1` | atom reordering off or on | `src/engine.cu` |
| `LOUP_ROW_SORT` | `global` | every neighbour row sorted in global scratch, the path of rows over 4,096 edges | `src/engine.cu` |
| `LOUP_NB_GROUPS` | `0`, `1` | the lane-group neighbour-list kernels off, or on any GPU (default: on for `sm_120` and compute capability 8.0) | `src/engine.cu` |
| `LOUP_M1_CHUNK` | atoms, `0` | specialised kernels: the receiver chunk of the second layer's messages (`0`: one chunk) | `src/engine.cu` |
| `LOUP_POLAR_FOLD` | `0` | MACE-POLAR-1: the unfolded SCF and electron-energy GEMMs | `src/polar.cu` |
| `LOUP_CHAIN_FOLD` | `0`, `1` | `0`: whole systems of multi-layer models on the generic kernels without the skip-chain fold (tiled systems never fold); `1`: the fold on GPUs other than `sm_120` and compute capability 8.0 | `src/generic.cu` |
| `LOUP_SPECIES_TABLES` | `0` | layer 0's species-only GEMMs instead of the per-species tables (OMOL-0, MH-1) | `src/generic.cu` |
| `LOUP_CONV_FWD_VARIANT` | `0`, `1`, `2` | forces the generic conv forward variant | `src/generic.cu` |
| `LOUP_CONV_TUNE` | `0`, `1` | per-system timing of the conv forward variants off, or on for any GPU | `src/generic.cu` |
| `LOUP_NO_HYBRID` | `1` | no tensor-core hybrid GEMM cubin | `src/engine.cu` |
| `LOUP_HYBRID_CUBIN` | path | loads this GEMM cubin instead of the shipped one | `src/engine.cu` |
| `LOUP_LR_HYBRID` | `1` | the hybrid cubin for long-range models too | `src/engine.cu` |
| `LOUP_FORCE_TC` | `1` | every GEMM the tensor-core cubin can take runs on it; no plans are read or stored | `src/engine.cu` |
| `LOUP_OWN_EXT` | `0`, `1` | the `sm_120` own-SGEMM extensions off, or on any GPU | `src/engine.cu` |
| `LOUP_DEBUG_HIDE_KERNEL` | signature | hides the library's own kernels of that signature, so a plugin can be tested | `src/generic.cu` |
| `LOUP_TILE_GEMM_BUCKETS` | `0` | tiled call: times every tile row count's GEMMs | `src/engine.cu` |
| `LOUP_TILE_WAVEFRONT` | `0`, `1` | tiled call: forces the wavefront schedule off or on | `src/engine.cu` |
| `LOUP_TILE_KEEP_ACTIVE` | `0`, count | tiled call: no tile keeps its active edges across its sweeps, or at most that many slots | `src/engine.cu` |
| `LOUP_TILE_KEEP_COMPACT` | `0`, count | tiled call: no compact slots of kept active edges, or at most that many | `src/engine.cu` |
| `LOUP_TILE_KEEP_L0` | `0`, `1` | tiled call: each tile's first-layer messages kept for its backward sweep, off or on | `src/engine.cu` |
| `LOUP_TILE_HOST_LISTS` | `0`, `1` | tiled plan: the tile lists' edges on the device, or in pinned host memory | `src/engine.cu` |
| `LOUP_TILE_HOST_STAGE` | `0`, `1` | tiled call: host lists read over the bus, or copied into device buffers one sweep ahead | `src/engine.cu` |
| `LOUP_TILE_HOST_STATE` | `0`, `1` | tiled plan: the layer outputs and their gradients never in host memory, or there (a plan that would pin over 90% of the available host memory is refused) | `src/engine.cu` |
| `LOUP_TILE_HOST_PIPE` | `0` | tiled call: the host state's copies one after another, not overlapped with the sweeps | `src/engine.cu` |
| `LOUP_TILE_HOST_ALIAS` | `0` | tiled call: a separate gradient array in the host state | `src/engine.cu` |
| `LOUP_TILE_HOST_COMPACT` | `0` | tiled call: host lists in the full form (28 B per edge) instead of 8 B | `src/engine.cu` |
| `LOUP_TILE_DEVICE_COMPACT` | `0` | tiled call: device lists in the full form (28 B per edge) instead of 8 B | `src/engine.cu` |
| `LOUP_TILE_SHARE_LAYERS` | `0` | tiled call: every layer's buffers its own instead of one arena the layers share | `src/generic.cu` |
| `LOUP_VARIABLE_CELL` | `1` | sends a `loup_run` call through the variable-cell step | `src/engine.cu` |
| `LOUP_DEBUG_CELL_RATE` | number r | scales the cell and positions by 1 + r in every variable-cell step | `src/engine.cu` |
| `LOUP_MIN_AHEAD` | `0`, `1` | the minimiser's list check before or after the force pass | `src/engine.cu` |
| `LOUP_MD_REBUILD` | `host`, `fallback` | MD loop: every neighbour rebuild on the host, or the device rebuild's fallback to the host in every rebuild | `src/engine.cu` |
| `LOUP_D3_KSUM` | `direct` | D3 split sum: the direct k sum instead of the mesh | `src/dispersion/d3_pair.cu` |
| `LOUP_D3_PME_ORDER` | `4`, `5`, `6`, `8` | D3 mesh: the B-spline order | `src/dispersion/d3_pair.cu` |
| `LOUP_D3_ATM_PROJ` | `0` | D3 three-body term: the per-triangle C6 table walk | `src/dispersion/d3_pair.cu` |
| `LOUP_SWEEP_REPLAY` | any value | generic path: a whole system runs the sweep schedule of a tiled one, without the skip-chain fold (not for long-range models) | `src/generic.cu` |
| `LOUP_DEBUG_PACK` | any value | prints the packed tables' relative error per layer | `src/generic.cu`, `src/model_loader.cu` |
| `LOUP_DEBUG_ERRORS` | any value | prints every library error to standard error | `src/engine.cu` |
| `LOUP_DEBUG_GRAPHS`, `LOUP_DEBUG_GEMM`, `LOUP_DEBUG_GEMM_BUCKETS`, `LOUP_DEBUG_SPLINE`, `LOUP_DEBUG_TILE_TIMES` | any value | print diagnostics of CUDA graphs, GEMM plans, bucketed GEMM timings, kernel-variant timings and tiled-call phase times | `src/engine.cu`, `src/generic.cu` |
| `LOUP_NO_NVML` | `1` | the device's free memory from CUDA alone instead of the smaller of CUDA's and NVML's figures | `src/engine.cu` |
| `LOUP_DEBUG_PLAN` | `1` | prints the free device memory at the steps of a system's set-up and first call, and the memory plan's counters | `src/engine.cu` |
| `LOUP_DEBUG_PLAN_FREE` | MiB | the capacity plan counts at most this much device memory as free (tests of its thresholds) | `src/engine.cu` |
| `LOUP_DEBUG_EDGE_LIMIT` | integer | lowers the largest neighbour-edge count of a whole-system list, for the test of its refusal | `src/engine.cu` |
| `LOUPMD_EXPORT_FASTMACE` | path to a FastMACE checkout | the exporter takes its names from that checkout instead of the vendored copy | `loupmd/export/_fastmace.py` |
