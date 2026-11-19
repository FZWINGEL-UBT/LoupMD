#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LoupHandle LoupHandle;

typedef struct {
    double step_ms;                    // GPU time per step, excluding tune_ms
    int rebuilds;
    double final_potential_energy_ev;
    double final_kinetic_energy_ev;
    double final_temperature_k;
    double tune_ms;                    // one-time per-system kernel-variant timing at the first rebuild (not in step_ms)
} LoupMDStats;

// All int-returning calls return 0 on success and -1 on failure (message on stderr; no C++ exception escapes).
// After a failure caused by a CUDA error the device context may be unusable: destroy the handle.

// Create engine from a .loup bundle (python/export.py; python/loupmd.py's ensure_loup_bundle caches them) on CUDA
// device `device_id`. One path for every family (MACE-OMAT-0 / MP-0 / MPA-0 / MatPES, MACE-OMOL-0, MACE-OFF23/23b/24,
// MACE-MH-1 with one head per bundle, MACE-POLAR-1 S/M/L); bundles of the retired specialised OMAT-0-medium path
// (arch omat0_medium_spec, before JIT_DESIGN.md stage 5) are refused: re-export them. MACE-POLAR-1 is
// evaluated for neutral singlet systems in zero external field; its stress includes the k-space terms' cell
// dependence (MACE PR 1718), which the PolarMACE releases up to 0.3.16 leave out.
LoupHandle* loup_create(const char* loup_path, int device_id);
void loup_destroy(LoupHandle* h);

// Facts of the loaded model: the cutoff radius in A (0 on a null handle), and the atomic numbers of the bundle's
// species (returns their count and writes at most `max` of them to out_atomic_numbers, which may be NULL).
double loup_cutoff(const LoupHandle* h);
int loup_species(const LoupHandle* h, int* out_atomic_numbers, int max);

// Set system topology (species as atomic numbers Z[0..n_atoms-1], masses in amu).
// Internally groups atoms by species for contiguous GEMMs; for systems whose per-atom working set exceeds L2, atoms
// inside each species group are also reordered spatially at every neighbour-list rebuild. All inputs and outputs
// use the caller's atom order.
int loup_set_system(LoupHandle* h, int n_atoms, const int* atomic_numbers, const double* masses_amu, double skin);

// Single-point evaluation (positions [n_atoms, 3] in A, cell [3, 3] row-major in A). Replaces the positions, keeps
// MD velocities; the neighbour list is rebuilt only if the cell changed or an atom moved more than skin/2.
// Forces written to out_forces [n_atoms, 3] in eV/A (caller atom order).
// Stress [3, 3] in eV/A^3 (optional, pass nullptr to skip virial/stress accumulation).
int loup_evaluate(
    LoupHandle* h,
    const double* positions,
    const double* cell,
    int compute_stress,
    double* out_energy_ev,
    double* out_forces,
    double* out_stress
);

// Evaluate ahead (0 off, the default; 1 on): after a loup_evaluate call the engine queues the next call's force pass
// on its GPU stream, held back until that call has written its positions, so the call itself launches nothing. It
// is for a caller that integrates on the host and calls loup_evaluate every step, such as the LAMMPS pair style:
// measured on an RTX 5060 Ti under WSL 2, it took 2-7% off a LAMMPS step on cells of a few hundred atoms and nothing
// measurable off a 5,000-atom step. It starts after two calls in a row with one stress flag and
// an unchanged cell; a call that does not fit the queued pass, and every other call on the handle, lets the queued
// pass run first (its result is discarded), so results never depend on the setting.
// WARNING: between two loup_evaluate calls the handle's stream then holds work that cannot run. Other GPU work of
// the process in that time may wait for it forever: anything on CUDA's default stream (another handle being set up,
// another library such as PyTorch) and any device-wide synchronisation. Switch it on only when this handle's
// loup_evaluate calls are the process's only GPU work between them. Without driver support for a stream wait on host
// memory the setting has no effect. LOUP_EVAL_ARM=0 keeps it off for every handle; =1 switches it on for every handle.
int loup_set_evaluate_ahead(LoupHandle* h, int on);

// Initialize MD state (positions [n_atoms, 3] in A, velocities [n_atoms, 3] in ASE units, i.e. A per ASE time unit
// as returned by ase.Atoms.get_velocities(); cell [3, 3] in A). Velocities are zero after loup_set_system until
// this call sets them (a run after only loup_evaluate starts from rest).
int loup_set_state(LoupHandle* h, const double* positions, const double* velocities, const double* cell);

// The run and profile calls return -1 unless positions and a neighbour list exist (loup_set_state or loup_evaluate
// after loup_set_system, with the last rebuild completed).
// Run NVE Velocity-Verlet steps (`dt_fs` in fs). Every step checks the skin list against its new positions before
// its force pass and rebuilds it only when an atom has moved more than skin/2.
int loup_run_nve(LoupHandle* h, int n_steps, double dt_fs, LoupMDStats* out_stats);

// Run Langevin steps (ASE's Vanden-Eijnden-Ciccotti scheme; `dt_fs` in fs, `temp_k` in K, `friction_per_fs` in 1/fs,
// same as ASE's friction=f/units.fs). The noise stream continues across calls; it restarts only in loup_set_system.
int loup_run_langevin(
    LoupHandle* h,
    int n_steps,
    double dt_fs,
    double temp_k,
    double friction_per_fs,
    unsigned long long seed,
    LoupMDStats* out_stats
);

// Read back state in caller's original atom order (velocities in ASE units, as for loup_set_state).
int loup_get_state(LoupHandle* h, double* out_positions, double* out_velocities, double* out_forces, double* out_energy_ev);

// Timing breakdown across components for diagnostics (averages over `n_calls` force evaluations).
int loup_profile_force(LoupHandle* h, int warmup, int n_calls, double* out_total_us);

// Device-resident evaluation, for host codes that keep their atoms on the GPU and integrate themselves (the LAMMPS
// Kokkos pair style `loupmd/kk`, lammps/ in this repository). All pointers in LoupDeviceIO are device pointers on the
// engine's CUDA device.
#define LOUP_IO_ENERGY 1       /* out_energy_ev is filled (the call then waits for the force pass) */
#define LOUP_IO_VIRIAL 2       /* out_virial is filled (the call then waits for the force pass) */
#define LOUP_IO_ATOM_ENERGY 4  /* reserved: per-atom energies are not available yet (the call is refused) */
typedef struct {
    int n_rows;               // rows of the caller's arrays; must equal n_atoms of loup_set_system
    // Component c (0 = x, 1 = y, 2 = z) of row r is at base[r * row_stride + c * comp_stride], in doubles: a row-major
    // [n][3] array has strides (3, 1), a column-major [n_alloc][3] array (Kokkos::LayoutLeft) (1, n_alloc).
    const double* positions;  // A. Rows may be unwrapped (any periodic image); they must be continuous from call to
                              // call, since the skin check compares them with the positions of the last list build.
    long long pos_row_stride, pos_comp_stride;
    const int* ids;           // [n_rows]: the engine atom (index in loup_set_system's order, 0 .. n_atoms-1) each row
                              // holds; every atom exactly once. NULL: row r is atom r.
    double* forces;           // eV/A: the forces are ADDED to the rows (NULL: no forces written)
    long long f_row_stride, f_comp_stride;
    double* atom_energy;      // reserved, must be NULL
    int flags;                // LOUP_IO_* bits
} LoupDeviceIO;

// Queues one force evaluation behind the work already on `stream` (a cudaStream_t of the same device; NULL is the
// default stream) and makes `stream` wait for the forces, so work the caller queues on it afterwards sees them. `cell`
// is a host [3, 3] row-major matrix in A, as for loup_evaluate; a changed cell rebuilds the neighbour list.
// The call returns as soon as the step is queued and its neighbour-list decision is known; it waits for the result
// only when energy or virial is asked for. out_energy_ev (with LOUP_IO_ENERGY) is the potential energy in eV.
// out_virial (with LOUP_IO_VIRIAL) is the virial tensor -dE/d(strain) in eV, as [xx, yy, zz, xy, xz, yz] (the layout
// and sign of LAMMPS's `virial`; the stress of loup_evaluate is -virial / volume).
// loup_evaluate, loup_set_state and the run calls may be mixed with it on one handle; MD velocities are untouched.
int loup_compute_device(LoupHandle* h, const LoupDeviceIO* io, const double* cell, void* stream,
                        double* out_energy_ev, double* out_virial);

// Environment:
//   LOUP_CACHE_DIR  directory for the per-GPU GEMM plan cache (default $HOME/.cache/loupmd). Files are keyed by GPU
//                   name, compute capability, cuBLASLt version and a hash of this library; plans whose winner leads
//                   the runner-up by less than 3% are not persisted.
//   LOUP_RETUNE     if set, ignore the plan cache and re-time every GEMM shape (results are still saved).
//   LOUP_EVAL_ARM=0|1  loup_set_evaluate_ahead for every handle: 0 keeps it off, 1 switches it on (testing; see the
//                   warning at loup_set_evaluate_ahead).
//   LOUP_DEBUG_EVAL_TIMES=1  when a handle is destroyed, print where its loup_evaluate calls spent their time: to
//                   the launch, to the guard's publication, to the results, to the return, and between calls.
//   LOUP_REORDER=0|1  the spatial atom reorder off or on (testing).
//   LOUP_CONV_FWD_VARIANT, LOUP_CONV_TUNE  force or untune the conv forward variants (testing; python/test_gate.py).
//   LOUP_JIT=0|1    the run-time compiler (JIT_DESIGN.md): 0 off (built-in kernels only; a bundle whose product the
//                   library lacks is then refused), unset the kernels the library has no instance for (conv widths,
//                   products), 1 also layer 0's kernels. LOUP_NVRTC names the NVRTC library, LOUP_DEBUG_JIT=1 prints
//                   what is compiled or read from <LOUP_CACHE_DIR>/jit/, LOUP_JIT_PRODUCTS=1 compiles every product
//                   from the bundle (testing).
//   LOUP_ROW_SORT=global  sort every neighbour row in global scratch, the path of rows over 4,096 edges (testing).
//   LOUP_MD_REBUILD=host|fallback  MD neighbour rebuilds host-driven, or counted on the GPU and always finished on the
//                   host, instead of inside the step graph (testing).
//   LOUP_POLAR_FOLD=0|1  POLAR long range without (0) the load-time dot^T weight fold, testing the path other GPUs take,
//                   or with it (1) on any GPU (sm_120 and cc 8.0 fold by default).
//   LOUP_POLAR_ZBLK=0  POLAR long range: run the GEMMs of weight blocks certified exactly zero at load (POLAR-1-L's
//                   l = 2 SCF, electron-energy and source blocks), which are skipped by default (testing).
//   LOUP_POLAR_HFOLD=0|1|2|3  POLAR models: the degrees (bit l - 1 = degree l) whose hidden chain is folded at load into
//                   the long-range mixer and source and the next layers' weights; 0 forms every layer's output, the
//                   path other GPUs take (testing; sm_120 folds degrees 1 and 2 by default).
//   LOUP_CHAIN_FOLD=0|1  generic models (OMOL-0, MH-1, the OMAT-0 / MP-0 / MPtrj / OFF families on the generic path):
//                   0 skips the load-time skip-chain fold (the path other GPUs take), 1 folds on any GPU (testing;
//                   sm_120 and cc 8.0 fold by default).
//   LOUP_SPECIES_TABLES=0|1  generic models (OMOL-0, MH-1): 0 keeps layer 0's species-only GEMMs instead of per-species
//                   tables, 1 builds the tables on any GPU (testing; sm_120 and cc 8.0 build them by default).
//   LOUP_CONV_TUNE=0|1  generic models: 0 keeps the default conv forward variant (no per-system timing at the first
//                   rebuild); 1 times the variants on any GPU (testing; sm_120 times them by default).
//   LOUP_CONV_FWD_VARIANT=0|1|2  force a generic conv forward variant (testing; see python/test_gate.py).
//   LOUP_CONV_PRUNE=0  generic models: keep every conv path of a layer whose l >= 1 projections are exactly zero
//                   (MP-0 medium's second interaction), which by default runs only its output-degree-0 paths and a
//                   scalar-only product (testing).
//   LOUP_LIN_COMPACT=0  generic models: the dense projections and product kernels of kind-1 layers (MPtrj-2022.9, MP-0
//                   medium's layer 0) instead of their live channels, certified from exact zeros at load (testing).
//   LOUP_SENDER_BWD=0  generic models: the receiver-owned backward of a pruned conv layer (MP-0 medium's second
//                   interaction) instead of the sender-owned one (testing).
//   LOUP_NO_HYBRID=1  no tensor-core GEMM cubin (float32 SIMT GEMMs only).
//   LOUP_HYBRID_CUBIN=path  load this GEMM cubin instead of the default (cubins/gemm_tc_hyb_sm80.cubin, LoupMD's own
//                   cubins/src/gemm_tc.cu build with more tiles and beta support, on cc 8.0; FastMACE's
//                   cubins/tf32x3_hyb_sm<arch>.cubin elsewhere) (testing). GEMM plans are keyed by the cubin file.
//   LOUP_LR_HYBRID=1  load the GEMM cubin for long-range models too (testing; they run float32 SIMT GEMMs by default).
//   LOUP_OWN_EXT=0|1  force the own-SGEMM extensions (sgemm64 / sgemm32 candidates, N-fastest rasterisation; testing;
//                   on by default for sm_120, and for cc 8.0 when no GEMM cubin is loaded, e.g. long-range models).
//   LOUP_NB_GROUPS=0|1  force the lane-group neighbour-list kernels (testing; on for sm_120 and compute capability 8.0
//                   by default).
//   LOUP_FORCE_TC=1  every GEMM the tensor-core cubin can take runs on it (testing the cubin's accuracy on every
//                   eligible shape; no GEMM plans are loaded or saved).
//   LOUP_GATE_FUSE=0, LOUP_GATE_FUSE_FWD=0  generic models with gated (kind 4) layers (MH-1, OMOL-0, POLAR-1): keep the
//                   gate kernels instead of the gate backward in the W_2^T GEMMs' epilogue / layer 0's gated store in
//                   its projections' epilogue (testing; both run where a GEMM's plan is the own N-fastest sgemm128,
//                   i.e. on sm_120).
//   LOUP_DEBUG_SPLINE, LOUP_DEBUG_GEMM, LOUP_DEBUG_COMPACT  print the variant / GEMM tuning decisions and the compacted
//                   generic layers to stderr.

#ifdef __cplusplus
}
#endif
