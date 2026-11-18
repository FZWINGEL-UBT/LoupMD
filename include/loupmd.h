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
// device `device_id`. Two architectures: MACE-OMAT-0 medium on its specialised kernels, and the generic path for
// MACE-OMAT-0 small, MACE-OFF23/23b/24, MACE-MH-1 (one head per bundle) and MACE-POLAR-1 S/M/L. MACE-POLAR-1 is
// evaluated for neutral singlet systems in zero external field; its stress includes the k-space terms' cell
// dependence (MACE PR 1718), which the PolarMACE releases up to 0.3.16 leave out.
LoupHandle* loup_create(const char* loup_path, int device_id);
void loup_destroy(LoupHandle* h);

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

// Environment:
//   LOUP_CACHE_DIR  directory for the per-GPU GEMM plan cache (default $HOME/.cache/loupmd). Files are keyed by GPU
//                   name, compute capability, cuBLASLt version and a hash of this library; plans whose winner leads
//                   the runner-up by less than 3% are not persisted.
//   LOUP_RETUNE     if set, ignore the plan cache and re-time every GEMM shape (results are still saved).
//   LOUP_SPLINE_VARIANT, LOUP_WPRIME_THREADS, LOUP_REORDER  force kernel variants (testing; see python/test_gate.py).
//   LOUP_WPRIME_RCV=0|1  specialised OMAT-0 path: the edge-tiled (0) or receiver-owned (1) w' backward (testing; the
//                   receiver-owned kernel is the default on sm_120 and compute capability 8.0, and on other GPUs from
//                   512 atoms).
//   LOUP_M1_CHUNK=<atoms>|0  specialised OMAT-0 path: chunks of at most this many receivers for the interaction-1
//                   message buffers whenever the system is larger (default: chunks of at most 8,192 above 16,384
//                   atoms, in two alternating buffers; 0 = always one chunk) (testing).
//   LOUP_ROW_SORT=global  sort every neighbour row in global scratch, the path of rows over 4,096 edges (testing).
//   LOUP_MD_REBUILD=host|fallback  MD neighbour rebuilds host-driven, or counted on the GPU and always finished on the
//                   host, instead of inside the step graph (testing).
//   LOUP_POLAR_FOLD=0|1  POLAR long range without (0) the load-time dot^T weight fold, testing the path other GPUs take,
//                   or with it (1) on any GPU (sm_120 and cc 8.0 fold by default).
//   LOUP_POLAR_ZBLK=0  POLAR long range: run the GEMMs of weight blocks certified exactly zero at load (POLAR-1-L's
//                   l = 2 SCF and electron-energy blocks), which are skipped by default (testing).
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
//   LOUP_DEBUG_SPLINE, LOUP_DEBUG_GEMM  print the variant / GEMM tuning decisions to stderr.

#ifdef __cplusplus
}
#endif
