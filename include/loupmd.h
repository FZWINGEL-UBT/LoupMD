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
    int dangerous_builds;              // rebuilds triggered after an atom had already moved more than skin/2 while the
                                       // old list was in use (pipelined loop only; should be 0, else raise the skin)
    double tune_ms;                    // one-time per-system kernel-variant timing at the first rebuild (not in step_ms)
    int pipelined_steps;               // steps of this call run in the pipelined loop (the rest: one host sync each)
} LoupMDStats;

// All int-returning calls return 0 on success and -1 on failure (no C++ exception escapes); loup_last_error gives
// the reason. After a failure caused by a CUDA error the device context may be unusable: destroy the handle.

// The reason of the last failure, as "<entry point>: <message>".
//   h != NULL: the last call on `h`; "" if that call succeeded. Valid until the next call on `h`.
//   h == NULL: this thread's last loup_create, or call with a NULL handle; "" if the last loup_create on this
//              thread succeeded. Valid until this thread's next loup_create or failing call without a handle.
// The library prints nothing on a failure. LOUP_DEBUG_ERRORS (set to anything) also prints each message to stderr.
const char* loup_last_error(const LoupHandle* h);

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
// Run NVE Velocity-Verlet steps (`dt_fs` in fs). Rebuilds skin neighbour list only when max_disp > skin/2.
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
//   LOUP_POLAR_FOLD=0  POLAR long range without the load-time dot^T weight fold (testing the path other GPUs take).
//   LOUP_CHAIN_FOLD=0  generic models (OMOL-0) without the load-time skip-chain fold (testing the path other GPUs take).
//   LOUP_SPECIES_TABLES=0  generic models (OMOL-0, MH-1) with layer 0's species-only GEMMs instead of per-species tables.
//   LOUP_DEBUG_SPLINE, LOUP_DEBUG_GEMM  print the variant / GEMM tuning decisions to stderr.
//   LOUP_DEBUG_ERRORS  print every failure's message (loup_last_error) to stderr as well.

#ifdef __cplusplus
}
#endif
