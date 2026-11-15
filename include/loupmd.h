#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LoupHandle LoupHandle;

// The version of this C interface. It changes when a call or a struct of this header changes incompatibly; a
// caller built against another value must not use the library (loupmd/_core.py refuses it).
#define LOUP_ABI_VERSION 1
// The release this source belongs to (the same string as loupmd.__version__).
#define LOUP_VERSION "1.0.0a1"

int loup_abi_version(void);     // LOUP_ABI_VERSION of the loaded library
const char* loup_version(void); // LOUP_VERSION "+" 16 hex digits: a hash of the library file (the one in the
                                // GEMM plan cache's file names)

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

// What a handle runs: the model as loaded and the GPU it is on.
typedef struct {
    int abi;                    // LOUP_ABI_VERSION of the library
    int compute_capability;     // of the handle's GPU, major * 10 + minor (120 for sm_120)
    int n_layers;               // interaction layers of the model
    int has_long_range;         // 1 for MACE-POLAR-1 (k-space electrostatics), else 0
    int specialised;            // 1 on the specialised kernels, 0 on the generic path
    int n_species;              // elements the bundle was exported for
    double r_max;               // cutoff in A
    char gpu_name[64];
    char kernel_signature[160]; // the compiled kernels the bundle needs; ends in "..." if it did not fit
} LoupInfo;
int loup_get_info(const LoupHandle* h, LoupInfo* out);

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
// A run may be split into calls of any length at no extra cost: the captured step graphs are kept while the
// timestep, thermostat settings and seed stay the same, and a call that follows a run or an evaluation at the same
// positions starts from the forces already on the device.
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

// One run call for every integrator. The caller sets `size` to sizeof(LoupIntegrator), so a library can tell which
// fields a caller built against another header knows; zero every field it does not use.
enum {
    LOUP_NVE = 0,            // velocity Verlet
    LOUP_LANGEVIN = 1,       // ASE's Vanden-Eijnden-Ciccotti Langevin scheme
    LOUP_CSVR = 2,           // the kinds from here on are refused by name until their task builds them
    LOUP_NHC = 3,
    LOUP_BERENDSEN_NVT = 4,
    LOUP_BERENDSEN_NPT = 5,
    LOUP_BAOAB_NPT = 6,
    LOUP_MTK_NPT = 7
};
enum { LOUP_CELL_HYDROSTATIC = 0, LOUP_CELL_FULL = 1 };

typedef struct {
    unsigned size;                 // sizeof(LoupIntegrator) of the caller
    int kind;                      // LOUP_NVE ...
    double dt_fs;                  // timestep in fs
    double temp_k;                 // thermostat temperature in K
    double friction_per_fs;        // Langevin friction in 1/fs
    double tau_t_fs;               // thermostat time constant in fs (CSVR, Nose-Hoover chain, Berendsen)
    double pressure_gpa;           // barostat pressure in GPa
    double tau_p_fs;               // barostat time constant in fs
    double compressibility_per_gpa;// Berendsen NPT
    int chain_length;              // Nose-Hoover chain
    int cell_mode;                 // LOUP_CELL_HYDROSTATIC or LOUP_CELL_FULL
    int cell_mask[3];              // per-axis mask of the barostat
    int fix_com;                   // remove the centre-of-mass motion (refused until built)
    int thermo_every;              // write a thermo row every so many steps (loup_read_thermo); 0 = none
    unsigned long long seed;       // noise stream of the stochastic integrators
} LoupIntegrator;

// Runs `n_steps` steps of `it`. NVE reads dt_fs; Langevin reads dt_fs, temp_k, friction_per_fs and seed. The rules
// of loup_run_nve and loup_run_langevin apply: those two calls are this one with the struct filled in.
int loup_run(LoupHandle* h, const LoupIntegrator* it, int n_steps, LoupMDStats* out_stats);

// Thermodynamic output without a host round trip per row: with LoupIntegrator.thermo_every = k, every k-th step
// appends a row to a buffer on the device, inside the captured step.
typedef struct {
    long long step;       // MD steps completed since the last loup_set_state or loup_import_state on this handle
    double e_pot;         // eV, as loup_get_state's energy at that step
    double e_kin;         // eV
    double temp_k;        // 2 e_kin / (3 N k_B)
    double pressure_gpa;  // 0 until a step carries stress
    double volume_a3;     // 0 until a step carries stress
    double conserved;     // e_pot + e_kin: NVE's conserved quantity (thermostats add their terms with their task)
} LoupThermoRow;
// Copies up to `max_rows` of the oldest unread rows to `rows`, removes them from the buffer and sets *n_rows to
// their number. loup_read_thermo(h, NULL, 0, &n) only counts the unread rows. The buffer holds 4096 rows: a run
// call that would write more than its free part is refused before its first step. loup_set_state and
// loup_import_state discard unread rows.
int loup_read_thermo(LoupHandle* h, LoupThermoRow* rows, int max_rows, int* n_rows);

// Read back state in caller's original atom order (velocities in ASE units, as for loup_set_state).
int loup_get_state(LoupHandle* h, double* out_positions, double* out_velocities, double* out_forces, double* out_energy_ev);

// The state a restart needs, as one versioned block of bytes: positions, velocities and cell as the caller gave
// them (caller atom order), the step counter (the offset of the noise stream, so a resumed Langevin run continues
// its noise) and the displacement tracker. Forces and the neighbour list are not in it.
//   loup_export_state(h, NULL, 0, &n) sets n to the size in bytes. With a buffer it writes the block; a buffer
//   smaller than that is refused, and n still holds the size.
//   loup_import_state needs the same system on the handle (loup_set_system with the same atomic numbers in the
//   same order) and refuses a block of another atom count, species list, version or length, or a damaged one. It
//   rebuilds the neighbour list; the next run starts with a force pass.
int loup_export_state(LoupHandle* h, void* buffer, size_t capacity, size_t* written);
int loup_import_state(LoupHandle* h, const void* buffer, size_t size);

// How a force call is executed. Only LOUP_EXEC_WHOLE is built; the tiled modes come with the capacity stage.
enum { LOUP_EXEC_AUTO = 0, LOUP_EXEC_WHOLE = 1, LOUP_EXEC_TILED = 2, LOUP_EXEC_WAVEFRONT = 3 };

// The device memory a system of `n_atoms` with `edges_per_atom` neighbours per atom inside cutoff + skin would
// take, scaled from the buffers of the handle's current system, which must have a neighbour list (any size: a
// small cell of the same material will do). edges_per_atom <= 0 takes the current system's value.
typedef struct {
    size_t whole_bytes;          // every buffer of the whole-system call that scales with atoms or edges
    size_t tiled_global_bytes;   // 0 until the tiled mode is built
    size_t tile_workspace_bytes; // 0 until the tiled mode is built
    size_t free_bytes;           // free device memory now (cudaMemGetInfo), the current system's buffers included
                                 // in what is used
    int receivers;               // receiver cap of the tiled mode; 0 until it is built
    int mode;                    // LOUP_EXEC_WHOLE if the whole-system call fits with 10% headroom once the current
                                 // system is released, else LOUP_EXEC_TILED (the size needs the capacity mode)
} LoupMemoryPlan;
// Not counted: MACE-POLAR-1's long-range buffers, captured graphs and cuBLAS's own allocations.
int loup_plan_memory(LoupHandle* h, int n_atoms, double edges_per_atom, LoupMemoryPlan* out);

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
//   LOUP_DEBUG_GRAPHS  print a line to stderr for every capture of an MD step graph and for every force pass a run
//                      call makes before its first step (tests/test_md_graph_reuse.py).

#ifdef __cplusplus
}
#endif
