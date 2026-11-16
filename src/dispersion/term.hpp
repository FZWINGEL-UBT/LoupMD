// The engine's interface for a dispersion term (plan task S6-D03; built by S6-D04, design
// docs/design/dispersion.md). Not compiled into the library yet: no translation unit includes this header until
// the D3 term exists. It fixes what the force pass expects of any dispersion term, so the D3 pair sum, the
// three-body term and the long-range methods (tail, split, mesh) are entries behind one interface.
//
// Rules every term keeps, the same as the model's force pass (src/generic.hpp):
// - record() is capturable: no host synchronisation, no allocation, no host reads of device data. Everything that
//   allocates or depends on the cell's shape happens in prepare(), outside stream capture.
// - The term adds to the pass's outputs; it never overwrites them. Energies and stress are summed in float64,
//   forces in float32 (the pass's own accumulators), pair arithmetic in float32 (S6-D02 measured float32 pairs
//   with float64 sums within 1e-5 meV/atom of simple-dftd3).
// - Units at the interface: positions and cell in A, energy in eV, forces in eV/A, virial in eV (the pass divides
//   by the volume). Cutoffs arrive in bohr and are converted once in prepare().
// - The energy-only path (per-atom energies, MC energy differences) records the energy without forces or virial.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace loup::dispersion {

enum class Damping : int32_t { BJ = 0, Zero = 1, BJM = 2, ZeroM = 3 };
enum class LongRange : int32_t { None = 0, Tail = 1, Split = 2, Mesh = 3 };  // "auto" is resolved before the term

// One resolved record (loupmd.dispersion.resolve): the functional's parameters are filled in, "auto" is replaced
// by the method the setup check chose (S6-D07).
struct Settings {
    Damping damping = Damping::BJ;
    double s6 = 1.0, s8 = 0.0;
    double a1 = 0.0, a2 = 0.0;                    // BJ, BJM
    double rs6 = 0.0, rs8 = 1.0, alpha = 14.0;    // zero, zerom
    double beta = 0.0;                            // zerom
    double pair_cutoff_bohr = 40.0, cn_cutoff_bohr = 40.0;
    bool three_body = false;
    double three_body_cutoff_bohr = 40.0;
    double smoothing_bohr = 0.0;                  // 0: no switch
    LongRange long_range = LongRange::None;
    double mesh_spacing_A = 0.0;                  // 0: the method's default
};

// What the pass hands the term for one force evaluation. Device pointers, on the pass's stream.
struct Inputs {
    cudaStream_t stream;
    int n_atoms;
    const float* positions;     // [n, 3], A, the pass's own (possibly reordered) atom order
    const int32_t* numbers;     // [n] atomic numbers
    const double* cell;         // [3, 3] rows, A, device copy of the pass's cell (stays valid through replay)
    bool forces;                // false: the energy-only path
    bool virial;
};

struct Outputs {
    double* energy;             // [1], eV, added to
    double* atom_energy;        // [n] or null, eV, added to (per-atom energies, S6-T08)
    float* forces;              // [n, 3] or null, eV/A, added to
    double* virial;             // [9] or null, eV, added to
};

class Term {
public:
    virtual ~Term() = default;
    // Outside capture: allocate for n atoms and this cell, build bins and tables. Returns true when captured
    // graphs that contain record() must be re-recorded (the bin grid or a buffer changed).
    virtual bool prepare(int n_atoms, const double cell[9], cudaStream_t stream) = 0;
    // Inside capture: the kernels of one evaluation.
    virtual void record(const Inputs& in, const Outputs& out) = 0;
    // Device bytes held (the capacity planner counts them).
    virtual size_t device_bytes() const = 0;
};

// The factory S6-D04 implements: null with a message when the settings name a method that is not built.
Term* make_term(const Settings& settings, const char** why_not);

}  // namespace loup::dispersion
