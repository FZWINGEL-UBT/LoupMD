/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This file is part of the LoupMD interface to LAMMPS (https://github.com/FZWINGEL-UBT/LoupMD, lammps/).
   It is distributed under the GNU General Public License, as LAMMPS is.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef INTEGRATE_CLASS
// clang-format off
IntegrateStyle(loupmd,IntegrateLoupMD);
// clang-format on
#else

#ifndef LMP_INTEGRATE_LOUPMD_H
#define LMP_INTEGRATE_LOUPMD_H

#include "integrate.h"

#include <vector>

namespace LAMMPS_NS {

// The LoupMD engine integrates: between the steps at which LAMMPS writes output (thermo, dump, restart) the engine
// runs its own MD loop on the GPU, at the engine's own step time, and LAMMPS gets positions and velocities back at
// those steps only. Requires pair_style loupmd and no fix that acts during the steps.
//
//   run_style loupmd nve
//   run_style loupmd langevin T damp seed       (T in K, damp in ps as fix langevin's, the engine's Langevin scheme)
class IntegrateLoupMD : public Integrate {
 public:
  IntegrateLoupMD(class LAMMPS *, int, char **);

  void init() override;
  void setup(int) override;
  void setup_minimal(int) override;
  void run(int) override;
  void force_clear() override;

 protected:
  int langevin;                 // 0 nve, 1 langevin
  double t_target, damp;        // K, ps
  unsigned long long seed;
  class PairLoupMD *pair;
  int triclinic;
  std::vector<double> xbuf, vbuf;

  void checks();
  void reneighbor();            // LAMMPS's own bookkeeping after the atoms moved: wrap, ghosts, lists
  void forces(bigint);          // energy / virial flags of this step, then the pair style's compute
  void push_state();            // LAMMPS positions and velocities to the engine
  void pull_state();            // and back
};

}    // namespace LAMMPS_NS

#endif
#endif
