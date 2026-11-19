/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This file is part of the LoupMD interface to LAMMPS (https://github.com/FZWINGEL-UBT/LoupMD, lammps/).
   It is distributed under the GNU General Public License, as LAMMPS is.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(loupmd,PairLoupMD);
// clang-format on
#else

#ifndef LMP_PAIR_LOUPMD_H
#define LMP_PAIR_LOUPMD_H

#include "pair.h"

#include <string>
#include <vector>

struct LoupHandle;

namespace LAMMPS_NS {

// MACE models through the LoupMD engine (libloupmd.so, CUDA). The engine evaluates the whole periodic system on one
// GPU with its own neighbour list; LAMMPS's neighbour lists and ghost atoms are not used. One MPI process.
//
//   pair_style loupmd [device N] [skin S] [ghost yes|no]
//   pair_coeff * * model.loup El1 El2 ...      (one element per atom type, in type order)
class PairLoupMD : public Pair {
 public:
  PairLoupMD(class LAMMPS *);
  ~PairLoupMD() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  // for run_style loupmd, where the engine integrates: the engine holding the current system, and the cell
  LoupHandle *engine();
  void cell(double *c) const { box_to_cell(c); }

 protected:
  LoupHandle *handle;           // the engine (created in init_style)
  std::string bundle;           // pair_coeff's .loup file
  std::string bundle_loaded;    // the file `handle` was created from
  int device;                   // CUDA device; -1: default_device()
  int device_loaded;
  double skin_user;             // the engine's list skin in A; < 0: LAMMPS's neighbor skin
  double rmax;                  // the model's cutoff, A
  bool ghost;                   // report the model's cutoff to LAMMPS (ghost atoms, forward communication)
  std::vector<int> type2z;      // [ntypes + 1] atomic number of each atom type
  bool system_dirty;            // the engine has to be given the system again (init_style ran)
  std::vector<int> z_sent;      // atomic number by atom ID - 1 of the system the engine holds
  double skin_sent;
  std::vector<double> xbuf, fbuf;    // host positions / forces by atom ID - 1 (the non-Kokkos compute)

  virtual void allocate();
  virtual int default_device() { return 0; }
  void ensure_system();         // sends the system to the engine when it is new, or an atom count / type changed
  void box_to_cell(double *) const;
  void refuse_per_atom() const;
};

}    // namespace LAMMPS_NS

#endif
#endif
