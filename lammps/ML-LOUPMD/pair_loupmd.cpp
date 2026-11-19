// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This file is part of the LoupMD interface to LAMMPS (https://github.com/FZWINGEL-UBT/LoupMD, lammps/).
   It is distributed under the GNU General Public License, as LAMMPS is.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "pair_loupmd.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "update.h"

#include "loupmd.h"

#include <cstring>

using namespace LAMMPS_NS;

namespace {
const char *const kSymbols[] = {
    "X",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
    "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
    "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
    "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md",
    "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};
const int kNumSymbols = sizeof(kSymbols) / sizeof(kSymbols[0]);

int atomic_number(const char *symbol)
{
  for (int z = 1; z < kNumSymbols; z++)
    if (strcmp(symbol, kSymbols[z]) == 0) return z;
  return 0;
}
}    // namespace

/* ---------------------------------------------------------------------- */

PairLoupMD::PairLoupMD(LAMMPS *lmp) :
    Pair(lmp), handle(nullptr), device(-1), device_loaded(-1), skin_user(-1.0), rmax(0.0), ghost(true), ahead(false), system_dirty(true),
    skin_sent(-1.0), serial(0)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  no_virial_fdotr_compute = 1;    // the engine returns the virial, not pair forces
  centroidstressflag = CENTROID_NOTAVAIL;
}

/* ---------------------------------------------------------------------- */

PairLoupMD::~PairLoupMD()
{
  if (copymode) return;

  if (handle) loup_destroy(handle);
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

/* ---------------------------------------------------------------------- */

void PairLoupMD::allocate()
{
  allocated = 1;
  int np1 = atom->ntypes + 1;

  memory->create(setflag, np1, np1, "pair:setflag");
  memory->create(cutsq, np1, np1, "pair:cutsq");
  map = new int[np1];
}

/* ----------------------------------------------------------------------
   pair_style loupmd [device N] [skin S] [ghost yes|no] [ahead yes|no]
------------------------------------------------------------------------- */

void PairLoupMD::settings(int narg, char **arg)
{
  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "device") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "pair_style loupmd device", error);
      device = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (device < 0) error->all(FLERR, "Pair style loupmd: device must be >= 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "skin") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "pair_style loupmd skin", error);
      skin_user = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (skin_user < 0.0) error->all(FLERR, "Pair style loupmd: skin must be >= 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "ghost") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "pair_style loupmd ghost", error);
      ghost = utils::logical(FLERR, arg[iarg + 1], false, lmp) != 0;
      iarg += 2;
    } else if (strcmp(arg[iarg], "ahead") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "pair_style loupmd ahead", error);
      ahead = utils::logical(FLERR, arg[iarg + 1], false, lmp) != 0;
      iarg += 2;
    } else
      error->all(FLERR, "Unknown pair_style loupmd keyword: {}", arg[iarg]);
  }
}

/* ----------------------------------------------------------------------
   pair_coeff * * model.loup El1 El2 ...
------------------------------------------------------------------------- */

void PairLoupMD::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  if (narg != 3 + atom->ntypes)
    error->all(FLERR, "Pair style loupmd: use 'pair_coeff * * model.loup' followed by one element per atom type");
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Pair style loupmd: pair_coeff must start with '* *'");

  map_element2type(narg - 3, arg + 3);

  bundle = arg[2];
  type2z.assign(atom->ntypes + 1, 0);
  for (int i = 1; i <= atom->ntypes; i++) {
    // The engine evaluates every atom of the system, so no type can be left out (NULL).
    if (map[i] < 0) error->all(FLERR, "Pair style loupmd does not support NULL elements: it evaluates every atom");
    type2z[i] = atomic_number(elements[map[i]]);
    if (type2z[i] == 0) error->all(FLERR, "Pair style loupmd: '{}' is not an element symbol", elements[map[i]]);
  }
}

/* ---------------------------------------------------------------------- */

void PairLoupMD::init_style()
{
  if (strcmp(update->unit_style, "metal") != 0) error->all(FLERR, "Pair style loupmd requires metal units");
  if (atom->tag_enable == 0) error->all(FLERR, "Pair style loupmd requires atom IDs");
  if (comm->nprocs != 1)
    error->all(FLERR, "Pair style loupmd runs on one MPI process: the engine evaluates the whole system on one GPU");
  if (domain->dimension != 3 || !domain->xperiodic || !domain->yperiodic || !domain->zperiodic)
    error->all(FLERR, "Pair style loupmd requires a 3d system that is periodic in x, y and z (put an isolated "
                      "system in a box with more than 2 (cutoff + skin) of vacuum)");
  if (atom->natoms > MAXSMALLINT) error->all(FLERR, "Pair style loupmd: too many atoms");
  if (atom->natoms > 0 && !atom->tag_consecutive())
    error->all(FLERR, "Pair style loupmd requires consecutive atom IDs 1 .. N (use 'reset_atoms id')");
  if (bundle.empty()) error->all(FLERR, "Pair style loupmd: no pair_coeff given");

  const int dev = device >= 0 ? device : default_device();
  if (!handle || bundle != bundle_loaded || dev != device_loaded) {
    if (handle) loup_destroy(handle);
    handle = loup_create(bundle.c_str(), dev);
    if (!handle)
      error->one(FLERR, "Pair style loupmd: could not load {} on CUDA device {}: {}", bundle, dev,
                 loup_last_error(nullptr));
    bundle_loaded = bundle;
    device_loaded = dev;
    z_sent.clear();
  }
  rmax = loup_cutoff(handle);

  // `ahead yes`: the engine queues each next force pass ahead of the call that needs it, which takes the launch off
  // every step (3-7% of the step on cells of a few hundred atoms under WSL 2). Between two force calls the engine's
  // GPU stream then holds work that cannot run, and other GPU work of this process could wait for it forever. So it
  // is off by default, and never on with the KOKKOS or GPU package active: switch it on only when the engine is the
  // process's one GPU user.
  const bool other_gpu_user = lmp->kokkos != nullptr || !modify->get_fix_by_style("^GPU").empty();
  loup_set_evaluate_ahead(handle, ahead && !other_gpu_user ? 1 : 0);

  int zs[128];
  const int ns = loup_species(handle, zs, 128);
  for (int i = 1; i <= atom->ntypes; i++) {
    bool found = false;
    for (int s = 0; s < ns && s < 128; s++) found = found || zs[s] == type2z[i];
    if (!found) {
      std::string have;
      for (int s = 0; s < ns && s < 128; s++) have += std::string(s ? " " : "") + kSymbols[zs[s] < kNumSymbols ? zs[s] : 0];
      error->all(FLERR, "Pair style loupmd: element {} (atom type {}) is not in {}, which holds: {}",
                 kSymbols[type2z[i]], i, bundle, have);
    }
  }

  // the engine builds its own neighbour list: none is requested from LAMMPS
  system_dirty = true;
}

/* ---------------------------------------------------------------------- */

double PairLoupMD::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
  // The engine builds its own periodic neighbour list and needs no ghost atoms. With `ghost no` LAMMPS is told a
  // cutoff of zero, so (unless another style or comm_modify asks for more) it creates no ghost atoms and skips their
  // communication in every step.
  return ghost ? rmax : 0.0;
}

/* ----------------------------------------------------------------------
   The engine keeps the system by atom ID (its atom k is the atom with ID k + 1), so LAMMPS may sort and reorder its
   atoms freely. It is given the system again when the atom count, an atom's type or the skin changed; that is
   looked for when a run starts and on reneighbouring steps.
------------------------------------------------------------------------- */

void PairLoupMD::ensure_system()
{
  const int n = (int) atom->natoms;
  if (atom->nlocal != n)
    error->one(FLERR, "Pair style loupmd: {} of {} atoms are on this process", atom->nlocal, atom->natoms);
  if (!system_dirty && neighbor->ago != 0 && (int) z_sent.size() == n) return;

  const double skin = skin_user >= 0.0 ? skin_user : neighbor->skin;
  const tagint *tag = atom->tag;
  const int *type = atom->type;
  std::vector<int> z(n, 0);
  for (int i = 0; i < n; i++) {
    if (tag[i] < 1 || tag[i] > n)
      error->one(FLERR, "Pair style loupmd requires consecutive atom IDs 1 .. N (use 'reset_atoms id')");
    z[tag[i] - 1] = type2z[type[i]];
  }
  if (!system_dirty && skin == skin_sent && z == z_sent) return;

  std::vector<double> m(n, 0.0);
  for (int i = 0; i < n; i++) {
    if (z[i] == 0) error->one(FLERR, "Pair style loupmd requires consecutive atom IDs 1 .. N (use 'reset_atoms id')");
    m[tag[i] - 1] = atom->rmass ? atom->rmass[i] : atom->mass[type[i]];
  }
  // A run command sets system_dirty (init_style: the masses may be new). The same system again is not sent: the
  // engine would drop its list, its tuned plans and, under run_style loupmd, the thermostat's state.
  if (skin == skin_sent && z == z_sent && m == m_sent) {
    system_dirty = false;
    return;
  }
  if (loup_set_system(handle, n, z.data(), m.data(), skin) != 0)
    error->one(FLERR, "Pair style loupmd: the engine refused the system: {}", loup_last_error(handle));
  serial++;
  z_sent.swap(z);
  m_sent.swap(m);
  skin_sent = skin;
  system_dirty = false;
}

/* ---------------------------------------------------------------------- */

LoupHandle *PairLoupMD::engine()
{
  ensure_system();
  return handle;
}

/* ---------------------------------------------------------------------- */

void PairLoupMD::box_to_cell(double *cell) const
{
  // lattice vectors as rows: a = (xprd, 0, 0), b = (xy, yprd, 0), c = (xz, yz, zprd)
  cell[0] = domain->xprd; cell[1] = 0.0;          cell[2] = 0.0;
  cell[3] = domain->xy;   cell[4] = domain->yprd; cell[5] = 0.0;
  cell[6] = domain->xz;   cell[7] = domain->yz;   cell[8] = domain->zprd;
}

/* ---------------------------------------------------------------------- */

void PairLoupMD::refuse_per_atom() const
{
  if (eflag_atom || vflag_atom)
    error->all(FLERR, "Pair style loupmd does not provide per-atom energies or virials");
}

/* ----------------------------------------------------------------------
   Host arrays: positions go to the engine and forces come back through host memory (loup_evaluate). The Kokkos
   style loupmd/kk keeps both on the GPU.
------------------------------------------------------------------------- */

void PairLoupMD::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag, 0);
  refuse_per_atom();
  ensure_system();

  const int n = atom->nlocal;
  double **x = atom->x;
  double **f = atom->f;
  const tagint *tag = atom->tag;
  const imageint *image = atom->image;

  // Unwrapped positions: the engine's skin check needs them continuous across LAMMPS's re-wrapping into the box.
  xbuf.resize(3 * (size_t) n);
  fbuf.resize(3 * (size_t) n);
  for (int i = 0; i < n; i++) {
    double u[3];
    domain->unmap(x[i], image[i], u);
    const size_t k = 3 * (size_t) (tag[i] - 1);
    xbuf[k] = u[0];
    xbuf[k + 1] = u[1];
    xbuf[k + 2] = u[2];
  }

  double cell[9], stress[9], energy = 0.0;
  box_to_cell(cell);
  // The stress is formed in every call: steps that need it then run the engine's captured graph as the others do,
  // instead of re-capturing it.
  if (loup_evaluate(handle, xbuf.data(), cell, 1, &energy, fbuf.data(), stress) != 0)
    error->one(FLERR, "Pair style loupmd: the force evaluation failed: {}", loup_last_error(handle));

  for (int i = 0; i < n; i++) {
    const size_t k = 3 * (size_t) (tag[i] - 1);
    f[i][0] += fbuf[k];
    f[i][1] += fbuf[k + 1];
    f[i][2] += fbuf[k + 2];
  }

  if (eflag_global) eng_vdwl += energy;
  if (vflag_global) {
    // the engine's stress is dE/d(strain) / V; LAMMPS's virial is -dE/d(strain), as xx yy zz xy xz yz
    const double vol = domain->xprd * domain->yprd * domain->zprd;
    virial[0] -= stress[0] * vol;
    virial[1] -= stress[4] * vol;
    virial[2] -= stress[8] * vol;
    virial[3] -= stress[1] * vol;
    virial[4] -= stress[2] * vol;
    virial[5] -= stress[5] * vol;
  }
}
