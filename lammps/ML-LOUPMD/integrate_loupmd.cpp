// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This file is part of the LoupMD interface to LAMMPS (https://github.com/FZWINGEL-UBT/LoupMD, lammps/).
   It is distributed under the GNU General Public License, as LAMMPS is.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "integrate_loupmd.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "modify.h"
#include "neighbor.h"
#include "output.h"
#include "pair_loupmd.h"
#include "timer.h"
#include "update.h"

#include "loupmd.h"

#include <cstring>

using namespace LAMMPS_NS;

// LAMMPS's metal velocities are in A/ps; the engine's are in ASE units (A per ASE time unit; 1 fs is
// 0.09822694788464063 of them), so v[A/ps] = 1000 * 0.0982269... * v[ASE].
static constexpr double kVelLmpPerAse = 98.22694788464063;

/* ----------------------------------------------------------------------
   run_style loupmd nve | langevin T damp seed
------------------------------------------------------------------------- */

IntegrateLoupMD::IntegrateLoupMD(LAMMPS *lmp, int narg, char **arg) :
    Integrate(lmp, narg, arg), langevin(0), t_target(0.0), damp(0.0), seed(0), pair(nullptr), triclinic(0)
{
  if (narg < 1) error->all(FLERR, "run_style loupmd needs an ensemble: nve, or langevin T damp seed");
  if (strcmp(arg[0], "nve") == 0) {
    if (narg != 1) error->all(FLERR, "run_style loupmd nve takes no further arguments");
  } else if (strcmp(arg[0], "langevin") == 0) {
    if (narg != 4) error->all(FLERR, "run_style loupmd langevin needs: T damp seed");
    langevin = 1;
    t_target = utils::numeric(FLERR, arg[1], false, lmp);
    damp = utils::numeric(FLERR, arg[2], false, lmp);
    seed = (unsigned long long) utils::bnumeric(FLERR, arg[3], false, lmp);
    if (t_target < 0.0 || damp <= 0.0) error->all(FLERR, "run_style loupmd langevin: T must be >= 0 and damp > 0");
  } else
    error->all(FLERR, "Unknown run_style loupmd ensemble: {} (nve or langevin)", arg[0]);
}

/* ---------------------------------------------------------------------- */

void IntegrateLoupMD::init()
{
  Integrate::init();

  pair = dynamic_cast<PairLoupMD *>(force->pair);
  if (!pair) error->all(FLERR, "run_style loupmd requires pair_style loupmd (alone, not inside pair_style hybrid)");
  if (lmp->kokkos)
    error->all(FLERR, "run_style loupmd runs without the KOKKOS package switched on (no '-k on' / '-sf kk'): the "
                      "engine integrates on the GPU itself");
  if (force->bond || force->angle || force->dihedral || force->improper || force->kspace)
    error->all(FLERR, "run_style loupmd: the engine computes all forces, so bonded and kspace styles cannot be used");

  // as Verlet: how the pair style is asked for the virial, and the computes that need energy / virial on a step
  if (force->newton_pair) virial_style = VIRIAL_FDOTR;
  else virial_style = VIRIAL_PAIR;
  ev_setup();
  if (modify->get_fix_by_id("package_omp")) external_force_clear = 1;

  triclinic = domain->triclinic;
}

/* ----------------------------------------------------------------------
   the engine integrates: nothing LAMMPS does during the steps would take effect
------------------------------------------------------------------------- */

void IntegrateLoupMD::checks()
{
  if (modify->n_initial_integrate || modify->n_post_integrate || modify->n_pre_exchange || modify->n_pre_neighbor ||
      modify->n_post_neighbor || modify->n_pre_force || modify->n_pre_reverse || modify->n_post_force_any ||
      modify->n_final_integrate || modify->n_end_of_step)
    error->all(FLERR, "run_style loupmd: the engine integrates, so fixes that act during the steps cannot be used. "
                      "Remove fix nve / fix langevin and give the ensemble to run_style; for other fixes use the "
                      "default run_style with pair_style loupmd");
}

/* ---------------------------------------------------------------------- */

void IntegrateLoupMD::setup(int flag)
{
  if (comm->me == 0 && screen) {
    fputs("Setting up LoupMD run ...\n", screen);
    if (flag) {
      utils::print(screen, "  Unit style    : {}\n"
                           "  Current step  : {}\n"
                           "  Time step     : {}\n",
                   update->unit_style, update->ntimestep, update->dt);
      if (langevin) utils::print(screen, "  Ensemble      : langevin {} K, damp {} ps (the engine's integrator)\n", t_target, damp);
      else utils::print(screen, "  Ensemble      : nve (the engine's integrator)\n");
      timer->print_timeout(screen);
    }
  }

  checks();
  update->setupflag = 1;

  // setup domain, communication and neighboring as Verlet does

  atom->setup();
  modify->setup_pre_exchange();
  if (triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  domain->reset_box();
  comm->setup();
  if (neighbor->style) neighbor->setup_bins();
  comm->exchange();
  if (atom->sortfreq > 0) atom->sort();
  comm->borders();
  if (triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
  domain->image_check();
  domain->box_too_small_check();
  modify->setup_pre_neighbor();
  neighbor->build(1);
  modify->setup_post_neighbor();
  neighbor->ncalls = 0;

  // forces of the current state through the pair style, then hand the state to the engine

  force->setup();
  forces(update->ntimestep);
  modify->setup_pre_reverse(eflag, vflag);
  if (force->newton) comm->reverse_comm();

  modify->setup(vflag);
  output->setup(flag);
  update->setupflag = 0;

  push_state();
}

/* ----------------------------------------------------------------------
   setup without output; flag = 0: the previous run left everything current
------------------------------------------------------------------------- */

void IntegrateLoupMD::setup_minimal(int flag)
{
  checks();
  update->setupflag = 1;

  if (flag) {
    modify->setup_pre_exchange();
    if (triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    domain->reset_box();
    comm->setup();
    if (neighbor->style) neighbor->setup_bins();
    comm->exchange();
    comm->borders();
    if (triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
    domain->image_check();
    domain->box_too_small_check();
    modify->setup_pre_neighbor();
    neighbor->build(1);
    modify->setup_post_neighbor();
    neighbor->ncalls = 0;
  }

  forces(update->ntimestep);
  modify->setup_pre_reverse(eflag, vflag);
  if (force->newton) comm->reverse_comm();

  modify->setup(vflag);
  update->setupflag = 0;

  push_state();
}

/* ----------------------------------------------------------------------
   Blocks of engine steps up to the next step at which LAMMPS writes output; there the state comes back, LAMMPS
   wraps the atoms and rebuilds its ghosts, and the pair style evaluates energy, forces and virial for the output.
------------------------------------------------------------------------- */

void IntegrateLoupMD::run(int n)
{
  LoupHandle *engine = pair->engine();
  const double dt_fs = update->dt * 1000.0;
  const double friction_per_fs = langevin ? 1.0 / (damp * 1000.0) : 0.0;

  timer->init_timeout();
  bigint remaining = n;
  while (remaining > 0) {
    if (timer->check_timeout((int) (n - remaining))) {
      update->nsteps = (int) (n - remaining);
      break;
    }

    bigint nblock = remaining;
    if (output->next > update->ntimestep && output->next - update->ntimestep < nblock)
      nblock = output->next - update->ntimestep;

    timer->stamp();
    LoupMDStats stats;
    const int rc = langevin ? loup_run_langevin(engine, (int) nblock, dt_fs, t_target, friction_per_fs, seed, &stats)
                            : loup_run_nve(engine, (int) nblock, dt_fs, &stats);
    if (rc != 0) error->one(FLERR, "run_style loupmd: the engine's MD run failed (see the message above)");
    timer->stamp(Timer::PAIR);

    update->ntimestep += nblock;
    remaining -= nblock;

    if (update->ntimestep == output->next || remaining == 0) {
      pull_state();
      reneighbor();
      timer->stamp(Timer::COMM);
      forces(update->ntimestep);
      timer->stamp(Timer::PAIR);
      if (update->ntimestep == output->next) {
        output->write(update->ntimestep);
        timer->stamp(Timer::OUTPUT);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void IntegrateLoupMD::force_clear()
{
  if (external_force_clear) return;

  size_t n = atom->nlocal;
  if (force->newton) n += atom->nghost;
  if (n) memset(&atom->f[0][0], 0, 3 * sizeof(double) * n);
}

/* ---------------------------------------------------------------------- */

void IntegrateLoupMD::forces(bigint step)
{
  ev_set(step);
  force_clear();
  if (pair_compute_flag) force->pair->compute(eflag, vflag);
}

/* ---------------------------------------------------------------------- */

void IntegrateLoupMD::reneighbor()
{
  if (triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  comm->exchange();
  comm->borders();
  if (triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
  neighbor->build(1);
}

/* ----------------------------------------------------------------------
   The engine keeps atoms by ID (its atom k is the atom with ID k + 1) and positions unwrapped.
------------------------------------------------------------------------- */

void IntegrateLoupMD::push_state()
{
  LoupHandle *engine = pair->engine();
  const int n = atom->nlocal;
  double **x = atom->x;
  double **v = atom->v;
  const tagint *tag = atom->tag;
  const imageint *image = atom->image;

  xbuf.resize(3 * (size_t) n);
  vbuf.resize(3 * (size_t) n);
  for (int i = 0; i < n; i++) {
    double u[3];
    domain->unmap(x[i], image[i], u);
    const size_t k = 3 * (size_t) (tag[i] - 1);
    for (int c = 0; c < 3; c++) {
      xbuf[k + c] = u[c];
      vbuf[k + c] = v[i][c] / kVelLmpPerAse;
    }
  }
  double cell[9];
  pair->cell(cell);
  if (loup_set_state(engine, xbuf.data(), vbuf.data(), cell) != 0)
    error->one(FLERR, "run_style loupmd: the engine refused the state (see the message above)");
}

/* ---------------------------------------------------------------------- */

void IntegrateLoupMD::pull_state()
{
  LoupHandle *engine = pair->engine();
  const int n = atom->nlocal;
  double **x = atom->x;
  double **v = atom->v;
  const tagint *tag = atom->tag;
  imageint *image = atom->image;

  xbuf.resize(3 * (size_t) n);
  vbuf.resize(3 * (size_t) n);
  if (loup_get_state(engine, xbuf.data(), vbuf.data(), nullptr, nullptr) != 0)
    error->one(FLERR, "run_style loupmd: could not read the engine's state (see the message above)");

  // unwrapped positions with zero image flags, wrapped into the box by LAMMPS (which sets the flags)
  const imageint zero_image = ((imageint) IMGMAX << IMG2BITS) | ((imageint) IMGMAX << IMGBITS) | IMGMAX;
  for (int i = 0; i < n; i++) {
    const size_t k = 3 * (size_t) (tag[i] - 1);
    for (int c = 0; c < 3; c++) {
      x[i][c] = xbuf[k + c];
      v[i][c] = vbuf[k + c] * kVelLmpPerAse;
    }
    image[i] = zero_image;
    domain->remap(x[i], image[i]);
  }
}
