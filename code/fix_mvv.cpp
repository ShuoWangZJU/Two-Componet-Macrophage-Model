/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   www.cs.sandia.gov/~sjplimp/lammps.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Igor Pivkin piv@dam.brown.edu
------------------------------------------------------------------------- */

#include "mpi.h"
#include "comm.h"
#include "fix_mvv.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "respa.h"
#include "error.h"
#include "memory.h"
#include "stdio.h"
#include "string.h"
using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixMVV::FixMVV(LAMMPS *lmp, int narg, char **arg) : Fix(lmp,narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal fix mvv command");

  // perform initial allocation of atom-based arrays
  // register with atom class
  nlevels = 2;//1st level for velocity, 2nd for force
  f_level = NULL;
  grow_arrays(atom->nmax);
  atom->add_callback(0);
  lambda=0.5;
}

/* ---------------------------------------------------------------------- */

FixMVV::~FixMVV()
{
  // if atom class still exists:
  //   unregister this fix so atom class doesn't invoke it any more

  if (atom) atom->delete_callback(id,0);

  // delete locally stored arrays

  memory->destroy(f_level);
}

/* ---------------------------------------------------------------------- */

int FixMVV::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixMVV::init()
{
  dtv = update->dt;
  dtf = update->dt * force->ftm2v;
  
}

/* ---------------------------------------------------------------------- */

void FixMVV::initial_integrate(int vflag)
{


  double dtfm;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  //fprintf(stderr,"FixMVV\n");
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      dtfm = dtf / mass[type[i]];

      //save previous velocity and force in f_level array      
      f_level[i][0][0]=v[i][0];
      f_level[i][0][1]=v[i][1];
      f_level[i][0][2]=v[i][2];
      f_level[i][1][0]=f[i][0];
      f_level[i][1][1]=f[i][1];
      f_level[i][1][2]=f[i][2];

      //now integrate
      x[i][0] += dtv * v[i][0];
      x[i][1] += dtv * v[i][1];
      x[i][2] += dtv * v[i][2];
      x[i][0] += dtv * 0.5 * dtfm * f[i][0];
      x[i][1] += dtv * 0.5 * dtfm * f[i][1];
      x[i][2] += dtv * 0.5 * dtfm * f[i][2];

      v[i][0] += lambda * dtfm * f[i][0];
      v[i][1] += lambda * dtfm * f[i][1];
      v[i][2] += lambda * dtfm * f[i][2];
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixMVV::final_integrate()
{
  double dtfm;

  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      dtfm = dtf / mass[type[i]];
      v[i][0] = f_level[i][0][0] + 0.5 * dtfm * (f[i][0] + f_level[i][1][0]);
      v[i][1] = f_level[i][0][1] + 0.5 * dtfm * (f[i][1] + f_level[i][1][1]);
      v[i][2] = f_level[i][0][2] + 0.5 * dtfm * (f[i][2] + f_level[i][1][2]);
    }
    //fprintf(stderr,"v and f_lev: %lf %lf\n",v[i][0],f_level[i][0][0]);

  }
}

/* ---------------------------------------------------------------------- */

void FixMVV::initial_integrate_respa(int ilevel, int ilevel2, int ilevel3)
{

}

/* ---------------------------------------------------------------------- */

void FixMVV::final_integrate_respa(int ilevel)
{

}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays 
------------------------------------------------------------------------- */

void FixMVV::grow_arrays(int nmax)
{
  // f_level = memory->grow_3d_double_array(f_level,nmax,nlevels,3,"fix_mvv:f_level");
  memory->create(f_level,nmax,nlevels,3,"fix_mvv:f_level");
  //fprintf(stderr,"FixMVV::grow_arrays(%d)\n",nmax);
}

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays 
------------------------------------------------------------------------- */

void FixMVV::copy_arrays(int i, int j)
{
  //fprintf(stderr,"FixMVV:copy array %d -> ",nlevels);
  for (int k = 0; k < nlevels; k++) {
    f_level[j][k][0] = f_level[i][k][0];
    f_level[j][k][1] = f_level[i][k][1];
    f_level[j][k][2] = f_level[i][k][2];
  }
  //fprintf(stderr,"#\n");
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays 
------------------------------------------------------------------------- */

double FixMVV::memory_usage()
{
  double bytes;
  bytes = atom->nmax*nlevels*3* sizeof(double);
  return bytes;

}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc 
------------------------------------------------------------------------- */

int FixMVV::pack_exchange(int i, double *buf)
{
  //fprintf(stderr,"Pakc_exchange is called!\n");
  int m = 0;
  for (int k = 0; k < nlevels; k++) {
    buf[m++] = f_level[i][k][0];
    buf[m++] = f_level[i][k][1];
    buf[m++] = f_level[i][k][2];
  }
  return m;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based arrays from exchange with another proc 
------------------------------------------------------------------------- */

int FixMVV::unpack_exchange(int nlocal, double *buf)
{
  int m = 0;
  for (int k = 0; k < nlevels; k++) {
    f_level[nlocal][k][0] = buf[m++];
    f_level[nlocal][k][1] = buf[m++];
    f_level[nlocal][k][2] = buf[m++];
  }
  return m;
}
