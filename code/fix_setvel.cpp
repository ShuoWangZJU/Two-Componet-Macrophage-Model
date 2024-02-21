/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <string.h>
#include <stdlib.h>
#include "fix_setvel.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "region.h"
#include "respa.h"
#include "input.h"
#include "comm.h"
#include "random_park.h"
#include "universe.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "force.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixSetVel::FixSetVel(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 9) error->all(FLERR,"Illegal fix setvel command");

  ind_const = force->inumeric(FLERR,arg[3]);
  xvalue = force->numeric(FLERR,arg[4]);
  yvalue = force->numeric(FLERR,arg[5]);
  zvalue = force->numeric(FLERR,arg[6]);
  step_each = force->inumeric(FLERR,arg[7]);
  int seed = force->inumeric(FLERR,arg[8]);

  // optional args

  random = NULL;
  iregion = -1;
  idregion = NULL;

  if (ind_const == 0)
    random = new RanPark(lmp,seed + comm->me);

  int iarg = 9;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix setvel command");
      iregion = domain->find_region(arg[iarg+1]);
      if (iregion == -1)
        error->all(FLERR,"Region ID for fix setvel does not exist");
      int n = strlen(arg[iarg+1]) + 1;
      idregion = new char[n];
      strcpy(idregion,arg[iarg+1]);
      iarg += 2;
    } else error->all(FLERR,"Illegal fix setvel command");
  }
}

/* ---------------------------------------------------------------------- */

FixSetVel::~FixSetVel()
{
  delete [] idregion;
  delete random;
}

/* ---------------------------------------------------------------------- */

int FixSetVel::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixSetVel::init()
{
    // set index and check validity of region

  if (iregion >= 0) {
    iregion = domain->find_region(idregion);
    if (iregion == -1)
      error->all(FLERR,"Region ID for fix setvel does not exist");
  }
}

/* ---------------------------------------------------------------------- */

void FixSetVel::post_integrate()
{
  int i;
  double factor;

  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  double *mass = atom->mass;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  if (update->ntimestep%step_each == 0){
    if (ind_const) {
      for (i = 0; i < nlocal; i++)
        if (mask[i] & groupbit) {
          if (iregion >= 0 &&
              !domain->regions[iregion]->match(x[i][0],x[i][1],x[i][2]))
            continue;

          v[i][0] = xvalue;
          v[i][1] = yvalue;
          v[i][2] = zvalue;
	}
    } else {
      for (i = 0; i < nlocal; i++)
        if (mask[i] & groupbit) {
          if (iregion >= 0 &&
              !domain->regions[iregion]->match(x[i][0],x[i][1],x[i][2]))
            continue;
          
          factor = sqrt(xvalue/mass[type[i]]);
          v[i][0] = factor*random->gaussian();
          v[i][1] = factor*random->gaussian();
          v[i][2] = factor*random->gaussian();
	}
    }
  }
}

/* ---------------------------------------------------------------------- */