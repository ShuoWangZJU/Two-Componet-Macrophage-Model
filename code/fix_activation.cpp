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

#include <math.h>
#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include "fix_activation.h"
#include "update.h"
#include "respa.h"
#include "atom.h"
#include "atom_vec.h"
#include "bond.h"
#include "force.h"
#include "group.h"
#include "pair.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define BIG 1.0e20

/* ---------------------------------------------------------------------- */

FixActivation::FixActivation(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg != 10) error->all(FLERR,"Illegal fix activation command");

  MPI_Comm_rank(world,&me);

  nevery = force->inumeric(FLERR,arg[3]);
  if (nevery <= 0) error->all(FLERR,"Illegal fix activation command");
  
  char gn[50];
  sprintf(gn,arg[4]);
  int igroup_trigger = group->find(arg[4]);
  if (igroup_trigger == -1) error->all(FLERR,"Could not find trigger group ID for fix Activation command");
  groupbit_trigger = group->bitmask[igroup_trigger];
  
  int igroup_active = group->find_or_create(arg[5]);
  groupbit_active = group->bitmask[igroup_active];
  inversebit_active = group->inversemask[igroup_active];
  
  int igroup_active_prt = group->find_or_create(arg[6]);
  groupbit_active_prt = group->bitmask[igroup_active_prt];
  inversebit_active_prt = group->inversemask[igroup_active_prt];
  
  inactive_type = force->inumeric(FLERR,arg[7]);
  active_type = force->inumeric(FLERR,arg[8]);
  r_active = force->numeric(FLERR,arg[9]);

  // error check

  if (atom->molecular != 1)
    error->all(FLERR,"Cannot use fix activation with non-molecular systems");

  nmax = 0;
}

/* ---------------------------------------------------------------------- */

FixActivation::~FixActivation()
{
  
}

/* ---------------------------------------------------------------------- */

int FixActivation::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixActivation::init()
{
  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;

  // need a half neighbor list, built every Nevery steps
 
  dt = update->dt;
  
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;
  neighbor->requests[irequest]->occasional = 1;

  lastcheck = -1;
}

/* ---------------------------------------------------------------------- */

void FixActivation::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixActivation::post_integrate()
{
  int i,j,k,m,n,ii,jj,inum,jnum,itype,jtype,n1,n2,n3,possible,i1,i2,type_bond, jb, jc;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq,xi[3],rmton;
  int *ilist,*jlist,*numneigh,**firstneigh;
  tagint *slist;

  if (update->ntimestep % nevery) return;
  
  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;
  int nprocs = comm->nprocs;
  me = comm->me;

  double **x = atom->x;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;

  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  int *type = atom->type;
  neighbor->build_one(list,1);
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  int **bond_type = atom->bond_type;
  
  trigger_num_local = 0;
  memory->create(trigger_list_local,atom->nmax,"activation:trigger_list_local");
  memory->create(trigger_x_local,atom->nmax,"activation:trigger_x_local");
  memory->create(trigger_y_local,atom->nmax,"activation:trigger_y_local");
  memory->create(trigger_z_local,atom->nmax,"activation:trigger_z_local");
  for (i = 0; i < nlocal; i++) {
	if (mask[i] & groupbit_trigger) {
	  xi[0] = x[i][0];
	  xi[1] = x[i][1];
	  xi[2] = x[i][2];
	  domain->unmap(xi,atom->image[i]);
	  trigger_list_local[trigger_num_local] = atom->tag[i];
	  trigger_x_local[trigger_num_local] = xi[0];
	  trigger_y_local[trigger_num_local] = xi[1];
	  trigger_z_local[trigger_num_local] = xi[2];
	  trigger_num_local++;
	}
  }
  MPI_Allreduce(&trigger_num_local,&trigger_num_all,1,MPI_INT,MPI_SUM,world);
  memory->create(trigger_num_proc,nprocs,"activation:trigger_num_proc");
  memory->create(displs,nprocs,"activation:displs");
  if (nprocs > 1) MPI_Allgather(&trigger_num_local, 1, MPI_INT, trigger_num_proc, 1, MPI_INT, world);
  
  int kn;
  kn = 0;
  for (i = 0; i < nprocs; i++){
    displs[i] = kn;
    kn += trigger_num_proc[i];
  }
  memory->create(trigger_list_all,kn,"activation:trigger_list_all");
  memory->create(trigger_x_all,kn,"activation:trigger_x_all");
  memory->create(trigger_y_all,kn,"activation:trigger_y_all");
  memory->create(trigger_z_all,kn,"activation:trigger_z_all");
  if (nprocs > 1) {
	MPI_Allgatherv(trigger_list_local, trigger_num_local, MPI_INT, trigger_list_all, trigger_num_proc, displs, MPI_INT, world);
	MPI_Allgatherv(trigger_x_local, trigger_num_local, MPI_DOUBLE, trigger_x_all, trigger_num_proc, displs, MPI_DOUBLE, world);
	MPI_Allgatherv(trigger_y_local, trigger_num_local, MPI_DOUBLE, trigger_y_all, trigger_num_proc, displs, MPI_DOUBLE, world);
	MPI_Allgatherv(trigger_z_local, trigger_num_local, MPI_DOUBLE, trigger_z_all, trigger_num_proc, displs, MPI_DOUBLE, world);
  }
  
  // set the mem atoms which are close to the protru atoms to be active
  for (i=0; i < nlocal; i++){
    if ((mask[i] & groupbit) && (type[i] == inactive_type || type[i] == active_type)){
	  mask[i] &= inversebit_active;
	  type[i] = inactive_type;    // reset all mem atoms to be inactive
	  xi[0] = x[i][0];
	  xi[1] = x[i][1];
	  xi[2] = x[i][2];
	  domain->unmap(xi,atom->image[i]);
	  for (j=0; j < trigger_num_all; j++) {
		rmton = sqrt((xi[0] - trigger_x_all[j]) * (xi[0] - trigger_x_all[j]) + (xi[1] - trigger_y_all[j]) * (xi[1] - trigger_y_all[j]) + (xi[2] - trigger_z_all[j]) * (xi[2] - trigger_z_all[j]));
		if (rmton < r_active) {
		  type[i] = active_type;
		  mask[i] |= groupbit_active;
		  break;
		}
	  }
	}
	else if ((mask[i] & groupbit) && !(mask[i] & groupbit_trigger) && type[i] != inactive_type && type[i] != active_type){
	  mask[i] &= inversebit_active_prt;
	  xi[0] = x[i][0];
	  xi[1] = x[i][1];
	  xi[2] = x[i][2];
	  domain->unmap(xi,atom->image[i]);
	  for (j=0; j < trigger_num_all; j++) {
		rmton = sqrt((xi[0] - trigger_x_all[j]) * (xi[0] - trigger_x_all[j]) + (xi[1] - trigger_y_all[j]) * (xi[1] - trigger_y_all[j]) + (xi[2] - trigger_z_all[j]) * (xi[2] - trigger_z_all[j]));
		if (rmton < r_active) {
		  mask[i] |= groupbit_active_prt;
		  break;
		}
	  }
	}
  }
  
  memory->destroy(trigger_list_local);
  memory->destroy(trigger_x_local);
  memory->destroy(trigger_y_local);
  memory->destroy(trigger_z_local);
  memory->destroy(trigger_list_all);
  memory->destroy(trigger_x_all);
  memory->destroy(trigger_y_all);
  memory->destroy(trigger_z_all);
  memory->destroy(trigger_num_proc);
  memory->destroy(displs);
}

/* ---------------------------------------------------------------------- */

void FixActivation::post_integrate_respa(int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_integrate();
}
