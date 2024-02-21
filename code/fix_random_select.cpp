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
#include "fix_random_select.h"
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

FixRandomSelect::FixRandomSelect(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  int i, m, n;
  
  if (narg != 7 && narg != 9 && narg != 11) error->all(FLERR,"Illegal fix random/select command");

  m = 3;
  nevery = force->inumeric(FLERR,arg[m++]);
  if (nevery <= 0) error->all(FLERR,"Illegal fix random/select command");
  
  group_num = group->count(igroup);
  if (strcmp(arg[m],"number") == 0) {
	m++;
	max_sel_num = force->inumeric(FLERR,arg[m++]);
	select_mode = 0;
  }
  else if (strcmp(arg[m],"percentage") == 0) {
	m++;
	n = force->numeric(FLERR,arg[m++]);
	max_sel_num = n * group_num;
	select_mode = 0;
  }
  else if (strcmp(arg[m],"probability") == 0) {
	m++;
	probability = force->numeric(FLERR,arg[m++]);
	select_mode = 1;
  }
  else error->all(FLERR,"Illegal fix random/select command");
  
  int igroup_selected = group->find_or_create(arg[m++]);
  groupbit_selected = group->bitmask[igroup_selected];
  inversebit_selected = group->inversemask[igroup_selected];
  
  if (narg == m){
    duration_flag = 0;
	exclude_flag = 0;
  }
  else if (narg == m + 2){
	if (strcmp(arg[m],"duration") == 0) {
      m++;
	  duration_flag= 1;
      duration_time = force->inumeric(FLERR,arg[m++]);
	}
	else if (strcmp(arg[m],"exclude") == 0) {
	  m++;
	  igroup_exclude = group->find(arg[m++]);
	  if (igroup_exclude == -1) error->all(FLERR,"Could not find exclude group ID for fix random/select command");
	  exclude_flag = 1;
	  groupbit_exclude = group->bitmask[igroup_exclude];
	}
	else error->all(FLERR,"Illegal fix random/select command");
  }
  else if (narg == m + 4){
	if (strcmp(arg[m],"duration") == 0 && strcmp(arg[m+2],"exclude") == 0) {
	  m++;
	  duration_flag= 1;
      duration_time = force->inumeric(FLERR,arg[m++]);
	  m++;
	  igroup_exclude = group->find(arg[m++]);
	  if (igroup_exclude == -1) error->all(FLERR,"Could not find exclude group ID for fix random/select command");
	  exclude_flag = 1;
	  groupbit_exclude = group->bitmask[igroup_exclude];
	}
	else error->all(FLERR,"Illegal fix random/select command");
  }
  else error->all(FLERR,"Illegal fix random/select command");

  // error check

  if (atom->molecular != 1)
    error->all(FLERR,"Cannot use fix random/select with non-molecular systems");

  maxatom = atom->nmax;
  comm_forward = 1;
  
  memory->create(initial_timestep,maxatom,"random/select:initial_timestep");
  int nlocal = atom->nlocal;
  int nlocalghost = atom->nlocal + atom->nghost;
  for (i = 0; i < nlocal; i++){
	initial_timestep[i] = -3;
  }
  fprintf(stderr, "In Fix RandomSelecet at proc %d when time = %d: class create \n", comm->me, update->ntimestep);
  existing_selected = 0;
}

/* ---------------------------------------------------------------------- */

FixRandomSelect::~FixRandomSelect()
{
  memory->destroy(initial_timestep);
}

/* ---------------------------------------------------------------------- */

int FixRandomSelect::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixRandomSelect::init()
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

void FixRandomSelect::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixRandomSelect::post_integrate()
{
  int i,j,k,m,n,ii,jj,inum,jnum,itype,jtype,n1,n2,n3,possible,i1,i2, jb, jc, selected_index, selected_ID, n_add, n_add_all, n_sub, n_sub_all;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq,xi[3],rmton;
  int *ilist,*jlist,*numneigh,**firstneigh;
  tagint *slist;
  
  int time = update->ntimestep;

  if (time % nevery) return;
  
  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;
  int nprocs = comm->nprocs;
  me = comm->me;

  double **x = atom->x;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int *type = atom->type;
  neighbor->build_one(list,1);
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  
  if (duration_flag) {
    for (i = 0; i < nlocal; i++) {
	  if (mask[i] & groupbit_selected) {
	    fprintf(stderr, "In Fix RandomSelecet 01 at proc %d when time = %d: atom %d, i = %d, initial_timestep[i] = %d, duration_time = %d \n", me, time, atom->tag[i], i, initial_timestep[i], duration_time);
	  }
    }
  }
  
  // selection, once at a time
  selected_index = -1;
  int count_exclude = 0;
  if (exclude_flag) count_exclude = group->count(igroup_exclude);
  int group_num_left = group_num - existing_selected - count_exclude;
  if (me == 0) {
    if (select_mode == 0 && existing_selected < max_sel_num) {
	  selected_index = rand() % group_num_left;
    }
    else if (select_mode == 1) {
	  for (int i = 0; i < group_num_left; i++) {
	    if (rand() % group_num_left < probability * group_num_left) {
	  	  selected_index = i;
		  break;
	    }
	  }
    }
  }
  MPI_Bcast(&selected_index, 1, MPI_INT, 0, world);
  
  // add selected atom into group_selected
  n_add = 0;
  if (selected_index >= 0) {
    // put all the group atoms into one list
    group_num_local = 0;
    memory->create(group_list_local,maxatom,"random/select:group_list_local");
    for (i = 0; i < nlocal; i++) {
	  if ((mask[i] & groupbit) && !(mask[i] & groupbit_selected) && (exclude_flag == 0 || (exclude_flag == 1 && !(mask[i] & groupbit_exclude)))) {
	    group_list_local[group_num_local] = atom->tag[i];
	    group_num_local++;  
	  }
    }
    MPI_Allreduce(&group_num_local,&group_num_all,1,MPI_INT,MPI_SUM,world);
    if (group_num_all != group_num_left) fprintf(stderr, "In Fix RandomSelecet at proc %d when time = %d: group_num_all = %d, group_num_left = %d \n", me, time, group_num_all, group_num_left);
    memory->create(group_num_proc,nprocs,"random/select:group_num_proc");
    memory->create(displs,nprocs,"random/select:displs");
    if (nprocs > 1) MPI_Allgather(&group_num_local, 1, MPI_INT, group_num_proc, 1, MPI_INT, world);
    
    int kn;
    kn = 0;
    for (i = 0; i < nprocs; i++){
      displs[i] = kn;
      kn += group_num_proc[i];
    }
    if (kn != group_num_left) fprintf(stderr, "In Fix RandomSelecet at proc %d when time = %d: kn = %d, group_num_left = %d \n", me, time, kn, group_num_left);
    memory->create(group_list_all,kn,"random/select:group_list_all");
    if (nprocs > 1) {
	  MPI_Allgatherv(group_list_local, group_num_local, MPI_INT, group_list_all, group_num_proc, displs, MPI_INT, world);
    }
	
	for (i = 0; i < nlocal; i++) {
	  if (atom->tag[i] == group_list_all[selected_index]) {
		fprintf(stderr, "In Fix RandomSelecet at proc %d when time = %d: atom %d was added to group selected, i = %d \n", me, time, atom->tag[i], i);
		mask[i] |= groupbit_selected;
		initial_timestep[i] = time;
		n_add++;
		break;
	  }
	}
    
    memory->destroy(group_list_local);
    memory->destroy(group_list_all);
    memory->destroy(group_num_proc);
    memory->destroy(displs);
  }
  MPI_Allreduce(&n_add,&n_add_all,1,MPI_INT,MPI_SUM,world);
  existing_selected += n_add_all;
  
  // remove selected atom from group_selected after duration time is up
  n_sub = 0;
  if (duration_flag) {
    for (i = 0; i < nlocal; i++) {
	  if (mask[i] & groupbit_selected) {
	    fprintf(stderr, "In Fix RandomSelecet 02 at proc %d when time = %d: atom %d, i = %d, initial_timestep[i] = %d, duration_time = %d \n", me, time, atom->tag[i], i, initial_timestep[i], duration_time);
		if (time > initial_timestep[i] + duration_time) {
		  fprintf(stderr, "In Fix RandomSelecet at proc %d when time = %d: atom %d was removed from group selected, i = %d, initial_timestep[i] = %d, duration_time = %d \n", me, time, atom->tag[i], i, initial_timestep[i], duration_time);
		  mask[i] &= inversebit_selected;
		  initial_timestep[i] = -2;
		  n_sub++;
		}
	  }
    }
  }
  MPI_Allreduce(&n_sub,&n_sub_all,1,MPI_INT,MPI_SUM,world);
  existing_selected -= n_sub_all;
}

/* ---------------------------------------------------------------------- */

void FixRandomSelect::post_integrate_respa(int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_integrate();
}


/* ---------------------------------------------------------------------- */

int FixRandomSelect::pack_forward_comm(int n, int *list, double *buf,
                                     int pbc_flag, int *pbc)
{
  int i,j,m;

  m = 0;

  for (i = 0; i < n; i++) {
    j = list[i];
	buf[m++] = ubuf(initial_timestep[j]).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixRandomSelect::unpack_forward_comm(int n, int first, double *buf)
{
  int i,j,k,m,last;
  int nlocal = atom->nlocal;

  m = 0;
  last = first + n;

  for (i = first; i < last; i++) {
	initial_timestep[i] = (int) ubuf(buf[m++]).i;
  }
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixRandomSelect::pack_exchange(int i, double *buf)
{
  int m, k;
  
  m = 0;
  buf[m++] = initial_timestep[i];
  return m;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based arrays from exchange with another proc
------------------------------------------------------------------------- */

int FixRandomSelect::unpack_exchange(int nlocal, double *buf)
{
  int m, k;
  
  m = 0;
  initial_timestep[nlocal] = static_cast<int> (buf[m++]);
  return m;
}