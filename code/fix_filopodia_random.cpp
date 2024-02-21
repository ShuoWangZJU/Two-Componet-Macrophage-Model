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

#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "fix.h"
#include "fix_filopodia_random.h"
#include "fix_addlipid.h"
#include "atom.h"
#include "atom_vec.h"
#include "atom_vec_ellipsoid.h"
#include "group.h"
#include "force.h"
#include "update.h"
#include "comm.h"
#include "molecule.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "fix.h"
#include "compute.h"
#include "input.h"
#include "variable.h"
#include "domain.h"
#include "random_park.h"
#include "random_mars.h"
#include "math_extra.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum{ATOM,MOLECULE};
enum{ONE,RANGE,POLY};
enum{LAYOUT_UNIFORM,LAYOUT_NONUNIFORM,LAYOUT_TILED};    // several files

#define EPSILON 0.001
#define SMALL 1.0e-10

/* ---------------------------------------------------------------------- */

FixFilopodiaRandom::FixFilopodiaRandom(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  int arg_min = 16;
  if (narg < arg_min) error->all(FLERR,"Illegal fix filopodia/random command");

  // required args
  // arg      0           1                 2                3          4      5       6          7           8           9             10              11                 12                 13                14          15             16              17                     18+
  // fix   fix_ID   skeleton_group   filopodia/random   probability   N_max   seed   Nevery   filo_type   filo_num   no_adh_num   bond_filo_type   bond_ske_type   angle_filo_90_type   angle_filo_180_type   reverse   reverse_time   addlipid_ID   addlipid_fix_ID    add filo atoms into groups...
  int m = 3;
  int add_per, all_num;
  probability = force->numeric(FLERR,arg[m++]);
  N_max = force->inumeric(FLERR,arg[m++]);
  seed = force->inumeric(FLERR,arg[m++]);
  nevery = force->inumeric(FLERR,arg[m++]);
  filo_type = force->inumeric(FLERR,arg[m++]);
  filo_num = force->inumeric(FLERR,arg[m++]);
  no_adh_num = force->inumeric(FLERR,arg[m++]);
  bond_filo_type = force->inumeric(FLERR,arg[m++]);
  // bond_filo_X2_type = force->inumeric(FLERR,arg[m++]);
  bond_ske_type = force->inumeric(FLERR,arg[m++]);
  angle_filo_90_type = force->inumeric(FLERR,arg[m++]);
  angle_filo_180_type = force->inumeric(FLERR,arg[m++]);
  
  mem_type = force->inumeric(FLERR,arg[m++]);
  prt_type = force->inumeric(FLERR,arg[m++]);
  
  char *str = (char *) "pseudopod_grow_point";
  int igroup_pseu = group->find_or_create(str);
  groupbit_pseu = group->bitmask[igroup_pseu];
  inversebit_pseu = group->inversemask[igroup_pseu];
  
  int i, igroup_added;
  memory->create(groupbit_added,32,"fix_filopodia_random:groupbit_added");
  reverse_flag = 0;
  group_added_flag = 0;
  add_lipid_flag = 0;
  if (m < narg) {
	if ((strcmp(arg[m],"reverse") == 0) && m + 1 < narg) {
	  reverse_flag = 1;
	  m++;
	  reverse_time = force->inumeric(FLERR,arg[m++]);
	  if (reverse_time <= 0) error->all(FLERR,"Illegal reverse time for fix filopodia/random command");
	}
  }
  if (m < narg) {
	if ((strcmp(arg[m],"addlipid_ID") == 0) && m + 1 < narg) {
	  add_lipid_flag = 1;
	  m++;
	  char *fixid = arg[m++];
	  ifix = modify->find_fix(fixid);
	  if (ifix < 0) error->all(FLERR,"Illegal fix id for fix addlipid");
	}
  }
  if (m < narg) {
	if ((strcmp(arg[m],"dynamic_bond") == 0) && m + 1 < narg) {
	  bond_dyn_flag = 1;
	  m++;
	  bond_dyn_type = force->inumeric(FLERR,arg[m++]);
	}
  }
  if (m < narg) {
	if ((strcmp(arg[m],"added_group") == 0) && m + 1 < narg) {
	  group_added_flag = 1;
	  m++;
	  N_added_group = narg - m;
	  for (i = 0; i < N_added_group; i++) {
		igroup_added = group->find_or_create(arg[m++]);
		groupbit_added[i] = group->bitmask[igroup_added];
	  }
	}
	else error->all(FLERR,"Invalid arg in fix filopodia/random command");
  }

  // error check

  if (filo_type <= 0 || filo_type > atom->ntypes) error->all(FLERR,"Invalid atom type in fix filopodia/random command");
  if (no_adh_num >= filo_num) error->all(FLERR,"No_adh_num must be less than filo_num in fix filopodia/random command");
  if (seed <= 0) error->all(FLERR,"Seed must be a positive integer in fix filopodia/random command");
  seed += comm->me;
  random = new RanMars(lmp,seed);

  // check if there are fix addlipid
  /* if (add_lipid exist) {
	if (add_lipid is after filopodia) error->all(FLERR,"Fix filopodia must come after fix addlipid when they both exist");
	add_lipid_flag = 1;
  }
  else add_lipid_flag = 0; */
  // add_lipid_flag = 1;
  
  // find max atom IDs

  // find_maxid();
  group_num = group->count(igroup);
  pseu_num = 0;
  memory->create(pseu_atom_ID,N_max,"fix_filopodia_random:pseu_atom_ID");
  memory->create(pseu_filo_num,N_max,"fix_filopodia_random:pseu_filo_num");
  memory->create(filo_atom_ID,N_max,filo_num,"fix_filopodia_random:filo_atom_ID");
  memory->create(start_time,N_max,"fix_filopodia_random:start_time");
  memory->create(prt_bond_atom_ID,N_max,"fix_filopodia_random:prt_bond_atom_ID");
  memory->create(ske_bond_num,N_max,"fix_filopodia_random:ske_bond_num");
  memory->create(ske_bond_atom_ID,N_max,atom->bond_per_atom,"fix_filopodia_random:ske_bond_atom_ID");
  
  add_flag = 1;
  bond_prt_type = 1;
  ske_type = 5;
  force_reneighbor = 1;
  next_reneighbor = (update->ntimestep/nevery)*nevery + nevery;
}

/* ---------------------------------------------------------------------- */

FixFilopodiaRandom::~FixFilopodiaRandom()
{
  memory->destroy(groupbit_added);
  memory->destroy(pseu_atom_ID);
  memory->destroy(pseu_filo_num);
  memory->destroy(filo_atom_ID);
  memory->destroy(start_time);
  memory->destroy(prt_bond_atom_ID);
  memory->destroy(ske_bond_num);
  memory->destroy(ske_bond_atom_ID);
}

/* ---------------------------------------------------------------------- */

int FixFilopodiaRandom::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixFilopodiaRandom::init()
{
  int i,j,k,ii,kn;
  int nlocal = atom->nlocal;
  int nprocs = comm->nprocs;
  int *mask = atom->mask;
  
  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  
  if (domain->triclinic)
    error->all(FLERR,"Cannot use fix filopodia with triclinic box");

  // need a half neighbor list, built every Nevery steps
 
  dt = update->dt;
  
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;
  neighbor->requests[irequest]->occasional = 1;
  
  for (i = 0; i < N_max; i++) {
	pseu_atom_ID[i] = 0;
    pseu_filo_num[i] = 0;
    for (j = 0; j < filo_num; j++) filo_atom_ID[i][j] = 0;
	start_time[i] = 0;
    prt_bond_atom_ID[i] = 0;
    ske_bond_num[i] = 0;
    for (j = 0; j < atom->bond_per_atom; j++) ske_bond_atom_ID[i][j] = 0;
  }
  
  delay_by_addlipid = 0;
  pseu_num = 0;

  // lastcheck = -1;
}

/* ---------------------------------------------------------------------- */

void FixFilopodiaRandom::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ----------------------------------------------------------------------
   perform lipid insertion
------------------------------------------------------------------------- */

void FixFilopodiaRandom::pre_exchange()
{
  int i,j,k,l,m,n,p,q,ii,jj,kk,mm,add_num_local,sub_num_local,flag;
  double dd;
  double xk[3];
  int *group_add_index;
  int *original_j;
  int *original_k;
  double **xnew;
  double **vnew;
  int *typenew;
  int *create_angle_aatom1_local;
  int *create_angle_aatom2_local;
  int *create_angle_aatom3_local;
  int *create_angle_type_local;
  int *change_angle_aatom1_local;
  int *change_angle_aatom_new_local;
  int *change_angle_position_local;
  int *create_angle_num_proc;
  int *change_angle_num_proc;
  int *create_displs;
  int *change_displs;
  int *create_angle_aatom1;
  int *create_angle_aatom2;
  int *create_angle_aatom3;
  int *create_angle_type;
  int *change_angle_aatom1;
  int *change_angle_aatom_new;
  int *change_angle_position;
  int *group_filo_added_num_local;
  int *group_filo_added_num;
  int *filo_added_ID_local;
  int *filo_added_ID;
  
  int *ilist,*jlist,*numneigh,**firstneigh;
  
  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int *num_angle = atom->num_angle;
  int **angle_type = atom->angle_type;
  tagint **angle_atom1 = atom->angle_atom1;
  tagint **angle_atom2 = atom->angle_atom2;
  tagint **angle_atom3 = atom->angle_atom3;
  int timenow = update->ntimestep;
  int me = comm->me;
  
  int n1,n3,*slist;
  int **nspecial = atom->nspecial;
  int **special = atom->special;

  // just return if should not be called on this timestep
  // if (reverse_flag == 0 || timenow < reverse_time) {
    // if (!add_flag) return;
    if (add_lipid_flag) {
	  FixAddLipid *fixptr = (FixAddLipid *) modify->fix[ifix];
	  int add_lipid_num = fixptr->count_global;
	  if (add_lipid_num) {
	    if (update->ntimestep == next_reneighbor) {
		  delay_by_addlipid = 1;
		  next_reneighbor = update->ntimestep + nevery;
		}
	    return;
	  }
	  else {
	    if (!delay_by_addlipid) {
	  	  if (update->ntimestep != next_reneighbor) return;
	    }
		next_reneighbor = update->ntimestep + nevery;
	    delay_by_addlipid = 0;
	  }
    }
    else {
      if (update->ntimestep != next_reneighbor) return;
	  next_reneighbor = update->ntimestep + nevery;
    }
    
    // fprintf(stderr, "In fix filopodia at proc %d when time = %d: add_flag = %d\n", comm->me, timenow, add_flag);
    
    // acquire updated ghost atom positions
    // necessary b/c are calling this after integrate, but before Verlet comm
    
    comm->forward_comm();
    
    double **x = atom->x;
    double **v = atom->v;
    int *type = atom->type;
    tagint *tag = atom->tag;
    int *mask = atom->mask;
    double *rmass = atom->rmass;
    double *vfrac = atom->vfrac;
    int *molecule = atom->molecule;
    AtomVecEllipsoid *avec_ellipsoid = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
    //AtomVecEllipsoid::Bonus *bonus = atom->avec->bonus;
    int *ellipsoid = atom->ellipsoid;
    double *sublo = domain->sublo;
    double *subhi = domain->subhi;
    int nlocal = atom->nlocal;
    int nlocalghost = atom->nlocal + atom->nghost;
    int nprocs = comm->nprocs;
    
    //int test01 = 0;
    //int test02, test03, test04, test05;
    //MPI_Allreduce(&test01,&test02,1,MPI_INT,MPI_SUM,world);
    //fprintf(stderr, "In Fix filopodia at proc %d when time = %d: local test01 = %d, test02 = %d\n", comm->me, timenow, test01, test02);
	
	// find if there is new pseudopod created
	int selected_index = -1;
    // int count_exclude = 0;
    // if (exclude_flag) count_exclude = group->count(igroup_exclude);
    // int group_num_left = group_num - existing_selected - count_exclude;
	int group_num_left = group_num - pseu_num;
	double p_rand = 0.0;
	double p_min = 0.0;
    if (me == 0 && pseu_num < N_max) {
	  // selected_index = rand() % group_num_left;
	  for (i = 0; i < group_num_left; i++) {
	    p_rand = random->uniform();
		p_min = probability / (1 - i * probability);
		// fprintf(stderr, "In Fix FilopodiaRandom 01 when time = %d: p_rand = %f, p_min = %f, i = %d\n", timenow, p_rand, p_min, i);
		if (p_rand < p_min) {    // make sure the probabilities of every group atom are the same
	  	  fprintf(stderr, "In Fix FilopodiaRandom 02 when time = %d: p_rand = %f, p_min = %f, selected_index = %d\n", timenow, p_rand, p_min, i);
		  selected_index = i;
		  break;
	    }
	  }
    }
	// fprintf(stderr, "In Fix FilopodiaRandom 01 at proc %d when time = %d: selected_index = %d\n", me, timenow, selected_index);
    MPI_Bcast(&selected_index, 1, MPI_INT, 0, world);
	// fprintf(stderr, "In Fix FilopodiaRandom 02 at proc %d when time = %d: selected_index = %d\n", me, timenow, selected_index);
    
    // add selected atom into group pseudopod
    // n_add = 0;
	int group_num_local, group_num_all;
	int *group_list_local;
    int *group_list_all;
    int *group_num_proc;
    int *displs;
    if (selected_index >= 0) {
      // put all the group atoms into one list
      group_num_local = 0;
      memory->create(group_list_local,atom->nmax,"fix_filopodia_random:group_list_local");
      for (i = 0; i < nlocal; i++) {
	    if (mask[i] & groupbit) {
	      ii = tag[i];
		  flag = 1;
		  for (j = 0; j < pseu_num; j++) {
			if (ii == pseu_atom_ID[j]) {
			  flag = 0;
			  break;
			}
		  }
		  if (flag) {
			group_list_local[group_num_local] = ii;
	        group_num_local++;
		  }
	    }
      }
      MPI_Allreduce(&group_num_local,&group_num_all,1,MPI_INT,MPI_SUM,world);
      if (group_num_all != group_num_left) fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: group_num_all = %d, group_num_left = %d \n", me, timenow, group_num_all, group_num_left);
      memory->create(group_num_proc,nprocs,"fix_filopodia_random:group_num_proc");
      memory->create(displs,nprocs,"fix_filopodia_random:displs");
      if (nprocs > 1) MPI_Allgather(&group_num_local, 1, MPI_INT, group_num_proc, 1, MPI_INT, world);
      
      int kn;
      kn = 0;
      for (i = 0; i < nprocs; i++){
        displs[i] = kn;
        kn += group_num_proc[i];
      }
      if (kn != group_num_left) fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: kn = %d, group_num_left = %d \n", me, timenow, kn, group_num_left);
      memory->create(group_list_all,kn,"fix_filopodia_random:group_list_all");
      if (nprocs > 1) {
	    MPI_Allgatherv(group_list_local, group_num_local, MPI_INT, group_list_all, group_num_proc, displs, MPI_INT, world);
      }
	  
	  int pseu_atom_ID_new = 0;
	  for (i = 0; i < nlocal; i++) {
	    if (tag[i] == group_list_all[selected_index]) {
	  	  fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: New pseudopod was created from atom %d, i = %d \n", me, timenow, atom->tag[i], i);
	  	  mask[i] |= groupbit_pseu;
		  pseu_atom_ID_new = tag[i];
		  // pseu_filo_num[pseu_num] = 0;
		  // for (j = 0; j < filo_num; j++) filo_atom_ID[pseu_num][j] = 0;
	  	  break;
	    }
	  }
	  MPI_Allreduce(&pseu_atom_ID_new,&pseu_atom_ID[pseu_num],1,MPI_INT,MPI_SUM,world);
	  start_time[pseu_num] = timenow;
	  
      memory->destroy(group_list_local);
      memory->destroy(group_list_all);
      memory->destroy(group_num_proc);
	  
	  int *ske_bond_atom_ID_local, *ske_bond_num_proc;
	  memory->create(ske_bond_atom_ID_local,atom->bond_per_atom,"fix_filopodia_random:ske_bond_atom_ID_local");
	  memory->create(ske_bond_num_proc,nprocs,"fix_filopodia_random:ske_bond_num_proc");
	  int ske_bond_num_local = 0;
	  int prt_bond_atom_ID_local = 0;
	  for (i = 0; i < nlocal; i++) {
		if (tag[i] == pseu_atom_ID[pseu_num]) {
		  for (j = 0; j < num_bond[i]; j++) {
			if (bond_type[i][j] == bond_ske_type) {
			  ske_bond_atom_ID_local[ske_bond_num_local] = bond_atom[i][j];
			  ske_bond_num_local++;
		    }
		    else {
			  prt_bond_atom_ID_local = bond_atom[i][j];
		    }
		  }
		}
		else {
		  for (j = 0; j < num_bond[i]; j++) {
		    if (bond_atom[i][j] == pseu_atom_ID[pseu_num]) {
			  ske_bond_atom_ID_local[ske_bond_num_local] = tag[i];
			  ske_bond_num_local++;
		    }
		  }
	    }
	  }
	  MPI_Allreduce(&prt_bond_atom_ID_local,&prt_bond_atom_ID[pseu_num],1,MPI_INT,MPI_SUM,world);
	  MPI_Allreduce(&ske_bond_num_local,&ske_bond_num[pseu_num],1,MPI_INT,MPI_SUM,world);
	  // fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: ske_bond_num_local = %d, pseu_num = %d, ske_bond_num[pseu_num] = %d\n", comm->me, timenow, ske_bond_num_local, pseu_num, ske_bond_num[pseu_num]);
	  if (nprocs > 1) MPI_Allgather(&ske_bond_num_local, 1, MPI_INT, ske_bond_num_proc, 1, MPI_INT, world);
	  kn = 0;
	  for (ii = 0; ii < nprocs; ii++){
	    displs[ii] = kn;
	    kn += ske_bond_num_proc[ii];
	  }
	  if (nprocs > 1) MPI_Allgatherv(ske_bond_atom_ID_local, ske_bond_num_local, MPI_INT, ske_bond_atom_ID[pseu_num], ske_bond_num_proc, displs, MPI_INT, world);
	  
	  pseu_num++;
	  
	  memory->destroy(ske_bond_atom_ID_local);
	  memory->destroy(ske_bond_num_proc);
      memory->destroy(displs);
    }
	
	// the number of pseudopods that need to add atoms to (1), remove atoms from (2) or take no actions (0)
	int pseu_add_num = 0;
	int pseu_sub_num = 0;
	int pseu_non_num = 0;
	int *action_flag;
	memory->create(action_flag,pseu_num,"fix_filopodia_random:action_flag");
	for (i = 0; i < pseu_num; i++) {
	  action_flag[i] = 0;
	  if (reverse_flag) {
		if (timenow > start_time[i] + reverse_time) {
		  if (pseu_filo_num[i] > 0) {
		    pseu_sub_num++;
		    action_flag[i] = 2;
		  }
		  else {
			pseu_non_num++;
			action_flag[i] = 0;
		  }
		}
		else {
		  if (pseu_filo_num[i] < filo_num) {
			pseu_add_num++;
			action_flag[i] = 1;
		  }
		  else {
			pseu_non_num++;
			action_flag[i] = 0;
		  }
		}
	  }
	  else {
		if (pseu_filo_num[i] < filo_num) {
		  pseu_add_num++;
		  action_flag[i] = 1;
	    }
		else {
		  pseu_non_num++;
		  action_flag[i] = 0;
		}
	  }
	  // fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: i = %d, action_flag[i] = %d, pseu_num = %d, start_time[i] = %d, reverse_time = %d\n", comm->me, timenow, i, action_flag[i], pseu_num, start_time[i], reverse_time);
	}
	
	// find the insert lipid atom
    memory->create(group_add_index,pseu_add_num,"fix_filopodia_random:group_add_index");
    memory->create(original_j,pseu_add_num,"fix_filopodia_random:original_j");
    memory->create(xnew,pseu_add_num,3,"fix_filopodia_random:xnew");
    memory->create(vnew,pseu_add_num,3,"fix_filopodia_random:vnew");
    memory->create(typenew,pseu_add_num,"fix_filopodia_random:typenew");
    add_num_local = 0;
	sub_num_local = 0;
    for (i = 0; i < nlocal; i++) {
	  if (mask[i] & groupbit_pseu) {
	    flag = 0;
		for (ii = 0; ii < pseu_num; ii++) {
	  	  if (atom->tag[i] == pseu_atom_ID[ii]) {
			if (action_flag[ii] == 1) {
	  	      group_add_index[add_num_local] = ii;
			  flag = 1;
	  	      // fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: i = %d, tag[i] = %d, ii = %d, pseu_num = %d, add_num_local = %d\n", comm->me, timenow, i, atom->tag[i], ii, pseu_num, add_num_local);
			  break;
			}
			else if (action_flag[ii] == 2) {
			  flag = 2;
	  	      break;
			}
	  	  }
	    }
		if (flag == 1) {
	      for (k = 0; k < num_bond[i]; k++) {
	  	    if (bond_type[i][k] != bond_ske_type) {
	  	      original_j[add_num_local] = bond_atom[i][k];
	  	      j = atom->map(original_j[add_num_local]);
	  	      break;
	  	    }
	      }
	      // j = atom->map(prt_bond_atom_ID[group_add_index[add_num_local]]);
	      // fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: i = %d, j = %d, original_j[add_num_local] = %d, nlocal = %d\n", comm->me, timenow, i, j, original_j[add_num_local], nlocal);
	      xnew[add_num_local][0] = (x[i][0] + x[j][0]) / 2;
	      xnew[add_num_local][1] = (x[i][1] + x[j][1]) / 2;
	      xnew[add_num_local][2] = (x[i][2] + x[j][2]) / 2;
	      vnew[add_num_local][0] = (v[i][0] + v[j][0]) / 2;
	      vnew[add_num_local][1] = (v[i][1] + v[j][1]) / 2;
	      vnew[add_num_local][2] = (v[i][2] + v[j][2]) / 2;
	      typenew[add_num_local] = type[i];
	      add_num_local++;
		}
		else if (flag == 2) {
		  sub_num_local++;
		}
	  }
    }
	
	int *num_bond_new = atom->num_bond;
    int **bond_type_new = atom->bond_type;
    tagint **bond_atom_new = atom->bond_atom;
    int *num_angle_new = atom->num_angle;
    int **angle_type_new = atom->angle_type;
    tagint **angle_atom1_new = atom->angle_atom1;
    tagint **angle_atom2_new = atom->angle_atom2;
    tagint **angle_atom3_new = atom->angle_atom3;
	
	int potential_j;
	double dmin;
	if (reverse_flag) {
	  // find the bead at the top of the filopod at the start of retraction, and remove the angle
	  for (m = 0; m < nlocal; m++) {
	    for (ii = 0; ii < pseu_num; ii++) {
	      if (pseu_filo_num[ii] == filo_num) {
	    	mm = atom->tag[m];
	    	if (mm == filo_atom_ID[ii][0]) {
	    	  for (n = 0; n < num_angle_new[m]; n++) {
	    	    if (angle_type_new[m][n] == angle_filo_180_type) {
	    		  fprintf(stderr, "In Fix filopodia reverse at proc %d when time = %d: 180 angle %d-%d-%d was deleted\n", comm->me, timenow, angle_atom1_new[m][n], angle_atom2_new[m][n], angle_atom3_new[m][n]);
	    		  for (jj = n; jj < num_angle_new[m]-1; jj++) {
	    		    angle_atom1_new[m][jj] = angle_atom1_new[m][jj+1];
	    		    angle_atom2_new[m][jj] = angle_atom2_new[m][jj+1];
	    		    angle_atom3_new[m][jj] = angle_atom3_new[m][jj+1];
					angle_type_new[m][jj] = angle_type_new[m][jj+1];
	    		  }
	    		  num_angle_new[m]--;
	    		  n--;
	    		  atom->nangles--;
	    	    }
	    	  }
	    	}
	      }
	    }
	  }
	  
	  // i: pseu_atom_ID[ii], k: filo_atom_ID[ii][pseu_filo_num[ii]-1], j: filo_atom_ID[ii][pseu_filo_num[ii]-2]
      // find i, and change bond i-k to i-j, change angle ske-i-k to ske-i-j when pseu_filo_num[ii] > 1
	  //          or change bond i-k to i-j, change bond type to bond_prt_type, delete angle ske-i-k when pseu_filo_num[ii] = 1
	  for (m = 0; m < nlocal; m++) {
	    if (mask[m] & groupbit_pseu) {
	      for (ii = 0; ii < pseu_num; ii++) {
	    	i = atom->tag[m];
	  	    if (i == pseu_atom_ID[ii] && action_flag[ii] == 2) {
	  	      j = filo_atom_ID[ii][pseu_filo_num[ii]-2];
	  		  k = filo_atom_ID[ii][pseu_filo_num[ii]-1];
	  		  for (n = 0; n < num_bond_new[m]; n++) {
	  		    if (bond_atom_new[m][n] == k) {
	  		  	  if (pseu_filo_num[ii] > 1) {
	  		  	    bond_atom_new[m][n] = j;
	  		  	    fprintf(stderr, "In Fix FilopodiaRandom reverse at proc %d when time = %d: bond %d-%d was changed to %d-%d\n", comm->me, timenow, i, k, i, j);
	  		  	  }
	  		  	  else {
	  		  	    // bond_atom_new[m][n] = prt_bond_atom_ID[ii];
	  		  	    // bond_type_new[m][n] = bond_prt_type;
	  		  	    // fprintf(stderr, "In Fix FilopodiaRandom reverse at proc %d when time = %d: bond %d-%d was changed to %d-%d\n", comm->me, timenow, i, k, i, prt_bond_atom_ID[ii]);
					
					potential_j = -1;
				    dmin = 100000.0;
				    for (mm = 0; mm < nlocal; mm++) {
				      if (type[mm] == mem_type) {
				        dd = (x[m][0] - x[mm][0]) * (x[m][0] - x[mm][0]) + (x[m][1] - x[mm][1]) * (x[m][1] - x[mm][1]) + (x[m][2] - x[mm][2]) * (x[m][2] - x[mm][2]);
				        if (dd < dmin) {
				      	dmin = dd;
				      	potential_j = mm;
				        }
				      }
				    }
				    if (potential_j > 0) {
				      bond_atom_new[m][n] = tag[potential_j];
				      bond_type_new[m][n] = bond_prt_type;
				      type[potential_j] = prt_type;
				      fprintf(stderr, "In Fix filopodia reverse at proc %d when time = %d: bond %d-%d was changed to %d-%d\n", comm->me, timenow, i, k, i, tag[potential_j]);
				    }
	  		  	  }
	  		  	  break;
	  		    }
	  		  }
	  		  for (n = 0; n < num_angle_new[m]; n++) {
	  		    if (angle_type_new[m][n] == angle_filo_90_type) {
	  		  	  if (angle_atom3_new[m][n] != k) error->all(FLERR,"angle_atom3 is not consistent with k");
	  		  	  if (pseu_filo_num[ii] == 1) {
	  		  	    fprintf(stderr, "In Fix FilopodiaRandom reverse at proc %d when time = %d: angle %d-%d-%d was deleted\n", comm->me, timenow, angle_atom1_new[m][n], angle_atom2_new[m][n], angle_atom3_new[m][n]);
	  		  	    for (jj = n; jj < num_angle_new[m]-1; jj++) {
	  		  	  	angle_atom1_new[m][jj] = angle_atom1_new[m][jj+1];
	  		  	  	angle_atom2_new[m][jj] = angle_atom2_new[m][jj+1];
	  		  	  	angle_atom3_new[m][jj] = angle_atom3_new[m][jj+1];
					angle_type_new[m][jj] = angle_type_new[m][jj+1];
	  		  	    }
	  		  	    num_angle_new[m]--;
	  		  	    n--;
	  		  	    atom->nangles--;
	  		  	  }
	  		  	  else {
	  		  	    fprintf(stderr, "In Fix FilopodiaRandom reverse at proc %d when time = %d: angle %d-%d-%d was changed to %d-%d-%d\n", comm->me, timenow, angle_atom1_new[m][n], angle_atom2_new[m][n], angle_atom3_new[m][n], angle_atom1_new[m][n], angle_atom2_new[m][n], j);
	  		  	    angle_atom3_new[m][n] = j;
	  		  	  }
	  		    }
	  		  }
	  	    }
	  	  }
	    }
	  }
	  
	  // find k, and delete bond k-j, delete angle i-k-j
	  for (m = 0; m < nlocal; m++) {
	    for (ii = 0; ii < pseu_num; ii++) {
	      if ((action_flag[ii] == 2) && (pseu_filo_num[ii] > 0)) {
	  	    k = atom->tag[m];
	  	    if (k == filo_atom_ID[ii][pseu_filo_num[ii]-1]) {
	  	      atom->nbonds = atom->nbonds - num_bond_new[m];
	  	      num_bond_new[m] = 0;
	  	      atom->nangles = atom->nangles - num_angle_new[m];
	  	      num_angle_new[m] = 0;
	  	      fprintf(stderr, "In Fix FilopodiaRandom reverse at proc %d when time = %d: the bonds and angles for %d was deleted\n", comm->me, timenow, k);
	  	    }
	  	  }
	    }
	  }
	  
	  // find j, and change k-j-p to i-j-p when pseu_filo_num[ii] > 2
	  for (m = 0; m < nlocal; m++) {
	    for (ii = 0; ii < pseu_num; ii++) {
	      if ((action_flag[ii] == 2) && (pseu_filo_num[ii] > 2)) {
	  	    j = atom->tag[m];
	  	    if (j == filo_atom_ID[ii][pseu_filo_num[ii]-2]) {
	  	      i = pseu_atom_ID[ii];
	  		  for (n = 0; n < num_angle_new[m]; n++) {
	  		    if (angle_type_new[m][n] == angle_filo_180_type) {
	  		  	angle_atom1_new[m][n] = i;
	  		  	fprintf(stderr, "In Fix FilopodiaRandom reverse at proc %d when time = %d: angle %d-%d-%d was changed to %d-%d-%d\n", comm->me, timenow, angle_atom1_new[m][n], angle_atom2_new[m][n], angle_atom3_new[m][n], i, angle_atom2_new[m][n], angle_atom3_new[m][n]);
	  		    }
	  		  }
	  	    }
	  	  }
	    }
	  }
	  
	  // find q, and change atom type to ske_type
	  for (m = 0; m < nlocal; m++) {
	    for (ii = 0; ii < pseu_num; ii++) {
	      if ((action_flag[ii] == 2) && (pseu_filo_num[ii] > no_adh_num)) {
	  	    q = atom->tag[m];
	  	    if (q == filo_atom_ID[ii][pseu_filo_num[ii] - no_adh_num - 1]) {
	  	      type[m] = ske_type;
	  		  fprintf(stderr, "In Fix FilopodiaRandom reverse at proc %d when time = %d: the type of atom %d was changed to %d\n", comm->me, timenow, q, ske_type);
			  if (bond_dyn_flag) {
			    for (jj = 0; jj < atom->num_bond[m]; jj++) {
			  	  n = atom->map(atom->bond_atom[m][jj]);
			  	  if (atom->bond_type[m][jj] == bond_dyn_type) {
			  	    fprintf(stderr, "In Fix filopodia reverse at proc %d when time = %d: dynamic bond %d-%d was deleted\n", comm->me, timenow, q, atom->bond_atom[m][jj]);
			  	    l = atom->num_bond[m];
			  	    atom->bond_atom[m][jj] = atom->bond_atom[m][l-1];
			  	    atom->bond_type[m][jj] = atom->bond_type[m][l-1];
			  	    atom->bond_length[m][jj] = atom->bond_length[m][l-1];
			  	    atom->num_bond[m]--;
			  	    jj--;
			  	    
			  	    // remove n from special bond list for atom m
			  	    // atom n will also do this
			  	    slist = special[m];
			  	    n1 = nspecial[m][0];
			  	    n3 = nspecial[m][2];
			  	    for (kk = 0; kk < n1; kk++)
			  	      if (slist[kk] == n) break;
			  	    for (; kk < n3-1; kk++) slist[kk] = slist[kk+1];
			  	    nspecial[m][0]--;
			  	    nspecial[m][1]--;
			  	    nspecial[m][2]--;
			  	    
			  	    slist = special[n];
			  	    n1 = nspecial[n][0];
			  	    n3 = nspecial[n][2];
			  	    for (kk = 0; kk < n1; kk++)
			  	      if (slist[kk] == m) break;
			  	    for (; kk < n3-1; kk++) slist[kk] = slist[kk+1];
			  	    nspecial[n][0]--;
			  	    nspecial[n][1]--;
			  	    nspecial[n][2]--;
			  	  }
			    }
			  }
	  	    }
	  	  }
	    }
	  }
    }
    // clear ghost count and any ghost bonus data internal to AtomVec
    // same logic as beginning of Comm::exchange()
    // do it now b/c inserting atoms will overwrite ghost atoms
    
    atom->nghost = 0;
    atom->avec->clear_bonus();
	
	// remove atoms
	if (reverse_flag) {
      nlocal = atom->nlocal;
	  
	  for (m = 0; m < nlocal; m++) {
	    for (ii = 0; ii < pseu_num; ii++) {
	      k = atom->tag[m];
	  	  if ((action_flag[ii] == 2) && (k == filo_atom_ID[ii][pseu_filo_num[ii]-1])) {
	  	    atom->avec->copy(nlocal-1,m,1);
	  	    nlocal--;
	  	    fprintf(stderr, "In Fix FilopodiaRandom reverse at proc %d when time = %d: the atom %d was deleted\n", comm->me, timenow, filo_atom_ID[ii][pseu_filo_num[ii]-1]);
			filo_atom_ID[ii][pseu_filo_num[ii]-1] = 0;
	  	  }
	    }
	  }
	  
	  atom->nlocal = nlocal;
	}
	
	// add atoms
    
    bigint natoms_previous = atom->natoms;
    int nlocal_previous = atom->nlocal;
    
    for (i = 0; i < add_num_local; i++) {
	  xk[0] = xnew[i][0];
	  xk[1] = xnew[i][1];
	  xk[2] = xnew[i][2];
	  atom->avec->create_atom(typenew[i],xk);
	  
	  k = atom->nlocal-1;
	  if (group_added_flag) {
	    for (j = 0; j < N_added_group; j++) mask[k] |= groupbit_added[j];
	  }
	  rmass[k] = 0.238732;
	  vfrac[k] = 0.523599;
	  molecule[k] = 5;
	  v[k][0] = vnew[i][0];
	  v[k][1] = vnew[i][1];
	  v[k][2] = vnew[i][2];
	  // avec_ellipsoid->set_shape(i, 1, 1, 1);
	  // fprintf(stderr, "In fix addlipid at proc %d when time = %d: i = %d, ellipsoid[i] = %d, avec_ellipsoid->bonus[ellipsoid[i]].ilocal = %d\n", comm->me, timenow, i, ellipsoid[i], avec_ellipsoid->bonus[ellipsoid[i]].ilocal);
	  // nquati = avec_ellipsoid->bonus[ellipsoid[i]].quat;
	  // nquati[0] = xnew[ii][6];
	  // nquati[1] = xnew[ii][7];
	  // nquati[2] = xnew[ii][8];
	  // nquati[3] = xnew[ii][9];
    }
    //if (comm->me == 38) test01 = 1;
    //MPI_Allreduce(&test01,&test03,1,MPI_INT,MPI_SUM,world);
    //fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: local test01 = %d, test03 = %d\n", comm->me, timenow, test01, test03);
    
	nlocal = atom->nlocal;
    for (m = 0; m < modify->nfix; m++) {
      Fix *fix = modify->fix[m];
      if (fix->create_attribute)
        for (i = nlocal_previous; i < nlocal; i++)
          fix->set_arrays(i);
    }
    for (m = 0; m < modify->ncompute; m++) {
      Compute *compute = modify->compute[m];
      if (compute->create_attribute)
        for (i = nlocal_previous; i < nlocal; i++)
          compute->set_arrays(i);
    }
    for (i = nlocal_previous; i < nlocal; i++)
      input->variable->set_arrays(i);
    
    // set new total # of atoms and error check
    
    bigint nblocal = atom->nlocal;
    MPI_Allreduce(&nblocal,&atom->natoms,1,MPI_LMP_BIGINT,MPI_SUM,world);
    if (atom->natoms < 0 || atom->natoms >= MAXBIGINT)
      error->all(FLERR,"Too many total atoms");
    
    // add IDs for newly created atoms
    // check that atom IDs are valid
    
    if (atom->tag_enable) atom->tag_extend();
    atom->tag_check();
    
    // if global map exists, reset it
    // invoke map_init() b/c atom count has grown
    
    if (atom->map_style) {
      atom->map_init();
      atom->map_set();
    }
    
    for (i = nlocal_previous; i < nlocal; i++) {
	  fprintf(stderr, "In fix FilopodiaRandom at proc %d when time = %d: a filo atom %d at %f %f %f was added\n", comm->me, timenow, atom->tag[i], x[i][0], x[i][1], x[i][2]);
    }
    //if (comm->me == 38) test01 = 2;
    //MPI_Allreduce(&test01,&test04,1,MPI_INT,MPI_SUM,world);
    //fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: local test01 = %d, test04 = %d\n", comm->me, timenow, test01, test04);
    
	// add bonds and angles
    int create_angle_num, change_angle_num;
    int create_angle_num_local = 0;
    int change_angle_num_local = 0;
    memory->create(create_angle_aatom1_local,(1 + atom->bond_per_atom) * pseu_add_num,"fix_filopodia_random:create_angle_aatom1_local");
    memory->create(create_angle_aatom2_local,(1 + atom->bond_per_atom) * pseu_add_num,"fix_filopodia_random:create_angle_aatom2_local");
    memory->create(create_angle_aatom3_local,(1 + atom->bond_per_atom) * pseu_add_num,"fix_filopodia_random:create_angle_aatom3_local");
    memory->create(create_angle_type_local,(1 + atom->bond_per_atom) * pseu_add_num,"fix_filopodia_random:create_angle_type_local");
    memory->create(change_angle_aatom1_local,(1 + atom->bond_per_atom) * pseu_add_num,"fix_filopodia_random:change_angle_aatom1_local");
    memory->create(change_angle_aatom_new_local,(1 + atom->bond_per_atom) * pseu_add_num,"fix_filopodia_random:change_angle_aatom_new_local");
    memory->create(change_angle_position_local,(1 + atom->bond_per_atom) * pseu_add_num,"fix_filopodia_random:change_angle_position_local");
    memory->create(group_filo_added_num_local,pseu_num,"fix_filopodia_random:group_filo_added_num_local");
    memory->create(group_filo_added_num,pseu_num,"fix_filopodia_random:group_filo_added_num");
    memory->create(filo_added_ID_local,pseu_num,"fix_filopodia_random:filo_added_ID_local");
    memory->create(filo_added_ID,pseu_num,"fix_filopodia_random:filo_added_ID");
    //memory->create(filo_added_order_local,pseu_add_num,"fix_filopodia_random:filo_added_order_local");
    for (ii = 0; ii < pseu_num; ii++) {
	  group_filo_added_num_local[ii] = 0;
	  filo_added_ID_local[ii] = 0;
	  //filo_added_order_local[ii] = 0;
    }
    for (ii = 0; ii < add_num_local; ii++) {
	  jj = group_add_index[ii];
	  i = pseu_atom_ID[jj];
	  j = original_j[ii];
	  k = atom->tag[ii + nlocal_previous];
	  filo_added_ID_local[jj] = k;
	  //filo_added_order_local[jj] = pseu_filo_num[jj];
	  // change_bond_into_two(i, j, k, bond_filo_type);
	  m = atom->map(i);
	  for (n = 0; n < num_bond_new[m]; n++) {
	    if (bond_atom_new[m][n] == j) {
	  	bond_atom_new[m][n] = k;
	  	// if (pseu_filo_num[jj] == filo_num - 1) bond_type_new[m][n] = bond_filo_type;
	  	bond_type_new[m][n] = bond_filo_type;
	  	fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: bond %d-%d was changed to %d-%d\n", comm->me, timenow, i, j, i, k);
	    }
	  }
	  if (pseu_filo_num[jj] == 0) {
	    // create angle for i-k-j, 180 degree
	    //fprintf(stderr, "In Fix FilopodiaRandom loop 01 at proc %d when time = %d: pseu_filo_num[jj] = %d, create_angle_num_local = %d\n", comm->me, timenow, pseu_filo_num[jj], create_angle_num_local);
	    create_angle_aatom1_local[create_angle_num_local] = i;
	    create_angle_aatom2_local[create_angle_num_local] = k;
	    create_angle_aatom3_local[create_angle_num_local] = j;
	    create_angle_type_local[create_angle_num_local] = angle_filo_180_type;
	    //fprintf(stderr, "In Fix FilopodiaRandom loop 02 at proc %d when time = %d: create_angle_aatom_local[create_angle_num_local] = %d %d %d, create_angle_type_local[create_angle_num_local] = %d\n", comm->me, timenow, create_angle_aatom1_local[create_angle_num_local], create_angle_aatom2_local[create_angle_num_local], create_angle_aatom3_local[create_angle_num_local], create_angle_type_local[create_angle_num_local]);
	    create_angle_num_local++;
	    
	    
	    // create angle for ske-i-k, 90 degree
		for (kk = 0; kk < ske_bond_num[jj]; kk++) {
	  	  create_angle_aatom1_local[create_angle_num_local] = ske_bond_atom_ID[jj][kk];
	  	  create_angle_aatom2_local[create_angle_num_local] = i;
	  	  create_angle_aatom3_local[create_angle_num_local] = k;
	  	  create_angle_type_local[create_angle_num_local] = angle_filo_90_type;
	  	  //fprintf(stderr, "In Fix FilopodiaRandom loop 0%d at proc %d when time = %d: create_angle_aatom_local[create_angle_num_local] = %d %d %d, create_angle_type_local[create_angle_num_local] = %d\n", create_angle_num_local + 2, comm->me, timenow, create_angle_aatom1_local[create_angle_num_local], create_angle_aatom2_local[create_angle_num_local], create_angle_aatom3_local[create_angle_num_local], create_angle_type_local[create_angle_num_local]);
	  	  create_angle_num_local++;
	    }
	  }
	  else {
	    // delete bond between protein and cytoskeleton
		bond_type_new[ii + nlocal_previous][num_bond_new[ii + nlocal_previous]] = bond_filo_type;
        bond_atom_new[ii + nlocal_previous][num_bond_new[ii + nlocal_previous]] = j;
        num_bond_new[ii + nlocal_previous]++;
	    fprintf(stderr, "In Fix FilopodiaRandom change_bond_into_two at proc %d when time = %d: bond %d-%d was created\n", comm->me, timenow, k, j);
	    atom->nbonds++;
	  
		
		// create angle for i-k-j, 180 degree
	    create_angle_aatom1_local[create_angle_num_local] = i;
	    create_angle_aatom2_local[create_angle_num_local] = k;
	    create_angle_aatom3_local[create_angle_num_local] = j;
	    create_angle_type_local[create_angle_num_local] = angle_filo_180_type;
	    create_angle_num_local++;
	    
	    // change angle from ske-i-j to ske-i-k, 90 degree
	    change_angle_aatom1_local[change_angle_num_local] = i;
	    change_angle_aatom_new_local[change_angle_num_local] = k;
	    change_angle_position_local[change_angle_num_local] = 3;
	    change_angle_num_local++;
	    
	    // change angle from i-j-n to k-j-n, 180 degree
	    change_angle_aatom1_local[change_angle_num_local] = j;
	    change_angle_aatom_new_local[change_angle_num_local] = k;
	    change_angle_position_local[change_angle_num_local] = 1;
	    change_angle_num_local++;
	  }
	  group_filo_added_num_local[jj]++;
	  //fprintf(stderr, "In Fix FilopodiaRandom loop at proc %d when time = %d: group_filo_added_num_local[jj] = %d, jj = %d\n", comm->me, timenow, group_filo_added_num_local[jj], jj);
    }
    //fprintf(stderr, "In Fix FilopodiaRandom end of the loop at proc %d when time = %d\n", comm->me, timenow);
    //if (comm->me == 38) test01 = 4;
    //fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: local test01 = %d\n", comm->me, timenow, test01);
    //MPI_Allreduce(&test01,&test05,1,MPI_INT,MPI_SUM,world);
    //fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: local test01 = %d, test05 = %d\n", comm->me, timenow, test01, test05);
    
    //for (i = 0; i < create_angle_num_local; i++){
	  //fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: local angle %d-%d-%d would be created, angle type is %d\n", comm->me, timenow, create_angle_aatom1_local[i], create_angle_aatom2_local[i], create_angle_aatom3_local[i], create_angle_type_local[i]);
    //}
    //for (i = 0; i < change_angle_num_local; i++){
	  //fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: local the %dth angle atom of atom %d was changed to %d\n", comm->me, timenow, change_angle_position_local[i], change_angle_aatom1_local[i], change_angle_aatom_new_local[i]);
    //}
    // MPI_Barrier(world);
    //fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: create_angle_num_local = %d, change_angle_num_local = %d\n", comm->me, timenow, create_angle_num_local, change_angle_num_local);
    MPI_Allreduce(&create_angle_num_local,&create_angle_num,1,MPI_INT,MPI_SUM,world);
    MPI_Allreduce(&change_angle_num_local,&change_angle_num,1,MPI_INT,MPI_SUM,world);
    //int tmp;
    //MPI_Reduce(&create_angle_num_local,&tmp,1,MPI_INT,MPI_SUM,0,world);
    //if (comm->me == 0) fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: tmp = %d\n", comm->me, timenow, tmp);
    //fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: create_angle_num = %d, change_angle_num = %d\n", comm->me, timenow, create_angle_num, change_angle_num);
    memory->create(create_angle_num_proc,nprocs,"fix_filopodia_random:create_angle_num_proc");
    memory->create(change_angle_num_proc,nprocs,"fix_filopodia_random:change_angle_num_proc");
    memory->create(create_displs,nprocs,"fix_filopodia_random:create_displs");
    memory->create(change_displs,nprocs,"fix_filopodia_random:change_displs");
    if (nprocs > 1) {
	  MPI_Allgather(&create_angle_num_local, 1, MPI_INT, create_angle_num_proc, 1, MPI_INT, world);
	  MPI_Allgather(&change_angle_num_local, 1, MPI_INT, change_angle_num_proc, 1, MPI_INT, world);
	  //for (i = 0; i < nprocs; i++){
	  //  fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: i = %d, create_angle_num_proc[i] = %d, change_angle_num_proc[i] = %d\n", comm->me, timenow, i, create_angle_num_proc[i], change_angle_num_proc[i]);
	  //}
    }
    
    int kn01 = 0;
    int kn02 = 0;
    for (i = 0; i < nprocs; i++){
      create_displs[i] = kn01;
      kn01 += create_angle_num_proc[i];
	  change_displs[i] = kn02;
      kn02 += change_angle_num_proc[i];
	  // fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: i = %d, create_angle_num_proc[i] = %d, change_angle_num_proc[i] = %d\n", comm->me, timenow, i, create_angle_num_proc[i], change_angle_num_proc[i]);
    }
    
    memory->create(create_angle_aatom1,kn01,"fix_filopodia_random:create_angle_aatom1");
    memory->create(create_angle_aatom2,kn01,"fix_filopodia_random:create_angle_aatom1");
    memory->create(create_angle_aatom3,kn01,"fix_filopodia_random:create_angle_aatom1");
    memory->create(create_angle_type,kn01,"fix_filopodia_random:create_angle_type");
    memory->create(change_angle_aatom1,kn02,"fix_filopodia_random:change_angle_aatom1");
    memory->create(change_angle_aatom_new,kn02,"fix_filopodia_random:change_angle_aatom_new");
    memory->create(change_angle_position,kn02,"fix_filopodia_random:change_angle_position");
    if (nprocs > 1) {
	  MPI_Allgatherv(create_angle_aatom1_local, create_angle_num_local, MPI_INT, create_angle_aatom1, create_angle_num_proc, create_displs, MPI_INT, world);
	  MPI_Allgatherv(create_angle_aatom2_local, create_angle_num_local, MPI_INT, create_angle_aatom2, create_angle_num_proc, create_displs, MPI_INT, world);
	  MPI_Allgatherv(create_angle_aatom3_local, create_angle_num_local, MPI_INT, create_angle_aatom3, create_angle_num_proc, create_displs, MPI_INT, world);
	  MPI_Allgatherv(create_angle_type_local, create_angle_num_local, MPI_INT, create_angle_type, create_angle_num_proc, create_displs, MPI_INT, world);
	  MPI_Allgatherv(change_angle_aatom1_local, change_angle_num_local, MPI_INT, change_angle_aatom1, change_angle_num_proc, change_displs, MPI_INT, world);
	  MPI_Allgatherv(change_angle_aatom_new_local, change_angle_num_local, MPI_INT, change_angle_aatom_new, change_angle_num_proc, change_displs, MPI_INT, world);
	  MPI_Allgatherv(change_angle_position_local, change_angle_num_local, MPI_INT, change_angle_position, change_angle_num_proc, change_displs, MPI_INT, world);
    }
    
    for (i = 0; i < kn01; i++){
	  //fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: angle %d-%d-%d would be created, angle type is %d\n", comm->me, timenow, create_angle_aatom1[i], create_angle_aatom2[i], create_angle_aatom3[i], create_angle_type[i]);
	  //create_new_angle(create_angle_aatom1[i], create_angle_aatom2[i], create_angle_aatom3[i], create_angle_type[i]);
	  m = atom->map(create_angle_aatom2[i]);
	  if (m >= 0 && m < nlocal) {
	    //fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: m = %d\n", comm->me, timenow, m);
	    //fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: num_angle_new[m] = %d\n", comm->me, timenow, num_angle_new[m]);
	    //fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: angle_type_new[m][num_angle_new[m]] = %d\n", comm->me, timenow, angle_type_new[m][num_angle_new[m]]);
	    //fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: create_angle_type[i] = %d\n", comm->me, timenow, create_angle_type[i]);
	    angle_type_new[m][num_angle_new[m]] = create_angle_type[i];
        angle_atom1_new[m][num_angle_new[m]] = create_angle_aatom1[i];
        angle_atom2_new[m][num_angle_new[m]] = create_angle_aatom2[i];
        angle_atom3_new[m][num_angle_new[m]] = create_angle_aatom3[i];
        num_angle_new[m]++;
	    fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: angle %d-%d-%d was created, angle type is %d\n", comm->me, timenow, create_angle_aatom1[i], create_angle_aatom2[i], create_angle_aatom3[i], create_angle_type[i]);
        atom->nangles++;
	  }
    }
    for (i = 0; i < kn02; i++){
	  // fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: the %dth angle atom of atom %d was changed to %d\n", comm->me, timenow, change_angle_position[i], change_angle_aatom1[i], change_angle_aatom_new[i]);
	  // change_angle(change_angle_aatom1[i], change_angle_aatom_new[i], change_angle_position[i]);
	  m = atom->map(change_angle_aatom1[i]);
	  if (m >= 0 && m < nlocal) {
	    if (change_angle_position[i] == 1) {
	      for (n = 0; n < num_angle_new[m]; n++) {
	        angle_atom1_new[m][n] = change_angle_aatom_new[i];
	  	  fprintf(stderr, "In Fix FilopodiaRandom change_angle at proc %d when time = %d: the aatom1 of atom %d was changed to %d\n", comm->me, timenow, change_angle_aatom1[i], change_angle_aatom_new[i]);
	      }
	    }
	    else if (change_angle_position[i] == 3) {
	      for (n = 0; n < num_angle_new[m]; n++) {
	        angle_atom3_new[m][n] = change_angle_aatom_new[i];
	  	  fprintf(stderr, "In Fix FilopodiaRandom change_angle at proc %d when time = %d: the aatom3 of atom %d was changed to %d\n", comm->me, timenow, change_angle_aatom1[i], change_angle_aatom_new[i]);
	      }
	    }
	    else error->all(FLERR, "change_angle_position is not 1 or 3");
	  }
    }
    
    MPI_Allreduce(filo_added_ID_local,filo_added_ID,pseu_num,MPI_INT,MPI_SUM,world);
    for (ii = 0; ii < pseu_num; ii++) {
	  if (action_flag[ii] != 1) continue; 
	  if (pseu_filo_num[ii] < filo_num) {
	    filo_atom_ID[ii][pseu_filo_num[ii]] = filo_added_ID[ii];
	  }
      // fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: the filo atoms connected to %d are", comm->me, timenow, pseu_atom_ID[ii]);
	  // for (jj = 0; jj < pseu_filo_num[ii] + 1; jj++) {
	  //   fprintf(stderr, " %d", filo_atom_ID[ii][jj]);
	  // }
	  // fprintf(stderr, "\n");
	  if (pseu_filo_num[ii] < filo_num && pseu_filo_num[ii] >= no_adh_num) {
	    k = atom->map(filo_atom_ID[ii][pseu_filo_num[ii] - no_adh_num]);
	    // fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: k = %d, filo_atom_ID[ii][pseu_filo_num[ii] - no_adh_num] = %d\n", comm->me, timenow, k, filo_atom_ID[ii][pseu_filo_num[ii] - no_adh_num]);
	    if (k >= 0 && k < nlocal) {
	  	  type[k] = filo_type;
	  	  fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: the type of atom %d was changed to %d\n", comm->me, timenow, atom->tag[k], filo_type);
	    }
	  }
    }
    
    MPI_Allreduce(group_filo_added_num_local,group_filo_added_num,pseu_num,MPI_INT,MPI_SUM,world);
    add_flag = 0;
    for (ii = 0; ii < pseu_num; ii++) {
	  // fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: ii = %d, group_filo_added_num[ii] = %d\n", comm->me, timenow, ii, group_filo_added_num[ii]);
	  // fprintf(stderr, "In Fix FilopodiaRandom at proc %d when time = %d: pseu_filo_num[ii] = %d\n", comm->me, timenow, pseu_filo_num[ii]);
	  if (action_flag[ii] == 1) pseu_filo_num[ii] ++;
	  if (action_flag[ii] == 2) {
		pseu_filo_num[ii] --;
		if (pseu_filo_num[ii] == 0) {
		  
		  pseu_atom_ID[ii] = pseu_atom_ID[pseu_num - 1];
		  pseu_filo_num[ii] = pseu_filo_num[pseu_num - 1];
		  for (j = 0; j < filo_num; j++) filo_atom_ID[ii][j] = filo_atom_ID[pseu_num - 1][j];
		  start_time[ii] = start_time[pseu_num - 1];
		  prt_bond_atom_ID[ii] = prt_bond_atom_ID[pseu_num - 1];
		  ske_bond_num[ii] = ske_bond_num[pseu_num - 1];
		  for (j = 0; j < atom->bond_per_atom; j++) ske_bond_atom_ID[ii][j] = ske_bond_atom_ID[pseu_num - 1][j];
		  action_flag[ii] = action_flag[pseu_num - 1];
		  
		  pseu_atom_ID[pseu_num - 1] = 0;
		  pseu_filo_num[pseu_num - 1] = 0;
		  for (j = 0; j < filo_num; j++) filo_atom_ID[pseu_num - 1][j] = 0;
		  start_time[pseu_num - 1] = 0;
		  prt_bond_atom_ID[pseu_num - 1] = 0;
		  ske_bond_num[pseu_num - 1] = 0;
		  for (j = 0; j < atom->bond_per_atom; j++) ske_bond_atom_ID[pseu_num - 1][j] = 0;
		  action_flag[pseu_num - 1] = 0;
		  
		  pseu_num--;
		  ii--;
		}
	  }
    }
    
    // free local memory
    
    memory->destroy(group_add_index);
    memory->destroy(original_j);
    memory->destroy(xnew);
    memory->destroy(vnew);
    memory->destroy(typenew);
	memory->destroy(action_flag);
    memory->destroy(create_angle_aatom1_local);
    memory->destroy(create_angle_aatom2_local);
    memory->destroy(create_angle_aatom3_local);
    memory->destroy(create_angle_type_local);
    memory->destroy(change_angle_aatom1_local);
    memory->destroy(change_angle_aatom_new_local);
    memory->destroy(change_angle_position_local);
    memory->destroy(create_angle_num_proc);
    memory->destroy(change_angle_num_proc);
    memory->destroy(create_displs);
    memory->destroy(change_displs);
    memory->destroy(create_angle_aatom1);
    memory->destroy(create_angle_aatom2);
    memory->destroy(create_angle_aatom3);
    memory->destroy(create_angle_type);
    memory->destroy(change_angle_aatom1);
    memory->destroy(change_angle_aatom_new);
    memory->destroy(change_angle_position);
    memory->destroy(group_filo_added_num_local);
    memory->destroy(group_filo_added_num);
    memory->destroy(filo_added_ID_local);
    memory->destroy(filo_added_ID);
    //memory->destroy(filo_added_order_local);
  // }
}

/* ----------------------------------------------------------------------
   maxtag_all = current max atom ID for all atoms
   maxmol_all = current max molecule ID for all atoms
------------------------------------------------------------------------- */

/* void FixFilopodiaRandom::find_maxid()
{
  tagint *tag = atom->tag;
  tagint *molecule = atom->molecule;
  int nlocal = atom->nlocal;

  tagint max = 0;
  for (int i = 0; i < nlocal; i++) max = MAX(max,tag[i]);
  MPI_Allreduce(&max,&maxtag_all,1,MPI_LMP_TAGINT,MPI_MAX,world);
} */

/* ---------------------------------------------------------------------- */

//void FixFilopodiaRandom::change_bond_into_two(int batom1, int batom2, int batom_new, int btype)
//{
//  int m, n;
//
//  // check that 2 atoms exist
//
//  int nlocal = atom->nlocal;
//  int idx1 = atom->map(batom1);
//  int idx2 = atom->map(batom2);
//  int idx_new = atom->map(batom_new);
//
//  int count = 0;
//  if ((idx1 >= 0) && (idx1 < nlocal)) count++;
//  if ((idx2 >= 0) && (idx2 < nlocal)) count++;
//  if ((idx_new >= 0) && (idx_new < nlocal)) count++;
//
//  int allcount;
//  MPI_Allreduce(&count, &allcount, 1, MPI_INT, MPI_SUM, world);
//  if (allcount != 3) error->all(FLERR, "change_bond_into_two bond atoms do not exist");
//
//  // create bond once or 2x if newton_bond set
//
//  int *num_bond = atom->num_bond;
//  int **bond_type = atom->bond_type;
//  tagint **bond_atom = atom->bond_atom;
//
//  if ((m = idx1) >= 0) {
//    if (num_bond[m] == atom->bond_per_atom)
//      error->one(FLERR, "New bond exceeded bonds per atom in change_bond_into_two");
//    for (n = 0; n < num_bond[m]; n++) {
//	  if (bond_atom[m][n] == batom2) {
//		bond_atom[m][n] = batom_new;
//		bond_type[m][n] = btype;
//		fprintf(stderr, "In Fix filopodia change_bond_into_two at proc %d when time = %d: bond %d-%d was changed to %d-%d\n", comm->me, update->ntimestep, batom1, batom2, batom1, batom_new);
//	  }
//	}
//  }
//  
//  if ((m = idx_new) >= 0) {
//    if (num_bond[m] == atom->bond_per_atom)
//      error->one(FLERR, "New bond exceeded bonds per atom in change_bond_into_two");
//	bond_type[m][num_bond[m]] = btype;
//    bond_atom[m][num_bond[m]] = batom2;
//    num_bond[m]++;
//	fprintf(stderr, "In Fix filopodia change_bond_into_two at proc %d when time = %d: bond %d-%d was created\n", comm->me, update->ntimestep, batom2, batom_new);
//  }
//  atom->nbonds++;
//
///*   if (force->newton_bond) return;
//
//  if ((m = idx2) >= 0) {
//    if (num_bond[m] == atom->bond_per_atom)
//      error->one(FLERR, "New bond exceeded bonds per atom in change_bond_into_two");
//    bond_type[m][num_bond[m]] = btype;
//    bond_atom[m][num_bond[m]] = batom1;
//    num_bond[m]++;
//  } */
//}

/* ---------------------------------------------------------------------- */

//void FixFilopodiaRandom::create_new_angle(int aatom1, int aatom2, int aatom3, int atype)
//{
//  int m;
//
//  // check that 3 atoms exist
//
//  int nlocal = atom->nlocal;
//  int idx1 = atom->map(aatom1);
//  int idx2 = atom->map(aatom2);
//  int idx3 = atom->map(aatom3);
//
//  int count = 0;
//  if ((idx1 >= 0) && (idx1 < nlocal)) count++;
//  if ((idx2 >= 0) && (idx2 < nlocal)) count++;
//  if ((idx3 >= 0) && (idx3 < nlocal)) count++;
//
//  int allcount;
//  MPI_Allreduce(&count, &allcount, 1, MPI_INT, MPI_SUM, world);
//  if (allcount != 3) error->all(FLERR, "create_new_angle angle atoms do not exist");
//
//  // create angle once or 3x if newton_bond set
//
//  int *num_angle = atom->num_angle;
//  int **angle_type = atom->angle_type;
//  tagint **angle_atom1 = atom->angle_atom1;
//  tagint **angle_atom2 = atom->angle_atom2;
//  tagint **angle_atom3 = atom->angle_atom3;
//
//  if ((m = idx2) >= 0) {
//    if (num_angle[m] == atom->angle_per_atom)
//      error->one(FLERR, "New angle exceeded angles per atom in create_new_angle");
//    angle_type[m][num_angle[m]] = atype;
//    angle_atom1[m][num_angle[m]] = aatom1;
//    angle_atom2[m][num_angle[m]] = aatom2;
//    angle_atom3[m][num_angle[m]] = aatom3;
//    num_angle[m]++;
//	fprintf(stderr, "In Fix filopodia create_new_angle at proc %d when time = %d: angle %d-%d-%d was created, angle type is %d\n", comm->me, update->ntimestep, aatom1, aatom2, aatom3, atype);
//  }
//  atom->nangles++;
//
///*   if (force->newton_bond) return;
//
//  if ((m = idx1) >= 0) {
//    if (num_angle[m] == atom->angle_per_atom)
//      error->one(FLERR, "New angle exceeded angles per atom in create_new_angle");
//    angle_type[m][num_angle[m]] = atype;
//    angle_atom1[m][num_angle[m]] = aatom1;
//    angle_atom2[m][num_angle[m]] = aatom2;
//    angle_atom3[m][num_angle[m]] = aatom3;
//    num_angle[m]++;
//  }
//
//  if ((m = idx3) >= 0) {
//    if (num_angle[m] == atom->angle_per_atom)
//      error->one(FLERR, "New angle exceeded angles per atom in create_new_angle");
//    angle_type[m][num_angle[m]] = atype;
//    angle_atom1[m][num_angle[m]] = aatom1;
//    angle_atom2[m][num_angle[m]] = aatom2;
//    angle_atom3[m][num_angle[m]] = aatom3;
//    num_angle[m]++;
//  } */
//}

/* ---------------------------------------------------------------------- */

//void FixFilopodiaRandom::change_angle(int aatom2, int aatom_new, int position)
//{
//  if (position != 1 && position != 3) error->all(FLERR, "create_new_angle new aatom position is not 1 or 3");
//  
//  int m,n;
//
//  // check that 3 atoms exist
//
//  int nlocal = atom->nlocal;
//  int idx2 = atom->map(aatom2);
//  int idx_new = atom->map(aatom_new);
//
//  int count = 0;
//  if ((idx2 >= 0) && (idx2 < nlocal)) count++;
//  if ((idx_new >= 0) && (idx_new < nlocal)) count++;
//
//  int allcount;
//  MPI_Allreduce(&count, &allcount, 1, MPI_INT, MPI_SUM, world);
//  if (allcount != 2) error->all(FLERR, "create_new_angle angle atoms do not exist");
//
//  // create angle once or 3x if newton_bond set
//
//  int *num_angle = atom->num_angle;
//  int **angle_type = atom->angle_type;
//  tagint **angle_atom1 = atom->angle_atom1;
//  tagint **angle_atom2 = atom->angle_atom2;
//  tagint **angle_atom3 = atom->angle_atom3;
//
//  if ((m = idx2) >= 0) {
//    if (position == 1) {
//	  for (n = 0; n < num_angle[m]; n++) {
//	    angle_atom1[m][n] = aatom_new;
//		fprintf(stderr, "In Fix filopodia change_angle at proc %d when time = %d: the aatom1 of atom %d was changed to %d\n", comm->me, update->ntimestep, aatom2, aatom_new);
//	  }
//	}
//	else {
//	  for (n = 0; n < num_angle[m]; n++) {
//	    angle_atom3[m][n] = aatom_new;
//		fprintf(stderr, "In Fix filopodia change_angle at proc %d when time = %d: the aatom3 of atom %d was changed to %d\n", comm->me, update->ntimestep, aatom2, aatom_new);
//	  }
//	}
//  }
//  // atom->nangles++;
//
///*   if (force->newton_bond) return;
//
//  if ((m = idx2) >= 0) {
//    if (num_angle[m] == atom->angle_per_atom)
//      error->one(FLERR, "New angle exceeded angles per atom in create_new_angle");
//    angle_type[m][num_angle[m]] = atype;
//    angle_atom1[m][num_angle[m]] = aatom1;
//    angle_atom2[m][num_angle[m]] = aatom2;
//    angle_atom3[m][num_angle[m]] = aatom3;
//    num_angle[m]++;
//  }
//
//  if ((m = idx3) >= 0) {
//    if (num_angle[m] == atom->angle_per_atom)
//      error->one(FLERR, "New angle exceeded angles per atom in create_new_angle");
//    angle_type[m][num_angle[m]] = atype;
//    angle_atom1[m][num_angle[m]] = aatom1;
//    angle_atom2[m][num_angle[m]] = aatom2;
//    angle_atom3[m][num_angle[m]] = aatom3;
//    num_angle[m]++;
//  } */
//}