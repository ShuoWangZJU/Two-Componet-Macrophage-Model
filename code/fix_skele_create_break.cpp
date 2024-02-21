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
#include "fix_skele_create_break.h"
#include "update.h"
#include "respa.h"
#include "atom.h"
#include "atom_vec.h"
#include "bond.h"
#include "force.h"
#include "pair.h"
#include "group.h"
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
#define DELTA 16
/* ---------------------------------------------------------------------- */

FixSkeleCreateBreak::FixSkeleCreateBreak(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 11) error->all(FLERR,"Illegal fix SkeleCreateBreak command");

  MPI_Comm_rank(world,&me);

  nevery = force->inumeric(FLERR,arg[3]);
  if (nevery <= 0) error->all(FLERR,"Illegal fix SkeleCreateBreak command");
  
  break_bond_type = force->inumeric(FLERR,arg[4]);
  
  int igroup_target;
  char gn[50];
  sprintf(gn,arg[5]);
  igroup_target = group->find(arg[5]);
  if (igroup_target == -1) error->all(FLERR,"Could not find molecule group ID for fix ProtrusiveSkeleton command");
  groupbit_target = group->bitmask[igroup_target];
  
  r_break = force->numeric(FLERR,arg[6]);
  if (r_break < 0.0) error->all(FLERR,"Illegal fix SkeleCreateBreak command");
  r_breaksq = r_break*r_break;
  
  band3_bond_type = force->inumeric(FLERR,arg[7]);
  
  r_new = force->numeric(FLERR,arg[8]);
  
  d_new = force->numeric(FLERR,arg[9]);
  d_newsq = d_new*d_new;
  
  r_new_cut = force->numeric(FLERR,arg[10]);
  r_new_cutsq = r_new_cut*r_new_cut;

  force_reneighbor = 1;
  next_reneighbor = -1;
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;

  // error check

  if (atom->molecular != 1)
    error->all(FLERR,"Cannot use fix SkeleCreateBreak with non-molecular systems");

  // allocate arrays local to this fix

  nnmax = atom->nmax;
  maxbond_per_atom = atom->bond_per_atom;

  // perform initial allocation of atom-based arrays
  // register with Atom class
  // bond_count values will be initialized in setup()
  
  bond_count = NULL;
  bond_sub_full_count = NULL;
  bond_add_full_count = NULL;
  bond_sub_count = NULL;
  bond_add_count = NULL;
  bond_initial_count = NULL;
  skele_status = NULL;
  bond_sub_full = NULL;
  bond_add_full = NULL;
  bond_sub = NULL;
  bond_add = NULL;
  bond_atom_full = NULL;
  grow_arrays(nnmax);
  atom->add_callback(0);
  countflag = 0;

  // set comm sizes needed by this fix
  // forward is big due to comm of broken bonds and 1-2 neighbors

//   comm_forward = MAX(2,2+atom->maxspecial); old
  comm_forward = 7+5*maxbond_per_atom;
  comm_reverse = 4+4*maxbond_per_atom;

  // copy = special list for one atom
  // size = ms^2 + ms is sufficient
  // b/c in rebuild_special() neighs of all 1-2s are added,
  //   then a dedup(), then neighs of all 1-3s are added, then final dedup()
  // this means intermediate size cannot exceed ms^2 + ms

  // zero out stats

  breakcount = 0;
  breakcounttotal = 0;
}

/* ---------------------------------------------------------------------- */

FixSkeleCreateBreak::~FixSkeleCreateBreak()
{
  // unregister callbacks to this fix from Atom class

  atom->delete_callback(id,0);

  // delete locally stored arrays
  
  memory->destroy(bond_count);
  memory->destroy(bond_sub_full_count);
  memory->destroy(bond_add_full_count);
  memory->destroy(bond_sub_count);
  memory->destroy(bond_add_count);
  memory->destroy(bond_initial_count);
  memory->destroy(skele_status);
  memory->destroy(bond_sub_full);
  memory->destroy(bond_add_full);
  memory->destroy(bond_sub);
  memory->destroy(bond_add);
  memory->destroy(bond_atom_full);
}

/* ---------------------------------------------------------------------- */

int FixSkeleCreateBreak::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixSkeleCreateBreak::init()
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

void FixSkeleCreateBreak::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixSkeleCreateBreak::setup(int vflag)
{
  int i,j,k,m;

  // compute initial bond_count if this is first run
  // can't do this earlier, in constructor or init, b/c need ghost info

  if (countflag) return;
  countflag = 1;

  // count bonds stored with each bond I own
  // if newton bond is not set, just increment count on atom I
  // if newton bond is set, also increment count on atom J even if ghost
  // bond_count is long enough to tally ghost atom counts

  int *mask = atom->mask;
  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  int newton_bond = force->newton_bond;

  for (i = 0; i < nall; i++){
	  bond_count[i] = 0;
	  bond_sub_full_count[i] = 0;
	  bond_add_full_count[i] = 0;
	  bond_sub_count[i] = 0;
	  bond_add_count[i] = 0;
	  bond_initial_count[i] = 0;
	  skele_status[i] = 0;
	  for (k = 0; k < maxbond_per_atom; k++) {
		bond_sub_full[i][k] = 0;
		bond_add_full[i][k] = 0;
		bond_sub[i][k] = 0;
		bond_add[i][k] = 0;
		bond_atom_full[i][k] = 0;
	  }
  }

  for (i = 0; i < nlocal; i++)
    for (j = 0; j < num_bond[i]; j++) {
      if (bond_type[i][j] == break_bond_type) {
		skele_status[i] = 1;
		bond_atom_full[i][bond_count[i]] = bond_atom[i][j];
        bond_count[i]++;
        if (newton_bond) {
          m = atom->map(bond_atom[i][j]);
		  skele_status[m] = 1;
          if (m < 0) error->one(FLERR,"Fix SkeleCreateBreak needs ghost atoms from further away");
		  else if (m < nlocal) {
		    bond_atom_full[m][bond_count[m]] = atom->tag[i];
            bond_count[m]++;
		  }
		  else {
		    bond_add_full[m][bond_add_full_count[m]] = atom->tag[i];
            bond_add_full_count[m]++;
		  }
        }
      }
    }
	
/*   for (i = 0; i < nall; i++){
	if (mask[i] & groupbit){
	  if(i < nlocal) fprintf(stderr, "In fix_skele_create_break setup() 01 when timestep = %d at proc %d: local bond_count[i] = %d, bond_add_full_count[i] = %d, bond_initial_count[i] = %d, skele_status[i] = %d, atom->tag[i] = %d\n", update->ntimestep, comm->me, bond_count[i], bond_add_full_count[i], bond_initial_count[i], skele_status[i], atom->tag[i]);
	  else fprintf(stderr, "In fix_skele_create_break setup() 01 when timestep = %d at proc %d: ghost bond_count[i] = %d, bond_add_full_count[i] = %d, bond_initial_count[i] = %d, skele_status[i] = %d, atom->tag[i] = %d\n", update->ntimestep, comm->me, bond_count[i], bond_add_full_count[i], bond_initial_count[i], skele_status[i], atom->tag[i]);
	}
  } */
  
  if (newton_bond) comm->reverse_comm_fix(this);
  for (i = 0; i < nall; i++){
	  bond_initial_count[i] = bond_count[i];
  }
  comm->forward_comm_fix(this);
  
/*   for (i = 0; i < nall; i++){
	if (mask[i] & groupbit){
	  if(i < nlocal) fprintf(stderr, "In fix_skele_create_break setup() 02 when timestep = %d at proc %d: local bond_count[i] = %d, bond_add_full_count[i] = %d, bond_initial_count[i] = %d, skele_status[i] = %d, atom->tag[i] = %d\n", update->ntimestep, comm->me, bond_count[i], bond_add_full_count[i], bond_initial_count[i], skele_status[i], atom->tag[i]);
	  else fprintf(stderr, "In fix_skele_create_break setup() 02 when timestep = %d at proc %d: ghost bond_count[i] = %d, bond_add_full_count[i] = %d, bond_initial_count[i] = %d, skele_status[i] = %d, atom->tag[i] = %d\n", update->ntimestep, comm->me, bond_count[i], bond_add_full_count[i], bond_initial_count[i], skele_status[i], atom->tag[i]);
	}
  } */
}

/* ---------------------------------------------------------------------- */

void FixSkeleCreateBreak::post_integrate()
{
  int i,j,k,m,n,ii,jj,mm,nn,tmpflag,inum,jnum,itype,jtype,n1,n2,n3,possible,i1,i2,type_bond, jb, jc;
  double xtmp,ytmp,ztmp,delx,dely,delz,delx2,dely2,delz2,delx3,dely3,delz3,rsq,rminsq,rmnsq,rijsq,rmjsq,rnjsq,ri,ri2,ri3,cosmn,cosmjj,cosnjj;
  int *ilist,*jlist,*numneigh,**firstneigh;
  tagint *slist;

  if (update->ntimestep % nevery) return;

  // check that all procs have needed ghost atoms within ghost cutoff
  // only if neighbor list has changed since last check
  // needs to be <= test b/c neighbor list could have been re-built in
  //   same timestep as last post_integrate() call, but afterwards
  // NOTE: no longer think is needed, due to error tests on atom->map()
  // NOTE: if delete, can also delete lastcheck and check_ghosts()

  //if (lastcheck <= neighbor->lastcall) check_ghosts();

  // acquire updated ghost atom positions
  // necessary b/c are calling this after integrate, but before Verlet comm

  comm->forward_comm();

  // forward comm of bond_count, so ghosts have it

  comm->forward_comm_fix(this);

  // resize bond partner list and initialize it
  // probability array overlays distsq array
  // needs to be atom->nmax in length

  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;

  double **x = atom->x;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  double *bondlist_length = neighbor->bondlist_length;
  
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  double **bond_length = atom->bond_length;
  int *type = atom->type;
  neighbor->build_one(list,1);
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;


  //Break:
  
  // if (force->newton_bond)   comm->reverse_comm_fix(this);
  
/*   for (i = 0; i < nall; i++){
	if (mask[i] & groupbit){
	  if(i < nlocal) fprintf(stderr, "In fix_skele_create_break post_integrate() innitially 01 when timestep = %d at proc %d: local bond_count[%d] = %d, bond_add_full_count[%d] = %d, bond_initial_count[%d] = %d, skele_status[%d] = %d, protru_flag[%d] = %d, atom->tag[%d] = %d\n", update->ntimestep, comm->me, i, bond_count[i], i, bond_add_full_count[i], i, bond_initial_count[i], i, skele_status[i], i, protru_flag[i], i, atom->tag[i]);
	  else fprintf(stderr, "In fix_skele_create_break post_integrate() innitially 01 when timestep = %d at proc %d: ghost bond_count[%d] = %d, bond_add_full_count[%d] = %d, bond_initial_count[%d] = %d, skele_status[%d] = %d, protru_flag[%d] = %d, atom->tag[%d] = %d\n", update->ntimestep, comm->me, i, bond_count[i], i, bond_add_full_count[i], i, bond_initial_count[i], i, skele_status[i], i, protru_flag[i], i, atom->tag[i]);
	}
  } */
/*   for (i = 0; i < nall; i++){
	if (bond_count[i] != 0) {
	  if (i < nlocal) fprintf(stderr, "In fix_skele_create_break post_integrate()01 when timestep = %d at proc %d: the bond_count of local atom %d is %d, bond_atom_full[i][0] = %d, bond_atom_full[i][bond_count[i]-1] = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], bond_atom_full[i][0], bond_atom_full[i][bond_count[i]-1]);
	  else fprintf(stderr, "In fix_skele_create_break post_integrate()01 when timestep = %d at proc %d: the bond_count of ghost atom %d is %d, bond_atom_full[i][0] = %d, bond_atom_full[i][bond_count[i]-1] = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], bond_atom_full[i][0], bond_atom_full[i][bond_count[i]-1]);
	}
  } */
  
  nbreak = 0;
  ncreate = 0;
  for (i = 0; i < nlocal; i++){
	if (mask[i] & groupbit){
	  if (!num_bond[i]) continue;
	  for (m = 0; m < num_bond[i]; m++) {
	    if (bond_type[i][m] == band3_bond_type) {
	      xtmp = x[atom->map(bond_atom[i][m])][0];
	      ytmp = x[atom->map(bond_atom[i][m])][1];
	      ztmp = x[atom->map(bond_atom[i][m])][2];
		  //fprintf(stderr, "In fix_skele_create_break post_integrate() when timestep = %d at proc %d: the connecting band3 of atom %d is %d, x = %f, y = %f, z = %f\n", update->ntimestep, comm->me, atom->tag[i], bond_atom[i][m], xtmp, ytmp, ztmp);
	    }
	  }
	  
	  // skele_status[i] = 0;
	  rminsq = BIG;
	  for (j = 0; j < nlocal; j++){
	    if (mask[j] & groupbit_target){
	      delx = x[j][0] - xtmp;
	      dely = x[j][1] - ytmp;
	      delz = x[j][2] - ztmp;
	      rsq = delx*delx + dely*dely + delz*delz;
		  //fprintf(stderr, "In fix_skele_create_break post_integrate() when timestep = %d at proc %d: the distance between connecting band3 and RBC atom %d is %d, x = %f, y = %f, z = %f\n", update->ntimestep, comm->me, atom->tag[j], bond_atom[i][m], rsq, x[j][0], x[j][1], x[j][2]);
		  if (rsq < rminsq){
	      	rminsq = rsq;
	      }
	    }
	  }
	  
	  //fprintf(stderr, "In fix_skele_create_break post_integrate() when timestep = %d at proc %d: the rminsq of atom %d is %f\n", update->ntimestep, comm->me, atom->tag[i], rminsq);
	  
	  if (rminsq <= r_breaksq){
		if (skele_status[i] == 1 || skele_status[i] == 2) {
	  	  skele_status[i] = 3;
		}
		else if (skele_status[i] == 0) {
	  	  skele_status[i] = 5;
		}
	  }
	  else {
		if (skele_status[i] == 5) {
		  skele_status[i] = 0;
		}
	  }
	}
  }
  
  comm->forward_comm_fix(this);
  
/*   for (i = 0; i < nall; i++){
	if (mask[i] & groupbit){
	  if(i < nlocal) fprintf(stderr, "In fix_skele_create_break post_integrate() innitially 02 when timestep = %d at proc %d: local bond_count[%d] = %d, bond_add_full_count[%d] = %d, bond_initial_count[%d] = %d, skele_status[%d] = %d, protru_flag[%d] = %d, atom->tag[%d] = %d\n", update->ntimestep, comm->me, i, bond_count[i], i, bond_add_full_count[i], i, bond_initial_count[i], i, skele_status[i], i, protru_flag[i], i, atom->tag[i]);
	  else fprintf(stderr, "In fix_skele_create_break post_integrate() innitially 02 when timestep = %d at proc %d: ghost bond_count[%d] = %d, bond_add_full_count[%d] = %d, bond_initial_count[%d] = %d, skele_status[%d] = %d, protru_flag[%d] = %d, atom->tag[%d] = %d\n", update->ntimestep, comm->me, i, bond_count[i], i, bond_add_full_count[i], i, bond_initial_count[i], i, skele_status[i], i, protru_flag[i], i, atom->tag[i]);
	}
  } */
  
  // Break
  for (i = 0; i < nlocal; i++){
	if (mask[i] & groupbit){
//	  if (skele_status[i] == 1) {
//	  	  // if (bond_count[i] != 0) error->all(FLERR,"Bonds haven't been all broke in fix skele_create_break command");
//		  fprintf(stderr, "In fix_skele_create_break post_integrate() when timestep = %d at proc %d: the bond_count01 of atom %d is %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i]);
//	  }
	  for (m = 0; m < num_bond[i]; m++) {
	  	if (bond_type[i][m] == break_bond_type) {
		  j = atom->map(bond_atom[i][m]);
	  	  if (skele_status[i] == 3 || skele_status[j] == 3) {
	  	  	fprintf(stderr, "In fix_skele_create_break post_integrate() when timestep = %d at proc %d: the bond between atom %d and %d broke, the bond type is %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_atom[i][m], bond_type[i][m]);
	  	  	for (k = m; k < num_bond[i]-1; k++) {
	  	  	  bond_atom[i][k] = bond_atom[i][k+1];
	  	  	  bond_type[i][k] = bond_type[i][k+1];
	  	  	  bond_length[i][k] = bond_length[i][k+1];
	  	  	}
			for (k = 0; k < bond_count[i]; k++) {
			  if (bond_atom_full[i][k] == atom->tag[j]) {
			    for (ii = k; ii < bond_count[i]-1; ii++) {
			  	  bond_atom_full[i][ii] = bond_atom_full[i][ii+1];
			    }
			    bond_atom_full[i][bond_count[i]-1] = 0;
			    bond_count[i]--;
			    break;
			  }
			}
			if (force->newton_bond) {
			  if (j < nlocal) {
			    for (k = 0; k < bond_count[j]; k++) {
			      if (bond_atom_full[j][k] == atom->tag[i]) {
			        for (ii = k; ii < bond_count[j]-1; ii++) {
			      	  bond_atom_full[j][ii] = bond_atom_full[j][ii+1];
			        }
			        bond_atom_full[j][bond_count[j]-1] = 0;
			        bond_count[j]--;
			        break;
			      }
				}
			  }
			  else {
				bond_sub_full[j][bond_sub_full_count[j]] = atom->tag[i];
			    bond_sub_full_count[j]++;
			  }
			}
	  	  	num_bond[i]--;
	  	  	m--;
	  	  	nbreak++;
	  	  }
	  	}
	  }
	  
/* 	  if (skele_status[i] == 3) {
	  	  if (bond_count[i] != 0) error->all(FLERR,"Bonds haven't been all broken in fix skele_create_break command");
		  //fprintf(stderr, "In fix_skele_create_break post_integrate() when timestep = %d at proc %d: the bond_count02 of atom %d is %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i]);
	  } */
	}
  }
  
/*   for (i = 0; i < nall; i++){
	if (skele_status[i] == 3) {
	  if (i < nlocal) fprintf(stderr, "In fix_skele_create_break post_integrate()01 when timestep = %d at proc %d: the bond_count of local atom %d is %d, bond_add_full_count = %d, bond_sub_full_count = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], bond_add_full_count[i], bond_sub_full_count[i]);
	  else fprintf(stderr, "In fix_skele_create_break post_integrate()01 when timestep = %d at proc %d: the bond_count of ghost atom %d is %d, bond_add_full_count = %d, bond_sub_full_count = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], bond_add_full_count[i], bond_sub_full_count[i]);
	}
  } */
  
  if (force->newton_bond)   comm->reverse_comm_fix(this);
  
/*   for (i = 0; i < nall; i++){
	if (bond_count[i] != 0) {
	  if (i < nlocal) fprintf(stderr, "In fix_skele_create_break post_integrate()02 when timestep = %d at proc %d: the bond_count of local atom %d is %d, bond_atom_full[i][0] = %d, bond_atom_full[i][bond_count[i]-1] = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], bond_atom_full[i][0], bond_atom_full[i][bond_count[i]-1]);
	  else fprintf(stderr, "In fix_skele_create_break post_integrate()02 when timestep = %d at proc %d: the bond_count of ghost atom %d is %d, bond_atom_full[i][0] = %d, bond_atom_full[i][bond_count[i]-1] = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], bond_atom_full[i][0], bond_atom_full[i][bond_count[i]-1]);
	}
  } */

/*   for (i = 0; i < nall; i++){
	  bond_count[i] += bond_add_full_count[i];
	  bond_add_full_count[i] = 0;
  } */
  
/*   for (i = 0; i < nall; i++){
	if (mask[i] & groupbit){
	  if(i < nlocal) fprintf(stderr, "In fix_skele_create_break post_integrate() innitially 05 when timestep = %d at proc %d: local bond_count[%d] = %d, bond_add_full_count[%d] = %d, bond_initial_count[%d] = %d, skele_status[%d] = %d, protru_flag[%d] = %d, atom->tag[%d] = %d\n", update->ntimestep, comm->me, i, bond_count[i], i, bond_add_full_count[i], i, bond_initial_count[i], i, skele_status[i], i, protru_flag[i], i, atom->tag[i]);
	  else fprintf(stderr, "In fix_skele_create_break post_integrate() innitially 05 when timestep = %d at proc %d: ghost bond_count[%d] = %d, bond_add_full_count[%d] = %d, bond_initial_count[%d] = %d, skele_status[%d] = %d, protru_flag[%d] = %d, atom->tag[%d] = %d\n", update->ntimestep, comm->me, i, bond_count[i], i, bond_add_full_count[i], i, bond_initial_count[i], i, skele_status[i], i, protru_flag[i], i, atom->tag[i]);
	}
  } */
  
  //reevaluate skele_status
  for (i = 0; i < nlocal; i++){
	if (mask[i] & groupbit){
	  if (skele_status[i] == 1 || skele_status[i] == 2) {
		// fprintf(stderr, "In fix_skele_create_break reevaluate when timestep = %d at proc %d: the bond_count of local atom %d is %d, bond_initial_count = %d, skele_status = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], bond_initial_count[i], skele_status[i]);
		if (bond_count[i] < bond_initial_count[i] && bond_count[i] > 0) {
		  skele_status[i] = 2;
		  fprintf(stderr, "In fix_skele_create_break reevaluate 01 when timestep = %d at proc %d: the bond_count of local atom %d is %d, bond_initial_count = %d, skele_status = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], bond_initial_count[i], skele_status[i]);
		}
		else if (bond_count[i] == 0) {
		  skele_status[i] = 3;
		  fprintf(stderr, "In fix_skele_create_break reevaluate 02 when timestep = %d at proc %d: the bond_count of local atom %d is %d, bond_initial_count = %d, skele_status = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], bond_initial_count[i], skele_status[i]);
		}
	  }
	}
  }
  
  comm->forward_comm_fix(this);
  
/*   for (i = 0; i < nall; i++){
	if (bond_count[i] != 0) {
	  if (i < nlocal) fprintf(stderr, "In fix_skele_create_break post_integrate()03 when timestep = %d at proc %d: the bond_count of local atom %d is %d, bond_atom_full[i][0] = %d, bond_atom_full[i][bond_count[i]-1] = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], bond_atom_full[i][0], bond_atom_full[i][bond_count[i]-1]);
	  else fprintf(stderr, "In fix_skele_create_break post_integrate()03 when timestep = %d at proc %d: the bond_count of ghost atom %d is %d, bond_atom_full[i][0] = %d, bond_atom_full[i][bond_count[i]-1] = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], bond_atom_full[i][0], bond_atom_full[i][bond_count[i]-1]);
	}
  } */
  
  // creation part, based on the existing bonds: m——i and i——n, only form one bond at a time
  // the case that skele_status[i] = 1, skele_status[m] = 2, skele_status[n] = 2, in this case we connect m and n
  for (i = 0; i < nlocal; i++){
	if ((mask[i] & groupbit) && bond_count[i] > 1){    //atom i should belong to skeleton and have at least two skeleton bonds
	  if (skele_status[i] == 1) {
		// fprintf(stderr, "In fix_skele_create_break create test01 when timestep = %d at proc %d: atom->tag[i] = %d, bond_count[i] = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i]);
		for (mm = 0; mm < bond_count[i]-1; mm++) {
		  for (nn = mm+1; nn < bond_count[i]; nn++) {
			m = atom->map(bond_atom_full[i][mm]);
			n = atom->map(bond_atom_full[i][nn]);
			rmnsq = (x[m][0] - x[n][0]) * (x[m][0] - x[n][0]) + (x[m][1] - x[n][1]) * (x[m][1] - x[n][1]) + (x[m][2] - x[n][2]) * (x[m][2] - x[n][2]);
			if (skele_status[m] == 2 && skele_status[n] == 2 && rmnsq < r_new_cutsq && m != n) {    // make sure the distance between m and n is within the cutoff
			  // fprintf(stderr, "In fix_skele_create_break create test02 when timestep = %d at proc %d: atom->tag[i] = %d, bond_count[i] = %d, atom->tag[m] = %d, bond_count[m] = %d, atom->tag[n] = %d, bond_count[n] = %d, mm = %d, nn = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], atom->tag[m], bond_count[m], atom->tag[n], bond_count[n],mm,nn);
			  tmpflag = 1;
			  for (ii = 0; ii < bond_count[m]; ii++) {    // make sure there are no bond between m and n before forming new bond
			    // fprintf(stderr, "In fix_skele_create_break create test021 when timestep = %d at proc %d: ii = %d, bond_atom_full[m][ii] = %d, atom->tag[m] = %d, atom->tag[n] = %d.\n", update->ntimestep, comm->me, ii, bond_atom_full[m][ii], atom->tag[m], atom->tag[n]);
				if (bond_atom_full[m][ii] == atom->tag[n]) tmpflag = -1;
			  }
			  if (tmpflag == 1) {
			    delx = x[m][0] - x[i][0];
			    dely = x[m][1] - x[i][1];
			    delz = x[m][2] - x[i][2];
			    ri = sqrt(delx * delx + dely * dely + delz * delz);
			    delx2 = x[n][0] - x[i][0];
			    dely2 = x[n][1] - x[i][1];
			    delz2 = x[n][2] - x[i][2];
			    ri2 = sqrt(delx2 * delx2 + dely2 * dely2 + delz2 * delz2);
			    cosmn = (delx * delx2 + dely * dely2 + delz * delz2) / ri / ri2;    // cos(theta_mn), theta_mn is the angle between i->m and i->n
			    for (ii = 0; ii < bond_count[i]; ii++) {    // make sure n is one of the two closest i bond partners to m 
				  if (ii != mm && ii != nn) {
				    jj = atom->map(bond_atom_full[i][ii]);
				    delx3 = x[jj][0] - x[i][0];
				    dely3 = x[jj][1] - x[i][1];
				    delz3 = x[jj][2] - x[i][2];
				    ri3 = sqrt(delx3 * delx3 + dely3 * dely3 + delz3 * delz3);
				    cosmjj = (delx * delx3 + dely * dely3 + delz * delz3) / ri / ri3;    // cos(theta_mjj), theta_mjj is the angle between i->m and i->jj
				    if (cosmjj > cosmn) {    // if theta_mjj < theta_mn
					  cosnjj = (delx2 * delx3 + dely2 * dely3 + delz2 * delz3) / ri2 / ri3;    // cos(theta_njj), theta_njj is the angle between i->n and i->jj
					  if (cosnjj > cosmn) tmpflag --;
				  	  // break;
				    }
				  }
			    }
			  }
			  
			  if (tmpflag > 0) {
				fprintf(stderr, "In fix_skele_create_break create01 when timestep = %d at proc %d: the bond between atom %d and %d formed, bond_count[m] = %d, bond_count[n] = %d, atom->tag[i] = %d, bond_count[i] = %d.\n", update->ntimestep, comm->me, atom->tag[m], atom->tag[n], bond_count[m], bond_count[n], atom->tag[i], bond_count[i]);
			    if (m < nlocal) {    // forming new bond for m
				  bond_atom_full[m][bond_count[m]] = atom->tag[n];
				  bond_count[m]++;
				  bond_atom[m][num_bond[m]] = atom->tag[n];
				  bond_type[m][num_bond[m]] = break_bond_type;
				  bond_length[m][num_bond[m]] = r_new;
				  num_bond[m]++;
			    }
			    else {
				  bond_add_full[m][bond_add_full_count[m]] = atom->tag[n];
				  bond_add_full_count[m]++;
				  bond_add[m][bond_add_count[m]] = atom->tag[n];
				  bond_add_count[m]++;
			    }
			    if (n < nlocal) {    // forming new bond for n
				  bond_atom_full[n][bond_count[n]] = atom->tag[m];
				  bond_count[n]++;
			    }
			    else {
				  bond_add_full[n][bond_add_full_count[n]] = atom->tag[m];
				  bond_add_full_count[n]++;
			    }
				ncreate++;
				
				nn = bond_count[i];    // jump out of the whole loop and end the creation part
				mm = bond_count[i]-1;
				i = nlocal;
			  }
			}
		  }
		}
	  }
	}
  }
  MPI_Allreduce(&ncreate,&createcount,1,MPI_INT,MPI_SUM,world);
  
  /* // the case that skele_status[i] = 2, skele_status[m] = 2, skele_status[n] = 2, in this case we find a free skeleton bead j, and connect i——j, m——j, n——j
  if (createcount == 0) {    // if there are no bond forming in last case
    for (i = 0; i < nlocal; i++){
	  if ((mask[i] & groupbit) && bond_count[i] == bond_initial_count[i]-1){    //atom i should belong to skeleton and have at least two skeleton bonds
	    if (skele_status[i] == 2) {
		  // fprintf(stderr, "In fix_skele_create_break create test03 when timestep = %d at proc %d: atom->tag[i] = %d, bond_count[i] = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i]);
	  	  for (mm = 0; mm < bond_count[i]-1; mm++) {
	  	    for (nn = mm+1; nn < bond_count[i]; nn++) {
			  m = atom->map(bond_atom_full[i][mm]);
			  n = atom->map(bond_atom_full[i][nn]);
	  	  	  if (skele_status[m] == 2 && skele_status[n] == 2 && m != i && n != i && m != n) {
				// fprintf(stderr, "In fix_skele_create_break create test04 when timestep = %d at proc %d: atom->tag[i] = %d, bond_count[i] = %d, atom->tag[m] = %d, bond_count[m] = %d, atom->tag[n] = %d, bond_count[n] = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], atom->tag[m], bond_count[m], atom->tag[n], bond_count[n]);
		  	    xtmp = 0;
		  	    ytmp = 0;
		  	    ztmp = 0;
	  	  	    for (ii = 0; ii < bond_count[i]; ii++) {    //caculate the sum of the unit vectors of the existing bonds to get the opposite direction of new j
		  	  	  jj = atom->map(bond_atom_full[i][ii]);
				  rmnsq = (x[jj][0] - x[i][0]) * (x[jj][0] - x[i][0]) + (x[jj][1] - x[i][1]) * (x[jj][1] - x[i][1]) + (x[jj][2] - x[i][2]) * (x[jj][2] - x[i][2]);
		  	  	  xtmp += (x[jj][0] - x[i][0])/sqrt(rmnsq);
		  	  	  ytmp += (x[jj][1] - x[i][1])/sqrt(rmnsq);
		  	  	  ztmp += (x[jj][2] - x[i][2])/sqrt(rmnsq);
		  	    }
		  	    rmnsq = xtmp * xtmp + ytmp * ytmp + ztmp * ztmp;
		  	    xtmp = x[i][0] - r_new * xtmp / sqrt(rmnsq);    // the position of expexted new j
		  	    ytmp = x[i][1] - r_new * ytmp / sqrt(rmnsq);
		  	    ztmp = x[i][2] - r_new * ztmp / sqrt(rmnsq);
				rminsq = BIG;
				j = 0;
		  	    for (jj = 0; jj < nall; jj++) {
		  	  	  if ((mask[jj] & groupbit) && (skele_status[jj] == 0 || skele_status[jj] == 3) && jj != i && jj != m && jj != n) {
		  	  	    rsq = (x[jj][0] - xtmp) * (x[jj][0] - xtmp) + (x[jj][1] - ytmp) * (x[jj][1] - ytmp) + (x[jj][2] - ztmp) * (x[jj][2] - ztmp);
		  	  	    // fprintf(stderr, "In fix_skele_create_break create test05 when timestep = %d at proc %d: atom->tag[i] = %d, bond_count[i] = %d, atom->tag[m] = %d, bond_count[m] = %d, atom->tag[n] = %d, bond_count[n] = %d, xtmp = %f, ytmp = %f, ztmp = %f, rsq = %f.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], atom->tag[m], bond_count[m], atom->tag[n], bond_count[n],xtmp,ytmp,ztmp,rsq);
					if (rsq < d_newsq && rsq < rminsq) {
		  	  	  	  rminsq = rsq;
					  j = jj;
		  	  	    }
		  	  	  }
		  	    }
		  	    if (j != 0) {
		  	  	  rijsq = (x[i][0] - x[j][0]) * (x[i][0] - x[j][0]) + (x[i][1] - x[j][1]) * (x[i][1] - x[j][1]) + (x[i][2] - x[j][2]) * (x[i][2] - x[j][2]);
		  	  	  rmjsq = (x[m][0] - x[j][0]) * (x[m][0] - x[j][0]) + (x[m][1] - x[j][1]) * (x[m][1] - x[j][1]) + (x[m][2] - x[j][2]) * (x[m][2] - x[j][2]);
		  	  	  rnjsq = (x[n][0] - x[j][0]) * (x[n][0] - x[j][0]) + (x[n][1] - x[j][1]) * (x[n][1] - x[j][1]) + (x[n][2] - x[j][2]) * (x[n][2] - x[j][2]);
		  	  	  // fprintf(stderr, "In fix_skele_create_break create test06 when timestep = %d at proc %d: atom->tag[i] = %d, bond_count[i] = %d, atom->tag[m] = %d, bond_count[m] = %d, atom->tag[n] = %d, bond_count[n] = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], atom->tag[m], bond_count[m], atom->tag[n], bond_count[n]);
				  if (rijsq < r_new_cutsq && rmjsq < r_new_cutsq && rnjsq < r_new_cutsq) {    // make sure the distances between i j, m j, and n j are within the cutoff
		  	  	  	   
					fprintf(stderr, "In fix_skele_create_break create02 when timestep = %d at proc %d: the bond between atom %d, %d %d and %d formed, bond_count[i] = %d, bond_count[m] = %d, bond_count[n] = %d, bond_count[j] = %d.\n", update->ntimestep, comm->me, atom->tag[i], atom->tag[m], atom->tag[n], atom->tag[j], bond_count[i], bond_count[m], bond_count[n], bond_count[j]);
					bond_atom_full[i][bond_count[i]] = atom->tag[j];    // forming new bond for i
					bond_count[i]++;
					bond_atom[i][num_bond[i]] = atom->tag[j];
					bond_type[i][num_bond[i]] = break_bond_type;
					bond_length[i][num_bond[i]] = r_new;
					num_bond[i]++;
					ncreate++;
					
					if (m < nlocal) {    // forming new bond for m
					  bond_atom_full[m][bond_count[m]] = atom->tag[j];
					  bond_count[m]++;
					  bond_atom[m][num_bond[m]] = atom->tag[j];
					  bond_type[m][num_bond[m]] = break_bond_type;
					  bond_length[m][num_bond[m]] = r_new;
					  num_bond[m]++;
					}
					else {
					  bond_add_full[m][bond_add_full_count[m]] = atom->tag[j];
					  bond_add_full_count[m]++;
					  bond_add[m][bond_add_count[m]] = atom->tag[j];
					  bond_add_count[m]++;
					}
					ncreate++;
					
					if (n < nlocal) {    // forming new bond for n
					  bond_atom_full[n][bond_count[n]] = atom->tag[j];
					  bond_count[n]++;
					  bond_atom[n][num_bond[n]] = atom->tag[j];
					  bond_type[n][num_bond[n]] = break_bond_type;
					  bond_length[n][num_bond[n]] = r_new;
					  num_bond[n]++;
					}
					else {
					  bond_add_full[n][bond_add_full_count[n]] = atom->tag[j];
					  bond_add_full_count[n]++;
					  bond_add[n][bond_add_count[n]] = atom->tag[j];
					  bond_add_count[n]++;
					}
					ncreate++;
					
					if (j < nlocal) {    // forming new bond for j
					  bond_atom_full[j][bond_count[j]] = atom->tag[i];
					  bond_count[j]++;
					  bond_atom_full[j][bond_count[j]] = atom->tag[m];
					  bond_count[j]++;
					  bond_atom_full[j][bond_count[j]] = atom->tag[n];
					  bond_count[j]++;
					  skele_status[j] = 2;
					  bond_initial_count[j] = 6;
					}
					else {
					  bond_add_full[j][bond_add_full_count[j]] = atom->tag[i];
					  bond_add_full_count[j]++;
					  bond_add_full[j][bond_add_full_count[j]] = atom->tag[m];
					  bond_add_full_count[j]++;
					  bond_add_full[j][bond_add_full_count[j]] = atom->tag[n];
					  bond_add_full_count[j]++;
					}
					// skele_status[j] = 2;
					
					nn = bond_count[i];    // jump out of the whole loop and end the creation part
					mm = bond_count[i]-1;
					i = nlocal;
		  	  	  }
		  	    }
	  	  	  }
	  	    }
	  	  }
	    }
	  }
    }
  }
  MPI_Allreduce(&ncreate,&createcount,1,MPI_INT,MPI_SUM,world); */
  
  int pot_m, pot_j, pot_n, createcount_tmp;
  int pot_count = 0;
  int ncreate_tmp = 0;
  // the case that skele_status[i] = 1, skele_status[m] = 2, skele_status[n] = 2 and m has connected n, in this case we find a free skeleton bead j, and connect m——j, n——j
  if (createcount == 0) {    // if there are no bond forming in last case
    for (i = 0; i < nlocal; i++){
	  if ((mask[i] & groupbit) && bond_count[i] > 1){    //atom i should belong to skeleton and have at least two skeleton bonds
	    if (skele_status[i] == 1) {
	  	  // fprintf(stderr, "In fix_skele_create_break create test01 when timestep = %d at proc %d: atom->tag[i] = %d, bond_count[i] = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i]);
	  	  for (mm = 0; mm < bond_count[i]-1; mm++) {
	  	    for (nn = mm+1; nn < bond_count[i]; nn++) {
	  	  	  m = atom->map(bond_atom_full[i][mm]);
	  	  	  n = atom->map(bond_atom_full[i][nn]);
	  	  	  //rmnsq = (x[m][0] - x[n][0]) * (x[m][0] - x[n][0]) + (x[m][1] - x[n][1]) * (x[m][1] - x[n][1]) + (x[m][2] - x[n][2]) * (x[m][2] - x[n][2]);
	  	  	  if (skele_status[m] == 2 && skele_status[n] == 2 && m != n) {    // make sure the distance between m and n is within the cutoff
	  	  	    // fprintf(stderr, "In fix_skele_create_break create test02 when timestep = %d at proc %d: atom->tag[i] = %d, bond_count[i] = %d, atom->tag[m] = %d, bond_count[m] = %d, atom->tag[n] = %d, bond_count[n] = %d, mm = %d, nn = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], atom->tag[m], bond_count[m], atom->tag[n], bond_count[n],mm,nn);
	  	  	    tmpflag = -1;
	  	  	    for (ii = 0; ii < bond_count[m]; ii++) {    // make sure there are bond between m and n and i is the only bead that connects to both m and n before forming new bond
	  	  	      // fprintf(stderr, "In fix_skele_create_break create test021 when timestep = %d at proc %d: ii = %d, bond_atom_full[m][ii] = %d, atom->tag[m] = %d, atom->tag[n] = %d.\n", update->ntimestep, comm->me, ii, bond_atom_full[m][ii], atom->tag[m], atom->tag[n]);
	  	  	  	  if (bond_atom_full[m][ii] == atom->tag[n]) tmpflag = 1;
				  for (jj = 0; jj < bond_count[n]; jj++) {
					if (bond_atom_full[m][ii] == bond_atom_full[n][jj] && bond_atom_full[m][ii] != atom->tag[i]) {
					  tmpflag = -1;
					  jj = bond_count[n];
					  ii = bond_count[m];
					}
				  }
	  	  	    }
	  	  	    if (tmpflag == 1) {
	  	  	      delx = x[m][0] - x[i][0];
	  	  	      dely = x[m][1] - x[i][1];
	  	  	      delz = x[m][2] - x[i][2];
	  	  	      ri = sqrt(delx * delx + dely * dely + delz * delz);
	  	  	      delx2 = x[n][0] - x[i][0];
	  	  	      dely2 = x[n][1] - x[i][1];
	  	  	      delz2 = x[n][2] - x[i][2];
	  	  	      ri2 = sqrt(delx2 * delx2 + dely2 * dely2 + delz2 * delz2);
	  	  	      cosmn = (delx * delx2 + dely * dely2 + delz * delz2) / ri / ri2;    // cos(theta_mn), theta_mn is the angle between i->m and i->n
	  	  	      for (ii = 0; ii < bond_count[i]; ii++) {    // make sure n is one of the two closest i bond partners to m 
	  	  	  	    if (ii != mm && ii != nn) {
	  	  	  	      jj = atom->map(bond_atom_full[i][ii]);
	  	  	  	      delx2 = x[jj][0] - x[i][0];
	  	  	  	      dely2 = x[jj][1] - x[i][1];
	  	  	  	      delz2 = x[jj][2] - x[i][2];
	  	  	  	      ri2 = sqrt(delx2 * delx2 + dely2 * dely2 + delz2 * delz2);
	  	  	  	      cosmjj = (delx * delx2 + dely * dely2 + delz * delz2) / ri / ri2;    // cos(theta_mjj), theta_mjj is the angle between i->m and i->jj
	  	  	  	      if (cosmjj > cosmn) {    // if theta_mjj < theta_mn
	  	  	  	    	tmpflag --;
	  	  	  	    	// break;
	  	  	  	      }
	  	  	  	    }
	  	  	      }
	  	  	    } 
	  	  	    
	  	  	    if (tmpflag >= 0) {
	  	  	  	  xtmp = x[m][0] + x[n][0] - 2 * x[i][0];
		  	      ytmp = x[m][1] + x[n][1] - 2 * x[i][1];
		  	      ztmp = x[m][2] + x[n][2] - 2 * x[i][2];
		  	      rmnsq = xtmp * xtmp + ytmp * ytmp + ztmp * ztmp;
		  	      xtmp = x[i][0] + sqrt(3) * r_new * xtmp / sqrt(rmnsq);    // the position of expexted new j
		  	      ytmp = x[i][1] + sqrt(3) * r_new * ytmp / sqrt(rmnsq);
		  	      ztmp = x[i][2] + sqrt(3) * r_new * ztmp / sqrt(rmnsq);
				  rminsq = BIG;
				  j = 0;
		  	      for (jj = 0; jj < nall; jj++) {
		  	  	    if ((mask[jj] & groupbit) && (skele_status[jj] == 0 || skele_status[jj] == 3) && jj != i && jj != m && jj != n) {
		  	  	      rsq = (x[jj][0] - xtmp) * (x[jj][0] - xtmp) + (x[jj][1] - ytmp) * (x[jj][1] - ytmp) + (x[jj][2] - ztmp) * (x[jj][2] - ztmp);
		  	  	      // fprintf(stderr, "In fix_skele_create_break create test05 when timestep = %d at proc %d: atom->tag[i] = %d, bond_count[i] = %d, atom->tag[m] = %d, bond_count[m] = %d, atom->tag[n] = %d, bond_count[n] = %d, xtmp = %f, ytmp = %f, ztmp = %f, rsq = %f.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], atom->tag[m], bond_count[m], atom->tag[n], bond_count[n],xtmp,ytmp,ztmp,rsq);
				  	  if (rsq < d_newsq && rsq < rminsq) {
		  	  	    	rminsq = rsq;
						j = jj;
		  	  	      }
		  	  	    }
		  	      }
				  if (j != 0) {
				    rmjsq = (x[m][0] - x[j][0]) * (x[m][0] - x[j][0]) + (x[m][1] - x[j][1]) * (x[m][1] - x[j][1]) + (x[m][2] - x[j][2]) * (x[m][2] - x[j][2]);
		  	  	    rnjsq = (x[n][0] - x[j][0]) * (x[n][0] - x[j][0]) + (x[n][1] - x[j][1]) * (x[n][1] - x[j][1]) + (x[n][2] - x[j][2]) * (x[n][2] - x[j][2]);
		  	  	    // fprintf(stderr, "In fix_skele_create_break create test06 when timestep = %d at proc %d: atom->tag[i] = %d, bond_count[i] = %d, atom->tag[m] = %d, bond_count[m] = %d, atom->tag[n] = %d, bond_count[n] = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], atom->tag[m], bond_count[m], atom->tag[n], bond_count[n]);
				    if (rmjsq < r_new_cutsq && rnjsq < r_new_cutsq) {    // make sure the distances m j, and n j are within the cutoff
		  	  	      
				      fprintf(stderr, "In fix_skele_create_break create03 when timestep = %d at proc %d: the bond between atom %d, %d and %d formed, bond_count[i] = %d, bond_count[m] = %d, bond_count[n] = %d, bond_count[j] = %d.\n", update->ntimestep, comm->me, atom->tag[m], atom->tag[n], atom->tag[j], bond_count[i], bond_count[m], bond_count[n], bond_count[j]);
				      pot_m = m;
				      pot_n = n;
				      pot_j = j;
					  pot_count = bond_count[m] + bond_count[n];
					  ncreate_tmp++;
				      
				      nn = bond_count[i];    // jump out of the whole loop and end the creation part
				      mm = bond_count[i]-1;
				      i = nlocal;
		  	  	    }
				  }
	  	  	    }
	  	  	  }
	  	    }
	  	  }
	    }
	  }
    }
  }
  MPI_Allreduce(&ncreate_tmp,&createcount_tmp,1,MPI_INT,MPI_SUM,world);
  
  int *pot_count_procs;
  int nprocs = comm->nprocs;
  int iproc = -1;
  int count_max = 0;
  memory->create(pot_count_procs,nprocs,"fix_skele_create_break:pot_count_procs");
  if (createcount_tmp) {
	MPI_Allgather(&pot_count, 1, MPI_INT, pot_count_procs, 1, MPI_INT, world);
	for (i = 0; i < nprocs; i++) {
	  if (pot_count_procs[i] > count_max) {
		count_max = pot_count_procs[i];
		iproc = i;
	  }
	}
	if (comm->me == iproc) {
	  m = pot_m;
	  n = pot_n;
	  j = pot_j;
	  if (m < nlocal) {    // forming new bond for m
	    bond_atom_full[m][bond_count[m]] = atom->tag[j];
	    bond_count[m]++;
	    bond_atom[m][num_bond[m]] = atom->tag[j];
	    bond_type[m][num_bond[m]] = break_bond_type;
	    bond_length[m][num_bond[m]] = r_new;
	    num_bond[m]++;
	  }
	  else {
	    bond_add_full[m][bond_add_full_count[m]] = atom->tag[j];
	    bond_add_full_count[m]++;
	    bond_add[m][bond_add_count[m]] = atom->tag[j];
	    bond_add_count[m]++;
	  }
	  ncreate++;
	  
	  if (n < nlocal) {    // forming new bond for n
	    bond_atom_full[n][bond_count[n]] = atom->tag[j];
	    bond_count[n]++;
	    bond_atom[n][num_bond[n]] = atom->tag[j];
	    bond_type[n][num_bond[n]] = break_bond_type;
	    bond_length[n][num_bond[n]] = r_new;
	    num_bond[n]++;
	  }
	  else {
	    bond_add_full[n][bond_add_full_count[n]] = atom->tag[j];
	    bond_add_full_count[n]++;
	    bond_add[n][bond_add_count[n]] = atom->tag[j];
	    bond_add_count[n]++;
	  }
	  ncreate++;
	  
	  if (j < nlocal) {    // forming new bond for j
	    bond_atom_full[j][bond_count[j]] = atom->tag[m];
	    bond_count[j]++;
	    bond_atom_full[j][bond_count[j]] = atom->tag[n];
	    bond_count[j]++;
	    if (skele_status[j] == 0) bond_initial_count[j] = 6;
	    skele_status[j] = 2;
	  }
	  else {
	    bond_add_full[j][bond_add_full_count[j]] = atom->tag[m];
	    bond_add_full_count[j]++;
	    bond_add_full[j][bond_add_full_count[j]] = atom->tag[n];
	    bond_add_full_count[j]++;
	  }
	  // skele_status[j] = 2;
    }
  }
  
  
  
  
  if (force->newton_bond)   comm->reverse_comm_fix(this);
  
  //reevaluate skele_status
  for (i = 0; i < nlocal; i++){
	if (mask[i] & groupbit){
	  if (skele_status[i] == 2) {
		//fprintf(stderr, "In fix_skele_create_break reevaluate when timestep = %d at proc %d: the bond_count of local atom %d is %d, bond_initial_count = %d, skele_status = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], bond_initial_count[i], skele_status[i]);
		if (bond_count[i] >= bond_initial_count[i]) {
		  skele_status[i] = 1;
		  fprintf(stderr, "In fix_skele_create_break reevaluate 03 when timestep = %d at proc %d: the bond_count of local atom %d is %d, bond_initial_count = %d, skele_status = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], bond_initial_count[i], skele_status[i]);
		}
	  }
	  else if (skele_status[i] == 0) {
		//fprintf(stderr, "In fix_skele_create_break reevaluate when timestep = %d at proc %d: the bond_count of local atom %d is %d, bond_initial_count = %d, skele_status = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], bond_initial_count[i], skele_status[i]);
		if (bond_count[i] > 0) {
		  if (bond_count[i] >= 6) {
			skele_status[i] = 1;
			bond_initial_count[i] = bond_count[i];
		  }
		  else {
			skele_status[i] = 2;
			bond_initial_count[i] = 6;
		  }
		  fprintf(stderr, "In fix_skele_create_break reevaluate 04 when timestep = %d at proc %d: the bond_count of local atom %d is %d, bond_initial_count = %d, skele_status = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], bond_initial_count[i], skele_status[i]);
		}
	  }
	  else if (skele_status[i] == 3) {
		//fprintf(stderr, "In fix_skele_create_break reevaluate when timestep = %d at proc %d: the bond_count of local atom %d is %d, bond_initial_count = %d, skele_status = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], bond_initial_count[i], skele_status[i]);
		if (bond_count[i] > 0) {
		  if (bond_count[i] >= bond_initial_count[i]) {
			skele_status[i] = 1;
		  }
		  else {
			skele_status[i] = 2;
		  }
		  fprintf(stderr, "In fix_skele_create_break reevaluate 05 when timestep = %d at proc %d: the bond_count of local atom %d is %d, bond_initial_count = %d, skele_status = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], bond_initial_count[i], skele_status[i]);
		}
	  }
	}
  }
  
  // tally stats
  MPI_Allreduce(&nbreak,&breakcount,1,MPI_INT,MPI_SUM,world);
  breakcounttotal += breakcount;
  atom->nbonds -= breakcount;
  
  MPI_Allreduce(&ncreate,&createcount,1,MPI_INT,MPI_SUM,world);
  createcounttotal += createcount;
  atom->nbonds += createcount;

  // trigger reneighboring if any bonds were formed
  // this insures neigh lists will immediately reflect the topology changes
  // done if any bonds created

  if (breakcount || createcount) next_reneighbor = update->ntimestep;
  
  
  if (update->ntimestep == 15000) {
    for (i = 0; i < nall; i++){
	  if (mask[i] & groupbit){
	    if(i < nlocal) fprintf(stderr, "In fix_skele_create_break post_integrate() innitially 06 when timestep = %d at proc %d: local bond_count[%d] = %d, bond_add_full_count[%d] = %d, bond_initial_count[%d] = %d, skele_status[%d] = %d, atom->tag[%d] = %d\n", update->ntimestep, comm->me, i, bond_count[i], i, bond_add_full_count[i], i, bond_initial_count[i], i, skele_status[i], i, atom->tag[i]);
	    else fprintf(stderr, "In fix_skele_create_break post_integrate() innitially 06 when timestep = %d at proc %d: ghost bond_count[%d] = %d, bond_add_full_count[%d] = %d, bond_initial_count[%d] = %d, skele_status[%d] = %d, atom->tag[%d] = %d\n", update->ntimestep, comm->me, i, bond_count[i], i, bond_add_full_count[i], i, bond_initial_count[i], i, skele_status[i], i, atom->tag[i]);
	  }
    }
  }
  // DEBUG
  //print_bb();
  memory->destroy(pot_count_procs);
}

/* ----------------------------------------------------------------------
   insure all atoms 2 hops away from owned atoms are in ghost list
   this allows dihedral 1-2-3-4 to be properly created
     and special list of 1 to be properly updated
   if I own atom 1, but not 2,3,4, and bond 3-4 is added
     then 2,3 will be ghosts and 3 will store 4 as its finalpartner
------------------------------------------------------------------------- */

void FixSkeleCreateBreak::check_ghosts()
{
  int i,j,n;
  tagint *slist;

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int nlocal = atom->nlocal;

  int flag = 0;
  for (i = 0; i < nlocal; i++) {
    slist = special[i];
    n = nspecial[i][1];
    for (j = 0; j < n; j++)
      if (atom->map(slist[j]) < 0) flag = 1;
  }

  int flagall;
  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
  if (flagall) 
    error->all(FLERR,"Fix SkeleBreak needs ghost atoms from further away");
  lastcheck = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

void FixSkeleCreateBreak::post_integrate_respa(int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_integrate();
}

/* ---------------------------------------------------------------------- */

int FixSkeleCreateBreak::pack_forward_comm(int n, int *list, double *buf,
                                     int pbc_flag, int *pbc)
{
  int i,j,k,m;

  m = 0;

  for (i = 0; i < n; i++) {
    j = list[i];
	buf[m++] = ubuf(bond_count[j]).d;
	buf[m++] = ubuf(bond_sub_full_count[j]).d;
	buf[m++] = ubuf(bond_add_full_count[j]).d;
	buf[m++] = ubuf(bond_sub_count[j]).d;
	buf[m++] = ubuf(bond_add_count[j]).d;
	buf[m++] = ubuf(bond_initial_count[j]).d;
	buf[m++] = ubuf(skele_status[j]).d;
/* 	for (k = 0; k < bond_count[j]; k++) {
	  buf[m++] = bond_sub_full[j][k];
      buf[m++] = bond_add_full[j][k];
	  buf[m++] = bond_atom_full[j][k];
    } */
	for (k = 0; k < maxbond_per_atom; k++) {
	  buf[m++] = ubuf(bond_sub_full[j][k]).d;
      buf[m++] = ubuf(bond_add_full[j][k]).d;
	  buf[m++] = ubuf(bond_sub[j][k]).d;
      buf[m++] = ubuf(bond_add[j][k]).d;
	  buf[m++] = ubuf(bond_atom_full[j][k]).d;
      //if (bond_count[j] != 0) fprintf(stderr, "In fix_skele_create_break pack_forward_comm() when timestep = %d at proc %d: the bond_count of ghost atom %d is %d, bond_atom_full[j][%d] = %d, maxbond_per_atom = %d.\n", update->ntimestep, comm->me, atom->tag[j], bond_count[j], k, bond_atom_full[j][k], maxbond_per_atom);
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixSkeleCreateBreak::unpack_forward_comm(int n, int first, double *buf)
{
  int i,j,k,m,last;
  int nlocal = atom->nlocal;

  m = 0;
  last = first + n;

  for (i = first; i < last; i++){
	bond_count[i] = (int) ubuf(buf[m++]).i;
	bond_sub_full_count[i] = (int) ubuf(buf[m++]).i;
	bond_add_full_count[i] = (int) ubuf(buf[m++]).i;
	bond_sub_count[i] = (int) ubuf(buf[m++]).i;
	bond_add_count[i] = (int) ubuf(buf[m++]).i;
	bond_initial_count[i] = (int) ubuf(buf[m++]).i;
	skele_status[i] = (int) ubuf(buf[m++]).i;
/* 	for (k = 0; k < bond_count[i]; k++) {
      bond_sub_full[i][k] = (int) ubuf(buf[m++]).i;
      bond_add_full[i][k] = (int) ubuf(buf[m++]).i;
      bond_atom_full[i][k] = (int) ubuf(buf[m++]).i;
    } */
	for (k = 0; k < maxbond_per_atom; k++) {
      bond_sub_full[i][k] = (int) ubuf(buf[m++]).i;
      bond_add_full[i][k] = (int) ubuf(buf[m++]).i;
      bond_sub[i][k] = (int) ubuf(buf[m++]).i;
      bond_add[i][k] = (int) ubuf(buf[m++]).i;
      bond_atom_full[i][k] = (int) ubuf(buf[m++]).i;
	  //if (bond_count[i] != 0) fprintf(stderr, "In fix_skele_create_break unpack_forward_comm() when timestep = %d at proc %d: the bond_count of ghost atom %d is %d, bond_atom_full[i][%d] = %d, maxbond_per_atom = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], k, bond_atom_full[i][k], maxbond_per_atom);
    }
/*   	if (bond_count[i] != 0) {
  	  if (i < nlocal) fprintf(stderr, "In fix_skele_create_break unpack_forward_comm() when timestep = %d at proc %d: the bond_count of local atom %d is %d, bond_atom_full[i][0] = %d, bond_atom_full[i][bond_count[i]-1] = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], bond_atom_full[i][0], bond_atom_full[i][bond_count[i]-1]);
  	  else fprintf(stderr, "In fix_skele_create_break unpack_forward_comm() when timestep = %d at proc %d: the bond_count of ghost atom %d is %d, bond_atom_full[i][0] = %d, bond_atom_full[i][bond_count[i]-1] = %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i], bond_atom_full[i][0], bond_atom_full[i][bond_count[i]-1]);
  	} */
  }

}

/* ---------------------------------------------------------------------- */

int FixSkeleCreateBreak::pack_reverse_comm(int n, int first, double *buf)
{
  int i,k,m,last;

  m = 0;
  last = first + n;

  for (i = first; i < last; i++){
	buf[m++] = ubuf(bond_sub_full_count[i]).d;
	buf[m++] = ubuf(bond_add_full_count[i]).d;
	buf[m++] = ubuf(bond_sub_count[i]).d;
	buf[m++] = ubuf(bond_add_count[i]).d;
/* 	for (k = 0; k < bond_sub_full_count[i]; k++) {
	  buf[m++] = ubuf(bond_sub_full[i][k]).d;
	}
	for (k = 0; k < bond_add_full_count[i]; k++) {
	  buf[m++] = ubuf(bond_add_full[i][k]).d;
	} */
	for (k = 0; k < maxbond_per_atom; k++) {
	  buf[m++] = ubuf(bond_sub_full[i][k]).d;
	  buf[m++] = ubuf(bond_sub[i][k]).d;
	}
	for (k = 0; k < maxbond_per_atom; k++) {
	  buf[m++] = ubuf(bond_add_full[i][k]).d;
	  buf[m++] = ubuf(bond_add[i][k]).d;
	}
	//buf[m++] = ubuf(skele_status[i]).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixSkeleCreateBreak::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,k,m,ii,jj,bond_sub_full_count_tmp,bond_add_full_count_tmp,bond_sub_count_tmp,bond_add_count_tmp,bond_full_tmp,bond_tmp;
  int nlocal = atom->nlocal;
  
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  double **bond_length = atom->bond_length;
  
  m = 0;

  for (i = 0; i < n; i++) {
    j = list[i];
    bond_sub_full_count_tmp = (int) ubuf(buf[m++]).i;
	bond_add_full_count_tmp = (int) ubuf(buf[m++]).i;
    bond_sub_count_tmp = (int) ubuf(buf[m++]).i;
	bond_add_count_tmp = (int) ubuf(buf[m++]).i;
/* 	for (k = 0; k < bond_sub_full_count_tmp; k++) {
	  bond_full_tmp = (int) ubuf(buf[m++]).i;
	  for (jj = 0; jj < bond_count[j]; jj++) {
		if (bond_atom_full[j][jj] == bond_full_tmp) {
		  for (ii = jj; ii < bond_count[j]-1; ii++) {
		    bond_atom_full[j][ii] = bond_atom_full[j][ii+1];
		  }
		  bond_atom_full[j][bond_count[j]-1] = 0;
		  bond_count[j]--;
		  break;
		}
	  } 
	} */
	
	// if the atom receives the ghost info is the local atom
	if (j < nlocal) {
	  // receive sub info from ghost atoms
	  for (k = 0; k < maxbond_per_atom; k++) {
		// remove sub atoms from bond_atom_full list
	    bond_full_tmp = (int) ubuf(buf[m++]).i;
	    if (k < bond_sub_full_count_tmp) {
	      for (jj = 0; jj < bond_count[j]; jj++) {
	  	    if (bond_atom_full[j][jj] == bond_full_tmp) {
	  	      for (ii = jj; ii < bond_count[j]-1; ii++) {
	  	        bond_atom_full[j][ii] = bond_atom_full[j][ii+1];
	  	      }
	  	      bond_atom_full[j][bond_count[j]-1] = 0;
	  	      bond_count[j]--;
	  	      break;
	  	    }
	      }
	    }
		// remove sub atoms from bond_atom list
		bond_tmp = (int) ubuf(buf[m++]).i;
		if (k < bond_sub_count_tmp) {
	      for (jj = 0; jj < num_bond[j]; jj++) {
	  	    if (bond_atom[j][jj] == bond_tmp) {
	  	      for (ii = jj; ii < num_bond[j]-1; ii++) {
	  	        bond_atom[j][ii] = bond_atom[j][ii+1];
				bond_type[j][ii] = bond_type[j][ii+1];
				bond_length[j][ii] = bond_length[j][ii+1];
	  	      }
	  	      bond_atom[j][num_bond[j]-1] = 0;
	  	      num_bond[j]--;
	  	      break;
	  	    }
	      }
	    }
	  }
	  bond_sub_full_count[j] = 0;
	  bond_sub_count[j] = 0;
	  // receive add info from ghost atoms
	  for (k = 0; k < maxbond_per_atom; k++) {
		// append add atoms into bond_atom_full list
	    bond_full_tmp = (int) ubuf(buf[m++]).i;
	    if (k < bond_add_full_count_tmp) {
	      bond_atom_full[j][bond_count[j]] = bond_full_tmp;
	      bond_count[j]++;
	      //fprintf(stderr, "In fix_skele_create_break unpack_reverse_comm() when timestep = %d at proc %d: local bond_count[j] = %d, bond_add_full_count[j] = %d, skele_status[j] = %d, bond_atom_full[j][bond_count[j]-1] = %d, atom->tag[j] = %d\n", update->ntimestep, comm->me, bond_count[j], bond_add_full_count_tmp, skele_status[j], bond_atom_full[j][bond_count[j]-1], atom->tag[j]);
	    }
		// append add atoms into bond_atom list
	    bond_tmp = (int) ubuf(buf[m++]).i;
	    if (k < bond_add_count_tmp) {
	      bond_atom[j][num_bond[j]] = bond_tmp;
		  bond_type[j][num_bond[j]] = break_bond_type;
		  bond_length[j][num_bond[j]] = r_new;
	      num_bond[j]++;
	      //fprintf(stderr, "In fix_skele_create_break unpack_reverse_comm() when timestep = %d at proc %d: local bond_count[j] = %d, bond_add_full_count[j] = %d, skele_status[j] = %d, bond_atom_full[j][bond_count[j]-1] = %d, atom->tag[j] = %d\n", update->ntimestep, comm->me, bond_count[j], bond_add_full_count_tmp, skele_status[j], bond_atom_full[j][bond_count[j]-1], atom->tag[j]);
	    }
	  }
	  bond_add_full_count[j] = 0;
	  bond_add_count[j] = 0;
	}
	// if the atom receives the ghost info is the ghost atom
	else {
	  // receive sub info from ghost atoms
	  for (k = 0; k < maxbond_per_atom; k++) {
	    // append sub info into bond_sub_full list
		bond_full_tmp = (int) ubuf(buf[m++]).i;
	    if (k < bond_sub_full_count_tmp) {
	      bond_sub_full[j][bond_sub_full_count[j]] = bond_full_tmp;
		  bond_sub_full_count[j]++;
	    }
	    // append sub info into bond_sub list
		bond_tmp = (int) ubuf(buf[m++]).i;
	    if (k < bond_sub_count_tmp) {
	      bond_sub[j][bond_sub_count[j]] = bond_tmp;
		  bond_sub_count[j]++;
	    }
	  }
	  // receive add info from ghost atoms
	  for (k = 0; k < maxbond_per_atom; k++) {
	    // append add info into bond_add_full list
		bond_full_tmp = (int) ubuf(buf[m++]).i;
	    if (k < bond_add_full_count_tmp) {
	      bond_add_full[j][bond_add_full_count[j]] = bond_full_tmp;
	      bond_add_full_count[j]++;
	    }
	    // append add info into bond_add list
		bond_tmp = (int) ubuf(buf[m++]).i;
	    if (k < bond_add_count_tmp) {
	      bond_add[j][bond_add_count[j]] = bond_tmp;
	      bond_add_count[j]++;
	    }
	  }
	}
	// bond_sub_full_count[j] += bond_sub_full_count_tmp;
	//bond_sub_full_count[j] = 0;
/* 	for (k = 0; k < bond_add_full_count_tmp; k++) {
	  bond_atom_full[j][bond_count[j]] = (int) ubuf(buf[m++]).i;
	  bond_count[j]++;
	  fprintf(stderr, "In fix_skele_create_break unpack_reverse_comm() when timestep = %d at proc %d: local bond_count[j] = %d, bond_add_full_count[j] = %d, skele_status[j] = %d, bond_atom_full[j][bond_count[j]-1] = %d, atom->tag[j] = %d\n", update->ntimestep, comm->me, bond_count[j], bond_add_full_count_tmp, skele_status[j], bond_atom_full[j][bond_count[j]-1], atom->tag[j]);
	} */
/* 	for (k = 0; k < maxbond_per_atom; k++) {
	  bond_full_tmp = (int) ubuf(buf[m++]).i;
	  if (k < bond_add_full_count_tmp) {
	    bond_atom_full[j][bond_count[j]] = bond_full_tmp;
	    bond_count[j]++;
	    fprintf(stderr, "In fix_skele_create_break unpack_reverse_comm() when timestep = %d at proc %d: local bond_count[j] = %d, bond_add_full_count[j] = %d, skele_status[j] = %d, bond_atom_full[j][bond_count[j]-1] = %d, atom->tag[j] = %d\n", update->ntimestep, comm->me, bond_count[j], bond_add_full_count_tmp, skele_status[j], bond_atom_full[j][bond_count[j]-1], atom->tag[j]);
	  }
	} */
	//bond_add_full_count[j] = 0;
	//bond_sub_full_count_tmp = (int) ubuf(buf[m++]).i;
	//if ((bond_sub_full_count_tmp == 2 || bond_sub_full_count_tmp == 3) && (skele_status[j] == 1 || skele_status[j] == 2)) skele_status[j] = bond_sub_full_count_tmp;
  }
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixSkeleCreateBreak::grow_arrays(int nmax)
{
  memory->grow(bond_count,nmax,"skele/create/break:bond_count");
  memory->grow(bond_sub_full_count,nmax,"skele/create/break:bond_sub_full_count");
  memory->grow(bond_add_full_count,nmax,"skele/create/break:bond_add_full_count");
  memory->grow(bond_sub_count,nmax,"skele/create/break:bond_sub_count");
  memory->grow(bond_add_count,nmax,"skele/create/break:bond_add_count");
  memory->grow(bond_initial_count,nmax,"skele/create/break:bond_initial_count");
  memory->grow(skele_status,nmax,"skele/create/break:skele_status");
  memory->grow(bond_sub_full,nmax,maxbond_per_atom,"skele/create/break:bond_sub_full");
  memory->grow(bond_add_full,nmax,maxbond_per_atom,"skele/create/break:bond_add_full");
  memory->grow(bond_sub,nmax,maxbond_per_atom,"skele/create/break:bond_sub");
  memory->grow(bond_add,nmax,maxbond_per_atom,"skele/create/break:bond_add");
  memory->grow(bond_atom_full,nmax,maxbond_per_atom,"skele/create/break:bond_atom_full");
}

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays
------------------------------------------------------------------------- */

void FixSkeleCreateBreak::copy_arrays(int i, int j, int delflag)
{
  int k;
  
  bond_count[j] = bond_count[i];
  bond_sub_full_count[j] = bond_sub_full_count[i];
  bond_add_full_count[j] = bond_add_full_count[i];
  bond_sub_count[j] = bond_sub_count[i];
  bond_add_count[j] = bond_add_count[i];
  bond_initial_count[j] = bond_initial_count[i];
  skele_status[j] = skele_status[i];
/*   for (k = 0; k < bond_sub_full_count[j]; k++) {
    bond_sub_full[j][k] = bond_sub_full[i][k];
  }
  for (k = 0; k < bond_add_full_count[j]; k++) {
    bond_add_full[j][k] = bond_add_full[i][k];
  }
  for (k = 0; k < bond_count[j]; k++) {
    bond_atom_full[j][k] = bond_atom_full[i][k];
  } */
  for (k = 0; k < maxbond_per_atom; k++) {
    bond_sub_full[j][k] = bond_sub_full[i][k];
  }
  for (k = 0; k < maxbond_per_atom; k++) {
    bond_add_full[j][k] = bond_add_full[i][k];
  }
  for (k = 0; k < maxbond_per_atom; k++) {
	bond_sub[j][k] = bond_sub[i][k];
  }
  for (k = 0; k < maxbond_per_atom; k++) {
	bond_add[j][k] = bond_add[i][k];
  }
  for (k = 0; k < maxbond_per_atom; k++) {
    bond_atom_full[j][k] = bond_atom_full[i][k];
  }
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixSkeleCreateBreak::pack_exchange(int i, double *buf)
{
  int m, k;
  
  m = 0;
  buf[m++] = bond_count[i];
  buf[m++] = bond_sub_full_count[i];
  buf[m++] = bond_add_full_count[i];
  buf[m++] = bond_sub_count[i];
  buf[m++] = bond_add_count[i];
  buf[m++] = bond_initial_count[i];
  buf[m++] = skele_status[i];
/*   for (k = 0; k < bond_sub_full_count[i]; k++) {
    buf[m++] = bond_sub_full[i][k];
  }
  for (k = 0; k < bond_add_full_count[i]; k++) {
    buf[m++] = bond_add_full[i][k];
  }
  for (k = 0; k < bond_count[i]; k++) {
    buf[m++] = bond_atom_full[i][k];
  } */
  for (k = 0; k < maxbond_per_atom; k++) {
    buf[m++] = bond_sub_full[i][k];
  }
  for (k = 0; k < maxbond_per_atom; k++) {
    buf[m++] = bond_add_full[i][k];
  }
  for (k = 0; k < maxbond_per_atom; k++) {
    buf[m++] = bond_sub[i][k];
  }
  for (k = 0; k < maxbond_per_atom; k++) {
    buf[m++] = bond_add[i][k];
  }
  for (k = 0; k < maxbond_per_atom; k++) {
    buf[m++] = bond_atom_full[i][k];
  }
  return m;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based arrays from exchange with another proc
------------------------------------------------------------------------- */

int FixSkeleCreateBreak::unpack_exchange(int nlocal, double *buf)
{
  int m, k;
  
  m = 0;
  bond_count[nlocal] = static_cast<int> (buf[m++]);
  bond_sub_full_count[nlocal] = static_cast<int> (buf[m++]);
  bond_add_full_count[nlocal] = static_cast<int> (buf[m++]);
  bond_sub_count[nlocal] = static_cast<int> (buf[m++]);
  bond_add_count[nlocal] = static_cast<int> (buf[m++]);
  bond_initial_count[nlocal] = static_cast<int> (buf[m++]);
  skele_status[nlocal] = static_cast<int> (buf[m++]);
/*   for (k = 0; k < bond_sub_full_count[nlocal]; k++) {
    bond_sub_full[nlocal][k] = static_cast<int> (buf[m++]);
  }
  for (k = 0; k < bond_add_full_count[nlocal]; k++) {
    bond_add_full[nlocal][k] = static_cast<int> (buf[m++]);
  }
  for (k = 0; k < bond_count[nlocal]; k++) {
    bond_atom_full[nlocal][k] = static_cast<int> (buf[m++]);
  } */
  for (k = 0; k < maxbond_per_atom; k++) {
    bond_sub_full[nlocal][k] = static_cast<int> (buf[m++]);
  }
  for (k = 0; k < maxbond_per_atom; k++) {
    bond_add_full[nlocal][k] = static_cast<int> (buf[m++]);
  }
  for (k = 0; k < maxbond_per_atom; k++) {
    bond_sub[nlocal][k] = static_cast<int> (buf[m++]);
  }
  for (k = 0; k < maxbond_per_atom; k++) {
    bond_add[nlocal][k] = static_cast<int> (buf[m++]);
  }
  for (k = 0; k < maxbond_per_atom; k++) {
    bond_atom_full[nlocal][k] = static_cast<int> (buf[m++]);
  }
  return m;
}

/* ---------------------------------------------------------------------- */
//Her ist mir nocn unklar wofuer dieseFunktion gut ist
double FixSkeleCreateBreak::compute_vector(int n)
{
  printf("In compute_vector\n");
  if (n == 0) return (double) breakcount;
  return (double) breakcounttotal;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixSkeleCreateBreak::memory_usage()
{
  double bytes = 7 * nnmax * sizeof(int);
  bytes += 5 * nnmax * maxbond_per_atom * sizeof(int);
  return bytes;
}

/* ---------------------------------------------------------------------- */

void FixSkeleCreateBreak::print_bb()
{
  for (int i = 0; i < atom->nlocal; i++) {
    printf("TAG " TAGINT_FORMAT ": %d nbonds: ",atom->tag[i],atom->num_bond[i]);
    for (int j = 0; j < atom->num_bond[i]; j++) {
      printf(" " TAGINT_FORMAT,atom->bond_atom[i][j]);
    }
    printf("\n");
  }
}

/* ---------------------------------------------------------------------- */

void FixSkeleCreateBreak::print_copy(const char *str, tagint m, 
                              int n1, int n2, int n3, int *v)
{
  printf("%s " TAGINT_FORMAT ": %d %d %d nspecial: ",str,m,n1,n2,n3);
  for (int j = 0; j < n3; j++) printf(" %d",v[j]);
  printf("\n");
}
