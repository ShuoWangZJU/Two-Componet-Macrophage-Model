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
#include "fix_skele_break_group.h"
#include "update.h"
#include "respa.h"
#include "atom.h"
#include "atom_vec.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
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

FixSkeleBreakGroup::FixSkeleBreakGroup(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg != 6 && narg != 8) error->all(FLERR,"Illegal fix SkeleBreakGroup command");

  MPI_Comm_rank(world,&me);

  start_time = force->inumeric(FLERR,arg[3]);
  if (start_time < update->ntimestep) error->all(FLERR,"Illegal fix SkeleBreakGroup command");
  
  break_bond_type = force->inumeric(FLERR,arg[4]);
  
  int igroup_break;
  igroup_break = group->find(arg[5]);
  if (igroup_break == -1) error->all(FLERR,"Could not find break group ID for fix SkeleBreakGroup command");
  groupbit_break = group->bitmask[igroup_break];
  
  int igroup_excluded;
  group_excluded_flag = 0;
  if (narg == 8 && strcmp(arg[6],"excluded_group") == 0) {
	group_excluded_flag = 1;
	igroup_excluded = group->find(arg[7]);
	if (igroup_excluded == -1) error->all(FLERR,"Could not find exclude group ID for fix SkeleBreakGroup command");
	groupbit_excluded = group->bitmask[igroup_excluded];
  }
  
  force_reneighbor = 1;
  next_reneighbor = -1;
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;
  create_attribute = 1;
  
  // error check

  if (atom->molecular != 1)
    error->all(FLERR,"Cannot use fix SkeleBreakGroup with non-molecular systems");

  // perform initial allocation of atom-based arrays
  // register with Atom class
  // bond_count values will be initialized in setup()
  
  bond_count = NULL;
  // angle_count = NULL;
  // dihedral_count = NULL;
  // improper_count = NULL;
  bond_change_count = NULL;
  // angle_change_count = NULL;
  // dihedral_change_count = NULL;
  // improper_change_count = NULL;
  bond_initial_count = NULL;
  // angle_initial_count = NULL;
  // dihedral_initial_count = NULL;
  // improper_initial_count = NULL;
  skele_status = NULL;
  grow_arrays(atom->nmax);
  atom->add_callback(0);
  countflag = 0;

  // set comm sizes needed by this fix
  // forward is big due to comm of broken bonds and 1-2 neighbors

//   comm_forward = MAX(2,2+atom->maxspecial); old
  // comm_forward = 9;
  // comm_reverse = 3;
  comm_forward = 4;
  comm_reverse = 1;
  
  // allocate arrays local to this fix

  // nmax = 0;
 
  maxbreak = 0;

  // copy = special list for one atom
  // size = ms^2 + ms is sufficient
  // b/c in rebuild_special() neighs of all 1-2s are added,
  //   then a dedup(), then neighs of all 1-3s are added, then final dedup()
  // this means intermediate size cannot exceed ms^2 + ms

  // zero out stats

  break_bond_count = 0;
  break_bond_count_total = 0;
  break_angle_count = 0;
  break_angle_count_total = 0;
  break_dihedral_count = 0;
  break_dihedral_count_total = 0;
  break_improper_count = 0;
  break_improper_count_total = 0;
}

/* ---------------------------------------------------------------------- */

FixSkeleBreakGroup::~FixSkeleBreakGroup()
{
  // unregister callbacks to this fix from Atom class

  atom->delete_callback(id,0);

  // delete locally stored arrays
  
  memory->destroy(bond_count); 
  // memory->destroy(angle_count);
  // memory->destroy(dihedral_count);
  // memory->destroy(improper_count);
  memory->destroy(bond_change_count);
  // memory->destroy(angle_change_count);
  // memory->destroy(dihedral_change_count);
  // memory->destroy(improper_change_count);
  memory->destroy(bond_initial_count);
  // memory->destroy(angle_initial_count);
  // memory->destroy(dihedral_initial_count);
  // memory->destroy(improper_initial_count);
  memory->destroy(skele_status);
}

/* ---------------------------------------------------------------------- */

int FixSkeleBreakGroup::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixSkeleBreakGroup::init()
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

void FixSkeleBreakGroup::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixSkeleBreakGroup::setup(int vflag)
{
  int i,j,m,a1,a2,a3,d1,d2,d3,d4,i1,i2,i3,i4;

  // compute initial bond_count if this is first run
  // can't do this earlier, in constructor or init, b/c need ghost info

  if (countflag) return;
  countflag = 1;

  // count bonds stored with each bond I own
  // if newton bond is not set, just increment count on atom I
  // if newton bond is set, also increment count on atom J even if ghost
  // bond_count is long enough to tally ghost atom counts

  int *num_bond = atom->num_bond;
  int *num_angle = atom->num_angle;
  int *num_dihedral = atom->num_dihedral;
  int *num_improper = atom->num_improper;
  int **bond_type = atom->bond_type;
  int **angle_type = atom->angle_type;
  int **dihedral_type = atom->dihedral_type;
  int **improper_type = atom->improper_type;
  tagint **bond_atom = atom->bond_atom;
  tagint **angle_atom1 = atom->angle_atom1;
  tagint **angle_atom2 = atom->angle_atom2;
  tagint **angle_atom3 = atom->angle_atom3;
  tagint **dihedral_atom1 = atom->dihedral_atom1;
  tagint **dihedral_atom2 = atom->dihedral_atom2;
  tagint **dihedral_atom3 = atom->dihedral_atom3;
  tagint **dihedral_atom4 = atom->dihedral_atom4;
  tagint **improper_atom1 = atom->improper_atom1;
  tagint **improper_atom2 = atom->improper_atom2;
  tagint **improper_atom3 = atom->improper_atom3;
  tagint **improper_atom4 = atom->improper_atom4;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  int newton_bond = force->newton_bond;

  for (i = 0; i < nall; i++){
	  bond_count[i] = 0;
	  // angle_count[i] = 0;
	  // dihedral_count[i] = 0;
	  // improper_count[i] = 0;
	  bond_change_count[i] = 0;
	  // angle_change_count[i] = 0;
	  // dihedral_change_count[i] = 0;
	  // improper_change_count[i] = 0;
	  bond_initial_count[i] = 0;
	  // angle_initial_count[i] = 0;
	  // dihedral_initial_count[i] = 0;
	  // improper_initial_count[i] = 0;
	  skele_status[i] = 0;
  }
  
  // count bonds
  for (i = 0; i < nlocal; i++)
    for (j = 0; j < num_bond[i]; j++) {
      if (bond_type[i][j] == break_bond_type) {
        bond_change_count[i]++;
        if (newton_bond) {
          m = atom->map(bond_atom[i][j]);
          if (m < 0) 
            error->one(FLERR,"Fix SkeleBreak needs ghost atoms "
                       "from further away");
          bond_change_count[m]++;
        }
      }
    }
	
/*   int *mask = atom->mask;
  for (i = 0; i < nall; i++){
	if (mask[i] & groupbit){
	  if(i < nlocal) fprintf(stderr, "In fix_skele_break setup() 01 when timestep = %d at proc %d: local bond_count[%d] = %d, bond_change_count[%d] = %d, bond_initial_count[%d] = %d, skele_status[%d] = %d, protru_flag[%d] = %d, atom->tag[%d] = %d\n", update->ntimestep, comm->me, i, bond_count[i], i, bond_change_count[i], i, bond_initial_count[i], i, skele_status[i], i, protru_flag[i], i, atom->tag[i]);
	  else fprintf(stderr, "In fix_skele_break setup() 01 when timestep = %d at proc %d: ghost bond_count[%d] = %d, bond_change_count[%d] = %d, bond_initial_count[%d] = %d, skele_status[%d] = %d, protru_flag[%d] = %d, atom->tag[%d] = %d\n", update->ntimestep, comm->me, i, bond_count[i], i, bond_change_count[i], i, bond_initial_count[i], i, skele_status[i], i, protru_flag[i], i, atom->tag[i]);
	}
  } */
  
  // commflag = 1;
  // if (newton_bond) comm->reverse_comm_fix(this,1);
  if (newton_bond) comm->reverse_comm_fix(this);
  for (i = 0; i < nall; i++){
	  bond_count[i] += bond_change_count[i];
	  bond_change_count[i] = 0;
	  bond_initial_count[i] = bond_count[i];
  }
  // comm->forward_comm_fix(this,4);
  comm->forward_comm_fix(this);
  
  // count angles, dihedrals and impropers
  for (i = 0; i < nlocal; i++) {
	if (bond_count[i] > 0) {
	  for (j = 0; j < num_angle[i]; j++) {
		// angle_change_count[i]++;
		if (newton_bond) {
          a1 = atom->map(angle_atom1[i][j]);
          a2 = atom->map(angle_atom2[i][j]);
          a3 = atom->map(angle_atom3[i][j]);
          if (a1 < 0 || a2 < 0 || a3 < 0) 
            error->one(FLERR,"Fix SkeleBreak needs ghost atoms "
                       "from further away");
          // if (a1 != i) angle_change_count[a1]++;
          // if (a2 != i) angle_change_count[a2]++;
          // if (a3 != i) angle_change_count[a3]++;
        }
	  }
	  for (j = 0; j < num_dihedral[i]; j++) {
		// dihedral_change_count[i]++;
		if (newton_bond) {
          d1 = atom->map(dihedral_atom1[i][j]);
          d2 = atom->map(dihedral_atom2[i][j]);
          d3 = atom->map(dihedral_atom3[i][j]);
          d4 = atom->map(dihedral_atom4[i][j]);
          if (d1 < 0 || d2 < 0 || d3 < 0 || d4 < 0) 
            error->one(FLERR,"Fix SkeleBreak needs ghost atoms "
                       "from further away");
          // if (d1 != i) dihedral_change_count[d1]++;
          // if (d2 != i) dihedral_change_count[d2]++;
          // if (d3 != i) dihedral_change_count[d3]++;
          // if (d4 != i) dihedral_change_count[d4]++;
        }
	  }
	  for (j = 0; j < num_improper[i]; j++) {
		// improper_change_count[i]++;
		if (newton_bond) {
          i1 = atom->map(improper_atom1[i][j]);
          i2 = atom->map(improper_atom2[i][j]);
          i3 = atom->map(improper_atom3[i][j]);
          i4 = atom->map(improper_atom4[i][j]);
          if (i1 < 0 || i2 < 0 || i3 < 0 || i4 < 0) 
            error->one(FLERR,"Fix SkeleBreak needs ghost atoms "
                       "from further away");
          // if (i1 != i) improper_change_count[i1]++;
          // if (i2 != i) improper_change_count[i2]++;
          // if (i3 != i) improper_change_count[i3]++;
          // if (i4 != i) improper_change_count[i4]++;
        }
	  }
	}
  }
  
/*   commflag = 2;
  if (newton_bond) comm->reverse_comm_fix(this,3);
  for (i = 0; i < nall; i++){
	  angle_count[i] += angle_change_count[i];
	  angle_change_count[i] = 0;
	  angle_initial_count[i] = angle_count[i];
	  dihedral_count[i] += dihedral_change_count[i];
	  dihedral_change_count[i] = 0;
	  dihedral_initial_count[i] = dihedral_count[i];
	  improper_count[i] += improper_change_count[i];
	  improper_change_count[i] = 0;
	  improper_initial_count[i] = improper_count[i];
  }
  comm->forward_comm_fix(this,9); */
  
/*   for (i = 0; i < nall; i++){
	if (mask[i] & groupbit){
	  if(i < nlocal) fprintf(stderr, "In fix_skele_break setup() 02 when timestep = %d at proc %d: local bond_count[%d] = %d, bond_change_count[%d] = %d, bond_initial_count[%d] = %d, skele_status[%d] = %d, protru_flag[%d] = %d, atom->tag[%d] = %d\n", update->ntimestep, comm->me, i, bond_count[i], i, bond_change_count[i], i, bond_initial_count[i], i, skele_status[i], i, protru_flag[i], i, atom->tag[i]);
	  else fprintf(stderr, "In fix_skele_break setup() 02 when timestep = %d at proc %d: ghost bond_count[%d] = %d, bond_change_count[%d] = %d, bond_initial_count[%d] = %d, skele_status[%d] = %d, protru_flag[%d] = %d, atom->tag[%d] = %d\n", update->ntimestep, comm->me, i, bond_count[i], i, bond_change_count[i], i, bond_initial_count[i], i, skele_status[i], i, protru_flag[i], i, atom->tag[i]);
	}
  } */
}

/* ---------------------------------------------------------------------- */

void FixSkeleBreakGroup::post_integrate()
{
  int i,j,k,m,n,ii,jj,inum,jnum,itype,jtype,n1,n2,n3,possible,i1,i2,type_bond, jb, jc, flag;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq,rminsq;
  int *ilist,*jlist,*numneigh,**firstneigh;
  tagint *slist;

  // if (update->ntimestep % nevery) return;
  
  if (update->ntimestep != start_time) return;

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
  // commflag = 1;
  // comm->forward_comm_fix(this,4);
  // commflag = 2;
  // comm->forward_comm_fix(this,9);
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
  
  int *num_bond = atom->num_bond;
  int *num_angle = atom->num_angle;
  int *num_dihedral = atom->num_dihedral;
  int *num_improper = atom->num_improper;
  tagint **bond_atom = atom->bond_atom;
  tagint **angle_atom1 = atom->angle_atom1;
  tagint **angle_atom2 = atom->angle_atom2;
  tagint **angle_atom3 = atom->angle_atom3;
  tagint **dihedral_atom1 = atom->dihedral_atom1;
  tagint **dihedral_atom2 = atom->dihedral_atom2;
  tagint **dihedral_atom3 = atom->dihedral_atom3;
  tagint **dihedral_atom4 = atom->dihedral_atom4;
  tagint **improper_atom1 = atom->improper_atom1;
  tagint **improper_atom2 = atom->improper_atom2;
  tagint **improper_atom3 = atom->improper_atom3;
  tagint **improper_atom4 = atom->improper_atom4;
  int *type = atom->type;
  neighbor->build_one(list,1);
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  int **bond_type = atom->bond_type;
  int **angle_type = atom->angle_type;
  int **dihedral_type = atom->dihedral_type;
  int **improper_type = atom->improper_type;

  //Break:
  
  // if (force->newton_bond)   comm->reverse_comm_fix(this);
  
/*   for (i = 0; i < nall; i++){
	if (mask[i] & groupbit){
	  if(i < nlocal) fprintf(stderr, "In fix_skele_break post_integrate() innitially 01 when timestep = %d at proc %d: local bond_count[%d] = %d, bond_change_count[%d] = %d, bond_initial_count[%d] = %d, skele_status[%d] = %d, protru_flag[%d] = %d, atom->tag[%d] = %d\n", update->ntimestep, comm->me, i, bond_count[i], i, bond_change_count[i], i, bond_initial_count[i], i, skele_status[i], i, protru_flag[i], i, atom->tag[i]);
	  else fprintf(stderr, "In fix_skele_break post_integrate() innitially 01 when timestep = %d at proc %d: ghost bond_count[%d] = %d, bond_change_count[%d] = %d, bond_initial_count[%d] = %d, skele_status[%d] = %d, protru_flag[%d] = %d, atom->tag[%d] = %d\n", update->ntimestep, comm->me, i, bond_count[i], i, bond_change_count[i], i, bond_initial_count[i], i, skele_status[i], i, protru_flag[i], i, atom->tag[i]);
	}
  } */
  
  nbreak_bond = 0;
  nbreak_angle = 0;
  nbreak_dihedral = 0;
  nbreak_improper = 0;
  // fprintf(stderr, "In fix_skele_break when timestep = %d at proc %d: all 01\n", update->ntimestep, comm->me);
  // find the Band-3 beads that connect to the cytoskeleton beads, and calculate the distance from target
  for (i = 0; i < nlocal; i++){
	if (mask[i] & groupbit_break){
	  if (group_excluded_flag && (mask[i] & groupbit_excluded)) continue;
	  // if (!num_bond[i]) continue;
	  //flag = 0;
	  //for (m = 0; m < num_bond[i]; m++) {
	  //  if (bond_type[i][m] == band3_bond_type) {
	  //    xtmp = x[atom->map(bond_atom[i][m])][0];
	  //    ytmp = x[atom->map(bond_atom[i][m])][1];
	  //    ztmp = x[atom->map(bond_atom[i][m])][2];
		//  flag = 1;
		//  //fprintf(stderr, "In fix_skele_break post_integrate() when timestep = %d at proc %d: the connecting band3 of atom %d is %d, x = %f, y = %f, z = %f\n", update->ntimestep, comm->me, atom->tag[i], bond_atom[i][m], xtmp, ytmp, ztmp);
	  //  }
	  //}
	  //if (!flag) continue;
	  
	  // skele_status[i] = 0;
	  // rminsq = BIG;
	  // calculate distance
	  //for (j = 0; j < nall; j++){
	  //  if (mask[j] & groupbit_target){
	  //    delx = x[j][0] - xtmp;
	  //    dely = x[j][1] - ytmp;
	  //    delz = x[j][2] - ztmp;
	  //    rsq = delx*delx + dely*dely + delz*delz;
		//  //fprintf(stderr, "In fix_skele_break post_integrate() when timestep = %d at proc %d: the distance between connecting band3 and RBC atom %d is %d, x = %f, y = %f, z = %f\n", update->ntimestep, comm->me, atom->tag[j], bond_atom[i][m], rsq, x[j][0], x[j][1], x[j][2]);
		//  if (rsq < rminsq){
	  //    	rminsq = rsq;
	  //    }
	  //  }
	  //}
	  
	  //fprintf(stderr, "In fix_skele_break post_integrate() when timestep = %d at proc %d: the rminsq of atom %d is %f\n", update->ntimestep, comm->me, atom->tag[i], rminsq);
	  // assign status for to-be-broken cytoskeleton beads
	  // if (rminsq <= r_breaksq){
	  	skele_status[i] = 3;
	  // }
	}
  }
  // fprintf(stderr, "In fix_skele_break when timestep = %d at proc %d: all 02\n", update->ntimestep, comm->me);
  // commflag = 1;
  // comm->forward_comm_fix(this,4);
  comm->forward_comm_fix(this);
  
/*   for (i = 0; i < nall; i++){
	if (mask[i] & groupbit){
	  if(i < nlocal) fprintf(stderr, "In fix_skele_break post_integrate() innitially 02 when timestep = %d at proc %d: local bond_count[%d] = %d, bond_change_count[%d] = %d, bond_initial_count[%d] = %d, skele_status[%d] = %d, protru_flag[%d] = %d, atom->tag[%d] = %d\n", update->ntimestep, comm->me, i, bond_count[i], i, bond_change_count[i], i, bond_initial_count[i], i, skele_status[i], i, protru_flag[i], i, atom->tag[i]);
	  else fprintf(stderr, "In fix_skele_break post_integrate() innitially 02 when timestep = %d at proc %d: ghost bond_count[%d] = %d, bond_change_count[%d] = %d, bond_initial_count[%d] = %d, skele_status[%d] = %d, protru_flag[%d] = %d, atom->tag[%d] = %d\n", update->ntimestep, comm->me, i, bond_count[i], i, bond_change_count[i], i, bond_initial_count[i], i, skele_status[i], i, protru_flag[i], i, atom->tag[i]);
	}
  } */
  
  // fprintf(stderr, "In Fix fix_skele_break at proc %d when time = %d: initial\n", comm->me, update->ntimestep);
  // fprintf(stderr, "In fix_skele_break when timestep = %d at proc %d: all 03\n", update->ntimestep, comm->me);
  // break bonds, angles, dihedrals and impropers
  for (i = 0; i < nlocal; i++){
	if (mask[i] & groupbit){
//	  if (skele_status[i] == 1) {
//	  	  // if (bond_count[i] != 0) error->all(FLERR,"Bonds haven't been all broke in fix skele_break command");
//		  fprintf(stderr, "In fix_skele_break post_integrate() when timestep = %d at proc %d: the bond_count01 of atom %d is %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i]);
//	  }
	  // break cytoskeleton bonds
	  for (m = 0; m < num_bond[i]; m++) {
	  	if (bond_type[i][m] == break_bond_type) {
		  j = atom->map(bond_atom[i][m]);
	  	  if (skele_status[i] == 3 || skele_status[j] == 3) {
	  	  	fprintf(stderr, "In fix_skele_break post_integrate() when timestep = %d at proc %d: the bond between atom %d and %d broke, the bond type is %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_atom[i][m], bond_type[i][m]);
	  	  	for (k = m; k < num_bond[i]-1; k++) {
	  	  	  bond_atom[i][k] = bond_atom[i][k+1];
	  	  	  bond_type[i][k] = bond_type[i][k+1];
	  	  	}
	  	  	num_bond[i]--;
	  	  	m--;
	  	  	bond_change_count[i]--;
			if (force->newton_bond) bond_change_count[j]--;
	  	  	nbreak_bond++;
	  	  }
	  	}
	  }
	  
	  // break all angles
	  for (m = 0; m < num_angle[i]; m++) {
	  	if(skele_status[atom->map(angle_atom1[i][m])] == 3 || skele_status[atom->map(angle_atom2[i][m])] == 3 || skele_status[atom->map(angle_atom3[i][m])] == 3) {
	  	  fprintf(stderr, "In fix_skele_break post_integrate() when timestep = %d at proc %d: the angle between atom %d, %d and %d broke, the angle type is %d.\n", update->ntimestep, comm->me, angle_atom1[i][m], angle_atom2[i][m], angle_atom3[i][m], angle_type[i][m]);
	  	  for (k = m; k < num_angle[i]-1; k++) {
	  	    angle_atom1[i][k] = angle_atom1[i][k+1];
	  	    angle_atom2[i][k] = angle_atom2[i][k+1];
	  	    angle_atom3[i][k] = angle_atom3[i][k+1];
	  	    angle_type[i][k] = angle_type[i][k+1];
	  	  }
	  	  num_angle[i]--;
	  	  m--;
/* 	  	  angle_change_count[i]--;
		  if (force->newton_bond) {
			if (atom->map(angle_atom1[i][m]) != i) angle_change_count[atom->map(angle_atom1[i][m])]--;
			if (atom->map(angle_atom2[i][m]) != i) angle_change_count[atom->map(angle_atom2[i][m])]--;
			if (atom->map(angle_atom3[i][m]) != i) angle_change_count[atom->map(angle_atom3[i][m])]--;
	  	  } */
		  nbreak_angle++; 
	  	}
	  }
	  // break all dihedrals
	  for (m = 0; m < num_dihedral[i]; m++) {
	  	if(skele_status[atom->map(dihedral_atom1[i][m])] == 3 || skele_status[atom->map(dihedral_atom2[i][m])] == 3 || skele_status[atom->map(dihedral_atom3[i][m])] == 3 || skele_status[atom->map(dihedral_atom4[i][m])] == 3) {
	  	  fprintf(stderr, "In fix_skele_break post_integrate() when timestep = %d at proc %d: the dihedral between atom %d, %d, %d and %d broke, the dihedral type is %d.\n", update->ntimestep, comm->me, dihedral_atom1[i][m], dihedral_atom2[i][m], dihedral_atom3[i][m], dihedral_atom4[i][m], dihedral_type[i][m]);
	  	  for (k = m; k < num_dihedral[i]-1; k++) {
	  	    dihedral_atom1[i][k] = dihedral_atom1[i][k+1];
	  	    dihedral_atom2[i][k] = dihedral_atom2[i][k+1];
	  	    dihedral_atom3[i][k] = dihedral_atom3[i][k+1];
	  	    dihedral_atom4[i][k] = dihedral_atom4[i][k+1];
	  	    dihedral_type[i][k] = dihedral_type[i][k+1];
	  	  }
	  	  num_dihedral[i]--;
	  	  m--;
/* 	  	  dihedral_change_count[i]--;
		  if (force->newton_bond) {
			if (atom->map(dihedral_atom1[i][m]) != i) dihedral_change_count[atom->map(dihedral_atom1[i][m])]--;
			if (atom->map(dihedral_atom2[i][m]) != i) dihedral_change_count[atom->map(dihedral_atom2[i][m])]--;
			if (atom->map(dihedral_atom3[i][m]) != i) dihedral_change_count[atom->map(dihedral_atom3[i][m])]--;
			if (atom->map(dihedral_atom4[i][m]) != i) dihedral_change_count[atom->map(dihedral_atom4[i][m])]--;
	  	  } */
	  	  nbreak_dihedral++; 
	  	}
	  }
	  // break all impropers
	  for (m = 0; m < num_improper[i]; m++) {
	  	if(skele_status[atom->map(improper_atom1[i][m])] == 3 || skele_status[atom->map(improper_atom2[i][m])] == 3 || skele_status[atom->map(improper_atom3[i][m])] == 3 || skele_status[atom->map(improper_atom4[i][m])] == 3) {
	  	  fprintf(stderr, "In fix_skele_break post_integrate() when timestep = %d at proc %d: the improper between atom %d, %d, %d and %d broke, the improper type is %d.\n", update->ntimestep, comm->me, improper_atom1[i][m], improper_atom2[i][m], improper_atom3[i][m], improper_atom4[i][m], improper_type[i][m]);
	  	  for (k = m; k < num_improper[i]-1; k++) {
	  	    improper_atom1[i][k] = improper_atom1[i][k+1];
	  	    improper_atom2[i][k] = improper_atom2[i][k+1];
	  	    improper_atom3[i][k] = improper_atom3[i][k+1];
	  	    improper_atom4[i][k] = improper_atom4[i][k+1];
	  	    improper_type[i][k] = improper_type[i][k+1];
	  	  }
	  	  num_improper[i]--;
	  	  m--;
/* 	  	  improper_change_count[i]--;
		  if (force->newton_bond) {
			if (atom->map(improper_atom1[i][m]) != i) improper_change_count[atom->map(improper_atom1[i][m])]--;
			if (atom->map(improper_atom2[i][m]) != i) improper_change_count[atom->map(improper_atom2[i][m])]--;
			if (atom->map(improper_atom3[i][m]) != i) improper_change_count[atom->map(improper_atom3[i][m])]--;
			if (atom->map(improper_atom4[i][m]) != i) improper_change_count[atom->map(improper_atom4[i][m])]--;
	  	  } */
	  	  nbreak_improper++; 
	  	}
	  }
	  
/* 	  if (skele_status[i] == 3) {
	  	  // if (bond_count[i] != 0) error->all(FLERR,"Bonds haven't been all broke in fix skele_break command");
		  fprintf(stderr, "In fix_skele_break post_integrate() when timestep = %d at proc %d: the bond_count02 of atom %d is %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_count[i]);
	  } */
	}
  }
  // fprintf(stderr, "In fix_skele_break when timestep = %d at proc %d: all 04\n", update->ntimestep, comm->me);
/*   for (i = 0; i < nall; i++){
	if (mask[i] & groupbit){
	  if(i < nlocal) fprintf(stderr, "In fix_skele_break post_integrate() innitially 03 when timestep = %d at proc %d: local bond_count[%d] = %d, bond_change_count[%d] = %d, bond_initial_count[%d] = %d, skele_status[%d] = %d, protru_flag[%d] = %d, atom->tag[%d] = %d\n", update->ntimestep, comm->me, i, bond_count[i], i, bond_change_count[i], i, bond_initial_count[i], i, skele_status[i], i, protru_flag[i], i, atom->tag[i]);
	  else fprintf(stderr, "In fix_skele_break post_integrate() innitially 03 when timestep = %d at proc %d: ghost bond_count[%d] = %d, bond_change_count[%d] = %d, bond_initial_count[%d] = %d, skele_status[%d] = %d, protru_flag[%d] = %d, atom->tag[%d] = %d\n", update->ntimestep, comm->me, i, bond_count[i], i, bond_change_count[i], i, bond_initial_count[i], i, skele_status[i], i, protru_flag[i], i, atom->tag[i]);
	}
  } */
  // commflag = 1;
  // if (force->newton_bond)   comm->reverse_comm_fix(this, 1);
  // commflag = 2;
  // if (force->newton_bond)   comm->reverse_comm_fix(this,3);
  if (force->newton_bond)   comm->reverse_comm_fix(this);

/*   for (i = 0; i < nall; i++){
	if (mask[i] & groupbit){
	  if(i < nlocal) fprintf(stderr, "In fix_skele_break post_integrate() innitially 04 when timestep = %d at proc %d: local bond_count[%d] = %d, bond_change_count[%d] = %d, bond_initial_count[%d] = %d, skele_status[%d] = %d, protru_flag[%d] = %d, atom->tag[%d] = %d\n", update->ntimestep, comm->me, i, bond_count[i], i, bond_change_count[i], i, bond_initial_count[i], i, skele_status[i], i, protru_flag[i], i, atom->tag[i]);
	  else fprintf(stderr, "In fix_skele_break post_integrate() innitially 04 when timestep = %d at proc %d: ghost bond_count[%d] = %d, bond_change_count[%d] = %d, bond_initial_count[%d] = %d, skele_status[%d] = %d, protru_flag[%d] = %d, atom->tag[%d] = %d\n", update->ntimestep, comm->me, i, bond_count[i], i, bond_change_count[i], i, bond_initial_count[i], i, skele_status[i], i, protru_flag[i], i, atom->tag[i]);
	}
  } */
  // fprintf(stderr, "In fix_skele_break when timestep = %d at proc %d: nall = %d\n", update->ntimestep, comm->me, nall);
  // recount bonds, angles, dihedrals and impropers
  for (i = 0; i < nall; i++){
	bond_count[i] += bond_change_count[i];
	bond_change_count[i] = 0;
	// angle_count[i] += angle_change_count[i];
	// angle_change_count[i] = 0;
	// dihedral_count[i] += dihedral_change_count[i];
	// dihedral_change_count[i] = 0;
	// improper_count[i] += improper_change_count[i];
	// improper_change_count[i] = 0;
	//if (group_added_flag) {
	//  if (mask[i] & groupbit){
	//	mask[i] &= inversebit_broken;
	//	mask[i] &= inversebit_partly;
	//	mask[i] &= inversebit_intact;
	//	if (bond_count[i] == 0) mask[i] |= groupbit_broken;
	//	else if (bond_count[i] > 0 && bond_count[i] < bond_initial_count[i]) mask[i] |= groupbit_partly;
	//	else if (bond_count[i] == bond_initial_count[i]) mask[i] |= groupbit_intact;
	//  }
	//}
  }
  
/*   for (i = 0; i < nall; i++){
	if (mask[i] & groupbit){
	  if(i < nlocal) fprintf(stderr, "In fix_skele_break post_integrate() innitially 05 when timestep = %d at proc %d: local bond_count[%d] = %d, bond_change_count[%d] = %d, bond_initial_count[%d] = %d, skele_status[%d] = %d, protru_flag[%d] = %d, atom->tag[%d] = %d\n", update->ntimestep, comm->me, i, bond_count[i], i, bond_change_count[i], i, bond_initial_count[i], i, skele_status[i], i, protru_flag[i], i, atom->tag[i]);
	  else fprintf(stderr, "In fix_skele_break post_integrate() innitially 05 when timestep = %d at proc %d: ghost bond_count[%d] = %d, bond_change_count[%d] = %d, bond_initial_count[%d] = %d, skele_status[%d] = %d, protru_flag[%d] = %d, atom->tag[%d] = %d\n", update->ntimestep, comm->me, i, bond_count[i], i, bond_change_count[i], i, bond_initial_count[i], i, skele_status[i], i, protru_flag[i], i, atom->tag[i]);
	}
  } */
  // fprintf(stderr, "In fix_skele_break when timestep = %d at proc %d: all 06\n", update->ntimestep, comm->me);
  // commflag = 1;
  // comm->forward_comm_fix(this,4);
  // commflag = 2;
  // comm->forward_comm_fix(this,9);
  comm->forward_comm_fix(this);
  
  // tally stats
  MPI_Allreduce(&nbreak_bond,&break_bond_count,1,MPI_INT,MPI_SUM,world);
  break_bond_count_total += break_bond_count;
  atom->nbonds -= break_bond_count;
  
  MPI_Allreduce(&nbreak_angle,&break_angle_count,1,MPI_INT,MPI_SUM,world);
  break_angle_count_total += break_angle_count;
  atom->nangles -= break_angle_count;
  
  MPI_Allreduce(&nbreak_dihedral,&break_dihedral_count,1,MPI_INT,MPI_SUM,world);
  break_dihedral_count_total += break_dihedral_count;
  atom->ndihedrals -= break_dihedral_count;
  
  MPI_Allreduce(&nbreak_improper,&break_improper_count,1,MPI_INT,MPI_SUM,world);
  break_improper_count_total += break_improper_count;
  atom->nimpropers -= break_improper_count;
  // fprintf(stderr, "In fix_skele_break when timestep = %d at proc %d: all 07\n", update->ntimestep, comm->me);
  // trigger reneighboring if any bonds were formed
  // this insures neigh lists will immediately reflect the topology changes
  // done if any bonds created

  if (break_bond_count || break_angle_count || break_dihedral_count || break_improper_count) next_reneighbor = update->ntimestep;

/*   for (i = 0; i < nall; i++){
	if (mask[i] & groupbit){
	  if(i < nlocal) fprintf(stderr, "In fix_skele_break post_integrate() innitially 06 when timestep = %d at proc %d: local bond_count[%d] = %d, bond_change_count[%d] = %d, bond_initial_count[%d] = %d, skele_status[%d] = %d, protru_flag[%d] = %d, atom->tag[%d] = %d\n", update->ntimestep, comm->me, i, bond_count[i], i, bond_change_count[i], i, bond_initial_count[i], i, skele_status[i], i, protru_flag[i], i, atom->tag[i]);
	  else fprintf(stderr, "In fix_skele_break post_integrate() innitially 06 when timestep = %d at proc %d: ghost bond_count[%d] = %d, bond_change_count[%d] = %d, bond_initial_count[%d] = %d, skele_status[%d] = %d, protru_flag[%d] = %d, atom->tag[%d] = %d\n", update->ntimestep, comm->me, i, bond_count[i], i, bond_change_count[i], i, bond_initial_count[i], i, skele_status[i], i, protru_flag[i], i, atom->tag[i]);
	}
  } */
  // DEBUG
  //print_bb();
}

/* ----------------------------------------------------------------------
   insure all atoms 2 hops away from owned atoms are in ghost list
   this allows dihedral 1-2-3-4 to be properly created
     and special list of 1 to be properly updated
   if I own atom 1, but not 2,3,4, and bond 3-4 is added
     then 2,3 will be ghosts and 3 will store 4 as its finalpartner
------------------------------------------------------------------------- */

void FixSkeleBreakGroup::check_ghosts()
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

void FixSkeleBreakGroup::post_integrate_respa(int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_integrate();
}

/* ---------------------------------------------------------------------- */

int FixSkeleBreakGroup::pack_forward_comm(int n, int *list, double *buf,
                                     int pbc_flag, int *pbc)
{
  int i,j,k,m;

  m = 0;
  
/*   if (commflag == 1) {
    for (i = 0; i < n; i++) {
      j = list[i];
	  buf[m++] = ubuf(bond_count[j]).d;
	  buf[m++] = ubuf(bond_change_count[j]).d;
	  buf[m++] = ubuf(bond_initial_count[j]).d;
	  buf[m++] = ubuf(skele_status[j]).d;
    }
    return m;
  }
  else {
    for (i = 0; i < n; i++) {
      j = list[i];
	  buf[m++] = ubuf(angle_count[j]).d;
	  buf[m++] = ubuf(angle_change_count[j]).d;
	  buf[m++] = ubuf(angle_initial_count[j]).d;
	  buf[m++] = ubuf(dihedral_count[j]).d;
	  buf[m++] = ubuf(dihedral_change_count[j]).d;
	  buf[m++] = ubuf(dihedral_initial_count[j]).d;
	  buf[m++] = ubuf(improper_count[j]).d;
	  buf[m++] = ubuf(improper_change_count[j]).d;
	  buf[m++] = ubuf(improper_initial_count[j]).d;
    }
    return m;
  } */
  
  for (i = 0; i < n; i++) {
    j = list[i];
	buf[m++] = ubuf(bond_count[j]).d;
	buf[m++] = ubuf(bond_change_count[j]).d;
	buf[m++] = ubuf(bond_initial_count[j]).d;
	buf[m++] = ubuf(skele_status[j]).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixSkeleBreakGroup::unpack_forward_comm(int n, int first, double *buf)
{
  int i,j,m,last;

  m = 0;
  last = first + n;
  
/*   if (commflag == 1) {
    for (i = first; i < last; i++){
	  bond_count[i] = (int) ubuf(buf[m++]).i;
	  bond_change_count[i] = (int) ubuf(buf[m++]).i;
	  bond_initial_count[i] = (int) ubuf(buf[m++]).i;
	  skele_status[i] = (int) ubuf(buf[m++]).i;
    }
  }
  else {
	for (i = first; i < last; i++){
	  angle_count[i] = (int) ubuf(buf[m++]).i;
	  angle_change_count[i] = (int) ubuf(buf[m++]).i;
	  angle_initial_count[i] = (int) ubuf(buf[m++]).i;
	  dihedral_count[i] = (int) ubuf(buf[m++]).i;
	  dihedral_change_count[i] = (int) ubuf(buf[m++]).i;
	  dihedral_initial_count[i] = (int) ubuf(buf[m++]).i;
	  improper_count[i] = (int) ubuf(buf[m++]).i;
	  improper_change_count[i] = (int) ubuf(buf[m++]).i;
	  improper_initial_count[i] = (int) ubuf(buf[m++]).i;
    }
  } */
  
  for (i = first; i < last; i++){
	bond_count[i] = (int) ubuf(buf[m++]).i;
	bond_change_count[i] = (int) ubuf(buf[m++]).i;
	bond_initial_count[i] = (int) ubuf(buf[m++]).i;
	skele_status[i] = (int) ubuf(buf[m++]).i;
  }
}

/* ---------------------------------------------------------------------- */

int FixSkeleBreakGroup::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  
/*   if (commflag == 1) {
    for (i = first; i < last; i++) {
      buf[m++] = ubuf(bond_change_count[i]).d;
    }
    return m;
  }
  else {
    for (i = first; i < last; i++) {
      buf[m++] = ubuf(angle_change_count[i]).d;
      buf[m++] = ubuf(dihedral_change_count[i]).d;
      buf[m++] = ubuf(improper_change_count[i]).d;
    }
    return m;
  } */
  
  for (i = first; i < last; i++) {
    buf[m++] = ubuf(bond_change_count[i]).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixSkeleBreakGroup::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m,tmp1,tmp2;

  m = 0;
  
/*   if (commflag == 1) {
    for (i = 0; i < n; i++) {
      j = list[i];
      bond_change_count[j] += (int) ubuf(buf[m++]).i;
    }
  }
  else {
	for (i = 0; i < n; i++) {
      j = list[i];
      angle_change_count[j] += (int) ubuf(buf[m++]).i;
      dihedral_change_count[j] += (int) ubuf(buf[m++]).i;
      improper_change_count[j] += (int) ubuf(buf[m++]).i;
    }
  } */
  
  for (i = 0; i < n; i++) {
    j = list[i];
    bond_change_count[j] += (int) ubuf(buf[m++]).i;
  }
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixSkeleBreakGroup::grow_arrays(int nmax)
{
  // fprintf(stderr, "In fix skele/break at proc %d when time = %d: nmax = %d\n", comm->me, update->ntimestep, nmax);
  memory->grow(bond_count,nmax,"skele/break:bond_count");
  // fprintf(stderr, "In fix skele/break at proc %d when time = %d: after bond_count\n", comm->me, update->ntimestep);
  memory->grow(bond_change_count,nmax,"skele/break:bond_change_count");
  // fprintf(stderr, "In fix skele/break at proc %d when time = %d: after bond_change_count\n", comm->me, update->ntimestep);
  memory->grow(bond_initial_count,nmax,"skele/break:bond_initial_count");
  // fprintf(stderr, "In fix skele/break at proc %d when time = %d: after bond_initial_count\n", comm->me, update->ntimestep);
  // memory->grow(angle_count,nmax,"skele/break:angle_count");
  // fprintf(stderr, "In fix skele/break at proc %d when time = %d: after angle_count\n", comm->me, update->ntimestep);
  // memory->grow(angle_change_count,nmax,"skele/break:angle_change_count");
  // fprintf(stderr, "In fix skele/break at proc %d when time = %d: after angle_change_count\n", comm->me, update->ntimestep);
  // memory->grow(angle_initial_count,nmax,"skele/break:angle_initial_count");
  // fprintf(stderr, "In fix skele/break at proc %d when time = %d: after angle_initial_count\n", comm->me, update->ntimestep);
  // memory->grow(dihedral_count,nmax,"skele/break:dihedral_count");
  // fprintf(stderr, "In fix skele/break at proc %d when time = %d: after dihedral_count\n", comm->me, update->ntimestep);
  // memory->grow(dihedral_change_count,nmax,"skele/break:dihedral_change_count");
  // fprintf(stderr, "In fix skele/break at proc %d when time = %d: after dihedral_change_count\n", comm->me, update->ntimestep);
  // memory->grow(dihedral_initial_count,nmax,"skele/break:dihedral_initial_count");
  // fprintf(stderr, "In fix skele/break at proc %d when time = %d: after dihedral_initial_count\n", comm->me, update->ntimestep);
  // memory->grow(improper_count,nmax,"skele/break:improper_count");
  // fprintf(stderr, "In fix skele/break at proc %d when time = %d: after improper_count\n", comm->me, update->ntimestep);
  // memory->grow(improper_change_count,nmax,"skele/break:improper_change_count");
  // fprintf(stderr, "In fix skele/break at proc %d when time = %d: after improper_change_count\n", comm->me, update->ntimestep);
  // memory->grow(improper_initial_count,nmax,"skele/break:improper_initial_count");
  // fprintf(stderr, "In fix skele/break at proc %d when time = %d: after improper_initial_count\n", comm->me, update->ntimestep);
  memory->grow(skele_status,nmax,"skele/break:skele_status");
  // fprintf(stderr, "In fix skele/break at proc %d when time = %d: after skele_status\n", comm->me, update->ntimestep);
}

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays
------------------------------------------------------------------------- */

void FixSkeleBreakGroup::copy_arrays(int i, int j, int delflag)
{
  bond_count[j] = bond_count[i];
  bond_change_count[j] = bond_change_count[i];
  bond_initial_count[j] = bond_initial_count[i];
  // angle_count[j] = angle_count[i];
  // angle_change_count[j] = angle_change_count[i];
  // angle_initial_count[j] = angle_initial_count[i];
  // dihedral_count[j] = dihedral_count[i];
  // dihedral_change_count[j] = dihedral_change_count[i];
  // dihedral_initial_count[j] = dihedral_initial_count[i];
  // improper_count[j] = improper_count[i];
  // improper_change_count[j] = improper_change_count[i];
  // improper_initial_count[j] = improper_initial_count[i];
  skele_status[j] = skele_status[i];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixSkeleBreakGroup::pack_exchange(int i, double *buf)
{
  int m = 0;
  
  buf[m++] = bond_count[i];
  buf[m++] = bond_change_count[i];
  buf[m++] = bond_initial_count[i];
  // buf[m++] = angle_count[i];
  // buf[m++] = angle_change_count[i];
  // buf[m++] = angle_initial_count[i];
  // buf[m++] = dihedral_count[i];
  // buf[m++] = dihedral_change_count[i];
  // buf[m++] = dihedral_initial_count[i];
  // buf[m++] = improper_count[i];
  // buf[m++] = improper_change_count[i];
  // buf[m++] = improper_initial_count[i];
  buf[m++] = skele_status[i];
  return m;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based arrays from exchange with another proc
------------------------------------------------------------------------- */

int FixSkeleBreakGroup::unpack_exchange(int nlocal, double *buf)
{
  int m = 0;
  
  bond_count[nlocal] = static_cast<int> (buf[m++]);
  bond_change_count[nlocal] = static_cast<int> (buf[m++]);
  bond_initial_count[nlocal] = static_cast<int> (buf[m++]);
  // angle_count[nlocal] = static_cast<int> (buf[m++]);
  // angle_change_count[nlocal] = static_cast<int> (buf[m++]);
  // angle_initial_count[nlocal] = static_cast<int> (buf[m++]);
  // dihedral_count[nlocal] = static_cast<int> (buf[m++]);
  // dihedral_change_count[nlocal] = static_cast<int> (buf[m++]);
  // dihedral_initial_count[nlocal] = static_cast<int> (buf[m++]);
  // improper_count[nlocal] = static_cast<int> (buf[m++]);
  // improper_change_count[nlocal] = static_cast<int> (buf[m++]);
  // improper_initial_count[nlocal] = static_cast<int> (buf[m++]);
  skele_status[nlocal] = static_cast<int> (buf[m++]);
  return m;
}

/* ---------------------------------------------------------------------- */
//Her ist mir nocn unklar wofuer dieseFunktion gut ist
double FixSkeleBreakGroup::compute_vector(int n)
{
  printf("In compute_vector\n");
  if (n == 0) return (double) break_bond_count;
  return (double) break_bond_count_total;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixSkeleBreakGroup::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = 4 * nmax * sizeof(int);
  return bytes;
}

/* ---------------------------------------------------------------------- */

void FixSkeleBreakGroup::print_bb()
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

void FixSkeleBreakGroup::print_copy(const char *str, tagint m, 
                              int n1, int n2, int n3, int *v)
{
  printf("%s " TAGINT_FORMAT ": %d %d %d nspecial: ",str,m,n1,n2,n3);
  for (int j = 0; j < n3; j++) printf(" %d",v[j]);
  printf("\n");
}

/* ----------------------------------------------------------------------
   initialize one atom's storage values, called when atom is created
------------------------------------------------------------------------- */

void FixSkeleBreakGroup::set_arrays(int i)
{
  bond_count[i] = 0;
  // angle_count[i] = 0;
  // dihedral_count[i] = 0;
  // improper_count[i] = 0;
  bond_change_count[i] = 0;
  // angle_change_count[i] = 0;
  // dihedral_change_count[i] = 0;
  // improper_change_count[i] = 0;
  bond_initial_count[i] = 0;
  // angle_initial_count[i] = 0;
  // dihedral_initial_count[i] = 0;
  // improper_initial_count[i] = 0;
  skele_status[i] = 0;
}
