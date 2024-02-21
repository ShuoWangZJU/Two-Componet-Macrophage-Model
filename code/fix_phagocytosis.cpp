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
#include "fix_phagocytosis.h"
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

FixPhagocytosis::FixPhagocytosis(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 11) error->all(FLERR,"Illegal fix phagocytosis command");

  MPI_Comm_rank(world,&me);

  nevery = force->inumeric(FLERR,arg[3]);
  if (nevery <= 0) error->all(FLERR,"Illegal fix phagocytosis command");
  
  break_bond_type = force->inumeric(FLERR,arg[4]);
  
  int igroup_target;
  char gn[50];
  sprintf(gn,arg[5]);
  igroup_target = group->find(arg[5]);
  if (igroup_target == -1) error->all(FLERR,"Could not find molecule group ID for fix protrusive command");
  groupbit_target = group->bitmask[igroup_target];
  
  r_break = force->numeric(FLERR,arg[6]);
  if (r_break < 0.0) error->all(FLERR,"Illegal fix phagocytosis command");
  r_breaksq = r_break*r_break;
  
  int igroup_protru = group->find_or_create(arg[7]);
  groupbit_protru = group->bitmask[igroup_protru];
  inversebit_protru = group->inversemask[igroup_protru];
  
  r_protru_lo = force->numeric(FLERR,arg[8]);
  if (r_protru_lo < r_break) error->all(FLERR,"Illegal fix phagocytosis command");
  r_protru_losq = r_protru_lo*r_protru_lo;
  
  r_protru_hi = force->numeric(FLERR,arg[9]);
  if (r_protru_hi < r_protru_lo) error->all(FLERR,"Illegal fix phagocytosis command");
  r_protru_hisq = r_protru_hi*r_protru_hi;
  
  band3_bond_type = force->inumeric(FLERR,arg[10]);

  force_reneighbor = 1;
  next_reneighbor = -1;
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;

  // error check

  if (atom->molecular != 1)
    error->all(FLERR,"Cannot use fix phagocytosis with non-molecular systems");

  // perform initial allocation of atom-based arrays
  // register with Atom class
  // bondcount values will be initialized in setup()

  phago_flag = NULL;
  bondcount = NULL;
  grow_arrays(atom->nmax);
  atom->add_callback(0);
  countflag = 0;

  // set comm sizes needed by this fix
  // forward is big due to comm of broken bonds and 1-2 neighbors

//   comm_forward = MAX(2,2+atom->maxspecial); old
  comm_forward = 6;
  comm_reverse = 2;

  // allocate arrays local to this fix

  nmax = 0;
 
  maxbreak = 0;

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

FixPhagocytosis::~FixPhagocytosis()
{
  // unregister callbacks to this fix from Atom class

  atom->delete_callback(id,0);

  // delete locally stored arrays

  memory->destroy(phago_flag);
  memory->destroy(bondcount);
}

/* ---------------------------------------------------------------------- */

int FixPhagocytosis::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixPhagocytosis::init()
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

void FixPhagocytosis::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixPhagocytosis::setup(int vflag)
{
  int i,j,m;

  // compute initial bondcount if this is first run
  // can't do this earlier, in constructor or init, b/c need ghost info

  if (countflag) return;
  countflag = 1;

  // count bonds stored with each bond I own
  // if newton bond is not set, just increment count on atom I
  // if newton bond is set, also increment count on atom J even if ghost
  // bondcount is long enough to tally ghost atom counts

  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  int newton_bond = force->newton_bond;

  for (i = 0; i < nall; i++){
	  bondcount[i] = 0;
	  phago_flag[i] = 0;
  }

  for (i = 0; i < nlocal; i++)
    for (j = 0; j < num_bond[i]; j++) {
      if (bond_type[i][j] == break_bond_type) {
        bondcount[i]++;
        if (newton_bond) {
          m = atom->map(bond_atom[i][j]);
          if (m < 0) 
            error->one(FLERR,"Fix phagocytosis needs ghost atoms "
                       "from further away");
          bondcount[m]++;
        }
      }
    }
	
  int *mask = atom->mask;
  for (i = 0; i < nlocal; i++){
	if (mask[i] & groupbit){
	  fprintf(stderr, "In fix_phagocytosis setup() 01 when timestep = %d at proc %d: the phago_flag and bondcount of atom %d is %d %d\n", update->ntimestep, comm->me, atom->tag[i], phago_flag[i], bondcount[i]);
	}
  }
  if (newton_bond) comm->reverse_comm_fix(this);
  
  for (i = 0; i < nlocal; i++){
	if (mask[i] & groupbit){
	  fprintf(stderr, "In fix_phagocytosis setup() 02 when timestep = %d at proc %d: the phago_flag and bondcount of atom %d is %d %d\n", update->ntimestep, comm->me, atom->tag[i], phago_flag[i], bondcount[i]);
	}
  }
}

/* ---------------------------------------------------------------------- */

void FixPhagocytosis::post_integrate()
{
  int i,j,k,m,n,ii,jj,inum,jnum,itype,jtype,n1,n2,n3,possible,i1,i2,type_bond, jb, jc;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq,rminsq;
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

  // forward comm of bondcount, so ghosts have it

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
  
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  int *type = atom->type;
  neighbor->build_one(list,1);
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  int **bond_type = atom->bond_type;

  //Break:
  
  if (force->newton_bond)   comm->reverse_comm_fix(this);
  
//  for (i = 0; i < nlocal; i++){
//	if (mask[i] & groupbit){
//	  fprintf(stderr, "In fix_phagocytosis post_integrate() innitially 01 when timestep = %d at proc %d: the phago_flag and bondcount of atom %d is %d %d\n", update->ntimestep, comm->me, atom->tag[i], phago_flag[i], bondcount[i]);
//	}
//  }
  
  nbreak = 0;
  for (i = 0; i < nlocal; i++){
	mask[i] &= inversebit_protru;
	if (mask[i] & groupbit){
	  if (!num_bond[i]) continue;
	  for (m = 0; m < num_bond[i]; m++) {
	    if (bond_type[i][m] == band3_bond_type) {
	      xtmp = x[atom->map(bond_atom[i][m])][0];
	      ytmp = x[atom->map(bond_atom[i][m])][1];
	      ztmp = x[atom->map(bond_atom[i][m])][2];
		  //fprintf(stderr, "In fix_phagocytosis post_integrate() when timestep = %d at proc %d: the connecting band3 of atom %d is %d, x = %f, y = %f, z = %f\n", update->ntimestep, comm->me, atom->tag[i], bond_atom[i][m], xtmp, ytmp, ztmp);
	    }
	  }
	  
	  // phago_flag[i] = 0;
	  rminsq = BIG;
	  for (j = 0; j < nlocal; j++){
	    if (mask[j] & groupbit_target){
	      delx = x[j][0] - xtmp;
	      dely = x[j][1] - ytmp;
	      delz = x[j][2] - ztmp;
	      rsq = delx*delx + dely*dely + delz*delz;
		  //fprintf(stderr, "In fix_phagocytosis post_integrate() when timestep = %d at proc %d: the distance between connecting band3 and RBC atom %d is %d, x = %f, y = %f, z = %f\n", update->ntimestep, comm->me, atom->tag[j], bond_atom[i][m], rsq, x[j][0], x[j][1], x[j][2]);
		  if (rsq < rminsq){
	      	rminsq = rsq;
	      }
	    }
	  }
	  
	  //fprintf(stderr, "In fix_phagocytosis post_integrate() when timestep = %d at proc %d: the rminsq of atom %d is %f\n", update->ntimestep, comm->me, atom->tag[i], rminsq);
	  
	  if (rminsq <= r_breaksq){
	  	phago_flag[i] = 1;
	  }
	  else if (rminsq > r_protru_losq && rminsq <= r_protru_hisq){
		if (phago_flag[i] == 0) phago_flag[i] = 2;
	  }
	  else {
		if (phago_flag[i] == 2) phago_flag[i] = 0;
	  }
	}
  }
  
  comm->forward_comm_fix(this);
  
//  for (i = 0; i < nlocal; i++){
//	if (mask[i] & groupbit){
//	  fprintf(stderr, "In fix_phagocytosis post_integrate() innitially 02 when timestep = %d at proc %d: the phago_flag and bondcount of atom %d is %d %d\n", update->ntimestep, comm->me, atom->tag[i], phago_flag[i], bondcount[i]);
//	}
//  }
  
  for (i = 0; i < nlocal; i++){
	if (mask[i] & groupbit){
//	  if (phago_flag[i] == 1) {
//	  	  // if (bondcount[i] != 0) error->all(FLERR,"Bonds haven't been all broke in fix phagocytosis command");
//		  fprintf(stderr, "In fix_phagocytosis post_integrate() when timestep = %d at proc %d: the bondcount01 of atom %d is %d.\n", update->ntimestep, comm->me, atom->tag[i], bondcount[i]);
//	  }
	  for (m = 0; m < num_bond[i]; m++) {
	  	if (bond_type[i][m] == break_bond_type) {
	  	  if (phago_flag[i] == 1 || phago_flag[atom->map(bond_atom[i][m])] == 1) {
	  	  	fprintf(stderr, "In fix_phagocytosis post_integrate() when timestep = %d at proc %d: the bond between atom %d and %d broke, the bond type is %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_atom[i][m], bond_type[i][m]);
	  	  	for (k = m; k < num_bond[i]-1; k++) {
	  	  	  bond_atom[i][k] = bond_atom[i][k+1];
	  	  	  bond_type[i][k] = bond_type[i][k+1];
	  	  	}
	  	  	num_bond[i]--;
	  	  	m--;
	  	  	bondcount[i]--;
			if (force->newton_bond) bondcount[atom->map(bond_atom[i][m])]--;
	  	  	nbreak++; 
	  	  }
	  	}
	  }
	  
	  if (phago_flag[i] == 1) {
	  	  // if (bondcount[i] != 0) error->all(FLERR,"Bonds haven't been all broke in fix phagocytosis command");
		  fprintf(stderr, "In fix_phagocytosis post_integrate() when timestep = %d at proc %d: the bondcount02 of atom %d is %d.\n", update->ntimestep, comm->me, atom->tag[i], bondcount[i]);
	  }
	  
	  if(phago_flag[i] == 2){
		mask[i] |= groupbit_protru;
		for (m = 0; m < num_bond[i]; m++) {
		  if (bond_type[i][m] == band3_bond_type) {
			mask[atom->map(bond_atom[i][m])] |= groupbit_protru;
		    //fprintf(stderr, "In fix_phagocytosis post_integrate() when timestep = %d at proc %d: atom %d and %d are added into group to_protru, the map of latter one is %d.\n", update->ntimestep, comm->me, atom->tag[i], bond_atom[i][m], atom->map(bond_atom[i][m]));
		  }
		}
	  }
	}
  }
  
//  for (i = 0; i < nlocal; i++){
//	if (mask[i] & groupbit){
//	  fprintf(stderr, "In fix_phagocytosis post_integrate() innitially 03 when timestep = %d at proc %d: the phago_flag and bondcount of atom %d is %d %d\n", update->ntimestep, comm->me, atom->tag[i], phago_flag[i], bondcount[i]);
//	}
//  }
  
  if (force->newton_bond)   comm->reverse_comm_fix(this);
  
  // tally stats
  MPI_Allreduce(&nbreak,&breakcount,1,MPI_INT,MPI_SUM,world);
  breakcounttotal += breakcount;
  atom->nbonds -= breakcount;

  // trigger reneighboring if any bonds were formed
  // this insures neigh lists will immediately reflect the topology changes
  // done if any bonds created

  if (breakcount) next_reneighbor = update->ntimestep;

//  for (i = 0; i < nlocal; i++){
//	if (mask[i] & groupbit){
//	  fprintf(stderr, "In fix_phagocytosis post_integrate() innitially 04 when timestep = %d at proc %d: the phago_flag and bondcount of atom %d is %d %d\n", update->ntimestep, comm->me, atom->tag[i], phago_flag[i], bondcount[i]);
//	}
//  }
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

void FixPhagocytosis::check_ghosts()
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
    error->all(FLERR,"Fix phagocytosis needs ghost atoms from further away");
  lastcheck = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

void FixPhagocytosis::post_integrate_respa(int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_integrate();
}

/* ---------------------------------------------------------------------- */

int FixPhagocytosis::pack_forward_comm(int n, int *list, double *buf,
                                     int pbc_flag, int *pbc)
{
  int i,j,k,m;

  m = 0;

  for (i = 0; i < n; i++) {
    j = list[i];
	buf[m++] = ubuf(phago_flag[j]).d;
    buf[m++] = ubuf(bondcount[j]).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixPhagocytosis::unpack_forward_comm(int n, int first, double *buf)
{
  int i,j,m,last;

  m = 0;
  last = first + n;

  for (i = first; i < last; i++){
	  phago_flag[i] = (int) ubuf(buf[m++]).i;
	  bondcount[i] = (int) ubuf(buf[m++]).i;
  }

}

/* ---------------------------------------------------------------------- */

int FixPhagocytosis::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;

  for (i = first; i < last; i++){
	  buf[m++] = ubuf(phago_flag[i]).d;
	  buf[m++] = ubuf(bondcount[i]).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixPhagocytosis::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;

  for (i = 0; i < n; i++) {
    j = list[i];
	phago_flag[j] = (int) ubuf(buf[m++]).i;
    bondcount[j] = (int) ubuf(buf[m++]).i;
  }
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixPhagocytosis::grow_arrays(int nmax)
{
  memory->grow(phago_flag,nmax,"phagocytosis:phago_flag");
  memory->grow(bondcount,nmax,"phagocytosis:bondcount");
}

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays
------------------------------------------------------------------------- */

void FixPhagocytosis::copy_arrays(int i, int j, int delflag)
{
  phago_flag[j] = phago_flag[i];
  bondcount[j] = bondcount[i];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixPhagocytosis::pack_exchange(int i, double *buf)
{
  buf[0] = phago_flag[i];
  buf[1] = bondcount[i];
  return 1;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based arrays from exchange with another proc
------------------------------------------------------------------------- */

int FixPhagocytosis::unpack_exchange(int nlocal, double *buf)
{
  phago_flag[nlocal] = static_cast<int> (buf[0]);
  bondcount[nlocal] = static_cast<int> (buf[1]);
  return 1;
}

/* ---------------------------------------------------------------------- */
//Her ist mir nocn unklar wofuer dieseFunktion gut ist
double FixPhagocytosis::compute_vector(int n)
{
  printf("In compute_vector\n");
  if (n == 0) return (double) breakcount;
  return (double) breakcounttotal;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixPhagocytosis::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = 2 * nmax * sizeof(int);
  return bytes;
}

/* ---------------------------------------------------------------------- */

void FixPhagocytosis::print_bb()
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

void FixPhagocytosis::print_copy(const char *str, tagint m, 
                              int n1, int n2, int n3, int *v)
{
  printf("%s " TAGINT_FORMAT ": %d %d %d nspecial: ",str,m,n1,n2,n3);
  for (int j = 0; j < n3; j++) printf(" %d",v[j]);
  printf("\n");
}
