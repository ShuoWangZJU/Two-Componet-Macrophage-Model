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

#include "math.h"
#include "mpi.h"
#include "string.h"
#include "stdlib.h"
#include "fix_bond_AD.h"
#include "update.h"
#include "respa.h"
#include "atom.h"
#include "atom_vec.h"
#include "force.h"
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

// #define MIN(A,B) ((A) < (B)) ? (A) : (B)
// #define MAX(A,B) ((A) > (B)) ? (A) : (B)

/* ---------------------------------------------------------------------- */

FixBondAD::FixBondAD(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg != 16) error->all(FLERR,"Illegal fix bond/adh command");

	me = comm->me;

  nevery = 1;

  force_reneighbor = 1;
  next_reneighbor = -1;
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 1;

  iatomtype = atoi(arg[3]);
  jatomtype = atoi(arg[4]);
  double cutoff = atof(arg[5]);
  ligtype = atoi(arg[6]);
  btype = atoi(arg[7]);
	ks = atof(arg[8]);
	r0 = atof(arg[9]);
	kf0 = atof(arg[10]);
	sig = atof(arg[11]);
	temp = atof(arg[12]);
  Nlig = atoi(arg[13]);
  imaxbond = atoi(arg[14]);
  jmaxbond = atoi(arg[15]);

  if (iatomtype < 1 || iatomtype > atom->ntypes || 
      jatomtype < 1 || jatomtype > atom->ntypes)
    error->all(FLERR,"Invalid atom type in fix bond/adh command");
  if (cutoff < 0.0) error->all(FLERR,"Illegal fix bond/adh command");
  if (btype < 1 || btype > atom->nbondtypes)
    error->all(FLERR,"Invalid bond type in fix bond/adh command");

  cutsq = cutoff*cutoff;

  // error check

  if (atom->molecular == 0)
    error->all(FLERR,"Cannot use fix bond/adh with non-molecular systems");

  // initialize Marsaglia RNG with processor-unique seed

  int seed = 12345;
  random = new RanMars(lmp,seed + me);

  // perform initial allocation of atom-based arrays
  // register with Atom class
  // bondcount values will be initialized in setup()

  bondcount = NULL;
  grow_arrays(atom->nmax);
  atom->add_callback(0);
  countflag = 0;

  // set comm sizes needed by this fix

  comm_forward = 1;
  comm_reverse = 1;

  // allocate arrays local to this fix

  nmax = 0;

  // zero out stats

  createcount = 0;
  createcounttotal = 0;
}

/* ---------------------------------------------------------------------- */

FixBondAD::~FixBondAD()
{
  // unregister callbacks to this fix from Atom class

  atom->delete_callback(id,0);

  delete random;

  // delete locally stored arrays

  memory->sfree(bondcount);
}

/* ---------------------------------------------------------------------- */

int FixBondAD::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBondAD::init()
{
  // check cutoff for iatomtype,jatomtype

  if (force->pair == NULL || cutsq > force->pair->cutsq[iatomtype][jatomtype]) 
    error->all(FLERR,"Fix bond/adh cutoff is longer than pairwise cutoff");
  
  if (force->angle || force->dihedral || force->improper) {
    if (me == 0) 
      error->warning(FLERR,"Created bonds will not create angles, "
		     "dihedrals, or impropers");
  }

  // need a half neighbor list, built when ever re-neighboring occurs

  int irequest = neighbor->request((void *) this);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;
}

/* ---------------------------------------------------------------------- */

void FixBondAD::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixBondAD::setup(int vflag)
{
  int i,j,m;

  // compute initial bondcount if this is first run
  // can't do this earlier, like in constructor or init, b/c need ghost info

  if (countflag) return;
  countflag = 1;

  // count bonds stored with each bond I own
  // if newton bond is not set, just increment count on atom I
  // if newton bond is set, also increment count on atom J even if ghost
  // bondcount is long enough to tally ghost atom counts
  
  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  int **bond_atom = atom->bond_atom;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  int newton_bond = force->newton_bond;

  for (i = 0; i < nall; i++) bondcount[i] = 0;

  for (i = 0; i < nlocal; i++)
    for (j = 0; j < num_bond[i]; j++) {
      if (bond_type[i][j] == btype) {
        bondcount[i]++;
	if (newton_bond) {
	  m = atom->map(bond_atom[i][j]);
	  if (m < 0)
	    error->one(FLERR,"Could not count initial bonds in fix bond/adh");
	  bondcount[m]++;
	}
//fprintf(stderr,"In fix bond/AD setup(): timestep = %d, i = %d, j = %d, atom->tag[i] = %d, bond_type[i][j] = %d, bond_atom[i][j] = %d, bondcount[i] = %d,bondcount[j] = %d, newton_bond = %d\n",update->ntimestep, i, j, atom->tag[i],bond_type[i][j],bond_atom[i][j], bondcount[i],bondcount[j], newton_bond);
      }
    }

//  for (i = 0; i < nall; i++) fprintf(stderr,"In fix bond/AD setup(): timestep = %d, i = %d, atom->tag[i] = %d, bondcount[i] = %d, newton_bond = %d\n",update->ntimestep, i, atom->tag[i], bondcount[i], newton_bond);;

  // if newton_bond is set, need to communicate ghost counts

  if (newton_bond) comm->reverse_comm_fix(this);

	neighbor->build();
}

/* ---------------------------------------------------------------------- */

void FixBondAD::post_integrate()
{
  int i,j,m,ii,jj,inum,jnum,itype,jtype,tmp;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq,r;
  double vxtmp,vytmp,vztmp,delvx,delvy,delvz;
  int *ilist,*jlist,*numneigh,**firstneigh;

  // need updated ghost atom positions
  
  // fprintf(stderr,"In fix bond/AD post_integrate() start on proc %d when time = %d\n",comm->me,  update->ntimestep);

  comm->forward_comm();

  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;

  // loop over neighbors of my atoms
  // setup possible partner list of bonds to create

  double **x = atom->x;
  double **v = atom->v;
  int *tag = atom->tag;
  int *mask = atom->mask;
  int *type = atom->type;
  int nprocs = comm->nprocs;

  int **bond_type = atom->bond_type;
  int **bond_atom = atom->bond_atom;
  double **bond_length = atom->bond_length;
  int *num_bond = atom->num_bond;
  int newton_bond = force->newton_bond;

  int k,l,n1,n3,*slist;
  int **nspecial = atom->nspecial;
  int **special = atom->special;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  int ncreate = 0;
  double dtv = update->dt;

  //if (update->ntimestep > 3000) fprintf(stderr,"In fix bond/AD post_integrate() part01 on proc %d: timestep = %d\n", comm->me, update->ntimestep);

///////////////////////////////////this is for vWF//////////////////////////////////////////////

	//Alireza: GPIba-vWF
	//const double lscale = 1.0e-6, tscale = 3.083e-5, fscale = 2.63e-13, escale = 2.63e-19;
//	const double lscale = 1.0e-6, tscale = 1.802e-4, fscale = 4.5e-14, escale = 4.5e-20;
//	const double Ftrans = 1.0e-11/fscale, alpha = 1.0+0.05/2.25, kf01 = 5.8e0*tscale, kf02 = 52.0e0*tscale, 
//				 sigf1 = 1.58e-9/lscale, sigf2 = 1.31e-9/lscale, dgf1 = 6.17e-21/escale, dgf2 = 6.17e-21/escale;
////////////////////////////////////////////////////////////////////////////////////////////////
  if (atom->nmax > nmax) {

    memory->destroy(bondcount);

    nmax = atom->nmax;
    memory->create(bondcount,nmax,"bond/AD:bondcount");
  }
  
  //if (update->ntimestep > 3000) fprintf(stderr,"In fix bond/AD post_integrate() part02 on proc %d: timestep = %d\n", comm->me, update->ntimestep);
  
  int inum_local = 0;
  for (i = 0; i < nall; i++) {
	bondcount[i] = 0;
	if (mask[i] & groupbit && type[i] == iatomtype) inum_local++;
  }
  for (i = 0; i < nlocal; i++) {
    for (j = 0; j < num_bond[i]; j++) {
      if (bond_type[i][j] == btype) {
        bondcount[i]++;
		if (newton_bond) {
		  m = atom->map(bond_atom[i][j]);
		  if (m < 0) error->one(FLERR,"Could not count initial bonds in fix bond/adh");
		  bondcount[m]++;
		}
		//fprintf(stderr,"In fix bond/AD setup(): timestep = %d, i = %d, j = %d, atom->tag[i] = %d, bond_type[i][j] = %d, bond_atom[i][j] = %d, bondcount[i] = %d,bondcount[j] = %d, newton_bond = %d\n",update->ntimestep, i, j, atom->tag[i],bond_type[i][j],bond_atom[i][j], bondcount[i],bondcount[j], newton_bond);
      }
    }
  }
  
  //if (update->ntimestep > 3000) fprintf(stderr,"In fix bond/AD post_integrate() part03 on proc %d: timestep = %d\n", comm->me, update->ntimestep);
  
  //if (comm->me == 23) fprintf(stderr,"In fix bond/AD post_integrate() 01 proc = %d, bondcount[7242] = %d, tag[7242] = %d, timestep = %d \n", comm->me, bondcount[7242], tag[7242], update->ntimestep);
  //if (atom->map(429384) >= 0) fprintf(stderr,"In fix bond/AD post_integrate() 02 proc = %d, atom->map(429384) = %d, bondcount[atom->map(429384)] = %d, nlocal = %d, nall = %d, timestep = %d \n", comm->me, atom->map(429384), bondcount[atom->map(429384)], nlocal, nall, update->ntimestep);
  if (newton_bond) comm->reverse_comm_fix(this);
  
  //if (comm->me == 23) fprintf(stderr,"In fix bond/AD post_integrate() 03 proc = %d, bondcount[7242] = %d, tag[7242] = %d, timestep = %d \n", comm->me, bondcount[7242], tag[7242], update->ntimestep);
  //if (atom->map(429384) >= 0) fprintf(stderr,"In fix bond/AD post_integrate() 04 proc = %d, atom->map(429384) = %d, bondcount[atom->map(429384)] = %d, nlocal = %d, nall = %d, timestep = %d \n", comm->me, atom->map(429384), bondcount[atom->map(429384)], nlocal, nall, update->ntimestep);
  comm->forward_comm_fix(this);
  
  //if (update->ntimestep > 3000) fprintf(stderr,"In fix bond/AD post_integrate() part04 on proc %d: timestep = %d\n", comm->me, update->ntimestep);
  
  //if (comm->me == 23) fprintf(stderr,"In fix bond/AD post_integrate() 05 proc = %d, bondcount[7242] = %d, tag[7242] = %d, timestep = %d \n", comm->me, bondcount[7242], tag[7242], update->ntimestep);
  //if (atom->map(429384) >= 0) fprintf(stderr,"In fix bond/AD post_integrate() 06 proc = %d, atom->map(429384) = %d, bondcount[atom->map(429384)] = %d, nlocal = %d, nall = %d, timestep = %d \n", comm->me, atom->map(429384), bondcount[atom->map(429384)], nlocal, nall, update->ntimestep);
  
  int new_bond_num_local = 0;
  int new_bond_num;
  int *new_bond_atom1_local;
  int *new_bond_atom2_local;
  int *bondcount_atom1_local;
  int *bondcount_atom2_local;
  int *new_bond_num_proc;
  int *displs;
  int *new_bond_atom1;
  int *new_bond_atom2;
  int *bondcount_atom1;
  int *bondcount_atom2;
  memory->create(new_bond_atom1_local,9 * (1 + atom->bond_per_atom) * inum_local,"bond/AD:new_bond_atom1_local");
  memory->create(new_bond_atom2_local,9 * (1 + atom->bond_per_atom) * inum_local,"bond/AD:new_bond_atom2_local");
  memory->create(bondcount_atom1_local,9 * (1 + atom->bond_per_atom) * inum_local,"bond/AD:bondcount_atom1_local");
  memory->create(bondcount_atom2_local,9 * (1 + atom->bond_per_atom) * inum_local,"bond/AD:bondcount_atom2_local");
  
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;
	//if (update->ntimestep > 3000 && comm->me == 27) fprintf(stderr,"In fix bond/AD post_integrate() part01 on proc %d: timestep = %d, i = %d, atom->tag[i] = %d, num_bond[i] = %d, bondcount[i] = %d, x[i][0] = %f, x[i][1] = %f, x[i][2] = %f \n",me,  update->ntimestep, i, atom->tag[i], num_bond[i], bondcount[i], x[i][0], x[i][1], x[i][2]);
    itype = type[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    vxtmp = v[i][0];
    vytmp = v[i][1];
    vztmp = v[i][2];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    
    for (jj = 0; jj < jnum; jj++) {
      //if (update->ntimestep > 3000 && comm->me == 27) fprintf(stderr,"In fix bond/AD post_integrate() part02pre01 on proc %d: timestep = %d, i = %d, jj = %d\n",me, update->ntimestep, i, jj);
	  //if (update->ntimestep > 3000 && comm->me == 27) fprintf(stderr,"In fix bond/AD post_integrate() part02pre02 on proc %d: timestep = %d, i = %d, jlist[jj] = %d\n",me, update->ntimestep, i, jlist[jj]);
	  
	  j = jlist[jj];
      j &= NEIGHMASK;
      //if (update->ntimestep > 3000 && comm->me == 27) fprintf(stderr,"In fix bond/AD post_integrate() part02pre02 on proc %d: timestep = %d, i = %d, mask[j] = %d\n",me, update->ntimestep, i, mask[j]);
	  if (!(mask[j] & groupbit)) continue;
      //if (update->ntimestep > 3000 && comm->me == 27) fprintf(stderr,"In fix bond/AD post_integrate() part02pre on proc %d: timestep = %d, i = %d, j = %d, atom->tag[i] = %d, atom->tag[j] = %d\n",me, update->ntimestep, i, j, atom->tag[i],atom->tag[j]);
	  if (atom->molecule[i] == atom->molecule[j]) continue;
      jtype = type[j];
	  
	  tmp = 0;
	  for (k = 0; k < num_bond[i]; k++) {
	    if (bond_type[i][k] == btype && bond_atom[i][k] == atom->tag[j]) {
		  tmp = 1;
		  break;
		}
	  }
	  if (tmp) continue;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      if (rsq >= cutsq) continue;
			double r = sqrt(rsq);

      delvx = vxtmp - v[j][0];
      delvy = vytmp - v[j][1];
      delvz = vztmp - v[j][2];
      double vs = sqrt(delvx*delvx+delvy*delvy+delvz*delvz);

      //if (update->ntimestep > 3000 && comm->me == 27) fprintf(stderr,"In fix bond/AD post_integrate() part02 on proc %d: timestep = %d, i = %d, j = %d, atom->tag[i] = %d, atom->tag[j] = %d, bondcount[i] = %d,bondcount[j] = %d, itype = %d, jtype = %d, ligtype = %d, x[i][0] = %f, x[i][1] = %f, x[i][2] = %f, x[j][0] = %f, x[j][1] = %f, x[j][2] = %f \n",me, update->ntimestep, i, j, atom->tag[i],atom->tag[j],bondcount[i],bondcount[j], itype, jtype, ligtype, x[i][0], x[i][1], x[i][2], x[j][0], x[j][1], x[j][2]);

      if (itype == iatomtype && jtype == jatomtype){
	    if (bondcount[i] >= imaxbond || bondcount[j] >= jmaxbond) continue;
		bondforce = ks*fabs(r-r0);
/////////////////////////////////////////vWF////////////////////////////////////////////////////
/*				                                //Alireza: GPIba-vWF
                                if (bondforce < Ftrans)
					kf = kf01*pow(alpha,1.5)*(1.0+sigf1*bondforce/2.0/dgf1)*
							exp((dgf1/temp)*(1.0-alpha*(sigf1*bondforce/2.0/dgf1)*(sigf1*bondforce/2.0/dgf1)));
				else
					kf = kf02*pow(alpha,1.5)*(1.0+sigf2*bondforce/2.0/dgf2)*
							exp((dgf2/temp)*(1.0-alpha*(sigf2*bondforce/2.0/dgf2)*(sigf2*bondforce/2.0/dgf2)));
*/
////////////////////////////////////////////////////////////////////////////////////////////////
       	kf = kf0*exp(-0.5*sig*(r-r0)*(r-r0)/temp);
		exp((bondforce*(sig-0.5*fabs(r-r0)))/temp);
		
        double pp = 1.0 - exp(-kf*dtv);
		//if (update->ntimestep > 3000 && comm->me == 27) fprintf(stderr,"In fix bond/AD post_integrate() part03 on proc %d: timestep = %d, i = %d, j = %d, atom->tag[i] = %d, atom->tag[j] = %d, bondcount[i] = %d,bondcount[j] = %d, itype = %d, jtype = %d, ligtype = %d, pp = %f, kf = %f, r = %f, r0 = %f, x[i][0] = %f, x[i][1] = %f, x[i][2] = %f, x[j][0] = %f, x[j][1] = %f, x[j][2] = %f \n",me, update->ntimestep, i, j, atom->tag[i],atom->tag[j],bondcount[i],bondcount[j], itype, jtype, ligtype,pp, kf, r, r0, x[i][0], x[i][1], x[i][2], x[j][0], x[j][1], x[j][2]);
        for (m = 0; m < Nlig; m++){
          if (random->uniform() < pp){
			//if (bondforce < Ftrans)
			//	fprintf(stdout,"At timestep = %d, bond formed itype: %e %e %d %d kf01\n", update->ntimestep, r-r0, pp, atom->tag[i], atom->tag[j]);
			//else
			
			if (bondcount[i] >= imaxbond || bondcount[j] >= jmaxbond) break;
			new_bond_atom1_local[new_bond_num_local] = atom->tag[i];
			new_bond_atom2_local[new_bond_num_local] = atom->tag[j];
			bondcount_atom1_local[new_bond_num_local] = bondcount[i];
			bondcount_atom2_local[new_bond_num_local] = bondcount[j];
			new_bond_num_local++;
			bondcount[i]++;
			bondcount[j]++;
          }
        }
	  }
	  
      if (itype == jatomtype && jtype == iatomtype){
        if (bondcount[i] >= jmaxbond || bondcount[j] >= imaxbond) continue;
		bondforce = ks*fabs(r-r0);	//Alireza: GPIba-vWF
		// if (bondforce < Ftrans)
		//   kf = kf01*pow(alpha,1.5)*(1.0+sigf1*bondforce/2.0/dgf1)*
		//   exp((dgf1/temp)*(1.0-alpha*(sigf1*bondforce/2.0/dgf1)*(sigf1*bondforce/2.0/dgf1)));
		// else
		//   kf = kf02*pow(alpha,1.5)*(1.0+sigf2*bondforce/2.0/dgf2)*
		//   exp((dgf2/temp)*(1.0-alpha*(sigf2*bondforce/2.0/dgf2)*(sigf2*bondforce/2.0/dgf2)));
       	kf = kf0*exp(-0.5*sig*(r-r0)*(r-r0)/temp);
        
		double pp = 1.0 - exp(-kf*dtv);
		//if (update->ntimestep > 3000 && comm->me == 27) fprintf(stderr,"In fix bond/AD post_integrate() part05 on proc %d: timestep = %d, i = %d, j = %d, atom->tag[i] = %d, atom->tag[j] = %d, bondcount[i] = %d,bondcount[j] = %d, itype = %d, jtype = %d, ligtype = %d, pp = %f, kf = %f, r = %f, r0 = %f, x[i][0] = %f, x[i][1] = %f, x[i][2] = %f, x[j][0] = %f, x[j][1] = %f, x[j][2] = %f \n",me, update->ntimestep, i, j, atom->tag[i],atom->tag[j],bondcount[i],bondcount[j], itype, jtype, ligtype,pp, kf, r, r0, x[i][0], x[i][1], x[i][2], x[j][0], x[j][1], x[j][2]);
        for (m = 0; m < Nlig; m++){
          if (random->uniform() < pp){
			//if (bondforce < Ftrans)
			//	fprintf(stdout,"bond formed itype: %e %e %d %d kf01\n",r-r0,pp,atom->tag[i],atom->tag[j]);
			//else
			if (bondcount[i] >= jmaxbond || bondcount[j] >= imaxbond) break;
			new_bond_atom1_local[new_bond_num_local] = atom->tag[j];
			new_bond_atom2_local[new_bond_num_local] = atom->tag[i];
			bondcount_atom1_local[new_bond_num_local] = bondcount[j];
			bondcount_atom2_local[new_bond_num_local] = bondcount[i];
			new_bond_num_local++;
			bondcount[i]++;
			bondcount[j]++;
          }
      	}
      }
	}
  }
  // if (newton_bond) comm->reverse_comm_fix(this);
  // comm->forward_comm_fix(this);
  
  MPI_Allreduce(&new_bond_num_local,&new_bond_num,1,MPI_INT,MPI_SUM,world);
  memory->create(new_bond_num_proc,nprocs,"bond/AD:new_bond_num_proc");
  memory->create(displs,nprocs,"bond/AD:displs");
  if (nprocs > 1) {
    MPI_Allgather(&new_bond_num_local, 1, MPI_INT, new_bond_num_proc, 1, MPI_INT, world);
  }
  
  int kn = 0;
  for (i = 0; i < nprocs; i++){
    displs[i] = kn;
    kn += new_bond_num_proc[i];
  }
  memory->create(new_bond_atom1,kn,"bond/AD:new_bond_atom1");
  memory->create(new_bond_atom2,kn,"bond/AD:new_bond_atom2");
  memory->create(bondcount_atom1,kn,"bond/AD:bondcount_atom1");
  memory->create(bondcount_atom2,kn,"bond/AD:bondcount_atom2");
  if (nprocs > 1) {
	MPI_Allgatherv(new_bond_atom1_local, new_bond_num_local, MPI_INT, new_bond_atom1, new_bond_num_proc, displs, MPI_INT, world);
	MPI_Allgatherv(new_bond_atom2_local, new_bond_num_local, MPI_INT, new_bond_atom2, new_bond_num_proc, displs, MPI_INT, world);
	MPI_Allgatherv(bondcount_atom1_local, new_bond_num_local, MPI_INT, bondcount_atom1, new_bond_num_proc, displs, MPI_INT, world);
	MPI_Allgatherv(bondcount_atom2_local, new_bond_num_local, MPI_INT, bondcount_atom2, new_bond_num_proc, displs, MPI_INT, world);
  }
  
  int bondcounti, bondcountj;
  int tmpi, tmpj;
  for (k = 0; k < new_bond_num; k++) {
	// find the first appearance of atom 1 and 2 to obtain the initial bondcount
	for (l = 0; l < new_bond_num; l++) {
	  if (new_bond_atom1[k] == new_bond_atom1[l]) {
		bondcounti = bondcount_atom1[l];
		tmpi = l;
		break;
	  }
	}
	for (l = 0; l < new_bond_num; l++) {
	  if (new_bond_atom2[k] == new_bond_atom2[l]) {
		bondcountj = bondcount_atom2[l];
		tmpj = l;
		break;
	  }
	}
	// see if the bond numbers have reached maximum
	if (bondcounti >= imaxbond || bondcountj >= jmaxbond) continue;
	bondcount_atom1[tmpi]++;
	bondcount_atom2[tmpj]++;
	
	i = atom->map(new_bond_atom1[k]);
	j = atom->map(new_bond_atom2[k]);
	if (i >= 0 && i < nlocal) {
	  fprintf(stdout,"At timestep = %d, bond formed itype: %d %d %d %d kf02, i = %d, j = %d, nall = %d, nlocal = %d, proc = %d\n", update->ntimestep, atom->tag[i], atom->tag[j], bondcount_atom1[tmpi], bondcount_atom1[tmpi], i, j, nall, nlocal, comm->me);
	  bond_type[i][num_bond[i]] = btype;
	  bond_atom[i][num_bond[i]] = atom->tag[j];
	  bond_length[i][num_bond[i]] = r0;
	  num_bond[i]++;
	  
	  // bondcount[i]++;
	  // bondcount[j]++;
	  ncreate++;
	  // add a 1-2 neighbor to special bond list for atom I
	  // atom J will also do this
	  
	  slist = special[i];
	  n1 = nspecial[i][0];
	  n3 = nspecial[i][2];
	  if (n3 == atom->maxspecial) error->warning(FLERR,"New bond exceeded special list size in fix bond/adh");
	  for (k = n3; k > n1; k--) slist[k+1] = slist[k];
	  slist[n1] = atom->tag[j];
	  nspecial[i][0]++;
	  nspecial[i][1]++;
	  nspecial[i][2]++;
	  
	  slist = special[j];
	  n1 = nspecial[j][0];
	  n3 = nspecial[j][2];
	  if (n3 == atom->maxspecial) error->warning(FLERR,"New bond exceeded special list size in fix bond/adh");
	  for (k = n3; k > n1; k--) slist[k+1] = slist[k];
	  slist[n1] = atom->tag[i];
	  nspecial[j][0]++;
	  nspecial[j][1]++;
	  nspecial[j][2]++;
	}
  }
  
  
  //if (update->ntimestep > 3000) fprintf(stderr,"In fix bond/AD post_integrate() part05 on proc %d: timestep = %d\n", comm->me, update->ntimestep);
  
  // tally stats

  MPI_Allreduce(&ncreate,&createcount,1,MPI_INT,MPI_SUM,world);
  createcounttotal += createcount;
  atom->nbonds += createcount;
  
  //if (update->ntimestep > 3000) fprintf(stderr,"In fix bond/AD post_integrate() part06 on proc %d: timestep = %d\n", comm->me, update->ntimestep);

  // trigger reneighboring if any bonds were formed

  if (createcount) next_reneighbor = update->ntimestep;
  //fprintf(stderr,"In fix bond/AD post_integrate() end on proc %d when time = %d\n",comm->me,  update->ntimestep);
  
  memory->destroy(new_bond_atom1_local);
  memory->destroy(new_bond_atom2_local);
  memory->destroy(bondcount_atom1_local);
  memory->destroy(bondcount_atom2_local);
  memory->destroy(new_bond_num_proc);
  memory->destroy(displs);
  memory->destroy(new_bond_atom1);
  memory->destroy(new_bond_atom2);
  memory->destroy(bondcount_atom1);
  memory->destroy(bondcount_atom2);
}

/* ---------------------------------------------------------------------- */

int FixBondAD::pack_forward_comm(int n, int *list, double *buf,
			     int pbc_flag, int *pbc)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = ubuf(bondcount[j]).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixBondAD::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    bondcount[i] = (int) ubuf(buf[m++]).i;
  }
}

/* ---------------------------------------------------------------------- */

int FixBondAD::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;

  for (i = first; i < last; i++) 
    buf[m++] = ubuf(bondcount[i]).d;
  return m;
}

/* ---------------------------------------------------------------------- */

void FixBondAD::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;

  for (i = 0; i < n; i++) {
    j = list[i];
    bondcount[j] += (int) ubuf(buf[m++]).i;
  }
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays 
------------------------------------------------------------------------- */

void FixBondAD::grow_arrays(int nmax)
{
  //bondcount = (int *)
  //  memory->srealloc(bondcount,nmax*sizeof(int),"bond/adh:bondcount");
  memory->grow(bondcount,nmax,"bond/AD:bondcount");
}

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays 
------------------------------------------------------------------------- */

void FixBondAD::copy_arrays(int i, int j, int delflag)
{
  bondcount[j] = bondcount[i];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc 
------------------------------------------------------------------------- */

int FixBondAD::pack_exchange(int i, double *buf)
{
  buf[0] = bondcount[i];
  return 1;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based arrays from exchange with another proc 
------------------------------------------------------------------------- */

int FixBondAD::unpack_exchange(int nlocal, double *buf)
{
  bondcount[nlocal] = static_cast<int> (buf[0]);
  return 1;
}

/* ---------------------------------------------------------------------- */

double FixBondAD::compute_vector(int n)
{
  if (n == 1) return (double) createcount;
  return (double) createcounttotal;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays 
------------------------------------------------------------------------- */

double FixBondAD::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = nmax*2 * sizeof(int);
  bytes += nmax * sizeof(double);
  return bytes;
}
