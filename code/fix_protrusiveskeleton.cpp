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

/* ----------------------------------------------------------------------
   Contributing author: Shuhao Ma (ZJU)
------------------------------------------------------------------------- */

#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include "fix_protrusiveskeleton.h"
#include "atom.h"
#include "atom_masks.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "region.h"
#include "respa.h"
#include "input.h"
#include "comm.h"
#include "universe.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "group.h"
#include "math_extra.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixProtrusiveSkeleton::FixProtrusiveSkeleton(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg != 6 && narg != 8 && narg != 9 && narg != 11) error->all(FLERR,"Illegal fix ProtrusiveSkeleton command");

  int igroup_mol;
  char gn[50];

  scalar_flag = 1;
  vector_flag = 1;
  size_vector = 3;
  global_freq = 1;
  extscalar = 1;
  extvector = 1;
  niter = 0;

  sprintf(gn,arg[3]);
  igroup_mol = group->find(arg[3]);
  if (igroup_mol == -1) error->all(FLERR,"Could not find molecule group ID for fix ProtrusiveSkeleton command");
  groupbit_mol = group->bitmask[igroup_mol];

  Fvalue = force->numeric(FLERR,arg[4]);
  
  bond_prt_type = force->inumeric(FLERR,arg[5]);
  
  bystep_flag = 0;
  around_flag = 0;
  if (narg == 6){
	bystep_flag = 0;
	around_flag = 0;
  }
  else if (narg == 8){
	if (strcmp(arg[6],"around") == 0) {
      around_flag= 1;
      Raround = force->numeric(FLERR,arg[7]);
	}
	else error->all(FLERR,"Illegal fix ProtrusiveSkeleton command");
  }
  else if (narg == 9){
	if (strcmp(arg[6],"bystep") == 0) {
      bystep_flag= 1;
      Ndiv = force->inumeric(FLERR,arg[7]);
      finalstep = force->inumeric(FLERR,arg[8]);
      if (finalstep % Ndiv) error->all(FLERR,"The timestep is not an integer multiple of the increase times for  the gradual-mode fix ProtrusiveSkeleton command");
      divstep = finalstep/Ndiv;
      finalstep = update->ntimestep + divstep*(Ndiv-1);
      Fadd = Fvalue/Ndiv;
      Fvalue = 0.0;
	}
	else error->all(FLERR,"Illegal fix ProtrusiveSkeleton command");
  }
  else if (narg == 11){
	if (strcmp(arg[6],"bystep") == 0 && strcmp(arg[9],"around") == 0) {
      bystep_flag= 1;
      Ndiv = force->inumeric(FLERR,arg[7]);
      finalstep = force->inumeric(FLERR,arg[8]);
      if (finalstep % Ndiv) error->all(FLERR,"The timestep is not an integer multiple of the increase times for  the gradual-mode fix ProtrusiveSkeleton command");
      divstep = finalstep/Ndiv;
      finalstep = update->ntimestep + divstep*(Ndiv-1);
      Fadd = Fvalue/Ndiv;
      Fvalue = 0.0;
	  
	  around_flag= 1;
	  Raround = force->numeric(FLERR,arg[10]);
	}
	else error->all(FLERR,"Illegal fix ProtrusiveSkeleton command");
  }


/*  char fname[FILENAME_MAX];
  FILE *f_write;
    sprintf(fname,"test.dat");
    f_write = fopen(fname,"w");
    fprintf(f_write,"groupbit_mol=%f\n",groupbit_mol);
    fprintf(f_write,"Fvalue=%f\n",Fvalue);
    fclose(f_write);
*/

//fprintf(stderr,"groupbit_mol=%f\n",groupbit_mol);
//fprintf(stderr,"Fvalue=%f\n",Fvalue);

  // optional args

  
  nevery = 1;


  force_flag = 0;
  foriginal[0] = foriginal[1] = foriginal[2] = foriginal[3] = 0.0;

  maxatom = atom->nmax;
}

/* ---------------------------------------------------------------------- */

FixProtrusiveSkeleton::~FixProtrusiveSkeleton()
{
  //memory->destroy(ProtrusiveSkeleton;);
}

/* ---------------------------------------------------------------------- */

int FixProtrusiveSkeleton::setmask()
{
  datamask_read = datamask_modify = 0;

  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
  //fprintf(stderr,"setup finish\n");
}

/* ---------------------------------------------------------------------- */

void FixProtrusiveSkeleton::init()
{
  avec = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
  if (!avec) error->all(FLERR,"Pair fluidmembrane requires atom style ellipsoid");
}

/* ---------------------------------------------------------------------- */

void FixProtrusiveSkeleton::setup(int vflag)
{
  if (strstr(update->integrate_style,"verlet"))
    post_force(vflag);
  else {
    ((Respa *) update->integrate)->copy_flevel_f(nlevels_respa-1);
    post_force_respa(vflag,nlevels_respa-1,0);
    ((Respa *) update->integrate)->copy_f_flevel(nlevels_respa-1);
  }
}

/* ---------------------------------------------------------------------- */

void FixProtrusiveSkeleton::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixProtrusiveSkeleton::post_force(int vflag)
{
  int step;

  int i,j,n,m,ncount;
  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  imageint *image = atom->image;
  int nlocal = atom->nlocal;
  double xh[3],cm[3],cmt[4], r[3],n1[3],rnorm[3],xm[3],xn[3],nall[3],nallg[3];
  double nx,ny,nz,sinhalftheta,q12,rdotn1,rdotrnorm,rmton,rcton,wd;
  double masstotal,r1;
  double *nquat;

  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;

  //if (update->ntimestep % nevery) return;


  // foriginal[0] = "potential energy" for added force
  // foriginal[123] = force on atoms before extra force added

  foriginal[0] = foriginal[1] = foriginal[2] = foriginal[3] = 0.0;
  force_flag = 0;

  int n_add;
  if (bystep_flag){
    if (update->ntimestep > finalstep) n_add = Ndiv;
    else n_add = Ndiv - (finalstep - update->ntimestep)/divstep -1;
    Fvalue = n_add*Fadd;
  }

////////////////////////////////////////////////////////////// find cell center///////////////////////////////////////////////////////////////////////////
/*
  for(i=0; i<4; i++){
    cm[i] = 0.0;
    cmt[i] = 0.0;
  }

  fprintf(stderr,"post_force cmt\n");
  
  for (i=0; i<nlocal; i++){
    if (mask[i] & groupbit){
      for (j=0;j<3;j++) xh[j]=x[i][j];
      domain->unmap(xh,atom->image[i]);
      for (j=0;j<3;j++) cmt[j]+=xh[j];
      cmt[3]+=1.0;
    }

    MPI_Allreduce(&cmt[0],&cm[0],4,MPI_DOUBLE,MPI_SUM,world);
    for(j=0; j<3; i++) cm[j] /= cm[3];
  }

  char fname2[FILENAME_MAX];
  FILE *f_write2;
  sprintf(fname2,"test2.dat");
  f_write2 = fopen(fname2,"w");
  fprintf(f_write2,"center of mass is (%f,%f,%f) in step %d\n", cm[0], cm[1], cm[2], update->ntimestep);
  fclose(f_write2);
*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  ncount = group->count(igroup);


  masstotal = group->mass(igroup);
  group->xcm(igroup,masstotal,cm);

//fprintf(stderr,"center of mass is (%f,%f,%f) in step %d\n", cm[0], cm[1], cm[2], update->ntimestep);
  nall[0] = 0.0;
  nall[1] = 0.0;
  nall[2] = 0.0;
  
  for (n=0; n<nlocal; n++){
    if (mask[n] & groupbit_mol){

//////////////////////////////////////////////// Gov's protrusive force by surface unit normal vector/////////////////////////////////////////////////////      
/*
      //find molecule position

      for (j=0;j<3;j++){
        xh[j]=x[n][j];
      }
      domain->unmap(xh,atom->image[n]);
      for (j=0;j<3;j++){
        r[j]=xh[j]-cm[j];
      }

      r1=sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
      for (j=0;j<3;j++){
        rnorm[j]/=r1;
      }

      //find surface unit normal vector
      //Suppose the particle direction is the normal direction
      nquat=bonus[ellipsoid[n]].quat;
      q12=nquat[0]*nquat[0];
      sinhalftheta=1-sqrt(q12);
      nx=nquat[1]/sinhalftheta;
      ny=nquat[2]/sinhalftheta;
      nz=nquat[3]/sinhalftheta;

      n1[0]=nx;
      n1[1]=ny;
      n1[2]=nz;

      rdotn1=r[0]*n1[0]+r[1]*n1[1]+r[2]*n1[2];
      rdotrnorm=r[0]*rnorm[0]+r[1]*rnorm[1]+r[2]*rnorm[2];
      if (rdotn1<0){
        for (j=0;j<3;j++){
          n1[j]*=-1.0;
        }
      }
      rdotn1=abs(rdotn1);
	  
	  //suppose center to surface is the normal direction
      foriginal[0]-=Fvalue*rdotrnorm;
      f[n][0]+=Fvalue*rnorm[0];
      f[n][1]+=Fvalue*rnorm[1];
      f[n][2]+=Fvalue*rnorm[2];

*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////// Gov's protrusive force by particle directions/////////////////////////////////////////////////////////
/*
	  //find the unit normal vectos through quaternions
	  double xyz[4] = {0, 1, 0, 0};
	  double quatx[4], quatxquat[4];
	  nquat =  bonus[ellipsoid[n]].quat;
	  double nquatinv[4] = {nquat[0], -nquat[1], -nquat[2], -nquat[3]} ;     
	  MathExtra::quatquat(nquat, xyz, quatx);
	  MathExtra::quatquat(quatx, nquatinv, quatxquat);
	  n1[0] = quatxquat[1];
	  n1[1] = quatxquat[2];
	  n1[2] = quatxquat[3];
//	  fprintf(stderr, "quatxquat is %f, %f, %f, %f\n", quatxquat[0], quatxquat[1], quatxquat[2], quatxquat[3]);


      foriginal[0]-=Fvalue*(n1[0]*xh[0] + n1[1]*xh[1] + n1[2]*xh[2]);
      foriginal[1] += f[n][0];
      foriginal[2] += f[n][1];
      foriginal[3] += f[n][2];
      f[n][0]+=Fvalue*n1[0];
      f[n][1]+=Fvalue*n1[1];
      f[n][2]+=Fvalue*n1[2];

	  for (m=0; m<nlocal; m++){
	    if (mask[m] & groupbit){
		  if(m != n){
		    f[m][0]-=Fvalue*n1[0]/(ncount-1);
		    f[m][1]-=Fvalue*n1[1]/(ncount-1);
		    f[m ][2]-=Fvalue*n1[2]/(ncount-1);
		  }
		}
	  }
//	  fprintf(stderr, "bilayer number is %d\n", ncount);

*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////	 

////////////////////////////////////////////////////////// Pair-wise protrusive force/////////////////////////////////////////////////////////////////////
/*
	  xn[0] = x[n][0];
	  xn[1] = x[n][1];
	  xn[2] = x[n][2];
	  domain->unmap(xn,atom->image[n]);
	  for (m=0; m<nlocal; m++){
	    if (mask[m] & groupbit){
		  if(m != n){
		    xm[0] = x[m][0];
		    xm[1] = x[m][1];
		    xm[2] = x[m][2];
		    domain->unmap(xm,atom->image[m]);

		    rmton = sqrt((xn[0] - xm[0]) * (xn[0] - xm[0]) + (xn[1] - xm[1]) * (xn[1] - xm[1]) + (xn[2] - xm[2]) * (xn[2] - xm[2]));
		    n1[0] = (xn[0] - xm[0])/rmton;
		    n1[1] = (xn[1] - xm[1])/rmton;
		    n1[2] = (xn[2] - xm[2])/rmton;
		    f[n][0]+=Fvalue*n1[0];
		    f[n][1]+=Fvalue*n1[1];
		    f[n][2]+=Fvalue*n1[2];
		    f[m][0]-=Fvalue*n1[0];
		    f[m][1]-=Fvalue*n1[1];
		    f[m][2]-=Fvalue*n1[2];
		  }
	    }
	  }

*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	  int *num_bond_new = atom->num_bond;
      int **bond_type_new = atom->bond_type;
      tagint **bond_atom_new = atom->bond_atom;
	  double **bond_length_new = atom->bond_length;
	  
	  for (i = 0; i < num_bond_new[n]; i++) {
		if (bond_type_new[n][i] == bond_prt_type) {
		  fprintf(stderr, "In Fix ProtrusiveSkeleton at proc %d when time = %d: bond %d-%d was deleted\n", comm->me, update->ntimestep, atom->tag[n], bond_atom_new[n][i]);
		  for (m = i; m < num_bond_new[n] - 1; m++) {
			bond_atom_new[n][m] = bond_atom_new[n][m+1];
			bond_type_new[n][m] = bond_type_new[n][m+1];
			bond_length_new[n][m] = bond_length_new[n][m+1];
		  }
		  num_bond_new[n]--;
		  i--;
		  atom->nbonds--;
		}
	  }
	  
/////////////////////////////////////////////////// Pair-wise protrusive force for skeleton //////////////////////////////////////////////////////////////

	  xn[0] = x[n][0];
	  xn[1] = x[n][1];
	  xn[2] = x[n][2];
	  domain->unmap(xn,atom->image[n]);
	  rcton = sqrt((xn[0] - cm[0]) * (xn[0] - cm[0]) + (xn[1] - cm[1]) * (xn[1] - cm[1]) + (xn[2] - cm[2]) * (xn[2] - cm[2]));
	  n1[0] = (xn[0] - cm[0])/rcton;
	  n1[1] = (xn[1] - cm[1])/rcton;
	  n1[2] = (xn[2] - cm[2])/rcton;
	  f[n][0]+=Fvalue*n1[0];
	  f[n][1]+=Fvalue*n1[1];
	  f[n][2]+=Fvalue*n1[2];
	  nall[0] += n1[0];
	  nall[1] += n1[1];
	  nall[2] += n1[2];
	  
	  if (around_flag) {
	    for (m=0; m<nlocal; m++){
	      if (mask[m] & groupbit){
		    if (m != n){
		      xm[0] = x[m][0];
		      xm[1] = x[m][1];
		      xm[2] = x[m][2];
		      domain->unmap(xm,atom->image[m]);
		  	  rmton = sqrt((xn[0] - xm[0]) * (xn[0] - xm[0]) + (xn[1] - xm[1]) * (xn[1] - xm[1]) + (xn[2] - xm[2]) * (xn[2] - xm[2]));
			  if(rmton < Raround){
			    wd = 1 - rmton/Raround;
		        f[m][0] += Fvalue*n1[0]*wd;
		        f[m][1] += Fvalue*n1[1]*wd;
		        f[m][2] += Fvalue*n1[2]*wd;
		        nall[0] += n1[0]*wd;
				nall[1] += n1[1]*wd;
				nall[2] += n1[2]*wd;
			  }
		    }
	      }
	    }
	  }


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    }
  }
  
  // add the opposite force
  
  MPI_Allreduce(nall,nallg,3,MPI_DOUBLE,MPI_SUM,world);
  nallg[0] = nallg[0]*Fvalue/masstotal;
  nallg[1] = nallg[1]*Fvalue/masstotal;
  nallg[2] = nallg[2]*Fvalue/masstotal;
  for (m=0; m<nlocal; m++){
    if (mask[m] & groupbit){
	  f[m][0]-=nallg[0]*rmass[m];
	  f[m][1]-=nallg[1]*rmass[m];
	  f[m][2]-=nallg[2]*rmass[m];
	}
  }

}

/* ---------------------------------------------------------------------- */

void FixProtrusiveSkeleton::post_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixProtrusiveSkeleton::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   potential energy of added force
------------------------------------------------------------------------- */

double FixProtrusiveSkeleton::compute_scalar()
{
  // only sum across procs one time

  if (force_flag == 0) {
    MPI_Allreduce(foriginal,foriginal_all,4,MPI_DOUBLE,MPI_SUM,world);
    force_flag = 1;
  }
  return foriginal_all[0];
}

/* ----------------------------------------------------------------------
   return components of total force on fix group before force was changed
------------------------------------------------------------------------- */

double FixProtrusiveSkeleton::compute_vector(int n)
{
  // only sum across procs one time

  if (force_flag == 0) {
    MPI_Allreduce(foriginal,foriginal_all,4,MPI_DOUBLE,MPI_SUM,world);
    force_flag = 1;
  }
  return foriginal_all[n+1];
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

/*

double FixProtrusiveSkeleton::memory_usage()
{
  double bytes = 0.0;
  if (varflag == ATOM) bytes = atom->nmax*4 * sizeof(double);
  return bytes;
}

 ---------------------------------------------------------------------- */