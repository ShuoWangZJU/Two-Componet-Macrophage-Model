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
#include "fix_protrusiveskeleton_local.h"
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

FixProtrusiveSkeletonLocal::FixProtrusiveSkeletonLocal(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg != 11 && narg != 13 && narg != 14 && narg != 16 && narg != 17 && narg != 19) error->all(FLERR,"Illegal fix ProtrusiveSkeletonLocal command");

  int i, m;
  double r;
  
  m = 3;
  
  ske_bond_type = force->inumeric(FLERR,arg[m++]);
  
  igroup_ske = group->find(arg[m++]);
  if (igroup_ske == -1) error->all(FLERR,"Could not find membrane group ID for fix ProtrusiveSkeletonLocal command");
  groupbit_ske = group->bitmask[igroup_ske];
  
  igroup_mem = group->find(arg[m++]);
  if (igroup_mem == -1) error->all(FLERR,"Could not find membrane group ID for fix ProtrusiveSkeletonLocal command");
  groupbit_mem = group->bitmask[igroup_mem];
  
  igroup_prt = group->find(arg[m++]);
  if (igroup_prt == -1) error->all(FLERR,"Could not find membrane group ID for fix ProtrusiveSkeletonLocal command");
  groupbit_prt = group->bitmask[igroup_prt];
  
  Fvalue = force->numeric(FLERR,arg[m++]);
  
  if (strcmp(arg[m],"opposite") == 0) {
	m++;
	oppo_layer = force->inumeric(FLERR,arg[m++]);
	if (oppo_layer < 0) error->all(FLERR,"Illegal fix ProtrusiveSkeletonLocal command");
  }
  else error->all(FLERR,"Illegal fix ProtrusiveSkeletonLocal command");
  
  if (strcmp(arg[m],"direction_position") == 0) {
	direction_flag = 0;
	m++;
  }
  else if (strcmp(arg[m],"direction_orientation") == 0) {
	direction_flag = 1;
	m++;
  }
  else if (strcmp(arg[m],"direction_custom") == 0) {
	direction_flag = 2;
	m++;
	dir_custom[0] = force->numeric(FLERR,arg[m++]);
	dir_custom[1] = force->numeric(FLERR,arg[m++]);
	dir_custom[2] = force->numeric(FLERR,arg[m++]);
	r = sqrt(dir_custom[0] * dir_custom[0] + dir_custom[1] * dir_custom[1] + dir_custom[2] * dir_custom[2]);
	dir_custom[0] = dir_custom[0] / r;
	dir_custom[1] = dir_custom[1] / r;
	dir_custom[2] = dir_custom[2] / r;
  }
  else error->all(FLERR,"Illegal fix ProtrusiveSkeletonLocal command");
  
  bystep_flag = 0;
  around_flag = 0;
  if (narg == m){
	bystep_flag = 0;
	around_flag = 0;
  }
  else if (narg == m+2){
	if (strcmp(arg[m++],"around") == 0) {
      around_flag= 1;
      Raround = force->numeric(FLERR,arg[m++]);
	}
	else error->all(FLERR,"Illegal fix ProtrusiveSkeletonLocal command");
  }
  else if (narg == m+3){
	if (strcmp(arg[m++],"bystep") == 0) {
      bystep_flag= 1;
      Ndiv = force->inumeric(FLERR,arg[m++]);
      durationstep = force->inumeric(FLERR,arg[m++]);
      if (durationstep % Ndiv) error->all(FLERR,"The timestep is not an integer multiple of the increase times for  the gradual-mode fix ProtrusiveSkeletonLocal command");
      divstep = durationstep/Ndiv;
      durationstep = divstep*(Ndiv-1);
      Fadd = Fvalue/Ndiv;
      Fvalue = 0.0;
	}
	else error->all(FLERR,"Illegal fix ProtrusiveSkeletonLocal command");
  }
  else if (narg == m+5){
	if (strcmp(arg[m++],"bystep") == 0 && strcmp(arg[narg-2],"around") == 0) {
      bystep_flag= 1;
      Ndiv = force->inumeric(FLERR,arg[m++]);
      durationstep = force->inumeric(FLERR,arg[m++]);
      if (durationstep % Ndiv) error->all(FLERR,"The timestep is not an integer multiple of the increase times for  the gradual-mode fix ProtrusiveSkeletonLocal command");
      divstep = durationstep/Ndiv;
      durationstep = divstep*(Ndiv-1);
      Fadd = Fvalue/Ndiv;
      Fvalue = 0.0;
	  
	  around_flag= 1;
	  Raround = force->numeric(FLERR,arg[narg-1]);
	  m = m + 2;
	}
	else error->all(FLERR,"Illegal fix ProtrusiveSkeletonLocal command");
  }
  else error->all(FLERR,"Illegal fix ProtrusiveSkeletonLocal command");
  
  if (m != narg) error->all(FLERR,"Illegal fix ProtrusiveSkeletonLocal command");
  
  maxatom = atom->nmax;
  maxbond_per_atom = atom->bond_per_atom;
  
  //grow_arrays(maxatom);
  memory->create(bond_count,maxatom,"protrusiveskeleton/local:bond_count");
  memory->create(bond_change_count,maxatom,"protrusiveskeleton/local:bond_change_count");
  memory->create(bond_atom_full,maxatom,maxbond_per_atom,"protrusiveskeleton/local:bond_atom_full");
  memory->create(bond_atom_change,maxatom,maxbond_per_atom,"protrusiveskeleton/local:bond_atom_change");
  memory->create(initial_time,maxatom,"protrusiveskeleton/local:initial_time");
  atom->add_callback(0);
  
  int nlocalghost = atom->nlocal + atom->nghost;
  for (i = 0; i < nlocalghost; i++){
	initial_time[i] = -1;
  }
  
  comm_forward = 3 + 2 * maxbond_per_atom;
  comm_reverse = 1 + maxbond_per_atom;


/*  char fname[FILENAME_MAX];
  FILE *f_write;
    sprintf(fname,"test.dat");
    f_write = fopen(fname,"w");
    fprintf(f_write,"groupbit_mem=%f\n",groupbit_mem);
    fprintf(f_write,"Fvalue=%f\n",Fvalue);
    fclose(f_write);
*/

//fprintf(stderr,"groupbit_mem=%f\n",groupbit_mem);
//fprintf(stderr,"Fvalue=%f\n",Fvalue);

  // optional args

  
  nevery = 1;


  force_flag = 0;
  foriginal[0] = foriginal[1] = foriginal[2] = foriginal[3] = 0.0;

}

/* ---------------------------------------------------------------------- */

FixProtrusiveSkeletonLocal::~FixProtrusiveSkeletonLocal()
{
  // unregister callbacks to this fix from Atom class

  atom->delete_callback(id,0);
  
  // delete locally stored arrays
  
  memory->destroy(bond_count);
  memory->destroy(bond_change_count);
  memory->destroy(bond_atom_full);
  memory->destroy(bond_atom_change);
  memory->destroy(initial_time);
}

/* ---------------------------------------------------------------------- */

int FixProtrusiveSkeletonLocal::setmask()
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

void FixProtrusiveSkeletonLocal::init()
{
  avec = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
  if (!avec) error->all(FLERR,"Fix protrusiveskeleton/local requires atom style ellipsoid");
}

/* ---------------------------------------------------------------------- */

void FixProtrusiveSkeletonLocal::setup(int vflag)
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

void FixProtrusiveSkeletonLocal::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixProtrusiveSkeletonLocal::post_force(int vflag)
{
  int step;
  
  comm->forward_comm();

  int i,j,k,n,m,ncount,ii,mm,nn,bond_skele,count_in,count_out,count_tmp,flag,kn;
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int *mask = atom->mask;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  imageint *image = atom->image;
  int nlocal = atom->nlocal;
  int nlocalghost = atom->nlocal + atom->nghost;
  int nprocs = comm->nprocs;
  double xh[3],cm[3],cmt[4], r[3],n1[3],rnorm[3],xm[3],xn[3],nall[3],nallg[3];
  double nx,ny,nz,sinhalftheta,q12,rdotn1,rdotrnorm,rmton,rcton,wd;
  double masstotal,r1;
  double *nquat;
  
  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;

  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  int nowtime = update->ntimestep;

  //if (update->ntimestep % nevery) return;


  // foriginal[0] = "potential energy" for added force
  // foriginal[123] = force on atoms before extra force added

  foriginal[0] = foriginal[1] = foriginal[2] = foriginal[3] = 0.0;
  force_flag = 0;

  int n_add;
  for (i = 0; i < nlocal; i++) {
	if (mask[i] & groupbit) {
	  if (initial_time[i] < 0) {
		initial_time[i] = nowtime;
	  }
	}
	else {
	  initial_time[i] = -1;
	}
  }
  
///////////////////////////////////////////////////////////// find full bondlist /////////////////////////////////////////////////////////////////////////
  if (oppo_layer) {
	for (i = 0; i < nlocalghost; i++){
	  bond_count[i] = 0;
	  bond_change_count[i] = 0;
	  for (k = 0; k < maxbond_per_atom; k++) {
	    bond_atom_change[i][k] = 0;
	    bond_atom_full[i][k] = 0;
	  }
    }
    
    for (i = 0; i < nlocal; i++) {
	  if (mask[i] & groupbit_ske) {
        for (j = 0; j < num_bond[i]; j++) {
          if (bond_type[i][j] == ske_bond_type) {
	    	bond_atom_full[i][bond_count[i]] = bond_atom[i][j];
            bond_count[i]++;
            k = atom->map(bond_atom[i][j]);
            if (k < 0) error->one(FLERR,"Fix SkeleCreateBreak needs ghost atoms from further away");
	    	else if (k < nlocal) {
	    	  bond_atom_full[k][bond_count[k]] = atom->tag[i];
              bond_count[k]++;
	    	}
	    	else {
	    	  bond_atom_change[k][bond_change_count[k]] = atom->tag[i];
              bond_change_count[k]++;
	    	}
          }
        }
	  }
    }
    comm->reverse_comm_fix(this);
    comm->forward_comm_fix(this);
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

  
  ncount = group->count(igroup_ske);
  masstotal = group->mass(igroup_ske);
  group->xcm(igroup_ske,masstotal,cm);

//fprintf(stderr,"center of mass is (%f,%f,%f) in step %d\n", cm[0], cm[1], cm[2], update->ntimestep);
  nall[0] = 0.0;
  nall[1] = 0.0;
  nall[2] = 0.0;
  
  
  bond_bond_count_local = 0;
  memory->create(bond_in_atom,maxatom,"protrusiveskeleton/local:bond_in_atom");
  memory->create(bond_out_atom,maxatom,"protrusiveskeleton/local:bond_out_atom");
  memory->create(bond_tmp_atom,maxatom,"protrusiveskeleton/local:bond_tmp_atom");
  memory->create(bond_bond_atom_local,maxatom*maxbond_per_atom,"protrusiveskeleton/local:bond_bond_atom_local");
  memory->create(bond_bond_fx_local,maxatom*maxbond_per_atom,"protrusiveskeleton/local:bond_bond_fx_local");
  memory->create(bond_bond_fy_local,maxatom*maxbond_per_atom,"protrusiveskeleton/local:bond_bond_fy_local");
  memory->create(bond_bond_fz_local,maxatom*maxbond_per_atom,"protrusiveskeleton/local:bond_bond_fz_local");
  
/*   if (nowtime % 10000 == 0 && comm->me == 57) {
	for (i = nlocal; i < nlocalghost; i++) {
	  if (type[i] == 5) {
		fprintf(stderr, "In Fix protrusiveskeleton/local at proc %d when time = %d: ghost atom %d\n", comm->me, update->ntimestep, atom->tag[i]);
	  }
	}
  } */
  
  // find the membrane beads that connect to skeleton beads
  for (bond_skele=0; bond_skele<nlocal; bond_skele++) {
    if (mask[bond_skele] & groupbit_ske) {
	  for (mm = 0; mm < num_bond[bond_skele]; mm++) {
	    if ((bond_type[bond_skele][mm] != ske_bond_type) && (mask[atom->map(bond_atom[bond_skele][mm])] & groupbit)) {
	      n = atom->map(bond_atom[bond_skele][mm]);
		  // fprintf(stderr, "In Fix protrusiveskeleton/local at proc %d when time = %d: the protrusive force was applied on atom %d\n", comm->me, update->ntimestep, bond_atom[bond_skele][mm]);
		  
		  if (bystep_flag){
		    if (nowtime > durationstep + initial_time[n]) n_add = Ndiv;
		    else n_add = (nowtime - initial_time[n]) / divstep + 1;
		    Fvalue = n_add*Fadd;
		  }
///////////////////////////////////////// Pair-wise protrusive force by particle position for skeleton ///////////////////////////////////////////////////

/* 	  xn[0] = x[n][0];
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
	  } */


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////// Pair-wise protrusive force by particle position or direction for skeleton ////////////////////////////////////////////

	      xn[0] = x[n][0];
		  xn[1] = x[n][1];
		  xn[2] = x[n][2];
		  domain->unmap(xn,atom->image[n]);
		  
		  // determine the force direction by particle position relative to skeleton
		  if (direction_flag == 0) {
			rcton = sqrt((xn[0] - cm[0]) * (xn[0] - cm[0]) + (xn[1] - cm[1]) * (xn[1] - cm[1]) + (xn[2] - cm[2]) * (xn[2] - cm[2]));
			n1[0] = (xn[0] - cm[0]) / rcton;
			n1[1] = (xn[1] - cm[1]) / rcton;
			n1[2] = (xn[2] - cm[2]) / rcton;
		  }
		  else if (direction_flag == 1) {
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
		    
		    if (around_flag) {
	          for (m=0; m<nlocal; m++){
	            if ((mask[m] & groupbit_mem) || (mask[m] & groupbit_prt)) {
		          if (m != n){
		            xm[0] = x[m][0];
		            xm[1] = x[m][1];
		            xm[2] = x[m][2];
		            domain->unmap(xm,atom->image[m]);
		        	rmton = sqrt((xn[0] - xm[0]) * (xn[0] - xm[0]) + (xn[1] - xm[1]) * (xn[1] - xm[1]) + (xn[2] - xm[2]) * (xn[2] - xm[2]));
		      	    if (rmton < Raround) {
		      	      nquat =  bonus[ellipsoid[m]].quat;
			  		  nquatinv[0] = nquat[0];
			  		  nquatinv[1] = - nquat[1];
			  		  nquatinv[2] = - nquat[2];
			  		  nquatinv[3] = - nquat[3];
			  		  MathExtra::quatquat(nquat, xyz, quatx);
			  		  MathExtra::quatquat(quatx, nquatinv, quatxquat);
			  		  n1[0] += quatxquat[1];
			  		  n1[1] += quatxquat[2];
			  		  n1[2] += quatxquat[3];
		      	    }
		          }
	            }
	          }
			  rmton = sqrt(n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2]);
			  n1[0] = n1[0] / rmton;
			  n1[1] = n1[1] / rmton;
			  n1[2] = n1[2] / rmton;
	        }
		  }
		  else if (direction_flag == 2) {
			n1[0] = dir_custom[0];
			n1[1] = dir_custom[1];
			n1[2] = dir_custom[2];
		  }
		  
	      f[n][0] += Fvalue*n1[0];
	      f[n][1] += Fvalue*n1[1];
	      f[n][2] += Fvalue*n1[2];
		  // fprintf(stderr, "In Fix protrusiveskeleton/local at proc %d when time = %d: the force on n = %d was %f %f %f\n", comm->me, update->ntimestep, n, f[n][0], f[n][1], f[n][2]);
	      nall[0] += n1[0];
	      nall[1] += n1[1];
	      nall[2] += n1[2];
	      
	      if (around_flag) {
	        for (m=0; m<nlocal; m++){
	          if ((mask[m] & groupbit_mem) || (mask[m] & groupbit_prt)){
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
		  
		  if (oppo_layer) {
		    count_in = 0;
	        count_out = 1;
		    count_tmp = 0;
		    bond_out_atom[0] = atom->tag[bond_skele];
	        for (nn = 0; nn < oppo_layer; nn++) {
			  for (i = 0; i < count_out; i++) {
			    bond_tmp_atom[i] = bond_out_atom[i];
			    bond_in_atom[count_in++] = bond_out_atom[i];
			  }
			  count_tmp = count_out;
			  count_out = 0;
			  for (i = 0; i < count_tmp; i++) {
			    ii = atom->map(bond_tmp_atom[i]);
			    for (j = 0; j < bond_count[ii]; j++) {
			  	  flag = 1;
			  	  for (k = 0; k < count_in; k++) {
			  	    if (bond_atom_full[ii][j] == bond_in_atom[k]) {
			  	  	  flag = 0;
			  	  	  break;
			  	    }
			  	  }
			  	  if (flag) {
			  	    for (k = 0; k < count_out; k++) {
			  	      if (bond_atom_full[ii][j] == bond_out_atom[k]) {
			  	    	flag = 0;
			  	    	break;
			  	      }
			  	    }
			  	  }
			  	  if (flag) {
			  	    // if (count_out < 100) fprintf(stderr, "In Fix protrusiveskeleton/local at proc %d when time = %d: i = %d, j = %d, count_tmp = %d, count_out = %d, bond_tmp_atom[i] = %d, bond_count_global[bond_tmp_atom[i]] = %d, bond_atom_global[bond_tmp_atom[i]][j] = %d \n", comm->me, update->ntimestep, i, j, count_tmp, count_out, bond_tmp_atom[i], bond_count_global[bond_tmp_atom[i]], bond_atom_global[bond_tmp_atom[i]][j]);
			  	    bond_out_atom[count_out++] = bond_atom_full[ii][j];
			  	  }
			    }
			  }
		    }
		    
	        
	        for (i = 0; i < count_out; i++) {
			  bond_bond_atom_local[bond_bond_count_local] = bond_out_atom[i];
		      bond_bond_fx_local[bond_bond_count_local] = nall[0] * Fvalue / count_out;
		      bond_bond_fy_local[bond_bond_count_local] = nall[1] * Fvalue / count_out;
		      bond_bond_fz_local[bond_bond_count_local] = nall[2] * Fvalue / count_out;
		      bond_bond_count_local++;
	        }
		  }
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		  break;
        }
	  }
	}
  }
  
  // fprintf(stderr, "In Fix protrusiveskeleton/local at proc %d when time = %d: bond_bond_count_local = %d\n", comm->me, update->ntimestep, bond_bond_count_local);
  // add the opposite force on the bondbond atoms
  MPI_Allreduce(&bond_bond_count_local,&bond_bond_count_all,1,MPI_INT,MPI_SUM,world);
  // fprintf(stderr, "In Fix protrusiveskeleton/local at proc %d when time = %d: bond_bond_count_all = %d\n", comm->me, update->ntimestep, bond_bond_count_all);
  memory->create(bond_bond_count_proc,nprocs,"protrusiveskeleton/local:bond_bond_count_proc");
  memory->create(displs,nprocs,"protrusiveskeleton/local:displs");
  if (nprocs > 1) MPI_Allgather(&bond_bond_count_local, 1, MPI_INT, bond_bond_count_proc, 1, MPI_INT, world);
  
  kn = 0;
  for (i = 0; i < nprocs; i++){
    displs[i] = kn;
    kn += bond_bond_count_proc[i];
  }
  // fprintf(stderr, "In Fix protrusiveskeleton/local at proc %d when time = %d: kn = %d\n", comm->me, update->ntimestep, kn);
  memory->create(bond_bond_atom_all,kn,"protrusiveskeleton/local:bond_bond_atom_all");
  memory->create(bond_bond_fx_all,kn,"protrusiveskeleton/local:bond_bond_fx_all");
  memory->create(bond_bond_fy_all,kn,"protrusiveskeleton/local:bond_bond_fy_all");
  memory->create(bond_bond_fz_all,kn,"protrusiveskeleton/local:bond_bond_fz_all");
  if (nprocs > 1) {
	MPI_Allgatherv(bond_bond_atom_local, bond_bond_count_local, MPI_INT, bond_bond_atom_all, bond_bond_count_proc, displs, MPI_INT, world);
	MPI_Allgatherv(bond_bond_fx_local, bond_bond_count_local, MPI_DOUBLE, bond_bond_fx_all, bond_bond_count_proc, displs, MPI_DOUBLE, world);
	MPI_Allgatherv(bond_bond_fy_local, bond_bond_count_local, MPI_DOUBLE, bond_bond_fy_all, bond_bond_count_proc, displs, MPI_DOUBLE, world);
	MPI_Allgatherv(bond_bond_fz_local, bond_bond_count_local, MPI_DOUBLE, bond_bond_fz_all, bond_bond_count_proc, displs, MPI_DOUBLE, world);
  }
  if (kn != bond_bond_count_all) fprintf(stderr, "In Fix protrusiveskeleton/local at proc %d when time = %d: kn = %d, bond_bond_count_all = %d\n", comm->me, update->ntimestep, kn, bond_bond_count_all);
  
  if (oppo_layer) {
    for (i = 0; i < nlocal; i++){
	  for (j = 0; j < bond_bond_count_all; j++) {
	    if (atom->tag[i] == bond_bond_atom_all[j]) {
	  	  f[i][0] -= bond_bond_fx_all[j];
	      f[i][1] -= bond_bond_fy_all[j];
	      f[i][2] -= bond_bond_fz_all[j];
	  	// if (nowtime % 1000 == 0) fprintf(stderr, "In Fix protrusiveskeleton/local at proc %d when time = %d: the opposite force was applied on atom %d\n", comm->me, update->ntimestep, atom->tag[i]);
	    }
	  }
    }
  }
  
  memory->destroy(bond_in_atom);
  memory->destroy(bond_out_atom);
  memory->destroy(bond_tmp_atom);
  memory->destroy(bond_bond_atom_local);
  memory->destroy(bond_bond_fx_local);
  memory->destroy(bond_bond_fy_local);
  memory->destroy(bond_bond_fz_local);
  memory->destroy(bond_bond_count_proc);
  memory->destroy(displs);
  memory->destroy(bond_bond_atom_all);
  memory->destroy(bond_bond_fx_all);
  memory->destroy(bond_bond_fy_all);
  memory->destroy(bond_bond_fz_all);
  
}

/* ---------------------------------------------------------------------- */

void FixProtrusiveSkeletonLocal::post_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

int FixProtrusiveSkeletonLocal::pack_forward_comm(int n, int *list, double *buf,
                                     int pbc_flag, int *pbc)
{
  int i,j,k,m;

  m = 0;

  for (i = 0; i < n; i++) {
    j = list[i];
	buf[m++] = ubuf(bond_count[j]).d;
	buf[m++] = ubuf(bond_change_count[j]).d;
	buf[m++] = ubuf(initial_time[j]).d;
	for (k = 0; k < maxbond_per_atom; k++) {
	  buf[m++] = ubuf(bond_atom_full[j][k]).d;
	  buf[m++] = ubuf(bond_atom_change[j][k]).d;
      //if (bond_count[j] != 0) fprintf(stderr, "In fix_skele_create_break pack_forward_comm() when timestep = %d at proc %d: the bond_count of ghost atom %d is %d, bond_atom_full[j][%d] = %d, maxbond_per_atom = %d.\n", update->ntimestep, comm->me, atom->tag[j], bond_count[j], k, bond_atom_full[j][k], maxbond_per_atom);
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixProtrusiveSkeletonLocal::unpack_forward_comm(int n, int first, double *buf)
{
  int i,j,k,m,last;
  int nlocal = atom->nlocal;

  m = 0;
  last = first + n;

  for (i = first; i < last; i++){
	bond_count[i] = (int) ubuf(buf[m++]).i;
	bond_change_count[i] = (int) ubuf(buf[m++]).i;
	initial_time[i] = (int) ubuf(buf[m++]).i;
	for (k = 0; k < maxbond_per_atom; k++) {
      bond_atom_full[i][k] = (int) ubuf(buf[m++]).i;
	  bond_atom_change[i][k] = (int) ubuf(buf[m++]).i;
	}
  }
}

/* ---------------------------------------------------------------------- */

int FixProtrusiveSkeletonLocal::pack_reverse_comm(int n, int first, double *buf)
{
  int i,k,m,last;

  m = 0;
  last = first + n;

  for (i = first; i < last; i++){
	buf[m++] = ubuf(bond_change_count[i]).d;
	for (k = 0; k < maxbond_per_atom; k++) {
	  buf[m++] = ubuf(bond_atom_change[i][k]).d;
	}
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixProtrusiveSkeletonLocal::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,k,m,ii,jj,bond_change_count_tmp,bond_atom_change_tmp,bond_sub_count_tmp,bond_add_count_tmp,bond_full_tmp,bond_tmp;
  int nlocal = atom->nlocal;
  
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  
  m = 0;

  for (i = 0; i < n; i++) {
    j = list[i];
    bond_change_count_tmp = (int) ubuf(buf[m++]).i;
	
	// if the atom receives the ghost info is a local atom
	if (j < nlocal) {
	  // receive change info from ghost atoms
	  for (k = 0; k < maxbond_per_atom; k++) {
		// append changed atoms into bond_atom_full list
	    bond_atom_change_tmp = (int) ubuf(buf[m++]).i;
	    if (k < bond_change_count_tmp) {
	      bond_atom_full[j][bond_count[j]] = bond_atom_change_tmp;
	      bond_count[j]++;
	    }
	  }
	  bond_change_count[j] = 0;
	}
	// if the atom receives the ghost info is a ghost atom
	else {
	  // receive change info from ghost atoms
	  for (k = 0; k < maxbond_per_atom; k++) {
	    // append changed info into bond_atom_change list
		bond_atom_change_tmp = (int) ubuf(buf[m++]).i;
	    if (k < bond_change_count_tmp) {
	      bond_atom_change[j][bond_change_count[j]] = bond_atom_change_tmp;
	      bond_change_count[j]++;
	    }
	  }
	}
  }
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixProtrusiveSkeletonLocal::pack_exchange(int i, double *buf)
{
  int m, k;
  
  m = 0;
  buf[m++] = initial_time[i];
  return m;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based arrays from exchange with another proc
------------------------------------------------------------------------- */

int FixProtrusiveSkeletonLocal::unpack_exchange(int nlocal, double *buf)
{
  int m, k;
  
  m = 0;
  initial_time[nlocal] = static_cast<int> (buf[m++]);
  return m;
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

/* void FixProtrusiveSkeletonLocal::grow_arrays(int nmax)
{
  memory->grow(bond_count,nmax,"protrusiveskeleton/local:bond_count");
  memory->grow(bond_change_count,nmax,"protrusiveskeleton/local:bond_change_count");
  memory->grow(bond_atom_full,nmax,maxbond_per_atom,"protrusiveskeleton/local:bond_atom_full");
  memory->grow(bond_atom_change,nmax,maxbond_per_atom,"protrusiveskeleton/local:bond_atom_change");
} */

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays
------------------------------------------------------------------------- */

/* void FixProtrusiveSkeletonLocal::copy_arrays(int i, int j, int delflag)
{
  int k;
  
  bond_count[j] = bond_count[i];
  bond_change_count[j] = bond_change_count[i];
  for (k = 0; k < maxbond_per_atom; k++) {
	bond_atom_full[j][k] = bond_atom_full[i][k];
	bond_atom_change[j][k] = bond_atom_change[i][k];
  }
} */

/* ---------------------------------------------------------------------- */

void FixProtrusiveSkeletonLocal::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   potential energy of added force
------------------------------------------------------------------------- */

double FixProtrusiveSkeletonLocal::compute_scalar()
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

double FixProtrusiveSkeletonLocal::compute_vector(int n)
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

/* double FixProtrusiveSkeletonLocal::memory_usage()
{
  double bytes = 0.0;
  bytes = 2 * maxatom * sizeof(int);
  bytes += 2 * maxatom * maxbond_per_atom * sizeof(int);
  return bytes;
} */