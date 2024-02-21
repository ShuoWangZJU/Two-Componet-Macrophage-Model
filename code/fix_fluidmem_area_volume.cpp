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
#include <stdlib.h>
#include <string.h>
#include "fix_fluidmem_area_volume.h"
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
#define BIG 1.0e10
#define OVERLAP 0.2

/* ---------------------------------------------------------------------- */

FixFluidMemAreaVolume::FixFluidMemAreaVolume(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  int arg_min = 11;
  if (narg != arg_min) error->all(FLERR,"Illegal fix fluidmem/area/volume command");

  // arg      0           1                  2                 3       4        5               6               7                   8               9            10
  // fix   fix_ID   membrane_group   fluidmem/area/volume   N_near   d_max   k_area   total/average_area   desired_area   total/average_volume   k_volume   desired_volume
  
  if (!atom->rmass_flag)
    error->all(FLERR,"Fix fluidmem/area/volume requires atom attributes radius, rmass");

  // required args
  int m = 3;
  N_near = force->inumeric(FLERR,arg[m++]);
  d_max = force->numeric(FLERR,arg[m++]);
  ka = force->numeric(FLERR,arg[m++]);
  if (strcmp(arg[m],"total_area") == 0) {
	average_area_flag = 0;
	m++;
	a0 = force->numeric(FLERR,arg[m++]);
  }
  else if (strcmp(arg[m],"average_area") == 0) {
	average_area_flag = 1;
	m++;
	a0_ave = force->numeric(FLERR,arg[m++]);
  }
  else error->all(FLERR,"Illegal fix fluidmem/area/volume command");
  
  
  kv = force->numeric(FLERR,arg[m++]);
  if (strcmp(arg[m],"total_volume") == 0) {
	average_volume_flag = 0;
	m++;
	v0 = force->numeric(FLERR,arg[m++]);
  }
  else if (strcmp(arg[m],"average_volume") == 0) {
	average_volume_flag = 1;
	m++;
	v0_ave = force->numeric(FLERR,arg[m++]);
  }
  else error->all(FLERR,"Illegal fix fluidmem/area/volume command");
  
  neigh_flag = 0;
  in_plane_num = 0;
  
  comm_forward = 1;
  comm_reverse = 1;
}

/* ---------------------------------------------------------------------- */

FixFluidMemAreaVolume::~FixFluidMemAreaVolume()
{
  memory->destroy(group_atom_ID);
  memory->destroy(nearby_atom_ID);
  memory->destroy(nearby_atom_num);
  
  memory->destroy(tricount);
}

/* ---------------------------------------------------------------------- */

int FixFluidMemAreaVolume::setmask()
{
  int mask = 0;
  mask |= PRE_NEIGHBOR;
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

/* void FixFluidMemAreaVolume::init()
{
  if (domain->triclinic)
    error->all(FLERR,"Cannot use fix fluidmem/area/volume with triclinic box");
  
  int i, j, k, m;
  int near_num;
  double xi[3], xj[3], dij;
  double *d_min_tmp;
  double **x = atom->x;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;
  nlocal_group = 0;
  int nlocal_group_now = 0;
  
  comm->forward_comm();
  
  for (i = 0; i < nlocal; i++) {
	if (mask[i] & groupbit) {
	  nlocal_group++;
	}
  }
  memory->create(group_atom_ID,nlocal_group,"fix_fluidmem_area_volume:group_atom_ID");
  memory->create(nearby_atom_ID,nlocal_group,N_near,"fix_fluidmem_area_volume:nearby_atom_ID");
  memory->create(d_min_tmp,N_near,"fix_fluidmem_area_volume:d_min_tmp");
  
  nlocal_group_now = 0;
  for (i = 0; i < nlocal; i++) {
	if (mask[i] & groupbit) {
	  group_atom_ID[nlocal_group_now] = i;
	  xi[0] = x[i][0];
	  xi[1] = x[i][1];
	  xi[2] = x[i][2];
	  for (k = 0; k < N_near; k++) {
		nearby_atom_ID[nlocal_group_now][k] = -1;
		d_min_tmp[k] = BIG;
	  }
	  near_num = 0;
	  for (j = 0; j < nall; j++) {
		if (mask[j] & groupbit) {
		  xj[0] = x[j][0];
		  xj[1] = x[j][1];
		  xj[2] = x[j][2];
		  dij = sqrt((xi[0] - xj[0]) * (xi[0] - xj[0]) + (xi[1] - xj[1]) * (xi[1] - xj[1]) + (xi[2] - xj[2]) * (xi[2] - xj[2]));
		  if (dij > d_max) continue;
		  for (k = 0; k < N_near; k++) {
			if (dij < d_min_tmp[k]) {
			  if (nearby_atom_ID[nlocal_group_now][N_near - 1] < 0) near_num++;
			  for (m = N_near - 1; m > k; m--) {
				nearby_atom_ID[nlocal_group_now][m] = nearby_atom_ID[nlocal_group_now][m-1];
				d_min_tmp[m] = d_min_tmp[m-1];
			  }
			  nearby_atom_ID[nlocal_group_now][k] = j;
			  d_min_tmp[k] = dij;
			  break;
			}
		  }
		}
	  }
	  // if (near_num > N_near) error->all(FLERR,"Error when building nearby atom list in fix fluidmem/area/volume: near_num = %d, tag[i] = %d", near_num, tag[i]);
	  if (near_num > N_near) error->all(FLERR,"Error when building nearby atom list in fix fluidmem/area/volume");
	  if (near_num < N_near) fprintf(stderr, "In fix fluidmem/area/volume at proc %d when time = %d: near_num = %d, tag[i] = %d\n", comm->me, update->ntimestep, near_num, tag[i]);
	  for (k = 0; k < N_near; k++) {
		// if (d_min_tmp[k] > d_max) error->all(FLERR,"Error when building nearby atom list: d_min_tmp[k] = %f, k = %d, tag[i] = %d", d_min_tmp[k], k, tag[i]);
		// fprintf(stderr, "In fix fluidmem/area/volume at proc %d when time = %d: i = %d, tag[i], nearby_atom_ID[nlocal_group_now][k] = %d, tag[j] = %d\n", comm->me, update->ntimestep, i, tag[i], nearby_atom_ID[nlocal_group_now][k], tag[nearby_atom_ID[nlocal_group_now][k]]);
		if (d_min_tmp[k] > d_max || nearby_atom_ID[nlocal_group_now][k] < 0) error->all(FLERR,"Error when building nearby atom list");
	  }
	  nlocal_group_now++;
	}
  }
  
  memory->destroy(d_min_tmp);
} */

void FixFluidMemAreaVolume::init()
{
  if (domain->triclinic)
    error->all(FLERR,"Cannot use fix fluidmem/area/volume with triclinic box");
  
  int i;
  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  nlocal_group = 0;
  
  for (i = 0; i < nlocal; i++) {
	if (mask[i] & groupbit) {
	  nlocal_group++;
	}
  }
  memory->create(group_atom_ID,nlocal_group,"fix_fluidmem_area_volume:group_atom_ID");
  memory->create(nearby_atom_ID,nlocal_group,N_near,"fix_fluidmem_area_volume:nearby_atom_ID");
  memory->create(nearby_atom_num,nlocal_group,"fix_fluidmem_area_volume:nearby_atom_num");
  
  memory->create(tricount,atom->nmax,"fix_fluidmem_area_volume:tricount");
  fp = fopen("nearby_atom.txt","w");
  neigh_flag = 0;
}

/* ----------------------------------------------------------------------
   find the closest N_near atoms
------------------------------------------------------------------------- */

void FixFluidMemAreaVolume::pre_neighbor()
{
  int i, j, k, m;
  double xi[3], xj[3], dij;
  double *d_min_tmp;
  double **x = atom->x;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;
  int nlocal_group_now = 0;
  int in_plane_num_local = 0;
  
  for (i = 0; i < nlocal; i++) {
	if (mask[i] & groupbit) {
	  nlocal_group_now++;
	}
	tricount[i] = 0;
  }
  //for (i = 0; i < nall; i++) {
	//tricount[i] = 0;
  //}
  comm->forward_comm_fix(this);
  if (nlocal_group_now > nlocal_group) {
	memory->destroy(group_atom_ID);
	memory->destroy(nearby_atom_ID);
	memory->destroy(nearby_atom_num);
	memory->create(group_atom_ID,nlocal_group_now,"fix_fluidmem_area_volume:group_atom_ID");
    memory->create(nearby_atom_ID,nlocal_group_now,N_near,"fix_fluidmem_area_volume:nearby_atom_ID");
    memory->create(nearby_atom_num,nlocal_group_now,"fix_fluidmem_area_volume:nearby_atom_num");
  }
  memory->create(d_min_tmp,N_near,"fix_fluidmem_area_volume:d_min_tmp");
  
  nlocal_group_now = 0;
  in_plane_num = 0;
  for (i = 0; i < nlocal; i++) {
	if (mask[i] & groupbit) {
	  group_atom_ID[nlocal_group_now] = i;
	  xi[0] = x[i][0];
	  xi[1] = x[i][1];
	  xi[2] = x[i][2];
	  nearby_atom_num[nlocal_group_now] = 0;
	  for (k = 0; k < N_near; k++) {
		nearby_atom_ID[nlocal_group_now][k] = -1;
		d_min_tmp[k] = BIG;
	  }
	  for (j = 0; j < nall; j++) {
		if ((mask[j] & groupbit) && j != i) {
		  xj[0] = x[j][0];
		  xj[1] = x[j][1];
		  xj[2] = x[j][2];
		  dij = sqrt((xi[0] - xj[0]) * (xi[0] - xj[0]) + (xi[1] - xj[1]) * (xi[1] - xj[1]) + (xi[2] - xj[2]) * (xi[2] - xj[2]));
		  if (dij > d_max) continue;
		  for (k = 0; k < N_near; k++) {
			if (dij < d_min_tmp[k]) {
			  if (nearby_atom_ID[nlocal_group_now][N_near - 1] < 0) nearby_atom_num[nlocal_group_now]++;
			  for (m = N_near - 1; m > k; m--) {
				nearby_atom_ID[nlocal_group_now][m] = nearby_atom_ID[nlocal_group_now][m-1];
				d_min_tmp[m] = d_min_tmp[m-1];
			  }
			  nearby_atom_ID[nlocal_group_now][k] = j;
			  d_min_tmp[k] = dij;
			  break;
			}
		  }
		}
	  }
	  for (k = 0; k < nearby_atom_num[nlocal_group_now]; k++) {
		j = nearby_atom_ID[nlocal_group_now][k];
		tricount[i]++;
		tricount[j] = tricount[j] + 2;
	  }
	  // if (nearby_atom_num[nlocal_group_now] > N_near) error->all(FLERR,"Error when building nearby atom list in fix fluidmem/area/volume: nearby_atom_num[nlocal_group_now] = %d, tag[i] = %d", nearby_atom_num[nlocal_group_now], tag[i]);
	  if (nearby_atom_num[nlocal_group_now] > N_near) error->all(FLERR,"Error when building nearby atom list in fix fluidmem/area/volume");
	  if (nearby_atom_num[nlocal_group_now] > 2) {
	    if (nearby_atom_num[nlocal_group_now] < N_near) fprintf(stderr, "In fix fluidmem/area/volume at proc %d when time = %d: nearby_atom_num[nlocal_group_now] = %d, tag[i] = %d\n", comm->me, update->ntimestep, nearby_atom_num[nlocal_group_now], tag[i]);
	    for (k = 0; k < nearby_atom_num[nlocal_group_now]; k++) {
		  // if (d_min_tmp[k] > d_max) error->all(FLERR,"Error when building nearby atom list: d_min_tmp[k] = %f, k = %d, tag[i] = %d", d_min_tmp[k], k, tag[i]);
		  // fprintf(stderr, "In fix fluidmem/area/volume at proc %d when time = %d: i = %d, tag[i] = %d, nearby_atom_ID[nlocal_group_now][k] = %d, tag[j] = %d\n", comm->me, update->ntimestep, i, tag[i], nearby_atom_ID[nlocal_group_now][k], tag[nearby_atom_ID[nlocal_group_now][k]]);
		  if (d_min_tmp[k] > d_max || nearby_atom_ID[nlocal_group_now][k] < 0) error->all(FLERR,"Error when building nearby atom list");
	    }
		in_plane_num_local++;
	  }
	  else {
		nearby_atom_num[nlocal_group_now] = 0;
		for (k = 0; k < N_near; k++) {
		  nearby_atom_ID[nlocal_group_now][k] = -1;
		  d_min_tmp[k] = BIG;
		}
	  }
	  nlocal_group_now++;
	}
  }
  comm->reverse_comm_fix(this);
  int *type = atom->type;
  for (i = 0; i < nlocal; i++) {
	if (mask[i] & groupbit) {
	  j = tricount[i] - 9;
	  if (j < 4) j = 4;
	  if (j > 14) j = 14;
	  // type[i] = j;
	}
  }
  nlocal_group = nlocal_group_now;
  neigh_flag = 1;
  
  int nlocal_group_all;
  MPI_Allreduce(&in_plane_num_local,&in_plane_num,1,MPI_INT,MPI_SUM,world);
  MPI_Allreduce(&nlocal_group,&nlocal_group_all,1,MPI_INT,MPI_SUM,world);
  if (comm->me == 0) fprintf(stderr, "In fix fluidmem/area/volume pre_neighbor() when time = %d: in_plane_num = %d, nlocal_group_all = %d\n", update->ntimestep, in_plane_num, nlocal_group_all);
  
  if (average_area_flag) a0 = in_plane_num * a0_ave;
  if (average_volume_flag) v0 = in_plane_num * v0_ave;
  
  memory->destroy(d_min_tmp);
}

/* ----------------------------------------------------------------------
   apply force
------------------------------------------------------------------------- */

void FixFluidMemAreaVolume::pre_force(int vflag)
{
  int i, j, k, m, n, tmp, i1, i2, i3;
  double di0, rji_dot_ri_nor,rji_0,rji_m, ri_nor_dot_rnor_0m, angle_tmp;
  double d21x,d21y,d21z,d31x,d31y,d31z,d32x,d32y,d32z;
  double nx,ny,nz,nn,mx,my,mz,aa,vv;
  double area_local, volume_local, area_all, volume_all, coefa, coefv;
  double ri[3], ri_nor[3], rnor_0m[3], xx1[3],xx2[3],xx3[3], aa0[3][3],aa1[3][3],aa2[3][3], dir_ave[3], f_tmp[3];
  double fa[3][3], fv[3][3];
  double *angle_j, *nquati0, *nquati1, *nquati2;
  double **rji, **rji_prj;
  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  tagint *tag = atom->tag;
  AtomVecEllipsoid *avec_ellipsoid = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
  int *ellipsoid = atom->ellipsoid;
  int newton_bond = force->newton_bond;
  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;
  int nlocal_group_now = 0;
  
  if (!neigh_flag) return;
  
  memory->create(angle_j,N_near,"fix_fluidmem_area_volume:angle_j");
  memory->create(rji,N_near,3,"fix_fluidmem_area_volume:rji");
  memory->create(rji_prj,N_near,3,"fix_fluidmem_area_volume:rji_prj");
  
  // if (update->ntimestep > 187359) fprintf(stderr, "In fix fluidmem/area/volume at proc %d when time = %d: j = %d, i = %d, tag[i] = %d, nlocal = %d\n", comm->me, update->ntimestep, j, i, tag[i], nlocal);
  
  // find the order of the nearby atoms
  for (i = 0; i < nlocal; i++) {
	if (mask[i] & groupbit) {
	  k = -1;
	  for (m = 0; m < nlocal_group; m++) {
		if (group_atom_ID[m] == i) {
		  k = m;
		  break;
		}
	  }
	  if (k < 0) error->all(FLERR,"Couldn't find the group_atom_ID in fix fluidmem/area/volume");
	  
	  if (nearby_atom_num[k] < 3) continue;
	  
	  di0 = sqrt(x[i][0] * x[i][0] + x[i][1] * x[i][1] + x[i][2] * x[i][2]);
	  ri_nor[0] = x[i][0] / di0;    // the normal vector from zero point to i
	  ri_nor[1] = x[i][1] / di0;
	  ri_nor[2] = x[i][2] / di0;
	  if (di0 < EPSILON) fprintf(stderr, "In fix fluidmem/area/volume at proc %d when time = %d: di0 = %f, ri_nor[0] = %f, ri_nor[1] = %f, ri_nor[2] = %f\n", comm->me, update->ntimestep, di0, ri_nor[0], ri_nor[1], ri_nor[2]);
	  
	  for (m = 0; m < nearby_atom_num[k]; m++) {
		j = nearby_atom_ID[k][m];
		// fprintf(stderr, "In fix fluidmem/area/volume at proc %d when time = %d: j = %d, i = %d, tag[i] = %d, nlocal = %d\n", comm->me, update->ntimestep, j, i, tag[i], nlocal);
		rji[m][0] = x[j][0] - x[i][0];
		rji[m][1] = x[j][1] - x[i][1];
		rji[m][2] = x[j][2] - x[i][2];
		rji_dot_ri_nor = rji[m][0] * ri_nor[0] + rji[m][1] * ri_nor[1] + rji[m][2] * ri_nor[2];
		rji_prj[m][0] = rji[m][0] - rji_dot_ri_nor * ri_nor[0];
		rji_prj[m][1] = rji[m][1] - rji_dot_ri_nor * ri_nor[1];
		rji_prj[m][2] = rji[m][2] - rji_dot_ri_nor * ri_nor[2];
		if (m == 0) {
		  angle_j[m] = 1.0;
		  rji_0 = sqrt(rji_prj[m][0] * rji_prj[m][0] + rji_prj[m][1] * rji_prj[m][1] + rji_prj[m][2] * rji_prj[m][2]);
		}
		else {
		  rji_m = sqrt(rji_prj[m][0] * rji_prj[m][0] + rji_prj[m][1] * rji_prj[m][1] + rji_prj[m][2] * rji_prj[m][2]);
		  angle_j[m] = (rji_prj[0][0] * rji_prj[m][0] + rji_prj[0][1] * rji_prj[m][1] + rji_prj[0][2] * rji_prj[m][2]) / rji_0 / rji_m;
		  if (rji_0 < EPSILON || rji_m < EPSILON) fprintf(stderr, "In fix fluidmem/area/volume at proc %d when time = %d: rji_0 = %f, rji_m = %f, angle_j[m] = %f\n", comm->me, update->ntimestep, rji_0, rji_m, angle_j[m]);
		  if (angle_j[m] != 1.0 && angle_j[m] != -1.0) {
			rnor_0m[0] = rji_prj[0][1] * rji_prj[m][2] - rji_prj[0][2] * rji_prj[m][1];
			rnor_0m[1] = rji_prj[0][2] * rji_prj[m][0] - rji_prj[0][0] * rji_prj[m][2];
			rnor_0m[2] = rji_prj[0][0] * rji_prj[m][1] - rji_prj[0][1] * rji_prj[m][0];
			ri_nor_dot_rnor_0m = ri_nor[0] * rnor_0m[0] + ri_nor[1] * rnor_0m[1] + ri_nor[2] * rnor_0m[2];
			if (ri_nor_dot_rnor_0m < 0) angle_j[m] = -2.0 - angle_j[m];
		  }
		}
	  }
	  for (m = 0; m < nearby_atom_num[k] - 1; m++) {
		for (n = 0; n < nearby_atom_num[k] - 1 - m; n++) {
		  if (angle_j[n] < angle_j[n+1]) {
			angle_tmp = angle_j[n+1];
			angle_j[n+1] = angle_j[n];
			angle_j[n] = angle_tmp;
			tmp = nearby_atom_ID[k][n+1];
			nearby_atom_ID[k][n+1] = nearby_atom_ID[k][n];
			nearby_atom_ID[k][n] = tmp;
		  }
		}
	  }
	}
  }
  
  // caculate area and volume
  area_local = 0.0;
  volume_local = 0.0;
  for (i = 0; i < nlocal; i++) {
	if (mask[i] & groupbit) {
	  k = -1;
	  for (m = 0; m < nlocal_group; m++) {
		if (group_atom_ID[m] == i) {
		  k = m;
		  break;
		}
	  }
	  if (k < 0) error->all(FLERR,"Couldn't find the group_atom_ID in fix fluidmem/area/volume");
	  
	  if (nearby_atom_num[k] < 3) continue;
	  
	  for (m = 0; m < nearby_atom_num[k]; m++) {
		i1 = i;
		i2 = nearby_atom_ID[k][m];
		if (m < nearby_atom_num[k] - 1) i3 = nearby_atom_ID[k][m+1];
		else i3 = nearby_atom_ID[k][0];
		
		// 2-1 distance
		d21x = x[i2][0] - x[i1][0];
		d21y = x[i2][1] - x[i1][1];
		d21z = x[i2][2] - x[i1][2];
		
		// 3-1 distance
		d31x = x[i3][0] - x[i1][0];
		d31y = x[i3][1] - x[i1][1];
		d31z = x[i3][2] - x[i1][2];
		
		// 3-2 distance
		d32x = x[i3][0] - x[i2][0];
		d32y = x[i3][1] - x[i2][1];
		d32z = x[i3][2] - x[i2][2];
		
		// calculate normal
		nx = d21y*d31z - d31y*d21z;
		ny = d31x*d21z - d21x*d31z;
		nz = d21x*d31y - d31x*d21y;
		nn = sqrt(nx*nx + ny*ny + nz*nz);
		
		// calculate center
		domain->unmap(x[i1],atom->image[i1],xx1);
		domain->unmap(x[i2],atom->image[i2],xx2);
		domain->unmap(x[i3],atom->image[i3],xx3);
		
		mx =  xx1[0] + xx2[0] + xx3[0];
		my =  xx1[1] + xx2[1] + xx3[1];
		mz =  xx1[2] + xx2[2] + xx3[2];
		
		nquati0 = avec_ellipsoid->bonus[ellipsoid[i1]].quat;
		nquati1 = avec_ellipsoid->bonus[ellipsoid[i2]].quat;
		nquati2 = avec_ellipsoid->bonus[ellipsoid[i3]].quat;
		MathExtra::quat_to_mat_trans(nquati0,aa0);
		MathExtra::quat_to_mat_trans(nquati1,aa1);
		MathExtra::quat_to_mat_trans(nquati2,aa2);
		dir_ave[0] = aa0[0][0] + aa1[0][0] + aa2[0][0];
		dir_ave[1] = aa0[0][1] + aa1[0][1] + aa2[0][1];
		dir_ave[2] = aa0[0][2] + aa1[0][2] + aa2[0][2];
		angle_tmp = dir_ave[0] * mx + dir_ave[1] * my + dir_ave[2] * mz;
		if (angle_tmp < 0) {
		  nx = - nx;
		  ny = - ny;
		  nz = - nz;
		}
		
		// calculate area and volume
		aa = 0.5*nn;
		vv = (nx*mx + ny*my + nz*mz)/18.0;
		area_local += aa;
		volume_local += vv;
		// fprintf(stderr, "In fix fluidmem/area/volume at proc %d when time = %d: aa = %f, vv = %f, i1 = %d, i2 = %d, i3 = %d\n", comm->me, update->ntimestep, aa, vv, i1, i2, i3);
	  }
	}
  }
  MPI_Allreduce(&area_local,&area_all,1,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(&volume_local,&volume_all,1,MPI_DOUBLE,MPI_SUM,world);
  area_all = area_all / 3.0;
  volume_all = volume_all / 3.0;
  // if (!(area_all < 100000) || !(volume_all < 1000000)) fprintf(stderr, "In fix fluidmem/area/volume when time = %d: total area is %f, volume is %f, desired is %f %f\n", update->ntimestep, area_all, volume_all, a0, v0);
  if (comm->me == 0 && update->ntimestep % 1000 == 0) {
	fprintf(stderr, "In fix fluidmem/area/volume when time = %d: total area is %f, volume is %f, desired is %f %f\n", update->ntimestep, area_all, volume_all, a0, v0);
	fprintf(fp, "%d %f\n", update->ntimestep, area_all);
  }
  
//  if (update->ntimestep % 5000 == 0 && comm->me == 0) {
//	fprintf(fp, "TIMESTEP = %d\n", update->ntimestep);
//	fprintf(fp, "\n");
//  }
  // apply force
  for (i = 0; i < nlocal; i++) {
	if (mask[i] & groupbit) {
	  k = -1;
	  for (m = 0; m < nlocal_group; m++) {
		if (group_atom_ID[m] == i) {
		  k = m;
		  break;
		}
	  }
	  if (k < 0) error->all(FLERR,"Couldn't find the group_atom_ID in fix fluidmem/area/volume");
	  
	  if (nearby_atom_num[k] < 3) continue;
	  
	  f_tmp[0] = 0.0;
	  f_tmp[1] = 0.0;
	  f_tmp[2] = 0.0;
	  for (m = 0; m < nearby_atom_num[k]; m++) {
		i1 = i;
		i2 = nearby_atom_ID[k][m];
		if (m < nearby_atom_num[k] - 1) i3 = nearby_atom_ID[k][m+1];
		else i3 = nearby_atom_ID[k][0];
		
		// 2-1 distance
		d21x = x[i2][0] - x[i1][0];
		d21y = x[i2][1] - x[i1][1];
		d21z = x[i2][2] - x[i1][2];
		
		// 3-1 distance
		d31x = x[i3][0] - x[i1][0];
		d31y = x[i3][1] - x[i1][1];
		d31z = x[i3][2] - x[i1][2];
		
		// 3-2 distance
		d32x = x[i3][0] - x[i2][0];
		d32y = x[i3][1] - x[i2][1];
		d32z = x[i3][2] - x[i2][2];
		
		// calculate normal
		nx = d21y*d31z - d31y*d21z;
		ny = d31x*d21z - d21x*d31z;
		nz = d21x*d31y - d31x*d21y;
		nn = sqrt(nx*nx + ny*ny + nz*nz);
		
		// calculate center
		domain->unmap(x[i1],atom->image[i1],xx1);
		domain->unmap(x[i2],atom->image[i2],xx2);
		domain->unmap(x[i3],atom->image[i3],xx3);
		
		mx =  xx1[0] + xx2[0] + xx3[0];
		my =  xx1[1] + xx2[1] + xx3[1];
		mz =  xx1[2] + xx2[2] + xx3[2];
		
		nquati0 = avec_ellipsoid->bonus[ellipsoid[i1]].quat;
		nquati1 = avec_ellipsoid->bonus[ellipsoid[i2]].quat;
		nquati2 = avec_ellipsoid->bonus[ellipsoid[i3]].quat;
		MathExtra::quat_to_mat_trans(nquati0,aa0);
		MathExtra::quat_to_mat_trans(nquati1,aa1);
		MathExtra::quat_to_mat_trans(nquati2,aa2);
		dir_ave[0] = aa0[0][0] + aa1[0][0] + aa2[0][0];
		dir_ave[1] = aa0[0][1] + aa1[0][1] + aa2[0][1];
		dir_ave[2] = aa0[0][2] + aa1[0][2] + aa2[0][2];
		angle_tmp = dir_ave[0] * mx + dir_ave[1] * my + dir_ave[2] * mz;
		if (angle_tmp < 0) {
		  nx = - nx;
		  ny = - ny;
		  nz = - nz;
		}
		
		// calculate area and volume
		aa = 0.5*nn;
		vv = (nx*mx + ny*my + nz*mz)/18.0;
		
		if (nn > EPSILON) {
		  coefa = 0.5 * ka * (a0 - area_all) / a0 / nn;
		}
		else {
		  coefa = 0.0;
		}
		coefv = kv * (v0 - volume_all) / v0 / 18.0;
		
		if (a0 < EPSILON || nn < EPSILON) fprintf(stderr, "In fix fluidmem/area/volume at proc %d when time = %d: a0 = %f, nn = %f, coefa = %f, tag[i1] = %d, x[i1][0] = %f, x[i1][1] = %f, x[i1][2] = %f, tag[i2] = %d, x[i2][0] = %f, x[i2][1] = %f, x[i2][2] = %f, tag[i3] = %d, x[i3][0] = %f, x[i3][1] = %f, x[i3][2] = %f\n", comm->me, update->ntimestep, a0, nn, coefa, tag[i1], x[i1][0], x[i1][1], x[i1][2], tag[i2], x[i2][0], x[i2][1], x[i2][2], tag[i3], x[i3][0], x[i3][1], x[i3][2]);
		
		// calculate force for local and global area constraint
		fa[0][0] = coefa*(ny*d32z - nz*d32y);
		fa[0][1] = coefa*(nz*d32x - nx*d32z);    
		fa[0][2] = coefa*(nx*d32y - ny*d32x);
		fa[1][0] = coefa*(nz*d31y - ny*d31z);
		fa[1][1] = coefa*(nx*d31z - nz*d31x);
		fa[1][2] = coefa*(ny*d31x - nx*d31y);
		fa[2][0] = coefa*(ny*d21z - nz*d21y);
		fa[2][1] = coefa*(nz*d21x - nx*d21z);
		fa[2][2] = coefa*(nx*d21y - ny*d21x);
		
		// calculate force for volume constraint
		fv[0][0] = coefv*(nx + d32z*my - d32y*mz);
		fv[0][1] = coefv*(ny - d32z*mx + d32x*mz);    
		fv[0][2] = coefv*(nz + d32y*mx - d32x*my);
		fv[1][0] = coefv*(nx - d31z*my + d31y*mz);
		fv[1][1] = coefv*(ny + d31z*mx - d31x*mz);
		fv[1][2] = coefv*(nz - d31y*mx + d31x*my);
		fv[2][0] = coefv*(nx + d21z*my - d21y*mz);
		fv[2][1] = coefv*(ny - d21z*mx + d21x*mz);
		fv[2][2] = coefv*(nz + d21y*mx - d21x*my);
		
		f_tmp[0] += fa[0][0];
		f_tmp[1] += fa[0][1];
		f_tmp[2] += fa[0][2];
		
		// apply force to each of 3 atoms
		if (newton_bond || i1 < nlocal) {
		  f[i1][0] += fa[0][0]+fv[0][0];
		  f[i1][1] += fa[0][1]+fv[0][1];
		  f[i1][2] += fa[0][2]+fv[0][2];
		}
		
		if (newton_bond || i2 < nlocal) {
		  f[i2][0] += fa[1][0]+fv[1][0];
		  f[i2][1] += fa[1][1]+fv[1][1];
		  f[i2][2] += fa[1][2]+fv[1][2];
		}
		
		if (newton_bond || i3 < nlocal) {
		  f[i3][0] += fa[2][0]+fv[2][0];
		  f[i3][1] += fa[2][1]+fv[2][1];
		  f[i3][2] += fa[2][2]+fv[2][2];
		}
	  }
//	  if (update->ntimestep % 5000 == 0) {
//		fprintf(fp, "%d %d %d", tag[i], nearby_atom_num[k], tricount[i]);
//		for (m = 0; m < nearby_atom_num[k]; m++) {
//		  j = nearby_atom_ID[k][m];
//		  fprintf(fp, " %d", tag[j]);
//		}
//		for (m = 0; m < 3; m++) {
//		  fprintf(fp, " %f", f_tmp[m]);
//		}
//		fprintf(fp, "\n");
//	  }
	}
  }
  
//  if (update->ntimestep % 5000 == 0 && comm->me == 0) {
//	fprintf(fp, "\n");
//  }
  
  memory->destroy(angle_j);
  memory->destroy(rji);
  memory->destroy(rji_prj);
}

int FixFluidMemAreaVolume::pack_forward_comm(int n, int *list, double *buf,
                                     int pbc_flag, int *pbc)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = ubuf(tricount[j]).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixFluidMemAreaVolume::unpack_forward_comm(int n, int first, double *buf)
{
  int i,j,m,ns,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++)
    tricount[i] = (int) ubuf(buf[m++]).i;
}

/* ---------------------------------------------------------------------- */

int FixFluidMemAreaVolume::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;

  for (i = first; i < last; i++) {
    buf[m++] = ubuf(tricount[i]).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixFluidMemAreaVolume::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    tricount[j] += (int) ubuf(buf[m++]).i;
  }
}