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
#define OVERLAP 0.2

/* ---------------------------------------------------------------------- */

FixAddLipid::FixAddLipid(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  int arg_min = 14;
  if (narg != arg_min && narg != arg_min + 2) error->all(FLERR,"Illegal fix addlipid command");

  if (!atom->rmass_flag)
    error->all(FLERR,"Fix addlipid requires atom attributes radius, rmass");

  // required args
  int m = 3;
  int add_per, all_num;
  nevery = force->inumeric(FLERR,arg[m++]);
  lipid_type = force->inumeric(FLERR,arg[m++]);
  d_cut = force->numeric(FLERR,arg[m++]);
  d_break = force->numeric(FLERR,arg[m++]);
  a_min = force->numeric(FLERR,arg[m++]);
  r_detect = force->numeric(FLERR,arg[m++]);
  d_Bezier = force->numeric(FLERR,arg[m++]);
  near_num_limit = force->inumeric(FLERR,arg[m++]);
  a_near_min = force->numeric(FLERR,arg[m++]);
  
  if (strcmp(arg[m],"number") == 0) {
	m++;
	add_num = force->inumeric(FLERR,arg[m++]);
  }
  else if (strcmp(arg[m],"percentage") == 0) {
	m++;
	add_per = force->numeric(FLERR,arg[m++]);
	all_num = group->count(igroup);
	add_num = all_num * add_per / 100;
  }
  else error->all(FLERR,"Illegal fix addlipid command");
  
  vel_flag = 1;
  if (narg == arg_min + 2) {
	if (strcmp(arg[m],"velocity") == 0 && strcmp(arg[m+1],"no") == 0) {
	  vel_flag = 0;
    }
	else if (strcmp(arg[m],"velocity") == 0 && strcmp(arg[m+1],"yes") == 0) {
	  vel_flag = 1;
    }
	else error->all(FLERR,"Illegal fix addlipid command");
	m = m + 2;
  }
  
  add_flag = 1;
  added_num = 0;

  // error check on type

  if (lipid_type <= 0 || lipid_type > atom->ntypes) error->all(FLERR,"Invalid atom type in fix addlipid command");

  // find max atom IDs

  // find_maxid();
}

/* ---------------------------------------------------------------------- */

FixAddLipid::~FixAddLipid()
{
  
}

/* ---------------------------------------------------------------------- */

int FixAddLipid::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAddLipid::init()
{
  if (domain->triclinic)
    error->all(FLERR,"Cannot use fix addlipid with triclinic box");

  // need a half neighbor list, built every Nevery steps
 
  dt = update->dt;
  
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;
  neighbor->requests[irequest]->occasional = 1;

  // lastcheck = -1;
}

/* ---------------------------------------------------------------------- */

void FixAddLipid::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ----------------------------------------------------------------------
   perform lipid insertion
------------------------------------------------------------------------- */

void FixAddLipid::pre_exchange()
{
  int i,m,ii,inum,itype,j,jj,jnum,jtype,k,kk,knum,ktype,flag,count,near_count_i,near_count_j,near_count_mid, mask_tmp;
  double xi[3],xj[3],xk[3],rij[3],r,xmid[3],xmidk[3],xik[3],xjk[3],rik,rjk,rmidk,rmidk_ji,rmidk_v,nquatmid[4],nmidl[3],rmp_nor[3],vector_tmp[3],nip[3],njp[3],o[3],rijo[3],Bezier_mid[3],xBmk[3],rBmk;
  double a1[3][3],a2[3][3],a3[3][3],ni1[3],nj1[3],nk1[3],nmid1[3],r12hat[3],r13hat[3],r23hat[3],r43hat[3],ninj,nink,njnk,ni1rhat,nj1rhat,a_fluid_mem,nil_dot_mp,njl_dot_mp,tani_ij,tanj_ij,xip[3],xjq[3],p[3],q[3];
  double nik1rhat, nki1rhat, njk1rhat, nkj1rhat, nmidk1rhat, nkmid1rhat, nmidnk, a_ik, a_jk, a_midk, a_ik_sum, a_jk_sum, a_midk_sum, r_ik_sum, r_jk_sum, r_midk_sum;
  double xmidi[3], rmidi, r41hat[3], nmidni, nmidi1rhat, nimid1rhat, a_midi;
  double xmidj[3], rmidj, r42hat[3], nmidnj, nmidj1rhat, njmid1rhat, a_midj;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double *newcoord;
  double *nquati,*nquatj,*nquatk;

  // just return if should not be called on this timestep
  // fprintf(stderr, "In fix addlipid at proc %d when time = %d: add_flag = %d\n", comm->me, update->ntimestep, add_flag);
  count_global = 0;
  if ((update->ntimestep % nevery) || !add_flag) return;
  
  // acquire updated ghost atom positions
  // necessary b/c are calling this after integrate, but before Verlet comm

  comm->forward_comm();
  
  // find the insert lipid atom
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
  
  neighbor->build_one(list,1);
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  
  memory->create(xnew,atom->nmax,10,"fix_addlipid:xnew");
  memory->create(masknew,atom->nmax,"fix_addlipid:masknew");
  count = 0;
  /* for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;
    itype = type[i];
    xi[0] = x[i][0];
    xi[1] = x[i][1];
    xi[2] = x[i][2];
    jlist = firstneigh[i];
    jnum = numneigh[i];
	
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      if (!(mask[j] & groupbit)) continue;
      jtype = type[j];
	  rij[0] = xi[0] - x[j][0];
	  rij[1] = xi[1] - x[j][1];
	  rij[2] = xi[2] - x[j][2];
	  r = sqrt(rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2]);
	  if (r < d_cut || r > d_break) continue;
	  r12hat[0] = rij[0] / r;
	  r12hat[1] = rij[1] / r;
	  r12hat[2] = rij[2] / r;
	  xmid[0] = (xi[0] + x[j][0]) / 2;
	  xmid[1] = (xi[1] + x[j][1]) / 2;
	  xmid[2] = (xi[2] + x[j][2]) / 2;
	  if (xmid[0] < sublo[0] || xmid[0] > subhi[0] || xmid[1] < sublo[1] || xmid[1] > subhi[1] || xmid[2] < sublo[2] || xmid[2] > subhi[2]) continue;
	  flag = 1;
	  for (kk = 0; kk < jnum; kk++) {
		if (kk != jj) {
		  k = jlist[kk];
		  k &= NEIGHMASK;
		  if (!(mask[k] & groupbit)) continue;
		  ktype = type[k];
		  
		  xmidk[0] = x[k][0] - xmid[0];    // vector from mid point to k
		  xmidk[1] = x[k][1] - xmid[1];
		  xmidk[2] = x[k][2] - xmid[2];
		  rmidk_ji = xmidk[0] * r12hat[0] + xmidk[1] * r12hat[1] + xmidk[2] * r12hat[2];    // the length between mid point and the projection of k at the ji vector
		  if (tag[i] == 13915 && tag[j] == 14531) fprintf(stderr, "In fix addlipid at proc %d when time = %d: tag[k] = %d, rmidk_ji = %f, r = %f, x[k][0] = %f, x[k][1] = %f, x[k][2] = %f\n", comm->me, update->ntimestep, tag[k], rmidk_ji, r, x[k][0], x[k][1], x[k][2]);
		  if (rmidk_ji * rmidk_ji > r * r / 4) continue;
		  rmidk_v = sqrt((xmidk[0] - rmidk_ji * r12hat[0]) * (xmidk[0] - rmidk_ji * r12hat[0]) + (xmidk[1] - rmidk_ji * r12hat[1]) * (xmidk[1] - rmidk_ji * r12hat[1]) + (xmidk[2] - rmidk_ji * r12hat[2]) * (xmidk[2] - rmidk_ji * r12hat[2]));    // distance between k and the projection of k at the ji vector
		  if (tag[i] == 13915 && tag[j] == 14531) fprintf(stderr, "In fix addlipid at proc %d when time = %d: tag[k] = %d, rmidk_ji = %f, rmidk_v = %f\n", comm->me, update->ntimestep, tag[k], rmidk_ji, rmidk_v);
		  if (rmidk_v < r_detect * sqrt(1 - 4 * rmidk_ji * rmidk_ji / r / r)) {    // see if k is in the ellipsoid between i and j, long axis is r/2, short axis is r_detect
			flag = 0;
			break;
		  }
		}
	  }
	  for (kk = 0; kk < count; kk++) {
		xmidk[0] = xnew[kk][0] - xmid[0];    // vector from mid point to k
		xmidk[1] = xnew[kk][1] - xmid[1];
		xmidk[2] = xnew[kk][2] - xmid[2];
		rmidk_ji = xmidk[0] * r12hat[0] + xmidk[1] * r12hat[1] + xmidk[2] * r12hat[2];    // the length between mid point and the projection of k at the ji vector
		if (rmidk_ji * rmidk_ji > r * r / 4) continue;
		rmidk_v = sqrt((xmidk[0] - rmidk_ji * r12hat[0]) * (xmidk[0] - rmidk_ji * r12hat[0]) + (xmidk[1] - rmidk_ji * r12hat[1]) * (xmidk[1] - rmidk_ji * r12hat[1]) + (xmidk[2] - rmidk_ji * r12hat[2]) * (xmidk[2] - rmidk_ji * r12hat[2]));    // distance between k and the projection of k at the ji vector
		if (rmidk_v < r_detect * sqrt(1 - 4 * rmidk_ji * rmidk_ji / r / r)) {    // see if k is in the ellipsoid between i and j, long axis is r/2, short axis is r_detect
		  flag = 0;
		  break;
		}
	  }
	  if (flag) {
		xnew[count][0] = xmid[0];    // position
		xnew[count][1] = xmid[1];
		xnew[count][2] = xmid[2];
		xnew[count][3] = (v[i][0] + v[j][0]) / 2;    // velocity
		xnew[count][4] = (v[i][1] + v[j][1]) / 2;
		xnew[count][5] = (v[i][2] + v[j][2]) / 2;
		nquati = avec_ellipsoid->bonus[ellipsoid[i]].quat;
		nquatj = avec_ellipsoid->bonus[ellipsoid[j]].quat;
		nquatmid[0] = (nquati[0] + nquatj[0]) / 2;
		nquatmid[1] = (nquati[1] + nquatj[1]) / 2;
		nquatmid[2] = (nquati[2] + nquatj[2]) / 2;
		nquatmid[3] = (nquati[3] + nquatj[3]) / 2;
		MathExtra::qnormalize(nquatmid);
		xnew[count][6] = nquatmid[0];    // quaternions
		xnew[count][7] = nquatmid[1];
		xnew[count][8] = nquatmid[2];
		xnew[count][9] = nquatmid[3];
		count++;
		fprintf(stderr, "In fix addlipid at proc %d when time = %d: a lipid atom at %f %f %f was added, atom->tag[i] = %d, x[i][0] = %f, x[i][1] = %f, x[i][2] = %f, atom->tag[j] = %d, x[j][0] = %f, x[j][1] = %f, x[j][2] = %f\n", comm->me, update->ntimestep, xmid[0], xmid[1], xmid[2], atom->tag[i], x[i][0], x[i][1], x[i][2], atom->tag[j], x[j][0], x[j][1], x[j][2]);
	  }
	}
  } */
  // fprintf(stderr, "In fix addlipid at proc %d when time = %d: all 01\n", comm->me, update->ntimestep);
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;
    itype = type[i];
	if (itype == lipid_type) mask_tmp = mask[i];
    xi[0] = x[i][0];
    xi[1] = x[i][1];
    xi[2] = x[i][2];
    jlist = firstneigh[i];
    jnum = numneigh[i];
	
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      if (!(mask[j] & groupbit) || j == i) continue;
      jtype = type[j];
	  // if (itype != lipid_type && jtype != lipid_type) continue;
      xj[0] = x[j][0];
      xj[1] = x[j][1];
      xj[2] = x[j][2];
	  rij[0] = xj[0] - xi[0];
	  rij[1] = xj[1] - xi[1];
	  rij[2] = xj[2] - xi[2];
	  r = sqrt(rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2]);
	  // if ((tag[i] == 6746 && tag[j] == 6748) || (tag[i] == 6748 && tag[j] == 6746)) fprintf(stderr, "In fix addlipid at proc %d when time = %d: tag[i] = %d, tag[j] = %d, r = %f\n", comm->me, update->ntimestep, tag[i], tag[j], r);
	  if (r < d_cut || r > d_break) continue;
	  
	  nquati = avec_ellipsoid->bonus[ellipsoid[i]].quat;
	  nquatj = avec_ellipsoid->bonus[ellipsoid[j]].quat;
	  MathExtra::quat_to_mat_trans(nquati,a1);
	  MathExtra::quat_to_mat_trans(nquatj,a2);
      MathExtra::normalize3(rij,r12hat);
	  ni1[0]=a1[0][0];
      ni1[1]=a1[0][1];
      ni1[2]=a1[0][2];
      nj1[0]=a2[0][0];
      nj1[1]=a2[0][1];
      nj1[2]=a2[0][2];
	  ninj = MathExtra::dot3(ni1,nj1);
      ni1rhat = MathExtra::dot3(ni1,r12hat);
      nj1rhat = MathExtra::dot3(nj1,r12hat);
      // a_fluid_mem = ninj + (sint-ni1rhat)*(sint+nj1rhat) - 2.0*sint*sint;
	  a_fluid_mem = ninj - ni1rhat * nj1rhat;
	  // if (update->ntimestep % 1000 == 0) fprintf(stderr, "In fix addlipid at proc %d when time = %d: a_fluid_mem = %f, ninj = %f, ni1rhat = %f, nj1rhat = %f\n", comm->me, update->ntimestep, a_fluid_mem, ninj, ni1rhat, nj1rhat);
	  // if ((tag[i] == 6746 && tag[j] == 6748) || (tag[i] == 6748 && tag[j] == 6746)) fprintf(stderr, "In fix addlipid at proc %d when time = %d: tag[i] = %d, tag[j] = %d, a_fluid_mem = %f\n", comm->me, update->ntimestep, tag[i], tag[j], a_fluid_mem);
	  if (a_fluid_mem < a_min) continue;
		
	  // see if there are beads near the mid point of the Bezier curve between i and j
	  vector_tmp[0] = (ni1[0] + nj1[0]) / 2;    // get the mid vector of the direction of i and j
	  vector_tmp[1] = (ni1[1] + nj1[1]) / 2;
	  vector_tmp[2] = (ni1[2] + nj1[2]) / 2;
	  MathExtra::normalize3(vector_tmp,nmidl);
	  MathExtra::cross3(nmidl,r12hat,vector_tmp);    // get the normal direction rmp_nor of the mid plane between i and j
	  MathExtra::normalize3(vector_tmp,rmp_nor);
	  nil_dot_mp = MathExtra::dot3(ni1,rmp_nor);    // get nip and njp, the projection of ni1 and nj1 in the mid plane
	  njl_dot_mp = MathExtra::dot3(nj1,rmp_nor);
	  nip[0] = ni1[0] - nil_dot_mp * rmp_nor[0];
	  nip[1] = ni1[1] - nil_dot_mp * rmp_nor[1];
	  nip[2] = ni1[2] - nil_dot_mp * rmp_nor[2];
	  njp[0] = nj1[0] - njl_dot_mp * rmp_nor[0];
	  njp[1] = nj1[1] - njl_dot_mp * rmp_nor[1];
	  njp[2] = nj1[2] - njl_dot_mp * rmp_nor[2];
	  // tani_ij = tan(arccos(ni1rhat));    // the tan of the angle between the the ij vector and the direction of i or j
	  // tanj_ij = tan(arccos(nj1rhat));
	  // MathExtra::cross3(rmp_nor,r12hat,rijo);    // get the vector vertical to both rij and rmp
	  // o[0] = xi[0] + r * (tani_ij * r12hat[0] + rijo[0]) / (tani_ij - tanj_ij);    // get the position of o, the point of intersection of the vertical vector of the direction of i and j
	  // o[1] = xi[1] + r * (tani_ij * r12hat[1] + rijo[1]) / (tani_ij - tanj_ij);
	  // o[2] = xi[2] + r * (tani_ij * r12hat[2] + rijo[2]) / (tani_ij - tanj_ij);
	  // Bezier_mid[0] = xi[0] / 4 + xj[0] / 4 + o[0] / 2;    // get the position of the mid point of Bezier curve
	  // Bezier_mid[1] = xi[1] / 4 + xj[1] / 4 + o[1] / 2;
	  // Bezier_mid[2] = xi[2] / 4 + xj[2] / 4 + o[2] / 2;
	  MathExtra::cross3(rmp_nor,ni1,xip);    // get the vector from i to reference point p for Bezier curve
	  MathExtra::cross3(nj1,rmp_nor,xjq);    // get the vector from j to reference point q for Bezier curve
	  p[0] = xi[0] + r * xip[0] / 2;    // get the postion of reference point p for i
	  p[1] = xi[1] + r * xip[1] / 2;
	  p[2] = xi[2] + r * xip[2] / 2;
	  q[0] = xj[0] + r * xjq[0] / 2;    // get the postion of reference point q for j
	  q[1] = xj[1] + r * xjq[1] / 2;
	  q[2] = xj[2] + r * xjq[2] / 2;
	  Bezier_mid[0] = (xi[0] + p[0] * 3 + xj[0] + q[0] * 3) / 8;    // get the position of the mid point of Bezier curve
	  Bezier_mid[1] = (xi[1] + p[1] * 3 + xj[1] + q[1] * 3) / 8;
	  Bezier_mid[2] = (xi[2] + p[2] * 3 + xj[2] + q[2] * 3) / 8;
	  
	  xmid[0] = (xi[0] + xj[0]) / 2;
	  xmid[1] = (xi[1] + xj[1]) / 2;
	  xmid[2] = (xi[2] + xj[2]) / 2;
	  nquatmid[0] = (nquati[0] + nquatj[0]) / 2;
	  nquatmid[1] = (nquati[1] + nquatj[1]) / 2;
	  nquatmid[2] = (nquati[2] + nquatj[2]) / 2;
	  nquatmid[3] = (nquati[3] + nquatj[3]) / 2;
	  MathExtra::qnormalize(nquatmid);
	  MathExtra::quat_to_mat_trans(nquatmid,a3);
	  nmid1[0]=a3[0][0];
      nmid1[1]=a3[0][1];
      nmid1[2]=a3[0][2];
	  // if (xmid[0] < sublo[0] || xmid[0] > subhi[0] || xmid[1] < sublo[1] || xmid[1] > subhi[1] || xmid[2] < sublo[2] || xmid[2] > subhi[2]) continue;
	  
/* 	  xabove[0] = xmid[0] - r * nmidl[0] / 2;    // get the points that are r/2 away from mid point along the mid vector direction
	  xabove[1] = xmid[1] - r * nmidl[1] / 2;
	  xabove[2] = xmid[2] - r * nmidl[2] / 2;
	  xbelow[0] = xmid[0] + r * nmidl[0] / 2;
	  xbelow[1] = xmid[1] + r * nmidl[1] / 2;
	  xbelow[2] = xmid[2] + r * nmidl[2] / 2; */
	  
	  flag = 1;
	  near_count_i = 0;
	  near_count_j = 0;
	  a_ik_sum = 0;
	  a_jk_sum = 0;
	  r_ik_sum = 0;
	  r_jk_sum = 0;
	  a_midk_sum = 0;
	  r_midk_sum = 0;
	  for (k = 0; k < nlocalghost; k++) {
		if (k == j || k == i) continue;
		ktype = type[k];
		xk[0] = x[k][0];
        xk[1] = x[k][1];
        xk[2] = x[k][2];
		
		// see if there are beads between i and j
		xmidk[0] = xk[0] - xmid[0];    // vector from mid to k
		xmidk[1] = xk[1] - xmid[1];
		xmidk[2] = xk[2] - xmid[2];
		rmidk = sqrt(xmidk[0] * xmidk[0] + xmidk[1] * xmidk[1] + xmidk[2] * xmidk[2]);
		if (rmidk < OVERLAP) {    // see if k is too close to mid
		  flag = 0;
		  break;
		}
		
		if (!(mask[k] & groupbit)) continue;
		rmidk_ji = xmidk[0] * r12hat[0] + xmidk[1] * r12hat[1] + xmidk[2] * r12hat[2];    // the length between mid point and the projection of k at the ji vector
		// if ((tag[i] == 6746 && tag[j] == 6748) || (tag[i] == 6748 && tag[j] == 6746)) fprintf(stderr, "In fix addlipid at proc %d when time = %d: tag[k] = %d, rmidk_ji = %f, r = %f, x[k][0] = %f, x[k][1] = %f, x[k][2] = %f\n", comm->me, update->ntimestep, tag[k], rmidk_ji, r, x[k][0], x[k][1], x[k][2]);
		if (rmidk_ji * rmidk_ji > r * r / 4) continue;
		rmidk_v = sqrt((xmidk[0] - rmidk_ji * r12hat[0]) * (xmidk[0] - rmidk_ji * r12hat[0]) + (xmidk[1] - rmidk_ji * r12hat[1]) * (xmidk[1] - rmidk_ji * r12hat[1]) + (xmidk[2] - rmidk_ji * r12hat[2]) * (xmidk[2] - rmidk_ji * r12hat[2]));    // distance between k and the projection of k at the ji vector
		// if ((tag[i] == 6746 && tag[j] == 6748) || (tag[i] == 6748 && tag[j] == 6746)) fprintf(stderr, "In fix addlipid at proc %d when time = %d: tag[k] = %d, rmidk_ji = %f, rmidk_v = %f\n", comm->me, update->ntimestep, tag[k], rmidk_ji, rmidk_v);
		if (rmidk_v < r_detect * sqrt(1 - 4 * rmidk_ji * rmidk_ji / r / r)) {    // see if k is in the ellipsoid between i and j, long axis is r/2, short axis is r_detect
		  flag = 0;
		  break;
		}
		
		// see if k is near the mid point of Bezier curve
		xBmk[0] = xk[0] - Bezier_mid[0];    // vector from the mid point of Bezier curve to k
		xBmk[1] = xk[1] - Bezier_mid[1];
		xBmk[2] = xk[2] - Bezier_mid[2];
		rBmk = sqrt(xBmk[0] * xBmk[0] + xBmk[1] * xBmk[1] + xBmk[2] * xBmk[2]);
		if (rBmk < d_Bezier) {
		  flag = 0;
		  break;
		}
		
		// see if k is near the points that are above or below the mid point
/* 		rabove = sqrt((xk[0] - xabove[0]) * (xk[0] - xabove[0]) + (xk[1] - xabove[1]) * (xk[1] - xabove[1]) + (xk[2] - xabove[2]) * (xk[2] - xabove[2]));
		rbelow = sqrt((xk[0] - xbelow[0]) * (xk[0] - xbelow[0]) + (xk[1] - xbelow[1]) * (xk[1] - xbelow[1]) + (xk[2] - xbelow[2]) * (xk[2] - xbelow[2]));
		if (rabove < d_Bezier || rbelow < d_Bezier) {
		  flag = 0;
		  break;
		} */
		
		// count the beads near i and j, see if they have left the membrane alone
		nquatk = avec_ellipsoid->bonus[ellipsoid[k]].quat;
	    MathExtra::quat_to_mat_trans(nquatk,a3);
	    nk1[0]=a3[0][0];
        nk1[1]=a3[0][1];
        nk1[2]=a3[0][2];
		xik[0] = xk[0] - xi[0];    // vector from i to k
		xik[1] = xk[1] - xi[1];
		xik[2] = xk[2] - xi[2];
		rik = sqrt(xik[0] * xik[0] + xik[1] * xik[1] + xik[2] * xik[2]);
		if (rik < d_cut) {
		  near_count_i++;
		  
		  // calculate the weighted mean of the cos(theta) between i or j and nearby paticles
          MathExtra::normalize3(xik,r13hat);
	      nink = MathExtra::dot3(ni1,nk1);
          nik1rhat = MathExtra::dot3(ni1,r13hat);
          nki1rhat = MathExtra::dot3(nk1,r13hat);
	      a_ik = nink - nik1rhat * nki1rhat;
		  a_ik_sum += a_ik / rik;
		  r_ik_sum += 1 / rik;
		}
		xjk[0] = xk[0] - xj[0];    // vector from j to k
		xjk[1] = xk[1] - xj[1];
		xjk[2] = xk[2] - xj[2];
		rjk = sqrt(xjk[0] * xjk[0] + xjk[1] * xjk[1] + xjk[2] * xjk[2]);
		if (rjk < d_cut) {
		  near_count_j++;
		
		  // calculate the weighted mean of the cos(theta) between i or j and nearby paticles
          MathExtra::normalize3(xjk,r23hat);
	      njnk = MathExtra::dot3(nj1,nk1);
          njk1rhat = MathExtra::dot3(nj1,r23hat);
          nkj1rhat = MathExtra::dot3(nk1,r23hat);
	      a_jk = njnk - njk1rhat * nkj1rhat;
		  a_jk_sum += a_jk / rjk;
		  r_jk_sum += 1 / rjk;
		}
		if (rmidk < d_cut) {
		  near_count_mid++;
		
		  // calculate the weighted mean of the cos(theta) between i or j and nearby paticles
          MathExtra::normalize3(xmidk,r43hat);
	      nmidnk = MathExtra::dot3(nmid1,nk1);
          nmidk1rhat = MathExtra::dot3(nmid1,r43hat);
          nkmid1rhat = MathExtra::dot3(nk1,r43hat);
	      a_midk = nmidnk - nmidk1rhat * nkmid1rhat;
		  a_midk_sum += a_midk / rmidk;
		  r_midk_sum += 1 / rmidk;
		}
	  }
	  xmidi[0] = xi[0] - xmid[0];    // vector from mid to i
	  xmidi[1] = xi[1] - xmid[1];
	  xmidi[2] = xi[2] - xmid[2];
	  rmidi = sqrt(xmidi[0] * xmidi[0] + xmidi[1] * xmidi[1] + xmidi[2] * xmidi[2]);
	  if (rmidi < d_cut) {
	    near_count_mid++;
	  
	    // calculate the weighted mean of the cos(theta) between i or j and nearby paticles
        MathExtra::normalize3(xmidi,r41hat);
	    nmidni = MathExtra::dot3(nmid1,ni1);
        nmidi1rhat = MathExtra::dot3(nmid1,r41hat);
        nimid1rhat = MathExtra::dot3(ni1,r41hat);
	    a_midi = nmidni - nmidi1rhat * nimid1rhat;
	    a_midk_sum += a_midi / rmidi;
	    r_midk_sum += 1 / rmidi;
	  }
	  xmidj[0] = xj[0] - xmid[0];    // vector from mid to j
	  xmidj[1] = xj[1] - xmid[1];
	  xmidj[2] = xj[2] - xmid[2];
	  rmidj = sqrt(xmidj[0] * xmidj[0] + xmidj[1] * xmidj[1] + xmidj[2] * xmidj[2]);
	  if (rmidj < d_cut) {
	    near_count_mid++;
	  
	    // calculate the weighted mean of the cos(theta) between i or j and nearby paticles
        MathExtra::normalize3(xmidj,r42hat);
	    nmidnj = MathExtra::dot3(nmid1,nj1);
        nmidj1rhat = MathExtra::dot3(nmid1,r42hat);
        njmid1rhat = MathExtra::dot3(nj1,r42hat);
	    a_midj = nmidnj - nmidj1rhat * njmid1rhat;
	    a_midk_sum += a_midj / rmidj;
	    r_midk_sum += 1 / rmidj;
	  }
	  if (flag) {
		// if (near_count_i < near_num_limit && itype == lipid_type) flag = 0;
		// if (near_count_j < near_num_limit && jtype == lipid_type) flag = 0;
		if (near_count_i < near_num_limit) flag = 0;
		if (near_count_j < near_num_limit) flag = 0;
		if (near_count_mid < near_num_limit) flag = 0;
	  }
	  if (flag) {
		if (a_ik_sum / r_ik_sum < a_near_min) flag = 0;
		if (a_jk_sum / r_jk_sum < a_near_min) flag = 0;
		if (a_midk_sum / r_midk_sum < a_near_min) flag = 0;
	  }
	  if (flag) {
	    for (kk = 0; kk < count; kk++) {
		  xmidk[0] = xnew[kk][0] - xmid[0];    // vector from mid point to k
		  xmidk[1] = xnew[kk][1] - xmid[1];
		  xmidk[2] = xnew[kk][2] - xmid[2];
		  rmidk_ji = xmidk[0] * r12hat[0] + xmidk[1] * r12hat[1] + xmidk[2] * r12hat[2];    // the length between mid point and the projection of k at the ji vector
		  if (rmidk_ji * rmidk_ji > r * r / 4) continue;
		  rmidk_v = sqrt((xmidk[0] - rmidk_ji * r12hat[0]) * (xmidk[0] - rmidk_ji * r12hat[0]) + (xmidk[1] - rmidk_ji * r12hat[1]) * (xmidk[1] - rmidk_ji * r12hat[1]) + (xmidk[2] - rmidk_ji * r12hat[2]) * (xmidk[2] - rmidk_ji * r12hat[2]));    // distance between k and the projection of k at the ji vector
		  if (rmidk_v < r_detect * sqrt(1 - 4 * rmidk_ji * rmidk_ji / r / r)) {    // see if k is in the ellipsoid between i and j, long axis is r/2, short axis is r_detect
		    flag = 0;
		    break;
		  }
	    }
	  }
	  if (flag) {
		xnew[count][0] = xmid[0];    // position
		xnew[count][1] = xmid[1];
		xnew[count][2] = xmid[2];
		xnew[count][3] = (v[i][0] + v[j][0]) / 2;    // velocity
		xnew[count][4] = (v[i][1] + v[j][1]) / 2;
		xnew[count][5] = (v[i][2] + v[j][2]) / 2;
		xnew[count][6] = nquatmid[0];    // quaternions
		xnew[count][7] = nquatmid[1];
		xnew[count][8] = nquatmid[2];
		xnew[count][9] = nquatmid[3];
		// if (itype == lipid_type) masknew[count] = mask[i];
		// else if (jtype == lipid_type) masknew[count] = mask[j];
		count++;
		fprintf(stderr, "In fix addlipid at proc %d when time = %d: a lipid atom at %f %f %f was added, atom->tag[i] = %d, x[i][0] = %f, x[i][1] = %f, x[i][2] = %f, atom->tag[j] = %d, x[j][0] = %f, x[j][1] = %f, x[j][2] = %f\n", comm->me, update->ntimestep, xmid[0], xmid[1], xmid[2], atom->tag[i], x[i][0], x[i][1], x[i][2], atom->tag[j], x[j][0], x[j][1], x[j][2]);
	  }
	}
  }
  // fprintf(stderr, "In fix addlipid at proc %d when time = %d: all 02\n", comm->me, update->ntimestep);
  MPI_Allreduce(&count,&count_global,1,MPI_INT,MPI_SUM,world);
  if (count) fprintf(stderr, "In fix addlipid at proc %d when time = %d: count = %d, count_global = %d\n", comm->me, update->ntimestep, count, count_global);
  if (!count_global && comm->me == 0) fprintf(stderr, "In fix addlipid when time = %d: no new lipid was added\n", update->ntimestep);
  added_num += count_global;
  if (added_num >= add_num) add_flag = 0;    // stop adding atoms when added number exceed max number
  // fprintf(stderr, "In fix addlipid at proc %d when time = %d: all 03\n", comm->me, update->ntimestep);
  
  if (count_global) {
    // int *create_num_local;
	double *xnew0_local;
	double *xnew1_local;
	double *xnew2_local;
	double *xnew0;
	double *xnew1;
	double *xnew2;
    int *create_num_local;
	int *create_num_proc;
    int *displs;
	int nprocs = comm->nprocs;
	
	// memory->create(create_num_local,nprocs,"fix_addlipid:create_num_local");
	memory->create(xnew0_local,count,"fix_addlipid:xnew0_local");
	memory->create(xnew1_local,count,"fix_addlipid:xnew1_local");
	memory->create(xnew2_local,count,"fix_addlipid:xnew2_local");
	memory->create(create_num_proc,nprocs,"fix_addlipid:create_num_proc");
    memory->create(displs,nprocs,"fix_addlipid:displs");
	
	
	if (nprocs > 1) {
	  MPI_Allgather(&count, 1, MPI_INT, create_num_proc, 1, MPI_INT, world);
    }
	
	for (ii = 0; ii < count; ii++) {
	  xnew0_local[ii] = xnew[ii][0];
	  xnew1_local[ii] = xnew[ii][1];
	  xnew2_local[ii] = xnew[ii][2];
	}
	
	int kn = 0;
    for (i = 0; i < nprocs; i++){
      displs[i] = kn;
      kn += create_num_proc[i];
	}
	
	memory->create(xnew0,kn,"fix_filopodia:xnew0");
	memory->create(xnew1,kn,"fix_filopodia:xnew1");
	memory->create(xnew2,kn,"fix_filopodia:xnew2");
	
	if (nprocs > 1) {
	  MPI_Allgatherv(xnew0_local, count, MPI_DOUBLE, xnew0, create_num_proc, displs, MPI_DOUBLE, world);
	  MPI_Allgatherv(xnew1_local, count, MPI_DOUBLE, xnew1, create_num_proc, displs, MPI_DOUBLE, world);
	  MPI_Allgatherv(xnew2_local, count, MPI_DOUBLE, xnew2, create_num_proc, displs, MPI_DOUBLE, world);
	}
	
	// clear ghost count and any ghost bonus data internal to AtomVec
    // same logic as beginning of Comm::exchange()
    // do it now b/c inserting atoms will overwrite ghost atoms
    
    atom->nghost = 0;
    atom->avec->clear_bonus();
    
    // add atoms/molecules in one of 3 ways
    
    bigint natoms_previous = atom->natoms;
    int nlocal_previous = atom->nlocal;
    
    // find current max atom and molecule IDs on every insertion step
    
    // find_maxid();
    
	// box size
	double *lo, *hi;
	double xprd, yprd, zprd, deltax, deltay, deltaz, rr0, rr1, rr2, rr3, rr4, rr5, rr6;
	int pre_num;
	lo = domain->boxlo;
	hi = domain->boxhi;
	xprd = domain->xprd;
	yprd = domain->yprd;
	zprd = domain->zprd;
	// if (update->ntimestep % 1000) fprintf(stderr, "In fix addlipid at proc %d when time = %d: lo[0] = %f, lo[1] = %f, lo[2] = %f, hi[0] = %f, hi[1] = %f, hi[2] = %f, xprd = %f, yprd = %f, zprd = %f\n", comm->me, update->ntimestep, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], xprd, yprd, zprd);
    
	// insert new lipid atoms
    for (ii = 0; ii < count; ii++) {
	  xi[0] = xnew[ii][0];
	  xi[1] = xnew[ii][1];
	  xi[2] = xnew[ii][2];
	  
	  pre_num = displs[comm->me];
	  flag = 1;
	  for (jj = 0; jj < pre_num; jj++) {
		// deltax = xnew0[jj] - xi[0];
		// deltay = xnew1[jj] - xi[1];
		// deltaz = xnew2[jj] - xi[2];
		deltax = abs(xnew0[jj] - xi[0]);
		deltay = abs(xnew1[jj] - xi[1]);
		deltaz = abs(xnew2[jj] - xi[2]);
		if ((deltax - xprd) * (deltax - xprd) < OVERLAP * OVERLAP) deltax = abs(deltax - xprd);
		if ((deltay - yprd) * (deltay - yprd) < OVERLAP * OVERLAP) deltay = abs(deltay - yprd);
		if ((deltaz - zprd) * (deltaz - zprd) < OVERLAP * OVERLAP) deltaz = abs(deltaz - zprd);
		rr0 = sqrt(deltax * deltax + deltay * deltay + deltaz * deltaz);
		if (rr0 < OVERLAP) {
		  flag = 0;
		  break;
		}
		
		// rr1 = sqrt((xnew0[jj] - xi[0] - xprd) * (xnew0[jj] - xi[0] - xprd) + (xnew1[jj] - xi[1]) * (xnew1[jj] - xi[1]) + (xnew2[jj] - xi[2]) * (xnew2[jj] - xi[2]));
		// rr2 = sqrt((xnew0[jj] - xi[0] + xprd) * (xnew0[jj] - xi[0] + xprd) + (xnew1[jj] - xi[1]) * (xnew1[jj] - xi[1]) + (xnew2[jj] - xi[2]) * (xnew2[jj] - xi[2]));
		// rr3 = sqrt((xnew0[jj] - xi[0]) * (xnew0[jj] - xi[0]) + (xnew1[jj] - xi[1] - yprd) * (xnew1[jj] - xi[1] - yprd) + (xnew2[jj] - xi[2]) * (xnew2[jj] - xi[2]));
		// rr4 = sqrt((xnew0[jj] - xi[0]) * (xnew0[jj] - xi[0]) + (xnew1[jj] - xi[1] + yprd) * (xnew1[jj] - xi[1] + yprd) + (xnew2[jj] - xi[2]) * (xnew2[jj] - xi[2]));
		// rr5 = sqrt((xnew0[jj] - xi[0]) * (xnew0[jj] - xi[0]) + (xnew1[jj] - xi[1]) * (xnew1[jj] - xi[1]) + (xnew2[jj] - xi[2] - zprd) * (xnew2[jj] - xi[2] - zprd));
		// rr6 = sqrt((xnew0[jj] - xi[0]) * (xnew0[jj] - xi[0]) + (xnew1[jj] - xi[1]) * (xnew1[jj] - xi[1]) + (xnew2[jj] - xi[2] + zprd) * (xnew2[jj] - xi[2] + zprd));
		// if (rr1 < OVERLAP || rr2 < OVERLAP || rr3 < OVERLAP || rr4 < OVERLAP || rr5 < OVERLAP || rr6 < OVERLAP) {
		//   flag = 0;
		//   break;
		// }
	  }
	  
	  if (flag == 0) break;
	  atom->avec->create_atom(lipid_type,xi);
	  
	  i = atom->nlocal-1;
	  mask[i] = mask_tmp;
	  rmass[i] = 0.238732;
	  vfrac[i] = 0.523599;
	  molecule[i] = 0;
	  if (vel_flag) {
	    v[i][0] = xnew[ii][3];
	    v[i][1] = xnew[ii][4];
	    v[i][2] = xnew[ii][5];
	  }
	  avec_ellipsoid->set_shape(i, 1, 1, 1);
	  // fprintf(stderr, "In fix addlipid at proc %d when time = %d: i = %d, ellipsoid[i] = %d, avec_ellipsoid->bonus[ellipsoid[i]].ilocal = %d\n", comm->me, update->ntimestep, i, ellipsoid[i], avec_ellipsoid->bonus[ellipsoid[i]].ilocal);
	  nquati = avec_ellipsoid->bonus[ellipsoid[i]].quat;
	  nquati[0] = xnew[ii][6];
	  nquati[1] = xnew[ii][7];
	  nquati[2] = xnew[ii][8];
	  nquati[3] = xnew[ii][9];
    }
    
    // invoke set_arrays() for fixes/computes/variables
    //   that need initialization of attributes of new atoms
    // don't use modify->create_attributes() since would be inefficient
    //   for large number of atoms
    // note that for typical early use of create_atoms,
    //   no fixes/computes/variables exist yet
    
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
	  fprintf(stderr, "In fix addlipid at proc %d when time = %d: a lipid atom %d at %f %f %f was added\n", comm->me, update->ntimestep, atom->tag[i], x[i][0], x[i][1], x[i][2]);
    }
  }
  // fprintf(stderr, "In fix addlipid at proc %d when time = %d: all 04\n", comm->me, update->ntimestep);

  // free local memory

  memory->destroy(xnew);
  // fprintf(stderr, "In fix addlipid at proc %d when time = %d: all 05\n", comm->me, update->ntimestep);
}

/* ----------------------------------------------------------------------
   maxtag_all = current max atom ID for all atoms
   maxmol_all = current max molecule ID for all atoms
------------------------------------------------------------------------- */

/* void FixAddLipid::find_maxid()
{
  tagint *tag = atom->tag;
  tagint *molecule = atom->molecule;
  int nlocal = atom->nlocal;

  tagint max = 0;
  for (int i = 0; i < nlocal; i++) max = MAX(max,tag[i]);
  MPI_Allreduce(&max,&maxtag_all,1,MPI_LMP_TAGINT,MPI_MAX,world);
} */