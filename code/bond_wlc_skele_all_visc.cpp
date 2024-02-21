/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   www.cs.sandia.gov/~sjplimp/lammps.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// WLC bond potential

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "bond_wlc_skele_all_visc.h"
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "update.h"
#include "output.h"
#include "statistic.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "random_mars.h"
#include "MersenneTwister.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BondWLC_SKELE_ALL_VISC::BondWLC_SKELE_ALL_VISC(LAMMPS *lmp) : Bond(lmp) 
{
  random = NULL;
  mtrand = NULL;
}

/* ----------------------------------------------------------------------
   free all arrays 
------------------------------------------------------------------------- */

BondWLC_SKELE_ALL_VISC::~BondWLC_SKELE_ALL_VISC()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(temp);
    memory->destroy(r0);
    memory->destroy(lmax);
    memory->destroy(lambda);
    memory->destroy(gamc);
    memory->destroy(gamt);
    memory->destroy(sigc);
    memory->destroy(sigt);
  }

  if (random) delete random;

#ifdef MTRAND
  if (mtrand) delete mtrand;
#endif
}

/* ---------------------------------------------------------------------- */

void BondWLC_SKELE_ALL_VISC::compute(int eflag, int vflag)
{
  int i1,i2,n,type,k,l;
  double rr,ra,ebond,fbond;
  double dvx, dvy, dvz, vv, rsq;
  double fr[3];
  int n_stress = output->n_stress;
  double ff[6];

  ebond = 0.0;
  if(eflag || vflag) ev_setup(eflag, vflag);
  else evflag = 0;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  //double *bondlist_length = neighbor->bondlist_length;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;
  for (n = 0; n < nbondlist; n++) {
    
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];
    //l0 = bondlist_length[n]*spring_scale[type];

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];
    dvx = v[i1][0] - v[i2][0];
    dvy = v[i1][1] - v[i2][1];
    dvz = v[i1][2] - v[i2][2];

    // force from log term
    rsq = delx*delx + dely*dely + delz*delz;
    ra = sqrt(rsq);
	if(ra>r0[type]) rr = (ra-r0[type])/(lmax[type]-r0[type]);
	else rr = 0.0;
    vv = (delx*dvx + dely*dvy +  delz*dvz)/ra;
    if (rr >= 1.0) {
      char warning[128];
      sprintf(warning,"WLC bond too long: " BIGINT_FORMAT " "
              TAGINT_FORMAT " " TAGINT_FORMAT " %g",
              update->ntimestep,atom->tag[i1],atom->tag[i2],rr);
      error->warning(FLERR, warning, 0);
    }   
    //generate_wrr();
#ifdef MTRAND
    wrr[0] = sqrt(0.5*(delx*delx/rsq + 3.0))*(2.0*mtrand->rand()-1.0);
    wrr[1] = sqrt(0.5*(dely*dely/rsq + 3.0))*(2.0*mtrand->rand()-1.0);
    wrr[2] = sqrt(0.5*(delz*delz/rsq + 3.0))*(2.0*mtrand->rand()-1.0);
    wrr[3] = 2.0*mtrand->rand()-1.0;
#else
    wrr[0] = sqrt(0.5*(delx*delx/rsq + 3.0))*(2.0*random->uniform()-1.0);
    wrr[1] = sqrt(0.5*(dely*dely/rsq + 3.0))*(2.0*random->uniform()-1.0);
    wrr[2] = sqrt(0.5*(delz*delz/rsq + 3.0))*(2.0*random->uniform()-1.0);
    wrr[3] = 2.0*random->uniform()-1.0;
#endif

    fbond = - temp[type]*(0.25/(1.0-rr)/(1.0-rr)-0.25+rr)/lambda[type]/(ra-r0[type]) + (sigc[type]*wrr[3] - gamc[type]*vv)/ra;

    // force & energy
    if(eflag){
      ebond = 0.25*temp[type]*lmax[type]*(3.0*rr*rr-2.0*rr*rr*rr)/(1.0-rr)/lambda[type];
    }
    
    fr[0] = sigt[type]*wrr[0] - gamt[type]*dvx;
    fr[1] = sigt[type]*wrr[1] - gamt[type]*dvy;
    fr[2] = sigt[type]*wrr[2] - gamt[type]*dvz;
    //fr[0] = sigt[type]*wrr[0]/ra - gamt[type]*dvx;  // use with generate_wrr();
    //fr[1] = sigt[type]*wrr[1]/ra - gamt[type]*dvy;  // use with generate_wrr();
    //fr[2] = sigt[type]*wrr[2]/ra - gamt[type]*dvz;  // use with generate_wrr();

    // apply force to each of 2 atoms
    if (newton_bond || i1 < nlocal) {
      f[i1][0] += delx*fbond + fr[0];
      f[i1][1] += dely*fbond + fr[1];
      f[i1][2] += delz*fbond + fr[2];
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= delx*fbond + fr[0];
      f[i2][1] -= dely*fbond + fr[1];
      f[i2][2] -= delz*fbond + fr[2];
    }
//if(update->ntimestep>999 && update->ntimestep%500==0) fprintf(stderr, "In bond wlc/skele/all/visc compute() at timestep %d at proc %d: i1 = %d, i2=%d, atom->tag[i1] = %d, atom->tag[i2] = %d, fbond = %f, rr = %f, fwlc = %f, fx = %f, fy = %f, fz = %f, delx = %f, dely = %f, delz = %f, f[i1][0] = %f, f[i1][1] = %f, f[i1][2] = %f\n", update->ntimestep, comm->me, i1, i2, atom->tag[i1], atom->tag[i2], fbond, rr, temp[type]*(0.25/(1.0-rr)/(1.0-rr)-0.25+rr)/lambda[type]/(ra-r0[type]), delx*fbond + fr[0], dely*fbond + fr[1], delz*fbond + fr[2], delx, dely, delz, f[i1][0], f[i1][1], f[i1][2]);
    // virial contribution

    if (n_stress){
      ff[0] = delx*(delx*fbond + fr[0]);
      ff[1] = dely*(dely*fbond + fr[1]);
      ff[2] = delz*(delz*fbond + fr[2]);
      ff[3] = delx*(dely*fbond + fr[1]);
      ff[4] = delx*(delz*fbond + fr[2]);
      ff[5] = dely*(delz*fbond + fr[2]);
      for (k = 0; k < n_stress; k++){
        l = output->stress_id[k];
        if ((output->next_stat_calc[l] == update->ntimestep) && (output->last_stat_calc[l] != update->ntimestep))
          output->stat[l]->virial3(i1,i2,ff);
      }
    }

    if (evflag) ev_tally2(i1,i2,nlocal,newton_bond,ebond,fbond,fr,delx,dely,delz);
  }
}

/* ---------------------------------------------------------------------- */

void BondWLC_SKELE_ALL_VISC::allocate()
{
  allocated = 1;
  int n = atom->nbondtypes;
  int seed = 19;
  long unsigned int seed2;

  seed +=  comm->me;
#ifdef MTRAND
  if (mtrand) delete mtrand;
  seed2 = static_cast<long unsigned int>(seed);
  mtrand = new MTRand(&seed2);
  //mtrand = new MTRand();
#else
  if (random) delete random;
  random = new RanMars(lmp,seed);
#endif

  int individual = atom->individual;
  if (individual == 0)
    error->all(FLERR,"Individual has wrong value or is not set! Using bond wlc/pow/all/visc only possible with individual =1");  

  memory->create(temp,n+1,"bond:temp");
  memory->create(r0,n+1,"bond:r0");
  memory->create(lmax,n+1,"bond:lmax");
  memory->create(lambda,n+1,"bond:lambda");
  memory->create(gamc,n+1,"bond:gamc");
  memory->create(gamt,n+1,"bond:gamt");
  memory->create(sigc,n+1,"bond:sigc");
  memory->create(sigt,n+1,"bond:sigt");

  memory->create(setflag,n+1,"bond:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs from one line in input script for one or more types
------------------------------------------------------------------------- */

void BondWLC_SKELE_ALL_VISC::coeff(int narg, char **arg)
{
  if (narg != 7) error->all(FLERR,"Incorrect args in bond_coeff command");
  if (!allocated) allocate();

  int ilo,ihi;
  force->bounds(arg[0],atom->nbondtypes,ilo,ihi);

  double temp_one = force->numeric(FLERR,arg[1]);
  double r0_one = force->numeric(FLERR,arg[2]);
  double lmax_one = force->numeric(FLERR,arg[3]);
  double lambda_one = force->numeric(FLERR,arg[4]);
  double gamc_one = force->numeric(FLERR,arg[5]);
  double gamt_one = force->numeric(FLERR,arg[6]);

  if(gamt_one > 3*gamc_one)
    error->all(FLERR,"Gamma_t > 3*Gamma_c");

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    temp[i] = temp_one;
    r0[i] = r0_one;
    lmax[i] = lmax_one;
    lambda[i] = lambda_one;
    gamc[i] = gamc_one;
    gamt[i] = gamt_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args in bond_coeff command");
}

/* ---------------------------------------------------------------------- */

void BondWLC_SKELE_ALL_VISC::init_style()
{
  double sdtt = sqrt(update->dt);

  if (!allocated) error->all(FLERR,"Bond coeffs are not set");
  for (int i = 1; i <= atom->nbondtypes; i++){
    //if (setflag[i] == 0) error->all(FLERR,"All bond coeffs are not set");
    //if (gamt[i] > 3.0*gamc[i]) error->all(FLERR,"Gamma_t > 3*Gamma_c");
    sigc[i] = sqrt(2.0*temp[i]*(3.0*gamc[i]-gamt[i]))/sdtt;
    sigt[i] = 2.0*sqrt(gamt[i]*temp[i])/sdtt;
  } 
}

/* ---------------------------------------------------------------------- */

double BondWLC_SKELE_ALL_VISC::equilibrium_distance(int i)
{
  return r0[i];
}


/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file 
------------------------------------------------------------------------- */

void BondWLC_SKELE_ALL_VISC::write_restart(FILE *fp)
{
  fwrite(&temp[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&r0[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&lmax[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&lambda[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&gamc[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&gamt[1],sizeof(double),atom->nbondtypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them 
------------------------------------------------------------------------- */

void BondWLC_SKELE_ALL_VISC::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    fread(&temp[1],sizeof(double),atom->nbondtypes,fp);
    fread(&r0[1],sizeof(double),atom->nbondtypes,fp);
    fread(&lmax[1],sizeof(double),atom->nbondtypes,fp);
    fread(&lambda[1],sizeof(double),atom->nbondtypes,fp);
    fread(&gamc[1],sizeof(double),atom->nbondtypes,fp);
    fread(&gamt[1],sizeof(double),atom->nbondtypes,fp); 
  }
  MPI_Bcast(&temp[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&r0[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&lmax[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&lambda[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&gamc[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&gamt[1],atom->nbondtypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void BondWLC_SKELE_ALL_VISC::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nbondtypes; i++)
    fprintf(fp,"%d %g %g %g %g %g %g %g\n",i,temp[i],r0[i],lmax[i],lambda[i],gamc[i],gamt[i]);
}

/* ----------------------------------------------------------------------*/

double BondWLC_SKELE_ALL_VISC::single(int type, double rsq, int i, int j,
                        double &fforce)
{
  char warning[128];

  double rr = rsq;

  if (rr >= 1.0) {
    sprintf(warning,"WLC single bond too long: " BIGINT_FORMAT " %g",update->ntimestep,rr);
    error->warning(FLERR,warning,0);
  }

  fforce = 0.0;
  return 0.0;
}

/* ----------------------------------------------------------------------*/

void BondWLC_SKELE_ALL_VISC::generate_wrr()
{
  int i;
  double ww[3][3];
  double v1, v2, factor, ss;

  for (i=0; i<5; i++){
    ss = 100.0;
    while ( ss > 1.0 ){
#ifdef MTRAND
      v1 = 2.0 * mtrand->rand() - 1.0;
      v2 = 2.0 * mtrand->rand() - 1.0;
#else
      v1 = 2.0 * random->uniform() - 1.0;
      v2 = 2.0 * random->uniform() - 1.0;
#endif
      ss = v1*v1 + v2*v2;
    }
    factor = sqrt(-2.0 * log(ss)/ss);
    if (i < 3){
      ww[i][0] = factor*v1;
      ww[i][1] = factor*v2; 
    }
    else if (i == 3){
      ww[0][2] = factor*v1;
      ww[1][2] = factor*v2;
    }
    else
      ww[2][2] = factor*v1; 
  }
  wrr[3] = (ww[0][0]+ww[1][1]+ww[2][2])/3.0;
  wrr[0] = (ww[0][0]-wrr[3])*delx + 0.5*(ww[0][1]+ww[1][0])*dely + 0.5*(ww[0][2]+ww[2][0])*delz;
  wrr[1] = 0.5*(ww[1][0]+ww[0][1])*delx + (ww[1][1]-wrr[3])*dely + 0.5*(ww[1][2]+ww[2][1])*delz;
  wrr[2] = 0.5*(ww[2][0]+ww[0][2])*delx + 0.5*(ww[2][1]+ww[1][2])*dely + (ww[2][2]-wrr[3])*delz;
}

/* ----------------------------------------------------------------------*/
