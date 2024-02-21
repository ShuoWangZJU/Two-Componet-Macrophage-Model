/* ----------------------------------------------------------------------
  Dmitry Fedosov - 08/12/05  accumulation of statistics
 
  LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   www.cs.sandia.gov/~sjplimp/lammps.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

------------------------------------------------------------------------- */

#include <math.h>
#include "statistic_rho.h"
#include "atom.h"
#include "memory.h"
#include "comm.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
StatisticRho::StatisticRho(LAMMPS *lmp, int narg, char **arg)
  :Statistic(lmp,narg,arg)
{
  int i, j, k;
  memory->create(num,nx,ny,nz,"statistic_rho:num");
  memory->create(rho,nx,ny,nz,"statistic_rho:rho"); 
  for (i=0;i<nx;i++)
    for (j=0;j<ny;j++)
      for (k=0;k<nz;k++){
        num[i][j][k] = 0.0;
        rho[i][j][k] = 0.0;
      }
  num_step = 0;
}

/* ---------------------------------------------------------------------- */

StatisticRho::~StatisticRho()
{
  memory->destroy(num);
  memory->destroy(rho);
}

/* ---------------------------------------------------------------------- */

void StatisticRho::calc_stat()
{
  int *mask = atom->mask;
  double **x = atom->x;
  double *rh = atom->rho;

  for (int l=0; l<atom->nlocal; l++)
    if (mask[l] & groupbit)   
      if (map_index(x[l][0],x[l][1],x[l][2])){
        num[is][js][ks] += 1.0;
        rho[is][js][ks] += rh[l];
      }
  num_step++;

}

/* ---------------------------------------------------------------------- */

void StatisticRho:: write_stat(bigint step)
{
  int i, j, k, l;
  int total = nx*ny*nz;
  double x, y, z, rr, theta, vol;
  double *ntmp, *rtmp, *tmp;
  char f_name[FILENAME_MAX];

  memory->create(ntmp,total,"statistic_rho:ntmp");
  memory->create(rtmp,total,"statistic_rho:rtmp");
  memory->create(tmp,total,"statistic_rho:tmp");  

  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){
        tmp[l]=num[i][j][k]/num_step;
        num[i][j][k] = 0.0;
        ntmp[l] = 0.0;
        rtmp[l] = 0.0;
        l++;
      }
  MPI_Reduce(tmp,ntmp,total,MPI_DOUBLE,MPI_SUM,0,world);
  l = 0;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++){
        tmp[l]=rho[i][j][k]/num_step;
        rho[i][j][k] = 0.0;
	l++;
      }
  MPI_Reduce(tmp,rtmp,total,MPI_DOUBLE,MPI_SUM,0,world);
  num_step = 0;

  if (!(comm->me)){
    FILE* out_stat;
    sprintf(f_name,"%s."BIGINT_FORMAT".plt",fname,step); 
    out_stat=fopen(f_name,"w");
    fprintf(out_stat,"VARIABLES=\"x\",\"y\",\"z\",\"rho\"  \n");
    if (cyl_ind){
      fprintf(out_stat,"ZONE I=%d,J=%d,K=%d, F=POINT \n", ny, nz, nx);
      for (i=0; i<nx; i++)
        for (k=0; k<nz; k++)
          for (j=0; j<ny; j++){
            l = k*nx*ny + j*nx + i;
            x = xlo + (i+0.5)*dx;
            rr = (j+0.5)*dy;
            theta = k*dz;
            y = ylo + rr*cos(theta);
            z = zlo + rr*sin(theta);
            if (ntmp[l] > 0.0)
              fprintf(out_stat,"%lf %lf %lf %15.10lf \n",x, y, z, rtmp[l]/ntmp[l]);
            else
              fprintf(out_stat,"%lf %lf %lf 0.0 \n",x, y, z);
          }
    } else{
      fprintf(out_stat,"ZONE I=%d,J=%d,K=%d, F=POINT \n", nx, ny, nz);
      l = 0;
      for (k=0; k<nz; k++)
        for (j=0; j<ny; j++)
          for (i=0; i<nx; i++){
            x = xlo + (i+0.5)*dx;
            y = ylo + (j+0.5)*dy;
            z = zlo + (k+0.5)*dz;
            if (ntmp[l] > 0.0)
              fprintf(out_stat,"%lf %lf %lf %15.10lf \n",x, y, z, rtmp[l]/ntmp[l]);
            else
              fprintf(out_stat,"%lf %lf %lf 0.0 \n",x, y, z);
            l++;
          }
    }     
    fclose(out_stat);
  }
  memory->destroy(ntmp);
  memory->destroy(rtmp);
  memory->destroy(tmp);    
}

/* ---------------------------------------------------------------------- */

