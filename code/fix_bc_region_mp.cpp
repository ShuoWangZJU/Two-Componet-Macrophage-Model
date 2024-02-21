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

/* ----------------------------------------------------------------------
   Contributing authors: Shuo Wang
------------------------------------------------------------------------- */

#include "math.h"
#include "string.h"
#include "stdlib.h"
#include "domain.h"
#include "lattice.h"
#include "fix_bc_region_mp.h"
#include "atom.h"
#include "update.h"
#include "force.h"
#include "modify.h"
#include "memory.h"
#include "error.h"
#include "fix_mvv_dpd.h"
#include "domain.h"
#include "region.h"

using namespace LAMMPS_NS;
using namespace FixConst;

FixBcRegionMP::FixBcRegionMP(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
	if (narg < 8) error->all(FLERR,"Illegal fix bc/region/mp command");
	iregion = domain->find_region(arg[3]);
	BX0 = force->numeric(FLERR,arg[4]);
	BY0 = force->numeric(FLERR,arg[5]);
	BZ0 = force->numeric(FLERR,arg[6]);
	X0 = force->numeric(FLERR,arg[7]);
	if (iregion == -1) error->all(FLERR,"Fix bc/region/mp command region ID does not exist");

	int iarg;
  
}

/* ---------------------------------------------------------------------- */

FixBcRegionMP::~FixBcRegionMP()
{
	//memory->destroy_2d_double_array(shear);
}

/* ---------------------------------------------------------------------- */

int FixBcRegionMP::setmask()
{
	int mask = 0;
	mask |= INITIAL_INTEGRATE;
	return mask;
}

/* ---------------------------------------------------------------------- */

void FixBcRegionMP::init()
{
	int i,j;
	// set constants that depend on fix mvv

	for (i = 0; i < modify->nfix; i++)
		if (strcmp(modify->fix[i]->style,"bc/region/mp") == 0) break;
	if (i < modify->nfix)
	{
		for (int j = i; j < modify->nfix; j++)
			if (strcmp(modify->fix[j]->style,"mvv/dpd") == 0)
				error->all(FLERR,"BcRegion fix must come after mvv/dpd fix");
	}
  
	for (i = 0; i < modify->nfix; i++)
		if (strcmp(modify->fix[i]->style,"mvv/dpd") == 0) break;
	verlet = ((FixMvvDPD *) (modify->fix[i]))->verlet;
	//fprintf(stderr,"Fix BcBb mvv verlet: %lf\n",verlet);
	ifixmvv=i;
}

/* ---------------------------------------------------------------------- */

void FixBcRegionMP::setup(int vflag)
{
	
}

/* ---------------------------------------------------------------------- */
//bounce-back from the walls
void FixBcRegionMP::initial_integrate(int vflag)
{
	// loop over all my atoms
	// dx,dy,dz = signed distance from wall
	// skip atom if not close enough to wall
	//   if wall was set to NULL, it's skipped since lo/hi are infinity
	// compute force and torque on atom if close enough to wall
	//   via wall potential matched to pair potential
	// set shear if pair potential stores history
	//fprintf(stderr,"initial_integrate RUNNING!!!!!!!!!\n");
	double **x = atom->x;
	double **v = atom->v;
	double **f = atom->f;
    double **vest = atom->vest;
	int *mask = atom->mask;
    double *rmass = atom->rmass;
	double *mass = atom->mass;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	double pv[3];//previous velocity
	double px[3];//previous coords
	int i;
	double dtv = update->dt;
	double dtf = update->dt * force->ftm2v;
	double dtfm;
	dt = update->dt;
	double x0, y0, z0;
	double BX = domain->xprd;
	double BY = domain->yprd;
	double BZ = domain->zprd;
	double EPS = 0.5;

	for (int i = 0; i < nlocal; i++) 
	{
		if (mask[i] & groupbit) 
		{
			x0 = x[i][0] * BX0 / BX;
			y0 = x[i][1] * BY0 / BY;
			z0 = x[i][2] * BZ0 / BZ;
			if (domain->regions[iregion]->match(x0, y0, z0))
			{
				if (x0 < X0 + EPS)
				{
				    //get previous velocity -- this is true only for MVV!!!!!!
				    if (rmass) dtfm = dtf / rmass[i];
				    else dtfm = dtf / mass[type[i]];
				    pv[0] = v[i][0] - verlet * dtfm * f[i][0];
				    
				    //get previous coords
				    px[0] = x[i][0] - dtv * pv[0] - dtv * 0.5 * dtfm * f[i][0];
				    
				    fprintf(stderr, "In fix bc/region/mp when time = %d, atom %d entered the region, x[i][0] = %f, x[i][1] = %f, x[i][2] = %f, px[0] = %f, px[1] = %f, px[2] = %f\n", update->ntimestep, atom->tag[i], x[i][0], x[i][1], x[i][2], px[0], px[1], px[2]);
				    
				    //place it back -- replace with proper bounce later
				    x[i][0] = px[0];
				    
				    //update velocity
				    v[i][0] = -v[i][0];
				    
				    //update force
				    f[i][0] = -f[i][0];
				    
				    vest[i][0] = -vest[i][0];
				}
				else
				{
					//get previous velocity -- this is true only for MVV!!!!!!
				    if (rmass) dtfm = dtf / rmass[i];
				    else dtfm = dtf / mass[type[i]];
				    pv[1] = v[i][1] - verlet * dtfm * f[i][1];
				    pv[2] = v[i][2] - verlet * dtfm * f[i][2];
				    
				    //get previous coords
				    px[1] = x[i][1] - dtv * pv[1] - dtv * 0.5 * dtfm * f[i][1];
				    px[2] = x[i][2] - dtv * pv[2] - dtv * 0.5 * dtfm * f[i][2];
				    
				    fprintf(stderr, "In fix bc/region/mp when time = %d, atom %d entered the region, x[i][0] = %f, x[i][1] = %f, x[i][2] = %f, px[0] = %f, px[1] = %f, px[2] = %f\n", update->ntimestep, atom->tag[i], x[i][0], x[i][1], x[i][2], px[0], px[1], px[2]);
				    
				    //place it back -- replace with proper bounce later
				    x[i][1] = px[1];
				    x[i][2] = px[2];
				    
				    //update velocity
				    v[i][1] = -v[i][1];
				    v[i][2] = -v[i][2];
				    
				    //update force
				    f[i][1] = -f[i][1];
				    f[i][2] = -f[i][2];
				    
				    vest[i][1] = -vest[i][1];
				    vest[i][2] = -vest[i][2];
				}
			}
		}
	}
}

/* ---------------------------------------------------------------------- */

void FixBcRegionMP::post_force(int vflag)
{

}

/* ---------------------------------------------------------------------- */


/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixBcRegionMP::grow_arrays(int nmax)
{
	//shear = memory->grow_2d_double_array(shear,nmax,3,"fix_wall_gran:shear");
}

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays
------------------------------------------------------------------------- */

void FixBcRegionMP::copy_arrays(int i, int j)
{
	//shear[j][0] = shear[i][0];
	//shear[j][1] = shear[i][1];
	//shear[j][2] = shear[i][2];
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

int FixBcRegionMP::memory_usage(int nmax)
{
	int bytes = 0;
	//bytes += nmax * sizeof(int);
	//bytes += 3*nmax * sizeof(double);
	return bytes;
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixBcRegionMP::pack_exchange(int i, double *buf)
{
	int m = 0;
	//buf[m++] = shear[i][0];
	//buf[m++] = shear[i][1];
	//buf[m++] = shear[i][2];
	return m;
}

/* ----------------------------------------------------------------------
   unpack values into local atom-based arrays after exchange
------------------------------------------------------------------------- */

int FixBcRegionMP::unpack_exchange(int nlocal, double *buf)
{
	int m = 0;
	//shear[nlocal][0] = buf[m++];
	//shear[nlocal][1] = buf[m++];
	//shear[nlocal][2] = buf[m++];
	return m;
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for restart file
------------------------------------------------------------------------- */

int FixBcRegionMP::pack_restart(int i, double *buf)
{
	int m = 0;
	//buf[m++] = 4;
	//buf[m++] = shear[i][0];
	//buf[m++] = shear[i][1];
	//buf[m++] = shear[i][2];
	return m;
}

/* ----------------------------------------------------------------------
   unpack values from atom->extra array to restart the fix
------------------------------------------------------------------------- */

void FixBcRegionMP::unpack_restart(int nlocal, int nth)
{
	//double **extra = atom->extra;

	// skip to Nth set of extra values

	//int m = 0;
	//for (int i = 0; i < nth; i++) m += static_cast<int> (extra[nlocal][m]);
	//m++;

	//shear[nlocal][0] = extra[nlocal][m++];
	//shear[nlocal][1] = extra[nlocal][m++];
	//shear[nlocal][2] = extra[nlocal][m++];
}

/* ----------------------------------------------------------------------
   maxsize of any atom's restart data
------------------------------------------------------------------------- */

int FixBcRegionMP::maxsize_restart()
{
	return 0;
}

/* ----------------------------------------------------------------------
   size of atom nlocal's restart data
------------------------------------------------------------------------- */

int FixBcRegionMP::size_restart(int nlocal)
{
	return 0;
}


