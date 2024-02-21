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

#include "string.h"
#include "stdlib.h"
#include "fix_addforce_sum.h"
#include "atom.h"
#include "atom_masks.h"
#include "accelerator_kokkos.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "region.h"
#include "group.h"
#include "respa.h"
#include "input.h"
#include "comm.h"
#include "universe.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "force.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{NONE,CONSTANT,EQUAL,ATOM};

/* ---------------------------------------------------------------------- */

FixAddForceSum::FixAddForceSum(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 6) error->all(FLERR,"Illegal fix addforce/sum command");

  dynamic_group_allow = 1;
  scalar_flag = 1;
  vector_flag = 1;
  size_vector = 3;
  global_freq = 1;
  extscalar = 1;
  extvector = 1;
  niter = 0;
  adapt_ind = 0;

  xstr = ystr = zstr = NULL;

  if (strstr(arg[3],"v_") == arg[3]) {
    int n = strlen(&arg[3][2]) + 1;
    xstr = new char[n];
    strcpy(xstr,&arg[3][2]);
  } else {
    xvalue = force->numeric(FLERR,arg[3]);
    xstyle = CONSTANT;
  }
  if (strstr(arg[4],"v_") == arg[4]) {
    int n = strlen(&arg[4][2]) + 1;
    ystr = new char[n];
    strcpy(ystr,&arg[4][2]);
  } else {
    yvalue = force->numeric(FLERR,arg[4]);
    ystyle = CONSTANT;
  }
  if (strstr(arg[5],"v_") == arg[5]) {
    int n = strlen(&arg[5][2]) + 1;
    zstr = new char[n];
    strcpy(zstr,&arg[5][2]);
  } else {
    zvalue = force->numeric(FLERR,arg[5]);
    zstyle = CONSTANT;
  }

  // optional args

  nevery = 1;
  iregion = -1;
  idregion = NULL;
  estr = NULL;

  int iarg = 6;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"every") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix addforce/sum command");
      nevery = atoi(arg[iarg+1]);
      if (nevery <= 0) error->all(FLERR,"Illegal fix addforce/sum command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix addforce/sum command");
      iregion = domain->find_region(arg[iarg+1]);
      if (iregion == -1)
        error->all(FLERR,"Region ID for fix addforce/sum does not exist");
      int n = strlen(arg[iarg+1]) + 1;
      idregion = new char[n];
      strcpy(idregion,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"adapt") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix addforce/sum command");
      niter_max = force->inumeric(FLERR,arg[iarg+1]);
      adapt_every = force->inumeric(FLERR,arg[iarg+2]);
      q_targ = force->numeric(FLERR,arg[iarg+3]);
      adapt_ind = 1; 
      iarg += 4;
    } else if (strcmp(arg[iarg],"energy") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix addforce/sum command");
      if (strstr(arg[iarg+1],"v_") == arg[iarg+1]) {
        int n = strlen(&arg[iarg+1][2]) + 1;
        estr = new char[n];
        strcpy(estr,&arg[iarg+1][2]);
      } else error->all(FLERR,"Illegal fix addforce/sum command");
      iarg += 2;
    } else error->all(FLERR,"Illegal fix addforce/sum command");
  }

  force_flag = 0;
  foriginal[0] = foriginal[1] = foriginal[2] = foriginal[3] = 0.0;

  maxatom = atom->nmax;
  memory->create(sforce,maxatom,4,"addforce/sum:sforce");
}

/* ---------------------------------------------------------------------- */

FixAddForceSum::~FixAddForceSum()
{
  delete [] xstr;
  delete [] ystr;
  delete [] zstr;
  delete [] estr;
  delete [] idregion;
  memory->destroy(sforce);
}

/* ---------------------------------------------------------------------- */

int FixAddForceSum::setmask()
{
  datamask_read = datamask_modify = 0;

  int mask = 0;
  mask |= POST_FORCE;
  mask |= THERMO_ENERGY;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAddForceSum::init()
{
  // check variables

  if (xstr) {
    xvar = input->variable->find(xstr);
    if (xvar < 0)
      error->all(FLERR,"Variable name for fix addforce/sum does not exist");
    if (input->variable->equalstyle(xvar)) xstyle = EQUAL;
    else if (input->variable->atomstyle(xvar)) xstyle = ATOM;
    else error->all(FLERR,"Variable for fix addforce/sum is invalid style");
  }
  if (ystr) {
    yvar = input->variable->find(ystr);
    if (yvar < 0)
      error->all(FLERR,"Variable name for fix addforce/sum does not exist");
    if (input->variable->equalstyle(yvar)) ystyle = EQUAL;
    else if (input->variable->atomstyle(yvar)) ystyle = ATOM;
    else error->all(FLERR,"Variable for fix addforce/sum is invalid style");
  }
  if (zstr) {
    zvar = input->variable->find(zstr);
    if (zvar < 0)
      error->all(FLERR,"Variable name for fix addforce/sum does not exist");
    if (input->variable->equalstyle(zvar)) zstyle = EQUAL;
    else if (input->variable->atomstyle(zvar)) zstyle = ATOM;
    else error->all(FLERR,"Variable for fix addforce/sum is invalid style");
  }
  if (estr) {
    evar = input->variable->find(estr);
    if (evar < 0)
      error->all(FLERR,"Variable name for fix addforce/sum does not exist");
    if (input->variable->atomstyle(evar)) estyle = ATOM;
    else error->all(FLERR,"Variable for fix addforce/sum is invalid style");
  } else estyle = NONE;

  // set index and check validity of region

  if (iregion >= 0) {
    iregion = domain->find_region(idregion);
    if (iregion == -1)
      error->all(FLERR,"Region ID for fix addforce/sum does not exist");
  }

  if (xstyle == ATOM || ystyle == ATOM || zstyle == ATOM)
    varflag = ATOM;
  else if (xstyle == EQUAL || ystyle == EQUAL || zstyle == EQUAL)
    varflag = EQUAL;
  else varflag = CONSTANT;

  if (varflag == CONSTANT && estyle != NONE)
    error->all(FLERR,"Cannot use variable energy with "
               "constant force in fix addforce/sum");
  if ((varflag == EQUAL || varflag == ATOM) &&
      update->whichflag == 2 && estyle == NONE)
    error->all(FLERR,"Must use variable energy with fix addforce/sum");
  if (adapt_ind && varflag != CONSTANT)
    error->all(FLERR,"Cannot use adapt function with "
               "variable force in fix addforce/sum");

  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;
}

/* ---------------------------------------------------------------------- */

void FixAddForceSum::setup(int vflag)
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

void FixAddForceSum::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixAddForceSum::post_force(int vflag)
{
  int step;

  double **x = atom->x;
  double **f = atom->f;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *mask = atom->mask;
  imageint *image = atom->image;
  int nlocal = atom->nlocal;
  double masstotal;

  if (update->ntimestep % nevery) return;

  if (lmp->kokkos)
    atom->sync_modify(Host, (unsigned int) (F_MASK | MASK_MASK),
                      (unsigned int) F_MASK);

  // update region if necessary

  Region *region = NULL;
  if (iregion >= 0) {
    region = domain->regions[iregion];
    region->prematch();
  }

  // reallocate sforce array if necessary

  if ((varflag == ATOM || estyle == ATOM) && nlocal > maxatom) {
    maxatom = atom->nmax;
    memory->destroy(sforce);
    memory->create(sforce,maxatom,4,"addforce/sum:sforce");
  }

  // foriginal[0] = "potential energy" for added force
  // foriginal[123] = force on atoms before extra force added

  foriginal[0] = foriginal[1] = foriginal[2] = foriginal[3] = 0.0;
  force_flag = 0;

  // constant force
  // potential energy = - x dot f in unwrapped coords

  if (varflag == CONSTANT) {
	// calculate average force
	if (iregion >= 0) masstotal = group->mass(igroup,iregion);
	else masstotal = group->mass(igroup);
	
    double unwrap[3];
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        if (region && !region->match(x[i][0],x[i][1],x[i][2])) continue;
        domain->unmap(x[i],image[i],unwrap);
        foriginal[0] -= (xvalue*unwrap[0] + yvalue*unwrap[1] + zvalue*unwrap[2]) * rmass[i] / masstotal;
        foriginal[1] += f[i][0];
        foriginal[2] += f[i][1];
        foriginal[3] += f[i][2];
        f[i][0] += xvalue * rmass[i] / masstotal;
        f[i][1] += yvalue * rmass[i] / masstotal;
        f[i][2] += zvalue * rmass[i] / masstotal;
      }

    if (adapt_ind){
      step = update->ntimestep;
      if (step > 2 && step%adapt_every == 1 && niter < niter_max) 
        adapt_force();
    }

  // variable force, wrap with clear/add
  // potential energy = evar if defined, else 0.0
  // wrap with clear/add

  } else {

    modify->clearstep_compute();

    if (xstyle == EQUAL) xvalue = input->variable->compute_equal(xvar);
    else if (xstyle == ATOM)
      input->variable->compute_atom(xvar,igroup,&sforce[0][0],4,0);
    if (ystyle == EQUAL) yvalue = input->variable->compute_equal(yvar);
    else if (ystyle == ATOM)
      input->variable->compute_atom(yvar,igroup,&sforce[0][1],4,0);
    if (zstyle == EQUAL) zvalue = input->variable->compute_equal(zvar);
    else if (zstyle == ATOM)
      input->variable->compute_atom(zvar,igroup,&sforce[0][2],4,0);
    if (estyle == ATOM)
      input->variable->compute_atom(evar,igroup,&sforce[0][3],4,0);

    modify->addstep_compute(update->ntimestep + 1);

    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        if (region && !region->match(x[i][0],x[i][1],x[i][2])) continue;
        if (estyle == ATOM) foriginal[0] += sforce[i][3];
        foriginal[1] += f[i][0];
        foriginal[2] += f[i][1];
        foriginal[3] += f[i][2];
        if (xstyle == ATOM) f[i][0] += sforce[i][0];
        else if (xstyle) f[i][0] += xvalue;
        if (ystyle == ATOM) f[i][1] += sforce[i][1];
        else if (ystyle) f[i][1] += yvalue;
        if (zstyle == ATOM) f[i][2] += sforce[i][2];
        else if (zstyle) f[i][2] += zvalue;
      }
  }
}

/* ---------------------------------------------------------------------- */

void FixAddForceSum::post_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixAddForceSum::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   potential energy of added force
------------------------------------------------------------------------- */

double FixAddForceSum::compute_scalar()
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

double FixAddForceSum::compute_vector(int n)
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

double FixAddForceSum::memory_usage()
{
  double bytes = 0.0;
  if (varflag == ATOM) bytes = atom->nmax*4 * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   recalculate adaptive force
------------------------------------------------------------------------- */

/*void FixAddForceSum::adapt_force()
{ 
  int i,j;
  double dm[5],ff[3],vv,qq;
  char buf[BUFSIZ];
  char fname[FILENAME_MAX];
  FILE *f_read;
  
  niter++;
  if (comm->me == 0){
    ff[1] = 0.0;
    ff[2] = 0.0;
    j = universe->me/96 + 1;
    sprintf(fname,"vel_t%d.%d.plt",j,niter*adapt_every);
    f_read = fopen(fname,"r");
    if(f_read == (FILE*) NULL)
      error->one(FLERR,"Could not open vel_t file");

    fgets(buf,BUFSIZ,f_read);
    fgets(buf,BUFSIZ,f_read);
    fgets(buf,BUFSIZ,f_read);
    sscanf(buf,"%lf %lf %lf %lf",&dm[0],&dm[1],&dm[2],&vv);
    qq = vv/20;
    ff[0] = xvalue*q_targ/qq;
    fclose(f_read);
    if (logfile) fprintf(logfile,"x: j = %d, niter = %d, f = %f, %f; q = %f, %f \n",j,niter,xvalue,ff[0],qq,q_targ);
  }
  MPI_Bcast(&ff[0],3,MPI_DOUBLE,0,world);
  xvalue = ff[0];
  yvalue = ff[1];
  zvalue = ff[2];
}*/

/* ---------------------------------------------------------------------- */
/*
void FixAddForceSum::adapt_force()
{ 
  int i,j;
  double dm[5],ff[3],da,vv,qq;
  char buf[BUFSIZ];
  char fname[FILENAME_MAX];
  FILE *f_read;
  
  niter++;
  if (comm->me == 0){
    ff[2] = 0.0;
    sprintf(fname,"vel_end.%d.plt",niter*adapt_every);
    da = 32.0*11.0/50.0/16.0;
    f_read = fopen(fname,"r");
    if(f_read == (FILE*) NULL)
      error->one(FLERR,"Could not open vel_end file");

    qq = 0.0;
    fgets(buf,BUFSIZ,f_read);
    fgets(buf,BUFSIZ,f_read);
    for (i=0; i<50*16; i++){
      fgets(buf,BUFSIZ,f_read);
      sscanf(buf,"%lf %lf %lf %lf",&dm[0],&dm[1],&dm[2],&vv);
      qq += vv*da;
    }
    ff[0] = xvalue*q_targ/qq;
    fclose(f_read);
    if (logfile) fprintf(logfile,"x: niter = %d, f = %f, %f; q = %f, %f \n",niter,xvalue,ff[0],qq,q_targ);

    sprintf(fname,"vel_t.%d.plt",niter*adapt_every);
    f_read = fopen(fname,"r");
    if(f_read == (FILE*) NULL)
      error->one(FLERR,"Could not open vel_t file");
    fgets(buf,BUFSIZ,f_read);
    fgets(buf,BUFSIZ,f_read);
    fgets(buf,BUFSIZ,f_read);
    sscanf(buf,"%lf %lf %lf %lf %lf",&dm[0],&dm[1],&dm[2],&dm[3],&vv);
    fclose(f_read);
    ff[1] = yvalue - vv; 
    if (logfile) fprintf(logfile,"y: niter = %d, f = %f, %f; vv = %f \n",niter,yvalue,ff[1],vv);
  
  }
  MPI_Bcast(&ff[0],3,MPI_DOUBLE,0,world);
  xvalue = ff[0];
  yvalue = ff[1];
  zvalue = ff[2];
}
*/

void FixAddForceSum::adapt_force()
{ 
  int i,j, meshX, meshY, meshZ, nave;
  double dm[5],ff[3],da,vv,qq, vsum, meshLx, meshLy, meshLz, fdev, qdev;
  char buf[BUFSIZ];
  char fname[FILENAME_MAX];
  FILE *f_read;
  
  niter++;
  if(niter == 1) {
      xave = 0.0;
      yave = 0.0;
  }
  nave = 10;
  if (comm->me == 0){
    ff[2] = 0.0;
    j = static_cast<int>(universe->me/15.0) + 1;
    sprintf(fname,"vel_end%d.%d.plt",j,niter*adapt_every);

    //3D flow rate, the mesh area of the plane perpendicular to the flow direction 
    //da = 32.0*11.0/50.0/16.0; 
    
    //2D flow rate: the mesh length of the line perpendicular to the flow direction

    meshX = 1;
    meshY = 50;
    meshZ = 1;
    meshLx = 1.0;
    meshLy = 25.0;
    meshLz = 1.0;
    da = meshLy/meshY;

    f_read = fopen(fname,"r");
    if(f_read == (FILE*) NULL)
      error->one(FLERR,"Could not open vel_end file");

    qq = 0.0;
    vsum = 0.0;
    fgets(buf,BUFSIZ,f_read);
    fgets(buf,BUFSIZ,f_read);
    for (i=0; i < meshY; i++){
      fgets(buf,BUFSIZ,f_read);
      sscanf(buf,"%lf %lf %lf %lf",&dm[0],&dm[1],&dm[2],&vv);
      vsum += vv;
      qq += vv*da;
    }
    ff[0] = xvalue*q_targ/qq;
    qdev = fabs(q_targ - qq)/q_targ;
    if(niter > niter_max-nave) xave += ff[0];
    fclose(f_read);
    if (logfile) fprintf(logfile,"\nx: j = %d, niter = %d, da = %f; f = %f, %f; q = %f, %f; qdev = %f; vsum = %f; velfile = %s;\n",j,niter,da,xvalue,ff[0],qq,q_targ,qdev,vsum,fname);
    sprintf(fname,"vel_t%d.%d.plt",j,niter*adapt_every);
    f_read = fopen(fname,"r");
    if(f_read == (FILE*) NULL)
      error->one(FLERR,"Could not open vel_t file");
    fgets(buf,BUFSIZ,f_read);
    fgets(buf,BUFSIZ,f_read);
    fgets(buf,BUFSIZ,f_read);
    sscanf(buf,"%lf %lf %lf %lf %lf",&dm[0],&dm[1],&dm[2],&dm[3],&vv);
    //if (logfile) fprintf(logfile,"%s: \n %f %f %f %f %f\n", fname, dm[0], dm[1], dm[2], dm[3],vv);
    fclose(f_read);
    ff[1] = yvalue - vv; 
    if(niter > niter_max-nave) yave += ff[1];
    if (logfile) fprintf(logfile,"y: j = %d, niter = %d, da = %f; f = %f, %f; vv = %f; velfile = %s; \n",j,niter,da,yvalue,ff[1],vv,fname);

    if (logfile) fprintf(logfile,"%f %f %f %f\n\n",ff[0], ff[1], xave, yave);
    if(niter == niter_max){
        ff[0] = xave/nave;
        ff[1] = yave/nave;
        if (logfile) fprintf(logfile,"%f %f\n\n",ff[0], ff[1]);
    }
  }
  MPI_Bcast(&ff[0],3,MPI_DOUBLE,0,world);
  xvalue = ff[0];
  yvalue = ff[1];
  zvalue = ff[2];
  }

/* ---------------------------------------------------------------------- */
