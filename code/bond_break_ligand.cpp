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
   Contributing author: Alireza Yazdani/Dmitry Fedosov
------------------------------------------------------------------------- */
#include "math.h"
//#include "mpi.h"
#include "string.h"
#include "stdlib.h"
#include "bond_break_ligand.h"
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "update.h"
#include "output.h"
#include "statistic.h"
#include "integrate.h"
#include "force.h"
#include "modify.h"
#include "fix.h"
#include "memory.h"
#include "error.h"
#include "random_mars.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BondBreakLigand::BondBreakLigand(LAMMPS *lmp) : Bond(lmp)
{
  random = NULL;
  mtrand = NULL;
  
  fp = NULL;
  fp2 = NULL;
  fp = fopen("dynamic_bond_energy.txt","w");
  fp2 = fopen("bond.txt","w");
//  if (fp && comm->me == 0) {
//	fprintf(fp,"timestep energy \n");
//	fflush(fp);
//  }
}

/* ----------------------------------------------------------------------
   free all arrays 
------------------------------------------------------------------------- */

BondBreakLigand::~BondBreakLigand()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(bond_index);
    memory->destroy(ks);
    memory->destroy(r0);
    memory->destroy(temp);
    memory->destroy(dsig);
    memory->destroy(kr0);
    memory->destroy(rcs);
    memory->destroy(mu_targ);
    memory->destroy(qp); 
    memory->destroy(gamc);
    memory->destroy(gamt);
    memory->destroy(sigc);
    memory->destroy(sigt);
  }

  if (random) delete random;
#ifdef MTRAND
  if (mtrand) delete mtrand;
#endif

  if (fp && comm->me == 0) fclose(fp);
  if (fp2 && comm->me == 0) fclose(fp2);
}

/* ---------------------------------------------------------------------- */

void BondBreakLigand::compute(int eflag, int vflag)
{
  int i,i1,i2,n,m,l,type,atom1,cc;
  double rsq,r,dr,fforce,kr,pp,lmax,rr;
  double rlogarg,kph,l0,mu,lambda;
  double dvx, dvy, dvz, vv;
  char warning[128];
  double ff[6], ebond;
  
  double ebond_ad, ebond_local, ebond_global;
  int ad_bond_num_local, ad_bond_num_global;
  
  
  int n_stress = output->n_stress;
  // int stress_ind = force->stress_ind;
  // int n_stress_tot = force->n_stress_tot;
  // int *stress_list = force->stress_list;
  
  //fprintf(stderr,"In bond break/ligand compute() start on proc %d when time = %d\n",comm->me,  update->ntimestep);

  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

  double **x = atom->x;
	double **f = atom->f;
  double **v = atom->v;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  double *bondlist_length = neighbor->bondlist_length;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

	int k,n1,n3,*slist;
  int **nspecial = atom->nspecial;
  int **special = atom->special;

  //Alireza: GPIba-vWF
  //const double lscale = 1.0e-6, tscale = 3.08e-5, fscale = 2.63e-13, escale = 2.63e-19;
  const double lscale = 9.6e-7, tscale = 2.86533e-4, fscale = 4.608e-13, escale = 4.42368e-19;
  const double Ftrans = 20.0e-12/fscale, kr01 = 0.0047*tscale, kr02 = 0.0022*tscale, sigr1 = 2.52e-9/lscale, sigr2 = 1.62e-9/lscale;
  
  int breakcount, nbreak = 0;
  
  ebond_local = 0.0;
  ebond_global = 0.0;
  ad_bond_num_local = 0;
  
  int *bond_type_local;
  int *bond_atom1_local;
  int *bond_atom2_local;
  bond_type_local = (int *) memory->smalloc(nbondlist*sizeof(int),"break/ligand:bond_type_local");
  bond_atom1_local = (int *) memory->smalloc(nbondlist*sizeof(int),"break/ligand:bond_atom1_local");
  bond_atom2_local = (int *) memory->smalloc(nbondlist*sizeof(int),"break/ligand:bond_atom2_local");
  
  for (n = 0; n < nbondlist; n++) {
    
    cc = 0;
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];
    l0 = bondlist_length[n];
	// if (update->ntimestep > 20000) fprintf(stderr,"In bond break/ligand on proc %d: timestep = %d, type = %d, l0 = %f, i1 = %d, i2 = %d, nbondlist = %d\n", comm->me, update->ntimestep, type, l0, atom->tag[i1], atom->tag[i2], nbondlist);

//if (update->ntimestep > 20545 && update->ntimestep < 20555) fprintf(stderr,"In bond_break_ligand compute() on proc %d: timestep = %d, bondlist[n][0] = %d, bondlist[n][1] = %d, bondlist[n][2] = %d, n = %d, i1 = %d, i2 = %d, bondlist_length[n] = %f, nbondlist = %d, x[i1][0] = %f, x[i1][1] = %f, x[i1][2] = %f, x[i2][0] = %f, x[i2][1] = %f, x[i2][2] = %f \n", comm->me, update->ntimestep, bondlist[n][0],bondlist[n][1],bondlist[n][2], n, atom->tag[i1],atom->tag[i2],bondlist_length[n], nbondlist, x[i1][0], x[i1][1], x[i1][2], x[i2][0], x[i2][1], x[i2][2]);

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];
    dvx = v[i1][0] - v[i2][0];
    dvy = v[i1][1] - v[i2][1];
    dvz = v[i1][2] - v[i2][2];
    domain->minimum_image(delx,dely,delz);

    rsq = delx*delx + dely*dely + delz*delz;
    r = sqrt(rsq);
    if (bond_index[type] == 1){
      lmax = l0*r0[type];
      rr = 1.0/r0[type];  
      kph = pow(l0,qp[type])*temp[type]*(0.25/(1.0-rr)/(1.0-rr)-0.25+rr);
      mu = 0.25*sqrt(3.0)*(temp[type]*(-0.25/pow(1.0-rr,2) + 0.25 + 0.5*rr/pow(1.0-rr,3))/lmax/rr + kph*(qp[type]+1.0)/pow(l0,qp[type]+1.0));
      lambda = mu/mu_targ[type];
      kph = kph*mu_targ[type]/mu;
      rr = r/lmax; 
      rlogarg = pow(r,qp[type]+1.0); 
      vv = (delx*dvx + dely*dvy +  delz*dvz)/r;
     
      if (rr >= 1.0) {
        sprintf(warning,"WLC bond too long: %d %d %d %d %g",
                update->ntimestep,atom->tag[i1],atom->tag[i2],type,rr);
//        sprintf(warning,"WLC bond too long: %d %d %d %d %g %g %g %g %g %g",
//                update->ntimestep,atom->tag[i1],atom->tag[i2],type,rr,r,lmax,rsq,l0,r0[type]);
        error->warning(FLERR, warning, 0);
      }  
       
      generate_wrr();
      fforce = - temp[type]*(0.25/(1.0-rr)/(1.0-rr)-0.25+rr)/lambda/r + kph/rlogarg + (sigc[type]*wrr[3] - gamc[type]*vv)/r;

      // energy
      ebond = 0.0;
      if (eflag) {
          ebond += 0.25*temp[type]*lmax*(3.0*rr*rr-2.0*rr*rr*rr)/(1.0-rr)/lambda;
        if (qp[type] == 1.0)
          ebond -= kph*log(r);
        else
          ebond += kph/(qp[type]-1.0)/pow(r,qp[type]-1.0);
      }
	  //bond_type_local[ad_bond_num_local] = type;
	  //bond_atom1_local[ad_bond_num_local] = atom->tag[i1];
	  //bond_atom2_local[ad_bond_num_local] = atom->tag[i2];
	  //ad_bond_num_local++;
    }
    else if (bond_index[type] == 2) {
	  ebond = 0.0;
	  ebond_ad = 0.0;
	  
	  bondforce = ks[type]*fabs(r-r0[type]); //Alireza: GPIba-vWF
/*	  if (bondforce < Ftrans)
	  	kr = kr01*exp(sigr1*bondforce/temp[type]);
	  else
	  	kr = kr02*exp(sigr2*bondforce/temp[type]);
*/
      kr = kr0[type]*exp(-0.5*dsig[type]*(r-r0[type])*(r-r0[type])/temp[type]);
      pp = 1.0 - exp(-kr*dtv);
      fforce = 0.0;
	  m = 0;
      while (m < atom->num_bond[i1]){
  	  	atom1 = atom->map(atom->bond_atom[i1][m]);
	  	if (atom1 == i2) {
		  if (random->uniform() < pp || r > rcs[type]){
			fprintf(stdout,"At timestep = %d, bond broken %e %e %d %d, i1 = %d, i2 = %d\n", update->ntimestep, r-r0[type], pp, atom->tag[i1], atom->tag[i2], i1, i2);
			cc++;
			nbreak++;
	    	l = atom->num_bond[i1];
	    	atom->bond_atom[i1][m] = atom->bond_atom[i1][l-1];
	    	atom->bond_type[i1][m] = atom->bond_type[i1][l-1];
			atom->bond_length[i1][m] = atom->bond_length[i1][l-1];
	    	atom->num_bond[i1]--;

			// remove J from special bond list for atom I
			// atom J will also do this
		
			slist = special[i1];
			n1 = nspecial[i1][0];
			n3 = nspecial[i1][2];
			for (k = 0; k < n1; k++)
			  if (slist[k] == i2) break;
			for (; k < n3-1; k++) slist[k] = slist[k+1];
			nspecial[i1][0]--;
			nspecial[i1][1]--;
			nspecial[i1][2]--;
		
			slist = special[i2];
			n1 = nspecial[i2][0];
			n3 = nspecial[i2][2];
			for (k = 0; k < n1; k++)
			  if (slist[k] == i1) break;
			for (; k < n3-1; k++) slist[k] = slist[k+1];
			nspecial[i2][0]--;
			nspecial[i2][1]--;
			nspecial[i2][2]--;

	  	  }
		  else{
			dr = r - r0[type];
			// force & energy
	       	if (dr > 0.0) fforce += -2.0 * ks[type] * dr / r;
			if (eflag) {
			  ebond += rr*dr;
			  ebond_ad += rr*dr;
			}
			m++;
			bond_type_local[ad_bond_num_local] = type;
			bond_atom1_local[ad_bond_num_local] = atom->tag[i1];
			bond_atom2_local[ad_bond_num_local] = atom->tag[i2];
			ad_bond_num_local++;
			// if (update->ntimestep > 20000) fprintf(stderr,"In bond break/ligand on proc %d: timestep = %d, ad_bond_num_local = %d, i1 = %d, i2 = %d, nbondlist = %d\n", comm->me, update->ntimestep, ad_bond_num_local, atom->tag[i1], atom->tag[i2], nbondlist);
		  }
	    }
		else m++;
      }
      if (cc != 0)	neighbor->bond_reneigh = 1;
    }
	else fforce = 0.0;

    if (bond_index[type] == 1){
      if (newton_bond || i1 < nlocal){
        f[i1][0] += delx*fforce - gamt[type]*dvx + sigt[type]*wrr[0]/r;
        f[i1][1] += dely*fforce - gamt[type]*dvy + sigt[type]*wrr[1]/r;
        f[i1][2] += delz*fforce - gamt[type]*dvz + sigt[type]*wrr[2]/r;
      }

      if (newton_bond || i2 < nlocal){
        f[i2][0] -= delx*fforce - gamt[type]*dvx + sigt[type]*wrr[0]/r;
        f[i2][1] -= dely*fforce - gamt[type]*dvy + sigt[type]*wrr[1]/r;
        f[i2][2] -= delz*fforce - gamt[type]*dvz + sigt[type]*wrr[2]/r;
      }
    } 
		else if (bond_index[type] == 2){
      if (newton_bond || i1 < nlocal){
        f[i1][0] += delx*fforce;
        f[i1][1] += dely*fforce;
        f[i1][2] += delz*fforce;
		//fprintf(stderr, "In bond_break_ligand.cpp compute() i1 at timestep = %d at proc %d: i1 = %d, i2 = %d, atom->tag[i1] = %d, atom->tag[i2] = %d, r = %f, dr = %f, fforce = %f, f1x0 = %f, f1y0 = %f, f1z0 = %f, f1x = %f, f1y = %f, f1z = %f, x1 = %f, y1 = %f, z1 = %f \n", update->ntimestep, comm->me, i1, i2, atom->tag[i1], atom->tag[i2], r, dr, fforce, f[i1][0] - delx * fforce, f[i1][1] - delx * fforce, f[i1][2] - delx * fforce, f[i1][0], f[i1][1], f[i1][2], x[i1][0], x[i1][1], x[i1][2]);
      }

      if (newton_bond || i2 < nlocal){
        f[i2][0] -= delx*fforce;
        f[i2][1] -= dely*fforce;
        f[i2][2] -= delz*fforce;
		//fprintf(stderr, "In bond_break_ligand.cpp compute() i2 at timestep = %d at proc %d: i1 = %d, i2 = %d, atom->tag[i1] = %d, atom->tag[i2] = %d, r = %f, dr = %f, fforce = %f, f2x0 = %f, f2y0 = %f, f2z0 = %f, f2x = %f, f2y = %f, f2z = %f, x2 = %f, y2 = %f, z2 = %f \n", update->ntimestep, comm->me, i1, i2, atom->tag[i1], atom->tag[i2], r, dr, fforce, f[i2][0] - delx * fforce, f[i2][1] - delx * fforce, f[i2][2] - delx * fforce, f[i2][0], f[i2][1], f[i2][2], x[i2][0], x[i2][1], x[i2][2]);
      }
    }

    // virial contribution
    if (n_stress){
      ff[0] = delx*delx*fforce;
      ff[1] = dely*dely*fforce;
      ff[2] = delz*delz*fforce;
      ff[3] = delx*dely*fforce;
      ff[4] = delx*delz*fforce;
      ff[5] = dely*delz*fforce;

      for(k = 0; k < n_stress; k++){
        l = output->stress_id[k];
        if ((output->next_stat_calc[l] == update->ntimestep) && (output->last_stat_calc[l] != update->ntimestep))
          output->stat[l]->virial3(i1,i2,ff);
      }
    }

	if (evflag) {
	  ev_tally(i1,i2,nlocal,newton_bond,ebond,fforce,delx,dely,delz);
	  
	  ebond_local += ebond_ad;
	}
  }

  MPI_Allreduce(&nbreak,&breakcount,1,MPI_INT,MPI_SUM,world);
  atom->nbonds -= breakcount;
  
  MPI_Allreduce(&ebond_local,&ebond_global,1,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(&ad_bond_num_local,&ad_bond_num_global,1,MPI_INT,MPI_SUM,world);
  
  int nprocs = comm->nprocs;
  int *ad_bond_num_proc;
  int *displs;
  int *bond_type_all;
  int *bond_atom1_all;
  int *bond_atom2_all;
  int kn;
  ad_bond_num_proc = (int *) memory->smalloc(nprocs*sizeof(int),"break/ligand:ad_bond_num_proc");
  displs = (int *) memory->smalloc(nprocs*sizeof(int),"break/ligand:displs");
  
  if (nprocs > 1) MPI_Allgather(&ad_bond_num_local, 1, MPI_INT, ad_bond_num_proc, 1, MPI_INT, world);
  kn = 0;
  for (i = 0; i < nprocs; i++){
    displs[i] = kn;
    kn += ad_bond_num_proc[i];
  }
  bond_type_all = (int *) memory->smalloc(kn*sizeof(int),"break/ligand:bond_type_all");
  bond_atom1_all = (int *) memory->smalloc(kn*sizeof(int),"break/ligand:bond_atom1_all");
  bond_atom2_all = (int *) memory->smalloc(kn*sizeof(int),"break/ligand:bond_atom2_all");
  if (nprocs > 1) {
	MPI_Allgatherv(bond_type_local, ad_bond_num_local, MPI_INT, bond_type_all, ad_bond_num_proc, displs, MPI_INT, world);
	MPI_Allgatherv(bond_atom1_local, ad_bond_num_local, MPI_INT, bond_atom1_all, ad_bond_num_proc, displs, MPI_INT, world);
	MPI_Allgatherv(bond_atom2_local, ad_bond_num_local, MPI_INT, bond_atom2_all, ad_bond_num_proc, displs, MPI_INT, world);
  }
  
  if (fp && comm->me == 0 && update->ntimestep % 1000 == 0) {
	fprintf(fp, "%d %f %d\n", update->ntimestep, ebond_global, ad_bond_num_global);
	fflush(fp);
  }
  
  if (fp2 && update->ntimestep % 1000 == 0) {
	if (comm->me == 0) {
	  fprintf(fp2, "ITEM: TIMESTEP\n");
	  fprintf(fp2, "%d\n", update->ntimestep);
	  fprintf(fp2, "ITEM: NUMBER OF ENTRIES\n");
	  fprintf(fp2, "%d\n", ad_bond_num_global);
	  fprintf(fp2, "ITEM: BOX BOUNDS pp pp pp\n");
	  fprintf(fp2, "%f %f\n", domain->boxlo[0],domain->boxhi[0]);
	  fprintf(fp2, "%f %f\n", domain->boxlo[1],domain->boxhi[1]);
	  fprintf(fp2, "%f %f\n", domain->boxlo[2],domain->boxhi[2]);
	  fprintf(fp2, "ITEM: ENTRIES c_b1[1] c_b1[2] c_b1[3]\n");
	  for (i = 0; i < ad_bond_num_global; i++) {
		fprintf(fp2, "%d %d %d\n", bond_type_all[i], bond_atom1_all[i], bond_atom2_all[i]);
	  }
	  fflush(fp2);
	}
	//for (n = 0; n < nbondlist; n++) {
	//  i1 = bondlist[n][0];
    //  i2 = bondlist[n][1];
    //  type = bondlist[n][2];
	//  if (bond_index[type] == 2) {
	//	while (m < atom->num_bond[i1]){
  	//  	  atom1 = atom->map(atom->bond_atom[i1][m]);
	//  	  if (atom1 == i2) {
	//	    fprintf(fp2, "%d %d %d\n", type, i1, i2);
	//		fflush(fp2);
	//	  }
	//	  m++;
	//	}
	//  }
	//}
	// fflush(fp2);
  }

  l = 0;
  MPI_Allreduce(&neighbor->bond_reneigh,&l,1,MPI_INT,MPI_MAX,world);
  neighbor->bond_reneigh = l;
  
  memory->destroy(bond_type_local);
  memory->destroy(bond_atom1_local);
  memory->destroy(bond_atom2_local);
  memory->destroy(ad_bond_num_proc);
  memory->destroy(displs);
  memory->destroy(bond_type_all);
  memory->destroy(bond_atom1_all);
  memory->destroy(bond_atom2_all);
  //fprintf(stderr,"In bond break/ligand compute() end on proc %d when time = %d\n",comm->me,  update->ntimestep);
}

/* ---------------------------------------------------------------------- */

void BondBreakLigand::allocate()
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

  memory->create(ks,n+1,"bond:ks");
  memory->create(r0,n+1,"bond:r0");
  memory->create(temp,n+1,"bond:temp");
  memory->create(dsig,n+1,"bond:dsig");
  memory->create(kr0,n+1,"bond:kr0");
  memory->create(rcs,n+1,"bond:rcs");
  memory->create(mu_targ,n+1,"bond:mu_targ");
  memory->create(qp,n+1,"bond:qp");
  memory->create(bond_index,n+1,"bond:bond_index");
  memory->create(gamc,n+1,"bond:gamc");
  memory->create(gamt,n+1,"bond:gamt");
  memory->create(sigc,n+1,"bond:sigc");
  memory->create(sigt,n+1,"bond:sigt");
  
  memory->create(setflag,n+1,"bond:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;
  
  
  // ks = (double *) memory->smalloc((n+1)*sizeof(double),"bond:ks");
  // r0 = (double *) memory->smalloc((n+1)*sizeof(double),"bond:r0");
  // temp = (double *) memory->smalloc((n+1)*sizeof(double),"bond:temp");
  // dsig = (double *) memory->smalloc((n+1)*sizeof(double),"bond:dsig");
  // kr0 = (double *) memory->smalloc((n+1)*sizeof(double),"bond:kr0");
  // rcs = (double *) memory->smalloc((n+1)*sizeof(double),"bond:rcs");
  // mu_targ = (double *) memory->smalloc((n+1)*sizeof(double),"bond:lambda");
  // qp = (double *) memory->smalloc((n+1)*sizeof(double),"bond:qp");
  // bond_index = (int *) memory->smalloc((n+1)*sizeof(int),"bond:bond_index");
  // gamc = (double *) memory->smalloc((n+1)*sizeof(double),"bond:gamc");
  // gamt = (double *) memory->smalloc((n+1)*sizeof(double),"bond:gamt");
  // sigc = (double *) memory->smalloc((n+1)*sizeof(double),"bond:sigc");
  // sigt = (double *) memory->smalloc((n+1)*sizeof(double),"bond:sigt");
  // 
  // setflag = (int *) memory->smalloc((n+1)*sizeof(int),"bond:setflag");
  // for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs from one line in input script
------------------------------------------------------------------------- */

void BondBreakLigand::coeff(int narg, char **arg)
{
  int count, i;
  double temp_one, rcs_one, mu_targ_one, qp_one, ks_one, r0_one, dsig_one, kr0_one, gamc_one, gamt_one;    

  if (!allocated) allocate();

  int ilo,ihi;
  force->bounds(arg[0],atom->nbondtypes,ilo,ihi);

  if (strcmp(arg[1],"rbc") == 0 || strcmp(arg[1],"plat") == 0) {
    if (narg != 8) error->all(FLERR,"Incorrect args in bond_coeff command");
    temp_one = force->numeric(FLERR,arg[2]);
    r0_one = force->numeric(FLERR,arg[3]);
    mu_targ_one = force->numeric(FLERR,arg[4]);
    qp_one = force->numeric(FLERR,arg[5]);
    gamc_one = force->numeric(FLERR,arg[6]);
    gamt_one = force->numeric(FLERR,arg[7]);

    count = 0;
    for (i = ilo; i <= ihi; i++) {
      bond_index[i] = 1;
      temp[i] = temp_one;
      r0[i] = r0_one;
      mu_targ[i] = mu_targ_one;
      qp[i] = qp_one;
      gamc[i] = gamc_one;
      gamt[i] = gamt_one;
      setflag[i] = 1;
      count++;
    }
    if (count == 0) error->all(FLERR,"Incorrect args in bond_coeff command");
  }
  else if (strcmp(arg[1],"ligand") == 0) {
    if (narg != 8) error->all(FLERR,"Incorrect args in bond_coeff command");
    ks_one = force->numeric(FLERR,arg[2]);
    r0_one = force->numeric(FLERR,arg[3]);
    kr0_one = force->numeric(FLERR,arg[4]);
    dsig_one = force->numeric(FLERR,arg[5]);
    temp_one = force->numeric(FLERR,arg[6]);
    rcs_one = force->numeric(FLERR,arg[7]);

    count = 0;
    for (i = ilo; i <= ihi; i++) {
      bond_index[i] = 2;
      ks[i] = ks_one;
      r0[i] = r0_one;
      kr0[i] = kr0_one; 
      dsig[i] = dsig_one;
      temp[i] = temp_one;
      rcs[i] = rcs_one;
      setflag[i] = 1;
      count++;
    }
    if (count == 0) error->all(FLERR,"Incorrect args in bond_coeff command");
  } else
    error->all(FLERR,"Incorrect bond definition in bond_coeff command");  
}

/* ----------------------------------------------------------------------
   error check and initialize all values needed for force computation
------------------------------------------------------------------------- */

void BondBreakLigand::init_style()
{
  double sdtt = sqrt(update->dt);

  if (!allocated) error->all(FLERR,"Bond coeffs are not set");
  for (int i = 1; i <= atom->nbondtypes; i++){
//    if (setflag[i] == 0) error->all(FLERR,"All bond coeffs are not set");
    if (bond_index[i] == 1){
      if (gamt[i] > 3.0*gamc[i]) error->all(FLERR,"Gamma_t > 3*Gamma_c");
      sigc[i] = sqrt(2.0*temp[i]*(3.0*gamc[i]-gamt[i]))/sdtt;
      sigt[i] = 2.0*sqrt(gamt[i]*temp[i])/sdtt;
    }
  }

  dtv = update->dt;
}

/* ----------------------------------------------------------------------
   return an equilbrium bond length 
------------------------------------------------------------------------- */

double BondBreakLigand::equilibrium_distance(int i)
{
  return r0[i];
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file 
------------------------------------------------------------------------- */

void BondBreakLigand::write_restart(FILE *fp)
{
  int i;

  for (i = 1; i <= atom->nbondtypes; i++) {
    fwrite(&bond_index[i],sizeof(int),1,fp);
    if (bond_index[i] == 1) {
      fwrite(&temp[i],sizeof(double),1,fp);
      fwrite(&r0[i],sizeof(double),1,fp);
      fwrite(&mu_targ[i],sizeof(double),1,fp);
      fwrite(&qp[i],sizeof(double),1,fp);
      fwrite(&gamc[i],sizeof(double),1,fp);
      fwrite(&gamt[i],sizeof(double),1,fp);
    } else {
      fwrite(&ks[i],sizeof(double),1,fp);
      fwrite(&r0[i],sizeof(double),1,fp);
      fwrite(&kr0[i],sizeof(double),1,fp);
      fwrite(&dsig[i],sizeof(double),1,fp);
      fwrite(&temp[i],sizeof(double),1,fp);
      fwrite(&rcs[i],sizeof(double),1,fp);
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them 
------------------------------------------------------------------------- */

void BondBreakLigand::read_restart(FILE *fp)
{
  int i;

  allocate();

	for (i = 1; i <= atom->nbondtypes; i++){
		if (comm->me == 0) {
       fread(&bond_index[i],sizeof(int),1,fp);  
       if (bond_index[i] == 1){
         fread(&temp[i],sizeof(double),1,fp);
         fread(&r0[i],sizeof(double),1,fp);
         fread(&mu_targ[i],sizeof(double),1,fp);
         fread(&qp[i],sizeof(double),1,fp);
         fread(&gamc[i],sizeof(double),1,fp);
         fread(&gamt[i],sizeof(double),1,fp);
       } else {
         fread(&ks[i],sizeof(double),1,fp);
         fread(&r0[i],sizeof(double),1,fp);
         fread(&kr0[i],sizeof(double),1,fp);
         fread(&dsig[i],sizeof(double),1,fp);
         fread(&temp[i],sizeof(double),1,fp);
         fread(&rcs[i],sizeof(double),1,fp);
       }
  	}
  	MPI_Bcast(&bond_index[i],1,MPI_INT,0,world);
		if (bond_index[i] == 1){
  		MPI_Bcast(&temp[i],1,MPI_DOUBLE,0,world);
  		MPI_Bcast(&r0[i],1,MPI_DOUBLE,0,world);
  		MPI_Bcast(&mu_targ[i],1,MPI_DOUBLE,0,world);
  		MPI_Bcast(&qp[i],1,MPI_DOUBLE,0,world);
  		MPI_Bcast(&gamc[i],1,MPI_DOUBLE,0,world);
  		MPI_Bcast(&gamt[i],1,MPI_DOUBLE,0,world);
		} else {
	  	MPI_Bcast(&ks[i],1,MPI_DOUBLE,0,world);
  		MPI_Bcast(&r0[i],1,MPI_DOUBLE,0,world);
	  	MPI_Bcast(&kr0[i],1,MPI_DOUBLE,0,world);
  		MPI_Bcast(&dsig[i],1,MPI_DOUBLE,0,world);
  		MPI_Bcast(&temp[i],1,MPI_DOUBLE,0,world);
  		MPI_Bcast(&rcs[i],1,MPI_DOUBLE,0,world);
		}
	}

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ---------------------------------------------------------------------- */

double BondBreakLigand::single(int type, double rsq, int i, int j, double &fforce)
{
  return 0.0; 
}

/* ----------------------------------------------------------------------*/

void BondBreakLigand::generate_wrr()
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
