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

#ifdef BOND_CLASS

BondStyle(break/ligand,BondBreakLigand)

#else

#ifndef LMP_BOND_BREAK_LIG_H
#define LMP_BOND_BREAK_LIG_H

#include "bond.h"
#include "stdio.h"

namespace LAMMPS_NS {

class BondBreakLigand : public Bond {
 public:
  BondBreakLigand(class LAMMPS *);
  ~BondBreakLigand();
  void compute(int, int);
  void coeff(int, char **);
  void init_style();
  double equilibrium_distance(int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  double single(int, double, int, int, double &);

 private:
  int *bond_index;
  double *ks,*r0,*temp,*dsig,*kr0,*rcs, *mu_targ, *qp;
  double dtv;
  double *gamc, *gamt, *sigc, *sigt;
  double bondforce;
  double wrr[4], delx, dely, delz;
  class RanMars *random;
  class MTRand *mtrand;

  void allocate();
  void generate_wrr();
  
  FILE *fp;
  FILE *fp2;
};

}

#endif
#endif
