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

#ifdef FIX_CLASS

FixStyle(mvv,FixMVV)

#else

#ifndef FIX_MVV_H
#define FIX_MVV_H

#include "fix.h"

namespace LAMMPS_NS {

class FixMVV : public Fix {
 public:
  FixMVV(class LAMMPS *, int, char **);
  ~FixMVV();
  int setmask();
  void init();
  void initial_integrate(int);
  void final_integrate();
  void initial_integrate_respa(int,int,int);
  void final_integrate_respa(int);

  void grow_arrays(int);
  void copy_arrays(int, int);
  double memory_usage();
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);

  double lambda;
  int nlevels;
  double ***f_level; 
 private:
  double dtv,dtf;


};

}

#endif
#endif
