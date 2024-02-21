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

FixStyle(bc/region/mp,FixBcRegionMP)

#else

#ifndef LMP_FIX_BC_REGION_MP_H
#define LMP_FIX_BC_REGION_MP_H

#include "fix.h"

namespace  LAMMPS_NS {

class FixBcRegionMP : public Fix {
 public:
  FixBcRegionMP(class LAMMPS *, int, char **);
  ~FixBcRegionMP();
  int setmask();
  void init();
  void setup(int);
  void post_force(int);
  void initial_integrate(int);

  void grow_arrays(int);
  void copy_arrays(int, int);
  int memory_usage(int);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);
  int pack_restart(int, double *);
  void unpack_restart(int, int);
  int size_restart(int);
  int maxsize_restart();

 private:
  int iregion,pairstyle;
  double dt;
  double verlet;
  int ifixmvv;
  double BX0, BY0, BZ0;
  double X0;

};

}
#endif
#endif
