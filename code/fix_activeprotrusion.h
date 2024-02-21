/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(activeprotrusion,FixActiveProtrusion)

#else

#ifndef LMP_FIX_ACTIVEPROTRUSION_H
#define LMP_FIX_ACTIVEPROTRUSION_H

#include "fix.h"

namespace LAMMPS_NS {

class FixActiveProtrusion : public Fix {
 public:
  FixActiveProtrusion(class LAMMPS *, int, char **);
  ~FixActiveProtrusion();
  int setmask();
  void init();
  void setup(int);
  void min_setup(int);
  void post_force(int);
  void post_force_respa(int, int, int);
  void min_post_force(int);
  double compute_scalar();
  double compute_vector(int);

 private:
  double Fvalue, Fadd, Raround;
  int bystep_flag, Ndiv, finalstep, divstep, around_flag;
  
  int varflag,iregion,adapt_ind,niter,niter_max,adapt_every;
  double foriginal[4],foriginal_all[4];
  int force_flag;
  int nlevels_respa;

  int maxatom;
  //double **protrusiveforce;
  
  int groupbit_mol;
  int groupbit_active;
  int inversebit_active;
  double r_active;
  int inactive_type;
  int active_type;

  class AtomVecEllipsoid *avec;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Region ID for fix addforce does not exist

Self-explanatory.

E: Variable name for fix addforce does not exist

Self-explanatory.

E: Variable for fix addforce is invalid style

Self-explanatory.

E: Cannot use variable energy with constant force in fix addforce

This is because for constant force, LAMMPS can compute the change
in energy directly.

E: Must use variable energy with fix addforce

Must define an energy vartiable when applyting a dynamic
force during minimization.

*/
