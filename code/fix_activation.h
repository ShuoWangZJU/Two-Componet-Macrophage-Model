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

FixStyle(activation,FixActivation)

#else

#ifndef LMP_FIX_ACTIVATION_H
#define LMP_FIX_ACTIVATION_H

#include "fix.h"

namespace LAMMPS_NS {

class FixActivation : public Fix {
 public:
  FixActivation(class LAMMPS *, int, char **);
  ~FixActivation();
  int setmask();
  void init();
  void init_list(int, class NeighList *);
  void post_integrate();
  void post_integrate_respa(int, int);

 private:
  int me;
  int groupbit_trigger;
  int groupbit_active;
  int inversebit_active;
  int groupbit_active_prt;
  int inversebit_active_prt;
  int inactive_type;
  int active_type;
  double r_active;
  int dt;
  
  int trigger_num_local, trigger_num_all;
  int *trigger_num_proc;
  int *trigger_list_local;
  int *trigger_list_all;
  double *trigger_x_local;
  double *trigger_y_local;
  double *trigger_z_local;
  double *trigger_x_all;
  double *trigger_y_all;
  double *trigger_z_all;
  int *displs;
  
  int nmax;
  class NeighList *list;
  int nlevels_respa;
  int lastcheck;
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