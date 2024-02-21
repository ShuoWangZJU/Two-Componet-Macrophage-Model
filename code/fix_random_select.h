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

FixStyle(random/select,FixRandomSelect)

#else

#ifndef LMP_FIX_RANDOM_SELECT_H
#define LMP_FIX_RANDOM_SELECT_H

#include "fix.h"

namespace LAMMPS_NS {

class FixRandomSelect : public Fix {
 public:
  FixRandomSelect(class LAMMPS *, int, char **);
  ~FixRandomSelect();
  int setmask();
  void init();
  void init_list(int, class NeighList *);
  void post_integrate();
  void post_integrate_respa(int, int);
  
  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);

 private:
  int me;
  int group_num, max_sel_num;
  int select_mode;
  int probability;
  int groupbit_selected;
  int inversebit_selected;
  int exclude_flag;
  int igroup_exclude;
  int groupbit_exclude;
  int duration_flag;
  int duration_time;
  int maxatom;
  int existing_selected;
  
  int group_num_local, group_num_all;
  int *group_num_proc;
  int *group_list_local;
  int *group_list_all;
  int *displs;
  int *initial_timestep;
  
  int nmax, dt;
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