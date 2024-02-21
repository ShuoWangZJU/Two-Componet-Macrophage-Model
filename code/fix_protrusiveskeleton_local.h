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

FixStyle(protrusiveskeleton/local,FixProtrusiveSkeletonLocal)

#else

#ifndef LMP_FIX_PROTRUSIVESKELETON_LOCAL_H
#define LMP_FIX_PROTRUSIVESKELETON_LOCAL_H

#include "fix.h"

namespace LAMMPS_NS {

class FixProtrusiveSkeletonLocal : public Fix {
 public:
  FixProtrusiveSkeletonLocal(class LAMMPS *, int, char **);
  ~FixProtrusiveSkeletonLocal();
  int setmask();
  void init();
  void setup(int);
  void min_setup(int);
  void post_force(int);
  void post_force_respa(int, int, int);
  
  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);
  //void grow_arrays(int);
  //void copy_arrays(int, int, int);
  void min_post_force(int);
  double compute_scalar();
  double compute_vector(int);
  //double memory_usage();

 private:
  double Fvalue, Fadd, Raround;
  int mem_type, ske_type, oppo_layer;
  int bystep_flag, Ndiv, durationstep, divstep, around_flag, direction_flag;
  double dir_custom[3];
  
  int varflag,iregion,adapt_ind,niter,niter_max,adapt_every;
  double foriginal[4],foriginal_all[4];
  int force_flag;
  int nlevels_respa;

  int maxatom;
  int maxbond_per_atom;
  int ske_bond_type;
  int groupbit_ske;
  int groupbit_mem;
  int groupbit_prt;
  int igroup_ske, igroup_mem, igroup_prt;
  
  int *bond_count; 
  int *bond_change_count;
  int **bond_atom_full;
  int **bond_atom_change;
  
  int *bond_in_atom;
  int *bond_out_atom;
  int *bond_tmp_atom;
  int bond_bond_count_local;
  int bond_bond_count_all;
  int *bond_bond_count_proc;
  int *displs;
  int *bond_bond_atom_local;
  int *bond_bond_atom_all;
  double *bond_bond_fx_local;
  double *bond_bond_fy_local;
  double *bond_bond_fz_local;
  double *bond_bond_fx_all;
  double *bond_bond_fy_all;
  double *bond_bond_fz_all;
  double *initial_time;

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
