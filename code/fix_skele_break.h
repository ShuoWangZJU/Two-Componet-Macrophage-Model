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

FixStyle(skele/break,FixSkeleBreak)

#else

#ifndef LMP_FIX_SKELE_BREAK_H
#define LMP_FIX_SKELE_BREAK_H

#include "fix.h"

namespace LAMMPS_NS {

class FixSkeleBreak : public Fix {
 public:
  FixSkeleBreak(class LAMMPS *, int, char **);
  ~FixSkeleBreak();
  int setmask();
  void init();
  void init_list(int, class NeighList *);
  void setup(int);
  void post_integrate();
  void post_integrate_respa(int, int);

  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  void grow_arrays(int);
  void copy_arrays(int, int, int);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);
  double compute_vector(int);
  double memory_usage();
  void set_arrays(int);

 private:
  int me;
  // int nevery;
  int break_bond_type;
  int band3_bond_type;
  int groupbit_target, groupbit_excluded, groupbit_broken, groupbit_partly, groupbit_intact, inversebit_broken, inversebit_partly, inversebit_intact;
  double r_break;
  double r_breaksq;
  
  int commflag, group_added_flag, group_excluded_flag;
  int *bond_count; 
  // int *angle_count;
  // int *dihedral_count;
  // int *improper_count;
  int *bond_change_count;
  // int *angle_change_count;
  // int *dihedral_change_count;
  // int *improper_change_count;
  int *bond_initial_count;
  // int *angle_initial_count;
  // int *dihedral_initial_count;
  // int *improper_initial_count;
  int *skele_status;
  int break_bond_count,break_bond_count_total;
  int break_angle_count,break_angle_count_total;
  int break_dihedral_count,break_dihedral_count_total;
  int break_improper_count,break_improper_count_total;
  int nbreak_bond,maxbreak;
  int nbreak_angle;
  int nbreak_dihedral;
  int nbreak_improper;
  int start_time;
  // int nmax;
  
  class NeighList *list;
  
  int countflag;
  int nlevels_respa;
  
  double dt;
  tagint lastcheck;

  void check_ghosts();

  // DEBUG

  void print_bb();
  void print_copy(const char *, tagint, int, int, int, int *);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Invalid atom type in fix bond/create/break command

Self-explanatory.

E: Invalid bond type in fix bond/create/break command

Self-explanatory.

E: Cannot use fix bond/create/break with non-molecular systems

Only systems with bonds that can be changed can be used.  Atom_style
template does not qualify.

E: Inconsistent iparam/jparam values in fix bond/create/break command

If itype and jtype are the same, then their maxbond and newtype
settings must also be the same.

E: Fix bond/create/break cutoff is longer than pairwise cutoff

This is not allowed because bond creation is done using the
pairwise neighbor list.

E: Fix bond/create/break requires special_bonds lj = 0,1,1

Self-explanatory.

E: Fix bond/create/break requires special_bonds coul = 0,1,1

Self-explanatory.

W: Created bonds will not create angles, dihedrals, or impropers

See the doc page for fix bond/create/break for more info on this
restriction.

E: Could not count initial bonds in fix bond/create/break

Could not find one of the atoms in a bond on this processor.

E: New bond exceeded bonds per atom in fix bond/create/break

See the read_data command for info on setting the "extra bond per
atom" header value to allow for additional bonds to be formed.

E: New bond exceeded special list size in fix bond/create/break

See the special_bonds extra command for info on how to leave space in
the special bonds list to allow for additional bonds to be formed.

*/
