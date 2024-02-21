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

#ifdef COMPUTE_CLASS

ComputeStyle(improper,ComputeImproper)

#else

#ifndef LMP_COMPUTE_IMPROPER_H
#define LMP_COMPUTE_IMPROPER_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeImproper : public Compute {
 public:
  ComputeImproper(class LAMMPS *, int, char **);
  ~ComputeImproper();
  void init();
  void compute_vector();

 private:
  int nsub;
  class ImproperHybrid *improper;
  double *emine;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Compute bond must use group all

Bond styles accumlate energy on all atoms.

E: Unrecognized bond style in compute bond command

Self-explanatory.

E: Energy was not tallied on needed timestep

You are using a thermo keyword that requires potentials to
have tallied energy, but they didn't on this timestep.  See the
variable doc page for ideas on how to make this work.

*/
