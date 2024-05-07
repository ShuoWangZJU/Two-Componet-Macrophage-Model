# Two-Componet-Macrophage-Model

The source code and examples for the paper [S. Wang, S. Ma, H. Li, M. Dao, X. Li, & G. E. Karniadakis. Two-component macrophage model for active phagocytosis with pseudopod formation. *Biophysical Journal*, 123, 1069-1084, 2024](https://doi.org/10.1016/j.bpj.2024.03.026).

## Compilation
The source code is developed on the basis of LAMMPS. To compile the code:
```
cd <working_copy>/code
make g++_openmpi
```

## Running examples
There are four examples in the `examples` directory, including:
1. Passive phagocytosis of a target particle without protrusion by a macrophage;
2. Active phagocytosis of a target particle with active protrusive forces by a macrophage;
3. Active phagocytosis of a target particle achieved by the growth and retraction of selected pseudopods from the macrophage membrane;
4. Active phagocytosis of a target particle achieved by the growth and retraction of randomly generated pseudopods from the macrophage membrane.

To run the examples:
```
cd <working_copy>/examples/<example_dir>
run -n N_proc ../../code/lmp_g++_openmpi -in in.cell >out.txt
```