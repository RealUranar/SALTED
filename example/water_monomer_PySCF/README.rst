Generate training data using PySCF
----------------------------------

In what follows, we describe how to generate the training electron density data using the PySCF quantum-chemistry program.

1. The following input arguments must be added to the :code:`inp.qm` section:

   :code:`qmcode:`: define the quantum-mechanical code as :code:`pyscf`

   :code:`path2qm`: set the path where the PySCF data are going to be saved
    
   :code:`qmbasis`: define the wave function basis set for the Kohn-Sham calculation (example: :code:`cc-pvqz`)

   :code:`functional`: define the functional for the Kohn-Sham calculation (example: :code:`b3lyp`)

2. Define the auxiliary basis set using the input variable :code:`dfbasis`, as provided in the :code:`inp.qm` section. This must be chosen consistently with the wave function basis set (example: :code:`RI-cc-pvqz`). Then, add this basis set information to SALTED by running:

   :code:`python3 -m salted.get_basis_info`

3. Run PySCF to compute the Kohn-Sham density matrices: 

   :code:`python3 -m salted.pyscf.run_pyscf`

4. From the computed density matrices, perform the density fitting on the selected auxiliary basis set by running: 

   :code:`python3 -m salted.pyscf.dm2df`

Indirect prediction of electrostatic energies
---------------------------------------------

From the predicted density coefficients, it is possible to validate the model computing the electrostatic energy and compare it against the reference PySCF values. 

1. Calculate the reference energies of the water molecules used for validation, by running:

   :code:`python3 -m salted.pyscf.electro_energy`

2. Calculate the energies derived from the predicted densities on the validation set and evaluate the error in kcal/mol, by running:

   :code:`python3 -m salted.pyscf.electro_error`
