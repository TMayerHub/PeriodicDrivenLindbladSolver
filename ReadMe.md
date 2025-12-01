# PeriodicDrivenLindbladSolver

**PeriodicDrivenLindbladSolver** is a Python solver developed during my master's thesis for studying **strongly correlated impurity systems** driven by periodic electric fields. The solver extends the **Auxiliary Master Equation Approach (AMEA)** using the **Lindblad formalism**, **superfermion representation**, and **Floquet theory** to treat correlation and driving effects in a non-perturbative way.  

This project lays the foundation for studying more complex environments and integrating the solver into **Dynamical Mean-Field Theory (DMFT)**, enabling simulations of correlated materials under monochromatic driving, relevant for photovoltaic and other periodically-driven setups.

---

## Features

- Compute **time-domain expectation values** required for all Green's functions.
- Transform results into **Floquet and Wigner Green's functions**.
- Integrates with **AMEA fitting procedure** to reproduce physical baths by optimizing Lindblad operator parameters.
- Parallelized Python implementation for efficient computations.
- Supports spin-symmetric and general Lindblad operator setups.

---
## Install
- run requirements.txt
- run pip install -e . to run files from any folder in the repository (like test or examples), otherwise always run the python files from the root folder, when using the periodicSolver module.

## Modules and Usage

This section describes the main classes/modules, their purpose, and how to use them in the solver workflow.

### 1. `augmented_basis`

**Purpose:**  
Defines an augmented Fock basis according to the quspin requirements.

**Key functionality:**
- Builds augmented Fock basis using QuSpin.
- Allows restriction to particle difference sectors

**Inputs:**  
- Parameters dictionary including:
  - Site count
  - sector
  - Spin symmetry flag

**Outputs:**  
- A quspin basis object. 

### 2. `Lindblad`

**Purpose:**  
Defines constructs the Lindblad operator on a given basis, with the system parameters.

**Key functionality:**
- Builds augmented Fock basis using QuSpin.
- Constructs Hamiltonian (`H`) and dissipator (`L_D`) according to system parameters.
- Supports spin-symmetric and general setups.

**Inputs:**  
- Parameters dictionary including:
  - Site count
  - Hopping terms
  - Interaction strength
  - Electric field amplitude and frequency
  - Bath couplings
  - Spin symmetry flag

**Outputs:**  
- `Lindblad` object capable of generating the time evolution operator.


### 3. `calculateGreensFunction_sites`

**Purpose:**  
Computes time-domain expectation values for Green's functions for different sites of the system.

**Key functionality:**
- Evolves the density vector `|ρ(t)⟩` using the Lindblad operator until the periodic steady state is reached.
- Applies creation/annihilation operators and calculates `<a_i†(t+τ) a_j(t)>` and `<a_i(t+τ) a_j†(t)>`.
- Parallelizes computations over all required time points.

**Inputs:**  
- Sites
- Spin component (`up`, `down`, `updown`)
- Time increment `dt`
- Convergence tolerance `ε`
- Maximum iterations
- Averaging periods
- Time steps
- Write-to-file options

**Outputs:**  
- Time arrays `t` and `τ`
- Expectation values `a_ij(t+τ,t)`


### 4. `FloquetSpace`

**Purpose:**  
Converts time-domain results into Floquet or Wigner representations.

**Key functionality:**
- Performs Wigner transform over one period `P`.
- Builds Floquet matrices or Wigner GF for selected components (`retarded`, `greater`, `lesser`, `Keldysh`).
- Returns frequency-domain Green’s functions.

**Inputs:**  
- Expectation values (or file path)
- Sites
- Components of GF
- Number of Floquet/Wigner modes

**Outputs:**  
- Dictionary with Floquet/Wigner matrices, frequency points, and site indices


### Analytic Solver (Optional)

This repository also includes the calculation of analytic solutions for special cases, useful for **benchmarking or validating** the Lindblad solver.  This includes the non-interacting non-driven case, 
the non-interacting driven case for one site, in timedomain and frequncy domain (bessel).
It is located in the `AnalyticSolutions/` folder and can be run independently.  


### Test (Optional)

A repository used for checking the functionally of each class in periodicSolver seperatly. 
Can be used for testing and can be helpful for a more in depth understanding of the code. 

---

# Examples
The minimal example shows, how to plot the zero Wigner mode given the system parameters. 
To get the Floquet matrix instead one simply has to exchange 
the function calculateWignerFromFile with calculateFloquetFromFile

The example current.py shows how to combine the fit of the hybritzization function with the solver, to calculate the current through the dot for different potentials. Read trough to comments carefully, 
lines of code have to be executed one after the other. Additionally, there is also a json file, containing fitted system parameters extracted for the fit. This can be used as an example input for the solver (see also current.py). 

---
# ToDo
- RAM efficient parallelization: at the moment the scipy Runge Kutta method is used for time evolution.
  This stores the whole state vector at each timestep, which is unnecessary, one only needs the 
  expectation value. Especially when parallelizing this can lead to an overload in RAM.
  Consider implementing a timeevolution that only stores the expectation at each timestep.
  As a quickfix if you run into overload: reduce the number of n_jobs in GreensFunction_sites.stepsGreaterLesser or reduce t_step=2, since after each t_step the Runge Kutta is interupted and 
  only the expectation value is saved
- weights in fit: Running the fit for the hybritization function without weight adjustment, often leads 
  to a large missmatch between the physical and auxilary non-interacting driven Green's function and often doesn't capture the main features, especially of the keldysh component. 
  For automatic fitting one might have to adjust the fitting procedure to better capture the Green's 
  function of a driven site. As a quick fix, always check the resulting non-interacting Green's function and adjust the weights accordingly (often a higher weight around the chemical potential 
  is helpful)

---
# Details
For more Details on the Algorithm and the Physics see my Master thesis:
 Mayer T. (2025) "Impurity solver for strongly correlated periodically driven systems" (Master thesis, TU Graz) [Link](https://online.tugraz.at/tug_online/wbAbs.showThesis?pThesisNr=85229&pOrgNr=2382)
