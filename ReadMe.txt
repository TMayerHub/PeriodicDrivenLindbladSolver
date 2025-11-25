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

## Modules and Usage

This section describes the main classes/modules, their purpose, and how to use them in the solver workflow.

**Key functionality:**
- Builds augmented Fock basis using QuSpin.
- Allows restriction to particle difference sectors

**Inputs:**  
- Parameters dictionary including:
  - Site count
  - sector
  - Spin symmetry flag

### 1. `augmented_basis`

**Purpose:**  
Defines an augmented Fock basis according to the quspin requirements.

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

---

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

---

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

---

### 5. `Fit wrapper`

**Purpose:**  
Automates fitting of Lindblad operator parameters to a target hybridization function.

**Key functionality:**
- Calls the existing AMEA fitting routines.
- Converts fit results into a dictionary compatible with the `Lindblad` class.
- Enables integration of fitted Lindblad operators into the full solver workflow.

**Inputs:**  
- Target hybridization function and parameters of the hybridization function as well as number of bath sites

**Outputs:**  
- Parameters necessary to initialize the Lindblad Solver

### Analytic Solver (Optional)

This repository also includes an **analytic solver** for special cases, useful for **benchmarking or validating** the Lindblad solver.  
It is located in the `AnalyticSolutions/` folder and can be run independently.  

### Minimal Working Example


### Examples
The example folder shows how to combine the fit with the solver, to calculate the current through the 
dot for different potentials. 
