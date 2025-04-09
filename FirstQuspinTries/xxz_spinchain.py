#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:24:38 2023

@author: theresa
"""

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions

##### define model parameters #####
L=3# system size
Jxy=np.sqrt(2.0) # xy interaction
Jzz_0=1.0 # zz interaction
hz=1.0/np.sqrt(3.0) # z external field

##### set up Heisenberg Hamiltonian in an enternal z-field #####
# compute spin-1/2 basis
basis = spin_basis_1d(L,pauli=False)
print(basis)
# define operators with OBC using site-coupling lists
J_zz = [[Jzz_0,i,i+1] for i in range(L-1)] # OBC
J_xy = [[Jxy/2.0,i,i+1] for i in range(L-1)] # OBC
h_z=[[hz,i] for i in range(L-1)]

# static and dynamic lists
static = [["+-",J_xy],["-+",J_xy],["zz",J_zz]]

dynamic=[]

# compute the time-dependent Heisenberg Hamiltonian
H_XXZ = hamiltonian(static,dynamic,basis=basis,dtype=np.float64)
H_XXZ=H_XXZ.toarray()
print(type(H_XXZ))