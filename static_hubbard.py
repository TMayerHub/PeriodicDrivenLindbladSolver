#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 14:59:40 2023

@author: theresa
"""

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spinful_fermion_basis_1d
import numpy as np # generic math functions

##### define model parameters #####
L=3# system size
T=np.sqrt(2.0)*np.ones(L)+1j#hopping right
T_c=np.conjugate(T) #hopping left
print(T_c.dtype)

i_middle=int(np.floor(L/2))
U=np.zeros(L) 
U[i_middle]=2 #onsite interaction

eps=1.0 #onsite energy

basis = spinful_fermion_basis_1d(L)
#print(basis)

##### define PBC site-coupling lists for operators
# define site-coupling lists
hop_left = [[T[i],i,(i+1)] for i in range(L-1)] # hopping to the right PBC
hop_right = [[T_c[i],(i+1),(i)] for i in range(L-1)] # hopping to the left PBC
int_list = [[U[i],i,i] for i in range(L)] # onsite interaction
# static and dynamic lists
static= [	
		["+-|", hop_left], # up hop left
		["+-|", hop_right], # up hop right
		["|+-", hop_left], # down hop left
		["|+-", hop_right], # down hop right
		["n|n", int_list], # onsite interaction
		]

dynamic=[]
###### construct Hamiltonian
H=hamiltonian(static,dynamic,dtype=np.complex128,basis=basis)