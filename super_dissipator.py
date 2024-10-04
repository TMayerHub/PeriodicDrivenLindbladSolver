#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:21:32 2023

@author: theresa
"""

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spinful_fermion_basis_1d
import numpy as np # generic math functions
import scipy
import matplotlib.pyplot as plt

##### define model parameters #####
L=3# system size
center=int(np.floor(L/2))
T=np.sqrt(2.0)*np.ones(L)+0j#hopping right
U=1
eps=1.0 #onsite energy
Gamma1=np.ones(L)*0.2
#Gamma1[center]=0
Gamma2=np.ones(L)*0.7
#Gamma2[center]=0

basis = spinful_fermion_basis_1d(2*L)


def defineH_super(L,T,U,eps,basis):
    if np.isscalar(T):
        T=np.ones(L)*T
    if np.isscalar(eps):
        eps=np.ones(L)*eps
        
    T_c=np.conjugate(T)
    i_middle=int(np.floor(L/2))

    #Hamiltonian acting on Fockspace
    hop_left = [[T[i],i,(i+1)] for i in range(L-1)] # hopping to the right
    hop_right = [[T_c[i],(i+1),(i)] for i in range(L-1)] # hopping to the left
    int_list = [[U,i_middle,i_middle] for i in range(L)] # onsite interaction
    eps_list=[[eps[i],i] for i in range(L)]
    
    #Hamiltonian acting on augmented Fockspace
    hop_left_a = [[-T[i-L],i,(i+1)] for i in range(L,2*L-1)] # hopping to the right
    hop_right_a = [[-T_c[i-L],(i+1),(i)] for i in range(L,2*L-1)] # hopping to the left
    int_list_a = [[-U,L+i_middle,L+i_middle] for i in range(L,2*L)] # onsite interaction
    eps_list_a=[[-eps[i-L],i] for i in range(L,2*L)]
    
    # static and dynamic lists
    static= [	
            #Hamiltonian acting on Fockspace
            
    		["+-|", hop_left], # up hop left
    		["+-|", hop_right], # up hop right
    		["|+-", hop_left], # down hop left
    		["|+-", hop_right], # down hop right
    		["n|n", int_list], # onsite interaction
            ["n|", eps_list], # onsite energy
            ["|n", eps_list], # onsite energy
            
            #Hamiltonian acting on augmented Fockspace
    		["+-|", hop_left_a], # up hop left
    		["+-|", hop_right_a], # up hop right
    		["|+-", hop_left_a], # down hop left
    		["|+-", hop_right_a], # down hop right
    		["n|n", int_list_a], # onsite interaction
            ["n|", eps_list_a], # onsite energy
            ["|n", eps_list_a], # onsite energy
    		]
    
    dynamic=[]
    ###### construct Hamiltonian
    return hamiltonian(static,dynamic,dtype=np.complex128,basis=basis)

#removes particels from the system
def define_Dissipators1(L,Gamma1,basis):
    part1 = [[-2j*Gamma1[i],i,(L+i)] for i in range(L)]
    part2 = [[-1*Gamma1[i],i,i] for i in range(L)]
    part3 = [[-1*Gamma1[i],L+i,L+i] for i in range(L)]
    
    static=[
        #spin up
        ["--|",part1],
        ["+-|",part2],
        ["+-|",part3],
        #spin down
        ["|--",part1],
        ["|+-",part2],
        ["|+-",part3],   
        ]
    dynamic=[]
    D1=hamiltonian(static,dynamic,dtype=np.complex128,basis=basis,check_herm=False)
    
    D1=D1.tocsr()
    
    return hamiltonian(static,dynamic,dtype=np.complex128,basis=basis,check_herm=False)

#add particels to the system
def define_Dissipators2(L,Gamma2,basis):
    part1 = [[-2j*Gamma2[i],i,(L+i)] for i in range(L)]
    part2 = [[-1*Gamma2[i],i,i] for i in range(L)]
    part3 = [[-1*Gamma2[i],L+i,L+i] for i in range(L)]
    #print(part1)
    #print(part2)
    #print(part3)
    static=[
        #spin up
        ["++|",part1],
        ["-+|",part2],
        ["-+|",part3],
        #spin down
        ["|++",part1],
        ["|-+",part2],
        ["|-+",part3],   
        ]
    
    dynamic=[]

    return hamiltonian(static,dynamic,dtype=np.complex128,basis=basis,check_herm=False)