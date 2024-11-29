#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:11:39 2024

@author: theresa
"""
from numba import carray,cfunc,jit # numba helper functions
from numba import uint32,int32, int64, uint64, complex128, boolean 
from augmented_basis import augmented_basis
from augmented_basis import to_extended_vector
from augmented_basis import to_reduced_vector
from quspin.operators import hamiltonian
from quspin.basis import spinful_fermion_basis_general
import numpy as np

@jit(int64(int64,int64),locals=dict(),nopython=True,nogil=True)
def _count_particles_64(state, site_ind):
    """
    Count the number of fermions (1's) in the binary configuration of the state up to site site_ind (0-indexed).
    This works for 64-bit integers.
    """
    # Create a mask to keep only the bits up to site_ind
    mask = (0xFFFFFFFFFFFFFFFF) >> (63 - site_ind)
    f_count = state & mask

    # Apply the bitwise trick to count the number of 1's
    f_count = f_count - ((f_count >> 1) & 0x5555555555555555)
    f_count = (f_count & 0x3333333333333333) + ((f_count >> 2) & 0x3333333333333333)
    f_count = (f_count + (f_count >> 4)) & 0x0F0F0F0F0F0F0F0F
    f_count = f_count + (f_count >> 8)
    f_count = f_count + (f_count >> 16)
    f_count = f_count + (f_count >> 32)

    # Return the result after shifting right by 56 bits
    return f_count & 0x7F

def _count_total_particles_64(state):
    """
    Count the number of fermions (1's) in the binary configuration of the state up to site site_ind (0-indexed).
    This works for 64-bit integers.
    see https://en.wikipedia.org/wiki/Hamming_weight for explanations
    """
    
    f_count=state
    # Apply the bitwise trick to count the number of 1's
    f_count = f_count - ((f_count >> np.uint64(1)) & np.uint64(0x5555555555555555))
    f_count = (f_count & np.uint64(0x3333333333333333)) + ((f_count >> np.uint64(2)) & np.uint64(0x3333333333333333))
    f_count = (f_count + (f_count >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    f_count = f_count + (f_count >> np.uint64(8))
    f_count = f_count + (f_count >> np.uint64(16))
    f_count = f_count + (f_count >> np.uint64(32))

    # Return the result after shifting right by 56 bits
    return f_count & np.uint64(0x7F)

def pre_check_state2(N,s,args):
    diff_up=args[1]
    diff_down=args[2]
    #get all the even bits (starting from back, so fock space)
    norm_space=s&np.uint64(0x5555555555555555)
    #get all the odd bits (starting from back so augmented space)
    augmented_space=s&np.uint64(0xAAAAAAAAAAAAAAAA)
    return args[0]==0 or _count_total_particles_64(norm_space) == diff_up+_count_total_particles_64(augmented_space)
    
def n(i,basis):
    #i counts from the middle
    if L%2:
        i_middle=L
    #even
    else:
        i_middle=L+1

    index=i
    n_list=[[1+0j,index],[1+0j,index+2*L]]
    n_list=[[1+0j,index]]
    
    static=[
        ["n", n_list],
        ]
    dynamic = []
    #print(static)
    n_operator=hamiltonian(static,dynamic,dtype=np.complex128,basis=basis,check_herm=False,check_symm=False,check_pcon=False)
    return n_operator

def a(i,basis):
    #i counts from the middle
    if L%2:
        i_middle=L
    #even
    else:
        i_middle=L+1

    index=i
    a_list=[[1+0j,index],[1+0j,index+2*L]]
    a_list=[[1+0j,index]]
    
    static=[
        ["-", a_list],
        ]
    dynamic = []
    
    a_operator=hamiltonian(static,dynamic,dtype=np.complex128,basis=basis,check_herm=False,check_symm=False,check_pcon=False)
    return a_operator

def a_dag(i,basis):
    #i counts from the middle
    if L%2:
        i_middle=L
    #even
    else:
        i_middle=L+1

    index=i
    adag_list=[[1+0j,index],[1+0j,index+2*L]]
    adag_list=[[1+0j,index]]
    
    static=[
        ["+", adag_list],
        ]
    dynamic = []
    
    adag_operator=hamiltonian(static,dynamic,dtype=np.complex128,basis=basis,check_herm=False,check_symm=False,check_pcon=False)
    return adag_operator



def test_operators(user_basis,general_basis,site):
    a_user=a(site,user_basis)
    adag_user=a_dag(site,user_basis)
    n_user=n(site,user_basis)
    
    a_general=a(site,general_basis)
    adag_general=a_dag(site,general_basis)
    n_general=n(site,general_basis)
    
    #N_sites=2**(4*L)
    N_sitesUser=user_basis.Ns
    N_sitesGeneral=general_basis.Ns
    for state in user_basis:
        state_array=np.zeros((N_sitesUser,1))
        state_array[user_basis.index(state)]=1
        state_user=a_user.dot(state_array)
        
        state_array=np.zeros((N_sitesUser,1))
        state_array[user_basis.index(state)]=1
        state_general=a_general.dot(state_array)
        index = np.nonzero(state_user)[0] 
        if len(index)>0:
            #print(index)
            #index=index[0]
            if not(state_user[index,0]==state_general[index,0]):
                print(user_basis.int_to_state(state),'            ',general_basis.int_to_state(state))
                print(user_basis.int_to_state(N_sites-1-index[0]),'            ',general_basis.int_to_state(N_sites-1-index[0]))
                print(state_user[index,0],'             ',state_general[index,0])
                print()
        
    for state in range(N_sites-1):
        state_array=np.zeros((N_sites,1))
        state_array[N_sites-1-state]=1
       
        state_user=adag_user.dot(state_array)
        state_general=adag_general.dot(state_array)
        index = np.nonzero(state_user)[0] 
        if len(index)>0:
            #print(index)
            #index=index[0]
            if not(state_user[index,0]==state_general[index,0]):
                print(user_basis.int_to_state(state),'            ',general_basis.int_to_state(state))
                print(user_basis.int_to_state(N_sites-1-index[0]),'            ',general_basis.int_to_state(N_sites-1-index[0]))
                print(state_user[index,0],'             ',state_general[index,0])
                print()
                
L=1
user_basis=augmented_basis(L,'1sector',[0,0])
general_basis = spinful_fermion_basis_general(2*L,simple_symm=False)
#%%
site=3
#test_operators(user_basis,general_basis,site)
print(user_basis)
args=[1,0,0]

#for state in user_basis:
    #print('loop')
    #print(user_basis.int_to_state(state))
    #print(user_basis.index(state))
    #print(pre_check_state(2*L, state, args))
    #print(pre_check_state2(2*L, state, args))

extended_basis=augmented_basis(L,'extended')
state_vector1=np.zeros(user_basis.Ns)

state_vector1[0]=1
state_vector1[3]=1
print(state_vector1)
extended1=to_extended_vector(state_vector1,user_basis,extended_basis)
state_vector2=np.array([state_vector1])
extended2=to_extended_vector(state_vector2,user_basis,extended_basis)
state_vector3=state_vector2.transpose()
print(state_vector3.shape)
extended3=to_extended_vector(state_vector3,user_basis,extended_basis)
print(extended1)
print(to_reduced_vector(extended1,user_basis,extended_basis))





