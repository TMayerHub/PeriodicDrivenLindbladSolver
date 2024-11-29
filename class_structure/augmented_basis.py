#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:26:50 2024

@author: theresa
"""

from __future__ import print_function, division
#
import sys,os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='1' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel
#
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)



from quspin.basis.user import user_basis # Hilbert space user basis
from quspin.basis.user import next_state_sig_64, pre_check_state_sig_64,op_sig_64,map_sig_64 # user basis data types signatures
from numba import carray,cfunc,jit # numba helper functions
from numba import uint32,int32, int64, uint64, complex128, boolean # numba data types
import numpy as np
from scipy.special import comb
from scipy.sparse import diags
import scipy
from scipy import sparse
from joblib import Parallel, delayed
from itertools import permutations



@cfunc(map_sig_64,
        locals=dict(s_up=uint64, s_down=uint64, ), )
def spin_sym(s,N,sign_ptr,args):
        s_down = s >> N//2
        s_up = s & args[0]
        return s >> N//2 | (s & args[0]) << N//2


@jit(int64(int64,int64),locals=dict(),nopython=True,nogil=True)
def _count_particles_64(state, site_ind):
    """
    Count the number of fermions (1's) in the binary configuration of the state up to site site_ind (0-indexed).
    This works for 64-bit integers.
    see https://en.wikipedia.org/wiki/Hamming_weight for explanations
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

@jit(uint64(uint64),locals=dict(),nopython=True,nogil=True)
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

@cfunc(op_sig_64,
 	locals=dict(sign=int32,n=int64,b=int64,f_count=uint32), )
def op(op_struct_ptr, op_str, site_ind, N, args):
    # using struct pointer to pass op_struct_ptr back to C++ see numba Records
    op_struct = carray(op_struct_ptr, 1)[0]
    err = 0
    #
    site_ind = N - site_ind - 1  # convention for QuSpin for mapping from bits to sites.
    #####
    f_count = _count_particles_64(op_struct.state, site_ind)
    #####
    sign = -1 if f_count &1 else 1
    n = (op_struct.state >> site_ind) & 1  # either 0 or 1
    b = 1 << site_ind
    #
    if op_str == 43:  # "+" is integer value 43 = ord("+")
        op_struct.matrix_ele *= 0.0 if n else sign
        op_struct.state ^= b  # create fermion
    elif op_str == 45:  # "-" is integer value 45 = ord("-")
        op_struct.matrix_ele *= sign*(-1) if n else 0.0
        op_struct.state ^= b  # create fermion
    elif op_str == 110:  # "n" is integer value 110 = ord("n")
        op_struct.matrix_ele *= n
    elif op_str == 73:  # "I" is integer value 73 = ord("I")
        pass
    else:
        op_struct.matrix_ele = 0
        err = -1
    #
    return err


@cfunc(next_state_sig_64,
	locals=dict(t=int64), )
def next_state(s,counter,N,args):
    """
    This function creates the next state
    recursively. Each state is created using the last one.
    After we have all possible states with a certain occupation/excitation we increase the
    quasiparticle number/exciation by one and go through the process again until we reached
    the user defined limit.
    """

    if(s==0): 
        return 0b1;
    
    for i in range(12):
        cond = 0
        for j in range(i): cond = cond | 1 << N-(j+1)
        if s == cond:
            return sum([2**k for k in range(i+1)])
        
    t = (s | (s - 1)) + 1
    return t | ((((t & (0-t)) // (s & (0-s))) >> 1) - 1)

next_state_args = np.array([], dtype=np.uint64)  # compulsory, even if empty


# python function to calculate the starting state to generate the particle conserving basis
def get_s0_pcon(N, Np):
    return 0

# python function to calculate the size of the particle-conserved basis,
# i.e. BEFORE applying pre_check_state and symmetry maps
def get_Ns_pcon(N, Np):
    return sum([comb(N,i,exact=True) for i in range(9)])

pcon_dict = dict(
    Np=(),
    next_state=next_state,
    next_state_args=next_state_args,
    get_Ns_pcon=get_Ns_pcon,
    get_s0_pcon=get_s0_pcon,
)


@cfunc(pre_check_state_sig_64,
    locals=dict(diff_up=uint64,diff_down=uint64,norm_space = uint64, augmented_space = uint64),)
def pre_check_state_sector(s,N,args):
        '''still need to add that sectors hold seperatly for spin up and spin down as well (more restrictive)'''
        diff_up=args[0]
        diff_down=args[1]
        #get all the even bits
        norm_space=s&uint64(0x5555555555555555)
        #get all the odd bits
        augmented_space=s&uint64(0xAAAAAAAAAAAAAAAA)
        

        total_equal=_count_total_particles_64(norm_space) == (diff_up+_count_total_particles_64(augmented_space))
        return total_equal

@cfunc(pre_check_state_sig_64,
    locals=dict(norm_space = uint64, augmented_space = uint64),)
def pre_check_state_extended(s,N,args):
        #get all the even bits
        norm_space=s&uint64(0x5555555555555555)
        #get all the odd bits
        augmented_space=s&uint64(0xAAAAAAAAAAAAAAAA)
        
        
        total_equal0=_count_total_particles_64(norm_space) == (0+_count_total_particles_64(augmented_space))
        total_equalp1=_count_total_particles_64(norm_space) == (1+_count_total_particles_64(augmented_space))
        total_equalm1=(1+_count_total_particles_64(norm_space)) == _count_total_particles_64(augmented_space)
        return total_equal0 or total_equalp1 or total_equalm1


    #else:
       # n_up_diff=args[0]
        #n_up_diff=args[0]
        
#pre_check_state_args = np.array([1,0,0], dtype=np.uint64)    

def augmented_basis(N,particle_sectors=None,sector=None):
    op_args=np.array([],dtype=np.uint64)
    op_dict=dict(op=op,op_args=op_args)
    noncommuting_bits = [(np.arange(N), -1)]
    if particle_sectors==None:
        #,noncommuting_bits=noncommuting_bits
        #pcon_dict=pcon_dict,
        return user_basis(np.uint64, 4*N, op_dict, allowed_ops = set("+-nI"))
    
    if particle_sectors == 'extended':
        pre_check_state_args = np.array([], dtype=np.uint64) 
        pre_check_states = (
            pre_check_state_extended,
            pre_check_state_args)
    
    if particle_sectors == '1sector':
        pre_check_state_args = np.array(sector, dtype=np.uint64)   
        pre_check_states = (
            pre_check_state_sector,
            pre_check_state_args)
        
    return user_basis(np.uint64, 4*N, op_dict, allowed_ops = set("+-nI"),noncommuting_bits=noncommuting_bits,pre_check_state=pre_check_states)



def get_extendedIndices(basis,extended_basis):
    #find index location of first occurrence of each value of interest
    #sorter = np.argsort(x)
    #sorter[np.searchsorted(x, vals, sorter=sorter)]
    
    sorter = np.argsort(extended_basis[:])
    ind_extended = sorter[np.searchsorted(extended_basis[:], basis[:], sorter=sorter)]
    return ind_extended


def to_extended_vector(state_vector,basis,extended_basis):

    v_shape=state_vector.shape
    if len(v_shape)==2:
        rows = v_shape[0]
        cols = v_shape[1]
        
        if rows == 1:
            extended_vector=np.zeros(extended_basis.Ns)
            extended_vector[get_extendedIndices(basis,extended_basis)]=state_vector[0]
            return extended_vector.reshape((1,extended_basis.Ns))
        
        if cols == 1:
            extended_vector=np.zeros(extended_basis.Ns)
            extended_vector[get_extendedIndices(basis,extended_basis)]=state_vector[:,0]
            return extended_vector.reshape((extended_basis.Ns,1))
    
    if len(v_shape) ==1:
        extended_vector=np.zeros(extended_basis.Ns)
        extended_vector[get_extendedIndices(basis,extended_basis)]=state_vector
        return extended_vector
    
def to_reduced_vector(state_vector,basis,extended_basis):
    #find index location of first occurrence of each value of interest
    #sorter = np.argsort(x)
    #sorter[np.searchsorted(x, vals, sorter=sorter)]
    v_shape=state_vector.shape
    if len(v_shape)==2:
        rows = v_shape[0]
        cols = v_shape[1]
        
        if rows == 1:
            state_vector=state_vector[0]
            reduced_vector=state_vector[get_extendedIndices(basis,extended_basis)]
            return reduced_vector.reshape((1,basis.Ns))
        
        if cols == 1:
            state_vector=state_vector[:,0]
            reduced_vector=state_vector[get_extendedIndices(basis,extended_basis)]
            return reduced_vector.reshape((basis.Ns,1))
    
    if len(v_shape) ==1:
        reduced_vector=state_vector[get_extendedIndices(basis,extended_basis)]
        return reduced_vector
    

