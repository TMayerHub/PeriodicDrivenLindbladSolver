#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:26:50 2024

@author: theresa
"""

from __future__ import print_function, division
#
import sys,os
#os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
#os.environ['OMP_NUM_THREADS']='4' # set number of OpenMP threads to run in parallel
#os.environ['MKL_NUM_THREADS']='4' # set number of MKL threads to run in parallel
#
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)



from quspin.basis.user import user_basis # Hilbert space user basis
from quspin.basis.user import next_state_sig_64, pre_check_state_sig_64,op_sig_64,map_sig_64 # user basis data types signatures
from numba import carray,cfunc,jit # numba helper functions
from numba import int32,uint32,int64, uint64, boolean # numba data types
import numpy as np
from scipy.special import comb

from joblib import Parallel, delayed




@cfunc(map_sig_64,
        locals=dict(s_up=uint64, s_down=uint64, ), )
def _spin_sym(s,N,sign_ptr,args):
    """
    Symmetry mapping function for spin states. Maps a given spin state into 
    its symmetric counterpart by switching the spinup and down states using
    bit-shifting. 
    
    Parameters:
    s : int
        The current spin state in binary representation.
    N : int
        Total number of sites.
    sign_ptr : predefined quspin structure
    notused here
    args : predefined quspin structure
        notused here
    Returns:
    int
        Spin symmetric counterpart
    """
    L2=int(N/2)
    s_down = s & np.uint64(2**(L2)-1)
    s_up = s>>np.uint64(L2)
    return s_up| s_down << N//2


@jit(int64(int64,int64),locals=dict(),nopython=True,nogil=True)
def _count_particles_64(state, site_ind):
    """
    Count the number of fermions (1's) in the binary configuration of the state up to site site_ind (0-indexed).
    This works for 64-bit integers.
    see https://en.wikipedia.org/wiki/Hamming_weight for explanations
    
    Parameters:
    state : int
        The binary representation of the state.
    site_ind : int
        The site index (0-indexed) up to which particles are counted.
    
    Returns:
    int
        Number of fermions (1's) up to the specified site index.
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
    Counts the total number of fermions (1's) in the binary configuration 
    of a state.
    
    Parameters:
    state : int
        The binary representation of the state.
    
    Returns:
    int
        Total number of fermions (1's) in the state.
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
    """
    Defines the action of quantum operators (+,-,n,1) on a state
    see quspin documation

    Parameters:
    op_struct_ptr : pointer
        Pointer to the operator structure, including the current state and matrix element.
    op_str : int
        Operator type ('+' = 43, '-' = 45, 'n' = 110, 'I' = 73).
    site_ind : int
        Site index where the operator is applied.
    N : int
        Total number of sites.
    args : tuple
        Additional arguments (not used here).
    
    Returns:
    int
        Error code: 0 for success, -1 for invalid operator.
    """
    
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
    Generates the next state recursively in a binary configuration space. 
    Used for creating all possible states.
    
    Parameters:
    s : int
        The current state in binary representation.
    counter : int
        Not used, necessary for quspin in the background.
    N : int
        Total number of sites.
    args : tuple
        Additional arguments (not used here).
    
    Returns:
    int
        The next state in binary representation.
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



def get_s0_pcon(N, Np):
    """
    Defines the starting state to create the basis
    
    Parameters:
    N : int
        Total number of sites.
    Np : int
        Number of particles.
    
    Returns:
    int
        The starting state (always 0 for this implementation).
    """
    return 0

# python function to calculate the size of the particle-conserved basis,
# i.e. BEFORE applying pre_check_state and symmetry maps
def get_Ns_pcon(N, Np):
    """
    Calculates the size of the particle-conserved basis before applying 
    symmetry or pre-check filters.
    
    Parameters:
    N : int
        Total number of sites.
    Np : int
        Number of particles.
    
    Returns:
    int
        Size of the particle-conserving basis.
    """
    return sum([comb(N,i,exact=True) for i in range(9)])


pcon_dict = dict(
    Np=(),
    next_state=next_state,
    next_state_args=next_state_args,
    get_Ns_pcon=get_Ns_pcon,
    get_s0_pcon=get_s0_pcon,
)


@cfunc(pre_check_state_sig_64,
    locals=dict(diff_up=int64,diff_down=int64,norm_space_up = uint64, augmented_space_up = uint64,norm_space_down = uint64, 
                augmented_space_down = uint64),)
def pre_check_state_sector1(s,N,args):
    """
    Pre-check function for filtering states in a specific sector. 
    Compares up and down spin components based on predefined differences.

    Parameters:
    s : int
        The current state in binary representation.
    N : int
        Total number of sites.
    args : array
        Array of differences for spin-up and spin-down components.

    Returns:
    bool
        True if the state satisfies the constraints, False otherwise.
    """
        #get all the even bits (odd sites)
        ##augmented_space=s&uint64(0x5555555555555555)
        #get all the odd bits (even sites)
        ##norm_space=s&uint64(0xAAAAAAAAAAAAAAAA)
        #shifting 2*L to the right => only the 2*L leftmost remain
        ##spin_up=norm_space >> np.uint64(2*L)
        #masking the left part of the state
        ##spin_down=norm_space & np.uint64(2**(2*L)-1)
    L=int(N/4)
        #0101010
        
    for i in range(0,2,2):
        diff_up=args[i]
        diff_down=args[i+1]
        norm_space_up=(s&uint64(0x5555555555555555))>>np.uint64(2*L)
        norm_space_down=(s&uint64(0x5555555555555555)) & np.uint64(2**(2*L)-1)
            
        #101010
        augmented_space_up=(s&uint64(0xAAAAAAAAAAAAAAAA))>>np.uint64(2*L)
        augmented_space_down=(s&uint64(0xAAAAAAAAAAAAAAAA))& np.uint64(2**(2*L)-1)
    
        equal_up=_count_total_particles_64(norm_space_up) == (diff_up+_count_total_particles_64(augmented_space_up))
        equal_down=_count_total_particles_64(norm_space_down) == (diff_down+_count_total_particles_64(augmented_space_down))
        if equal_up and equal_down:
            return True
            
    return False
    
@cfunc(pre_check_state_sig_64,
    locals=dict(diff_up=int64,diff_down=int64,norm_space_up = uint64, augmented_space_up = uint64,norm_space_down = uint64, 
                augmented_space_down = uint64),)
def pre_check_state_sector2(s,N,args):
        ''''''
        
        
        #get all the even bits (odd sites)
        ##augmented_space=s&uint64(0x5555555555555555)
        #get all the odd bits (even sites)
        ##norm_space=s&uint64(0xAAAAAAAAAAAAAAAA)
        #shifting 2*L to the right => only the 2*L leftmost remain
        ##spin_up=norm_space >> np.uint64(2*L)
        #masking the left part of the state
        ##spin_down=norm_space & np.uint64(2**(2*L)-1)
        L=int(N/4)
        #0101010
        
        for i in range(0,4,2):
            diff_up=args[i]
            diff_down=args[i+1]
            norm_space_up=(s&uint64(0x5555555555555555))>>np.uint64(2*L)
            norm_space_down=(s&uint64(0x5555555555555555)) & np.uint64(2**(2*L)-1)
            
            #101010
            augmented_space_up=(s&uint64(0xAAAAAAAAAAAAAAAA))>>np.uint64(2*L)
            augmented_space_down=(s&uint64(0xAAAAAAAAAAAAAAAA))& np.uint64(2**(2*L)-1)
    
            equal_up=_count_total_particles_64(norm_space_up) == (diff_up+_count_total_particles_64(augmented_space_up))
            equal_down=_count_total_particles_64(norm_space_down) == (diff_down+_count_total_particles_64(augmented_space_down))
            if equal_up and equal_down:
                return True
            
        return False

@cfunc(pre_check_state_sig_64,
    locals=dict(norm_space = uint64, augmented_space = uint64),)
def pre_check_state_extended(s,N,args):
        #odd sites 01010
        norm_space=s&uint64(0x5555555555555555)
        
        #even sites 10101
        augmented_space=s&uint64(0xAAAAAAAAAAAAAAAA)
        
        
        total_equal0=_count_total_particles_64(norm_space) == (0+_count_total_particles_64(augmented_space))
        total_equalp1=_count_total_particles_64(norm_space) == (1+_count_total_particles_64(augmented_space))
        total_equalm1=(1+_count_total_particles_64(norm_space)) == _count_total_particles_64(augmented_space)
        return total_equal0 or total_equalp1 or total_equalm1


    #else:
       # n_up_diff=args[0]
        #n_up_diff=args[0]
        
#pre_check_state_args = np.array([1,0,0], dtype=np.uint64)    

def augmented_basis(N,particle_sectors=None,sector=None,spin_sym=False):
    op_args=np.array([],dtype=np.uint64)
    op_dict=dict(op=op,op_args=op_args)
    noncommuting_bits = [(np.arange(N), -1)]
    sym_args = np.array([], dtype=np.uint64)
    maps = dict(spin_block=(_spin_sym, 2, -1/2, sym_args))
    if particle_sectors==None:
        #,noncommuting_bits=noncommuting_bits
        #pcon_dict=pcon_dict,
        if spin_sym:
            basis=user_basis(np.uint64, 4*N, op_dict, allowed_ops = set("+-nI"),**maps)
        else:
            basis=user_basis(np.uint64, 4*N, op_dict, allowed_ops = set("+-nI"))
        basis.sector_type = None
        basis.diff_sector = sector
        return basis
    
    if particle_sectors == 'extended':
        pre_check_state_args = np.array([], dtype=np.uint64) 
        pre_check_states = (
            pre_check_state_extended,
            pre_check_state_args)
        
        #noncommuting_bits=noncommuting_bits,
        if spin_sym:
            basis= user_basis(np.uint64, 4*N, op_dict, allowed_ops = set("+-nI"),pre_check_state=pre_check_states,noncommuting_bits=noncommuting_bits,**maps)
        else:
            basis = user_basis(np.uint64, 4*N, op_dict, allowed_ops = set("+-nI"),pre_check_state=pre_check_states)
        basis.sector_type = 'extended'
        basis.diff_sector = None
        return basis
    
    if particle_sectors == 'restricted':
        pre_check_state_args = np.array(sector, dtype=np.uint64) 
        if len(sector)==2:
            pre_check_states = (
                pre_check_state_sector1,
                pre_check_state_args)
        if len(sector)==4:
            pre_check_states = (
                pre_check_state_sector2,
                pre_check_state_args)
        if spin_sym:
            if len(sector)==2 and not((sector==[0,0])):
                raise ValueError('Cannot define a spin symmetric basis with unsymmetric restrictions')
            basis = user_basis(np.uint64, 4*N, op_dict, allowed_ops = set("+-nI"),pre_check_state=pre_check_states,noncommuting_bits=noncommuting_bits,**maps)
        else:
            basis = user_basis(np.uint64, 4*N, op_dict, allowed_ops = set("+-nI"),pre_check_state=pre_check_states)
        basis.sector_type = 'restricted'
        basis.diff_sector = sector
        return basis
        
    #return user_basis(np.uint64, 4*N, op_dict, allowed_ops = set("+-nI"),noncommuting_bits=noncommuting_bits,pre_check_state=pre_check_states)



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
    

