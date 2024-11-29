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
import sparse
from joblib import Parallel, delayed
from itertools import permutations

class augmented_basis(user_basis):
    
    def __init__(self,L,Nf=None,nf=None,s_symmetry=False):
        self.L=L
        self.Nf=Nf
        self.nf=nf
        self.s_symmetry=s_symmetry
        
        
        
    @cfunc(map_sig_64,
            locals=dict(s_up=uint64, s_down=uint64, ), )
    def spin_sym(s,N,sign_ptr,args):
            s_down = s >> N//2
            s_up = s & args[0]
            return s >> N//2 | (s & args[0]) << N//2
    
    @jit(uint64(uint64),locals=dict(state=uint64,),nopython=True,nogil=True)
    def reverse_mask(self,x):
        """
        Reverts a binary input.
        """
        state = x
        state = ((state & 0x5555555555555555) << 1) | ((state & 0xAAAAAAAAAAAAAAAA) >> 1)
        state = ((state & 0x3333333333333333) << 2) | ((state & 0xCCCCCCCCCCCCCCCC) >> 2)
        state = ((state & 0x0F0F0F0F0F0F0F0F) << 4) | ((state & 0xF0F0F0F0F0F0F0F0) >> 4)
        state = ((state & 0x00FF00FF00FF00FF) << 8) | ((state & 0xFF00FF00FF00FF00) >> 8)
        state = ((state & 0x0000FFFF0000FFFF) << 16) | ((state & 0xFFFF0000FFFF0000) >> 16)
        state = ((state & 0x00000000FFFFFFFF) << 32) | ((state & 0xFFFFFFFF00000000) >> 32)
        return state
    
    @jit(int64(int64,int64),locals=dict(),nopython=True,nogil=True)
    def _count_particles(self,state, site_ind):
        """
        Counts the numbers of ones in the binary input, which have the physical interpretation of particles.
        For more information on the binary representation look into the documentation of this code.
        """
        state = self.reverse_mask(state) >> (64 - site_ind)
        state = (state & 0x555555555555555) + ((state & 0xAAAAAAAAAAAAAAAA) >> 1)
        state = (state & 0x3333333333333333) + ((state & 0xCCCCCCCCCCCCCCCC) >> 2)
        state = (state & 0x0F0F0F0F0F0F0F0F) + ((state & 0xF0F0F0F0F0F0F0F0) >> 4)
        state = (state & 0x00FF00FF00FF00FF) + ((state & 0xFF00FF00FF00FF00) >> 8)
        state = (state & 0x0000FFFF0000FFFF) + ((state & 0xFFFF0000FFFF0000) >> 16)
        state = (state & 0x00000000FFFFFFFF) + ((state & 0xFFFFFFFF00000000) >> 32) # This last & isn't strictly necessary.
        return state
    
    @cfunc(op_sig_64,
     	locals=dict(sign=int32,n=int64,b=int64,f_count=uint32), )
    def op(self,op_struct_ptr,op_str,site_ind,N,args):
        """
        This function is called during the creation of the matrix. It defines which operators have which
        action on a state.
        """
        op_struct = carray(op_struct_ptr,1)[0]
        err = 0
        site_ind = N - site_ind - 1 # convention for QuSpin for mapping from bits to sites.
        
        f_count = self._count_particles(op_struct.state, site_ind) 
        sign = -1 if f_count&1 else 1
        if site_ind == 0:
            sign = 1
        
        n = (op_struct.state>>site_ind)&1 # either 0 or 1
        b = (1<<site_ind)
    
        if op_str==43: # "+" is integer value 43 = ord("+")
            op_struct.matrix_ele *= (0.0 if n else sign)
            op_struct.state ^= b # create fermion
     
        elif op_str==45: # "-" is integer value 45 = ord("-")
            op_struct.matrix_ele *= (sign if n else 0.0)
            op_struct.state ^= b # annihilate fermion
    
        elif op_str==110: # "n" is integer value 110 = ord("n")
           		op_struct.matrix_ele *= n
    
        elif op_str==73: # "I" is integer value 73 = ord("I")
         		pass
       
        else:
         		op_struct.matrix_ele = 0
         		err = -1
        return err
    
    
    
    def augmented_basis64(self,N,operators):
        op_args=np.array([],dtype=np.uint64)
        noncommuting_bits = [(np.arange(2*N),-1)] 
        op_dict=dict(op=operators,op_args=op_args)
    
        # create user basiss
        return user_basis(np.uint64,2*N,op_dict,allowed_ops=set("+-nI"),sps=2,noncommuting_bits=noncommuting_bits)
    
    
