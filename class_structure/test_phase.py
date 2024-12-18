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
import augmented_basis as ab
import numpy as np
import scipy

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

def pre_check_state_sector(s,N,args):
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
        print(args[-1])
        for i in range(0,args[-1],2):
            print(i)
            diff_up=args[i]
            diff_down=args[i+1]
            norm_space_up=(s&uint64(0x5555555555555555))>>np.uint64(2*L)
            norm_space_down=(s&uint64(0x5555555555555555)) & np.uint64(2**(2*L)-1)
            
            #101010
            augmented_space_up=(s&uint64(0xAAAAAAAAAAAAAAAA))>>np.uint64(2*L)
            augmented_space_down=(s&uint64(0xAAAAAAAAAAAAAAAA))& np.uint64(2**(2*L)-1)
    
            equal_up=_count_total_particles_64(norm_space_up) == (diff_up+_count_total_particles_64(augmented_space_up))
            print(equal_up)
            equal_down=_count_total_particles_64(norm_space_down) == (diff_down+_count_total_particles_64(augmented_space_down))
            print(equal_down)
            if equal_up and equal_down:
                print('equal')
                return True
            
        return False
    
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
                
def _spin_sym(s,N,sign_ptr,args):
        L2=int(N/2)
        s_down = s & np.uint64(2**(L2)-1)
        s_up = s>>np.uint64(L2)
        return not(s==(s_up| s_down << np.uint64(N//2)))
    

def _leftVacuum(basis,L):
    
    states=basis.states
    data=[]
    row_ind=[]
        #define a mask for even and odd states  
    shitf_1= np.uint64(1)    
    mask_even=np.uint64(int('01'*(4*L//2),2))
    mask_odd=mask_even << shitf_1
    for state in states:
            #print(bin(mask_odd))
        even_bits=state&mask_even
        odd_bits=(state&mask_odd) >> shitf_1
                
        if(even_bits==odd_bits):
            k=basis.index(format(state,'0{}b'.format(4*L)))
            n=int(int(state).bit_count()/2)
            if k==1:
                data.append(np.sqrt(2)*(-1j)**n)
            else:
                data.append((-1j)**n)
            row_ind.append(k)
            
    col_ind=np.zeros(len(data),dtype=int)
    data=np.array(data)
        #create csr matrix
    leftVacuum=scipy.sparse.csr_matrix((data,(row_ind,col_ind)),shape=(basis.Ns,1))
    norm=np.sqrt(leftVacuum.T.conjugate()@leftVacuum)
    print(norm[0,0])
    leftVacuum=leftVacuum/norm[0,0]
    return leftVacuum
    
L=1
user_basis=augmented_basis(L,spin_sym=True)
#print(user_basis)
user_basis=augmented_basis(L,'restricted',[1,0,0,1])
user_basis_sym=augmented_basis(L,'restricted',[0,0],spin_sym=True)
print(user_basis)
print(user_basis_sym)



user_basis=augmented_basis(L,'restricted',[-1,0])
print('-1')
print(user_basis)

user_basis=augmented_basis(L,'restricted',[1,0])
print('+1')
print(user_basis)

print(user_basis.blocks)
general_basis = spinful_fermion_basis_general(2*L,simple_symm=True)
print(general_basis.blocks)
#%%
site=3
#test_operators(user_basis,general_basis,site)
print(user_basis)
print(user_basis.diff_sector)
args=[1,0]
args.append(len(args))
print('args: ',args)
args=np.array(args, dtype=np.uint64) 
user_basis=augmented_basis(L)
for state in user_basis:
    print(user_basis.int_to_state(state))
    #print(user_basis.index(state))
     
    print(ab.pre_check_state_sector1(state, 4*L, args))
    
user_basis=augmented_basis(L,'restricted',[1,0,1,0])
print(user_basis)

extended_basis=augmented_basis(L,'extended')
state_vector1=np.zeros(user_basis.Ns)

state_vector1[0]=1
#state_vector1[2]=1
print(state_vector1)
extended1=to_extended_vector(state_vector1,user_basis,extended_basis)
state_vector2=np.array([state_vector1])
extended2=to_extended_vector(state_vector2,user_basis,extended_basis)
state_vector3=state_vector2.transpose()
print(state_vector3.shape)
extended3=to_extended_vector(state_vector3,user_basis,extended_basis)
print(extended1)
print(to_reduced_vector(extended1,user_basis,extended_basis))
#%%
user_basis=augmented_basis(L)
user_basis=augmented_basis(L,'restricted',[0,0])
for state in user_basis:
    print(user_basis.int_to_state(state))
    #print(user_basis.index(state))
    sym_state=_spin_sym(state,4*L,0,0)
    print(sym_state)
    print()

user_basis_sym=augmented_basis(L,'restricted',[0,0],spin_sym=True)
print(user_basis_sym)

lv_sym=_leftVacuum(user_basis_sym, L)
lv=_leftVacuum(user_basis, L)

print(lv_sym)
print(lv)

    
    


