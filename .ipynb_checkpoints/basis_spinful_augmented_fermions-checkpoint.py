#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 10:22:16 2023

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
#
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis.user import user_basis # Hilbert space user basis
from quspin.basis.user import next_state_sig_32,op_sig_32,map_sig_32,count_particles_sig_32 # user basis data types signatures
from numba import carray,cfunc,jit # numba helper functions
from numba import uint32,int32,complex64 # numba data types
import numpy as np
import scipy.sparse as sparse
import scipy
#
##### define model parameters #####
L=1# system size
N=2*L
center=int(np.floor(L/2))
T=np.sqrt(2.0)*np.ones(N)+0j#hopping right
U=1
eps=1.0 #onsite energy
Gamma1=np.ones(N)*0.2
#Gamma1[center]=0
Gamma2=np.ones(N)*0.7
#Gamma2[center]=0
############   create spinless fermion user basis object   #############

@jit(uint32(uint32,uint32,uint32),locals=dict(f_count=uint32,relevant_bits=uint32,total=uint32,shift=uint32),nopython=True,nogil=True)
def _get_phase_32(state,site_ind,N):
 	# auxiliary function to count number of fermions, i.e. 1's in bit configuration of the state, up to site site_ind
 	# CAUTION: 32-bit integers code only!
    
    #count bits from the left up to site
    if site_ind<int(N/2):

        #count bits from the left up to the site 
        shift=N-site_ind
        relevant_bits= (0xFFFFFFFF << shift)
        f_count = state & relevant_bits;
        f_count = f_count - ((f_count >> 1) & 0x55555555);
        f_count = (f_count & 0x33333333) + ((f_count >> 2) & 0x33333333);
        f_count=(((f_count + (f_count >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
        #print('from left: ',f_count);
        return f_count
    
        #fock_state=state >> int(N/2)
        # f_count = state & ((0x7FFFFFFF) >> (31 - site_ind));
        # f_count = f_count - ((f_count >> 1) & 0x55555555);
        # f_count = (f_count & 0x33333333) + ((f_count >> 2) & 0x33333333);
        # return (((f_count + (f_count >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24

    
    else: 
        #count total number of fermions
        # total = state & ((0x7FFFFFFF) >> (31 - N));
        # total = total - ((total >> 1) & 0x55555555);
        # total = (total & 0x33333333) + ((total >> 2) & 0x33333333);
        # total = (((total + (total >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
        #print('total: ',total)
        
        
        #count number of fermions in augmented space from left, up to site
        # state_a=state & 2**int(N/2)-1
        # shift=N-site_ind
        # relevant_bits= (0xFFFFFFFF << shift) & 0xFFFFFFFF
        # f_count = state_a & relevant_bits;
        # f_count = f_count - ((f_count >> 1) & 0x55555555);
        # f_count = (f_count & 0x33333333) + ((f_count >> 2) & 0x33333333);
        # f_count=(((f_count + (f_count >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
        #print('augmented from left: ',f_count);
        
        # f_count = state & ((0x7FFFFFFF) >> (31 - site_ind));
        # f_count = f_count - ((f_count >> 1) & 0x55555555);
        # f_count = (f_count & 0x33333333) + ((f_count >> 2) & 0x33333333);
        # f_count=(((f_count + (f_count >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24
        
        shift=N-site_ind
        relevant_bits= (0xFFFFFFFF << shift)
        f_count = state & relevant_bits;
        f_count = f_count - ((f_count >> 1) & 0x55555555);
        f_count = (f_count & 0x33333333) + ((f_count >> 2) & 0x33333333);
        f_count=(((f_count + (f_count >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
        
        total = state;
        total = total - ((total >> 1) & 0x55555555);
        total = (total & 0x33333333) + ((total >> 2) & 0x33333333);
        total=(((total + (total >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
        

        
        return f_count#+total

@cfunc(op_sig_32,
	locals=dict(s=int32,sign=complex64,n=int32,b=uint32,f_count=uint32), )
def op_32(op_struct_ptr,op_str,site_ind,N,args):
    #here N corresponds to 2*N
	# using struct pointer to pass op_struct_ptr back to C++ see numba Records
    op_struct = carray(op_struct_ptr,1)[0]
    err = 0
    site_ind_o=site_ind
    site_ind = N - site_ind - 1 # convention for QuSpin for mapping from bits to sites.
    f_count = _get_phase_32(op_struct.state,site_ind_o,N)
    sign = -1 if f_count&1 else 1
    sign = 1j*sign if site_ind_o >=int(N/2) else sign
    
	#####
    #site_ind = N - site_ind - 1 # convention for QuSpin for mapping from bits to sites.

    n = (op_struct.state>>site_ind)&1 # either 0 or 1
    b = (1<<site_ind)
	#
    if op_str==43: # "+" is integer value 43 = ord("+")
        op_struct.matrix_ele *= (0.0 if n else np.conjugate(sign))
        op_struct.state ^= b # create fermion

    elif op_str==45: # "-" is integer value 45 = ord("-")
        op_struct.matrix_ele *= (sign if n else 0.0)
        op_struct.state ^= b # annihilate fermion (exclusive or)
		
    elif op_str==110: # "n" is integer value 110 = ord("n")
        op_struct.matrix_ele *= n
            
    elif op_str==73: # "I" is integer value 73 = ord("I")
        pass

    else:
        op_struct.matrix_ele = 0
        err = -1
	#
    return err
op_args=np.array([],dtype=np.uint32)

def augmented_basis_32(N,operators):
    noncommuting_bits = [(np.arange(2*N),-1)] 
    op_dict=dict(op=operators,op_args=op_args)

    # create user basiss
    return user_basis(np.uint32,2*N,op_dict,allowed_ops=set("+-nI"),sps=2,noncommuting_bits=noncommuting_bits)

########################################################################






def defineH_super(N,T,U,eps,basis):
    L=int(np.floor(N/2))
    if np.isscalar(T):
        T=np.ones(N)*T
    if np.isscalar(eps):
        eps=np.ones(N)*eps
        
    T_c=np.conjugate(T)
    i_middle=int(np.floor(L/2))

    #Hamiltonian acting on Fockspace
    hop_left = [[T[i],i,(i+1)] for i in range(N-1)] # hopping to the right
    hop_right = [[T_c[i],(i+1),(i)] for i in range(N-1)] # hopping to the left
    int_list = [[U,i_middle,i_middle+L]] # onsite interaction
    eps_list=[[eps[i],i] for i in range(N)]
    
    #Hamiltonian acting on augmented Fockspace
    hop_left_a = [[-T[i-N],i,(i+1)] for i in range(N,2*N-1)] # hopping to the right
    hop_right_a = [[-T_c[i-N],(i+1),(i)] for i in range(N,2*N-1)] # hopping to the left
    int_list_a = [[-U,N+i_middle,N+i_middle+L]] # onsite interaction
    eps_list_a=[[-eps[i-N],i] for i in range(N,2*N)]
    
    # static and dynamic lists
    static= [	
            #Hamiltonian acting on Fockspace
            
    		["+-", hop_left], # hop left
    		["+-", hop_right], # hop right
    		["nn", int_list], # onsite interaction
            ["n", eps_list], # onsite energy
            
            #Hamiltonian acting on augmented Fockspace
    		["+-", hop_left_a], # hop left
    		["+-", hop_right_a], # hop right
    		["nn", int_list_a], # onsite interaction
            ["n", eps_list_a], # onsite energy
    		]
    
    dynamic=[]
    ###### construct Hamiltonian
    return hamiltonian(static,dynamic,dtype=np.complex128,basis=basis,check_symm=False,check_herm=False,check_pcon=False)

#removes particels from the system
def define_Dissipators1(N,Gamma1,basis):
    part1 = [[-2j*Gamma1[i],i,(N+i)] for i in range(N)]
    part2 = [[-1*Gamma1[i],i,i] for i in range(N)]
    part3 = [[-1*Gamma1[i],N+i,N+i] for i in range(N)]
    
    static=[
        ["--",part1],
        ["+-",part2],
        ["+-",part3], 
        ]
    dynamic=[]
    return hamiltonian(static,dynamic,dtype=np.complex128,basis=basis,check_herm=False,check_symm=False,check_pcon=False)

#add particels to the system
def define_Dissipators2(N,Gamma2,basis):
    part1 = [[-2j*Gamma2[i],i,(N+i)] for i in range(N)]
    part2 = [[-1*Gamma2[i],i,i] for i in range(N)]
    part3 = [[-1*Gamma2[i],N+i,N+i] for i in range(N)]
    static=[
        #spin up
        ["++",part1],
        ["-+",part2],
        ["-+",part3],
        ]
    
    dynamic=[]

    return hamiltonian(static,dynamic,dtype=np.complex128,basis=basis,check_herm=False,check_symm=False,check_pcon=False)

def get_Lindbladoperator(N,T,U,eps,Gamma1,Gamma2,basis=0):
    if basis==0:
        basis=augmented_basis_32(2*N,op_32)
    H_super=defineH_super(N,T,U,eps,basis)
    D1=define_Dissipators1(N,Gamma1,basis)
    D2=define_Dissipators2(N,Gamma2,basis)

    return -1j*H_super+D1+D2

def get_leftVacuum(L_static,N):
    basis=L_static.basis
    states=basis.states
    data=[]
    row_ind=[]
    for state in states:
        #shifting 2*L to the right => only the N leftmost remain
        fock_state=state >> N
        #mask with 0..(N)1...(N)
        augmented_state=state & 2**(N)-1

            
        if fock_state==augmented_state:
            #get the index corresponding to this state
            k=basis.index(state)

            #count number of occupied states in Fockspace
            #n=int(fock_state.bit_count())
            #calculate phase
            #data.append((-1j)**(n))
            data.append(1)
            row_ind.append(k)
    col_ind=np.zeros(len(data),dtype=int)
    data=np.array(data)
    
    #create csr matrix
    leftVacuum=scipy.sparse.csr_matrix((data,(row_ind,col_ind)))
    norm=np.sqrt(leftVacuum.conj().T@leftVacuum)
    leftVacuum=leftVacuum/norm[0,0]
    return leftVacuum

def n(N,i,rho,basis,leftVacuum):
    #i counts from the middle
    L=int(np.floor(N/2))
    i_center=int(np.floor(L/2))
    index=i_center+i
    
    n_list=[[1+0j,index],[1,index+L]]
    
    static=[
        ["n", n_list],
        ]
    dynamic = []
    
    n_operator=hamiltonian(static,dynamic,dtype=np.complex128,basis=basis,check_herm=False,check_symm=False,check_pcon=False)
    n_csr=n_operator.tocsr()
    rho_norm=leftVacuum.conj().T@rho
    rho=rho/rho_norm
    n_rho=n_csr@rho
    n_exp=leftVacuum.conj().T@n_rho

    return n_exp[0]

def exact_Diagonalization(L_static_csr):
    w,vl,vr=scipy.linalg.eig(L_static_csr.toarray(),left=True,right=True)
    w_min=np.argmin(abs(w))
    rho_inf=np.array([vr[:,w_min]]).T

    #column vector
    vl0=scipy.sparse.csc_array(np.transpose([vl[:,w_min]]))
    
    norm=leftVacuum.conj().T@rho_inf
    rho_inf=rho_inf/norm
    return vl0,rho_inf


#calculate the Lindbladoperator
basis=augmented_basis_32(N,op_32)
L_static=get_Lindbladoperator(N,T,U,eps,Gamma1,Gamma2,basis)
L_static_csr=L_static.tocsr()
L_s_a=L_static_csr.toarray()

#calculate leftVacuum
leftVacuum=get_leftVacuum(L_static,N)

vl0,rho_inf=exact_Diagonalization(L_static_csr)
print(vl0)
#print(leftVacuum)
#print(np.sum(abs(vl0-leftVacuum)))
ew=vl0.conj().T@L_static_csr
ew=ew.toarray()
print('lowest eigenvalue: ',scipy.linalg.norm(ew[0]))

#n_inf=n(N,0,rho_inf,basis,vl0)
n_inf=n(N,0,rho_inf,basis,leftVacuum)

print('n_analytic',2*Gamma2[0]/(Gamma1[0]+Gamma2[0]))
print('n_inf',n_inf)







































############# test operators

i=0
state=int('1111',2)
state_i=basis.index(state)
state_array = sparse.csc_array(([1],([state_i],[0])),shape=(16,1))
    
n_list=[[1+0j,i,i]]
    
static=[
    ["+-", n_list],
    ]

dynamic = []
    
n_operator=hamiltonian(static,dynamic,dtype=np.complex128,basis=basis,check_symm=False,check_pcon=False,check_herm=False )
#print(state)
#print(basis.int_to_state(state))
state_array=state_array.toarray()


v_out=n_operator.dot(state_array)
v_out=sparse.csc_array(v_out)
print(v_out)
#j=sparse.find(v_out)

#print(basis.int_to_state(basis[j[0][0]]))










