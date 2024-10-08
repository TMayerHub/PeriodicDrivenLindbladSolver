#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 14:59:40 2023

@author: theresa
"""

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spinful_fermion_basis_1d
import numpy as np # generic math functions
import scipy
import matplotlib.pyplot as plt

##### define model parameters #####
L=1# system size
center=int(np.floor(L/2))
T=np.sqrt(2.0)*np.ones(L)+0j#hopping right
U=1
eps=1.0 #onsite energy
Gamma1=np.ones(L)*0.2
#Gamma1[center]=0
Gamma2=np.ones(L)*0.7
#Gamma2[center]=0

#total lenght is N=4L, due to spin up and down
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

def get_Lindbladoperator(L,T,U,eps,Gamma1,Gamma2,basis=0):
    if basis==0:
        basis=spinful_fermion_basis_1d(2*L)
        print('not using symmetries')
    H_super=defineH_super(L,T,U,eps,basis)
    D1=define_Dissipators1(L,Gamma1,basis)
    D2=define_Dissipators2(L,Gamma2,basis)

    return -1j*H_super+D1+D2

def get_leftVacuum(L_static):
    basis=L_static.basis
    states=basis.states
    data=[]
    row_ind=[]
    for state in states:
        #shifting 2*L to the right => only the 2*L leftmost remain
        spin_up=state >> 2*L
        #mask with 0..(2*L)1...(2*L)
        spin_down=state & 2**(2*L)-1
        
        #check if spinup is the same
        spin_up_fock=spin_up >> L
        spin_up_augmented=spin_up & 2**(L)-1
        equal_up=spin_up_fock==spin_up_augmented
        
        if equal_up:      
            #check if spin down is the same
            spin_down_fock=spin_down >> L
            spin_down_augmented=spin_down & 2**(L)-1
            equal_down=spin_down_fock==spin_down_augmented
            
            if equal_down:
                #get the index corresponding to this state
                k=basis.index(format(spin_up,'0{}b'.format(2*L)),\
                              format(spin_down,'0{}b'.format(2*L)))

                #count number of occupied states in Fockspace
                n=int(spin_up_fock).bit_count()+int(spin_down_fock).bit_count()
                #print(format(spin_up_fock,'0{}b'.format(L)),\
                #              format(spin_down_fock,'0{}b'.format(L)))
                #print(n)
                #calculate phase
                #if n%2:
                    #data.append((-1j)**(n))
                #else:
                    #data.append((-1j)**(n)*(-1))
                data.append((1)**(n))
                #data.append((1)**n)
                row_ind.append(k)
                print(format(spin_up,'0{}b'.format(2*L)),format(spin_down,'0{}b'.format(2*L)))
                print(k,(-1j)**(n))
    col_ind=np.zeros(len(data),dtype=int)
    data=np.array(data)
    
    #create csr matrix
    leftVacuum=scipy.sparse.csr_matrix((data,(row_ind,col_ind)))
    norm=np.sqrt(leftVacuum.H@leftVacuum)
    leftVacuum=leftVacuum/norm[0,0]
    return leftVacuum
        

def differential_rho(t,rho_n):
    return L_static_csr@rho_n

def differential_rho_T(t,rho_n_T):
    return rho_n_T@L_static_csr.T

def n(i,rho,basis,leftVacuum):
    #i counts from the middle
    i_center=int(np.floor(L/2))
    index=i_center+i
    
    n_list=[[1+0j,index]]
    
    static=[
        ["n|", n_list],
        ["|n", n_list],
        ]
    dynamic = []
    
    n_operator=hamiltonian(static,dynamic,dtype=np.complex128,basis=basis)
    n_csr=n_operator.tocsr()
    rho_norm=leftVacuum.H@rho
    rho=rho/rho_norm
    n_rho=n_csr@rho
    n_exp=leftVacuum.H@n_rho

    return n_exp[0]

def exact_Diagonalization(L_static_csr):
    w,vl,vr=scipy.linalg.eig(L_static_csr.toarray(),left=True,right=True)
    #print(abs(w))
    w_min=np.argmin(abs(w))
    rho_inf=np.array([vr[:,w_min]]).T

    #column vector
    vl0=scipy.sparse.csc_array(np.transpose([vl[:,w_min]]))
    
    norm=leftVacuum.H@rho_inf
    rho_inf=rho_inf/norm
    
    return vl0,rho_inf
        
        
        
        
        
#print(basis)

#calculate the Lindbladoperator
L_static=get_Lindbladoperator(L,T,U,eps,Gamma1,Gamma2,basis)
L_static_csr=L_static.tocsr()




#calculate leftVacuum
leftVacuum=get_leftVacuum(L_static)

##test leftVacuum
vl0,rho_inf=exact_Diagonalization(L_static_csr)
# %%


vl0[abs(vl0)< 1e-15]=np.real(vl0[abs(vl0)< 1e-15])*0

print('difference',np.sum(abs(vl0-leftVacuum)))
print(np.sum(abs(vl0.H@L_static_csr)))
n_inf=n(0,rho_inf,basis,leftVacuum)
print('n_inf',n_inf)

print('vl0')
#print(vl0)
print('leftVacuum')
#print(leftVacuum)

# %%

#print('basis index:  ',basis.index(format(3,'0{}b'.format(2*L)),\
#              format(3,'0{}b'.format(2*L))))
for i_vl, data_vl in zip(vl0.indices,vl0.data):
    data_V=leftVacuum[i_vl].data
    #if leftVacuum[i_vl].data != 
    if not data_V:
        data_V=0
    else:
        data_V=data_V[0]
    if abs(data_vl-data_V)>1e-10:
        print(format(2**(4*L)-i_vl-1,'0{}b'.format(4*L)))
        print(i_vl,'    ',f"{data_vl:.3}", data_V)
# %%
    



#initial rho
rho0=leftVacuum.toarray()


#calculate timeevolution of rho
t0=0
tf=1e2
dt=0.01

t=np.linspace(t0,tf,int(tf/dt)+1)

#rho_0T=rho0.T
#rho=scipy.integrate.solve_ivp(differential_rho_T,(t0,tf),rho_0T[0]+0j,method='RK45',t_eval=t)
#rho=rho['y']

#n_exp=n(0,rho,basis,leftVacuum)
#print('n_time evolved: ',n_exp[-1])



print('n_analytic',2*Gamma2[0]/(Gamma1[0]+Gamma2[0]))



# plt.figure()
# plt.title('time dependence of n0')
# plt.plot(t,n_exp)




















