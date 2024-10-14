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
# %%
##### define model parameters #####
L=5# system size
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

# %%
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
    hop_left_a = [[-1*T[i-L],i,(i+1)] for i in range(L,2*L-1)] # hopping to the right
    hop_right_a = [[-1*T_c[i-L],(i+1),(i)] for i in range(L,2*L-1)] # hopping to the left
    int_list_a = [[-1*U,L+i_middle,L+i_middle] for i in range(L,2*L)] # onsite interaction
    eps_list_a=[[-1*eps[i-L],i] for i in range(L,2*L)]
    
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
    #add phase due to augmented space
    part1 = [[-2j*Gamma2[i],i,(L+i)] for i in range(L)]
    part2 = [[-1*Gamma2[i],i,i] for i in range(L)]
    part3 = [[-1*Gamma2[i],L+i,L+i] for i in range(L)]
    
    static=[
        #spin up
        ["++|",part1],
        ["-+|",part2],
        ["-+|",part3],
        #["|",part4],
        #spin down
        ["|++",part1],
        ["|-+",part2],
        ["|-+",part3],
        #["|",part4]
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

    #return -1j*H_super+D1+D2
    return H_super+1j*(D1+D2)  # multiply by -i due to .evolve structure

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

                if n%2:
                    data.append(1j*(-1)**L)
                else: data.append((-1)**(L-int(spin_up_fock).bit_count()))

                row_ind.append(k)

    col_ind=np.zeros(len(data),dtype=int)
    data=np.array(data)
    
    #create csr matrix
    leftVacuum=scipy.sparse.csr_matrix((data,(row_ind,col_ind)))
    norm=np.sqrt(leftVacuum.H@leftVacuum)
    leftVacuum=leftVacuum/norm[0,0]
    return leftVacuum
        

def differential_rho(t,rho_n):
    #return L_static_csr@rho_n
    return L_static.dot(rho_n)
    
def differential_rho_T(t,rho_nT):
    rho_n=np.array([rho_nT]).T
    print('rho_n',rho_n)
    return L_static.rdot(rho_n)
    #return rho_n_T@L_static_csr.T
    

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

def exact_Diagonalization(L_static):
    L_static_csr=L_static.as_sparse_format()
    w,vl,vr=scipy.linalg.eig(L_static_csr.toarray(),left=True,right=True)
    w_min=np.argmin(abs(w))

    rho_inf=np.array([vr[:,w_min]]).T

    #column vector
    vl0=scipy.sparse.csc_array(np.transpose([vl[:,w_min]]))
    
    norm=leftVacuum.H@rho_inf
    rho_inf=rho_inf/norm
    
    return vl0,rho_inf

def lowestEV(L_static):
    w,rho_inf=L_static.eigsh(k=1,sigma=0)
    print('w rho inf: ', w)
    norm=leftVacuum.H@rho_inf
    return rho_inf/norm
        
def compareVectors(v1,v2):
    print('difference',np.sum(abs(v1-v2)))
    print('direct comparison')
    correct=[]
    false=[]
    
    for i_1, data_1 in zip(v1.indices,v1.data):
        data_2=leftVacuum[i_1].data
        #if leftVacuum[i_vl].data != 
        if len(data_2)==0:
            data_2=0
        else:
            data_2=data_2[0]
        num=int(2**(4*L)-i_1-1).bit_count()/2
        if abs(data_1-data_2)>1e-10:
            #print(format(2**(4*L)-i_vl-1,'0{}b'.format(4*L)), num)
            #print(i_vl,'    ',f"{data_vl:.3}", data_V)
            false.append(num)
        else:
            if abs(data_1)>1e-10:
                correct.append(num)
        if abs(data_1)>1e-10:
            print(format(2**(4*L)-i_1-1,'0{}b'.format(4*L)), num)
            print(f"{data_1:.3}")
            print('')
    
    print(false)     
    print(correct)      
    print()  
        


# %%
#calculate the Lindbladoperator
L_static=get_Lindbladoperator(L,T,U,eps,Gamma1,Gamma2,basis)
print('define left vacuum')
leftVacuum=get_leftVacuum(L_static)

print('calculate EV')
#rho_inf=lowestEV(L_static)

# %%

#initial rho
rho0=leftVacuum#.toarray()


#calculate timeevolution of rho
t0=0
tf=1e2
dt=0.1

t=np.linspace(t0,tf,int(tf/dt)+1)
#print('calculate rdot: ',L_static.rdot(rho0))


rho0T=rho0.T.toarray()
#rho=scipy.integrate.solve_ivp(differential_rho_T,(t0,tf),rho_0T[0]+0j,method='RK45',t_eval=t)
#rho=rho['y']
#print('type: ',type(rho0T))
#print(rho0T[0])
#print('calculate rdot: ',L_static.rdot(rho0T[0]))
#print('calculate norm')
#rho_dot=L_static.dot(rho0)
#leftVacuumH=leftVacuum.H
#norm=leftVacuumH@rho_dot
#print('norm')
#print(norm.toarray())

#rho=scipy.integrate.solve_ivp(differential_rho_T,(t0,tf),rho0T[0]+0j,method='RK45',t_eval=t)
#rho=rho['y']

#check that inital is not multiplied by -i
rho=L_static.evolve(rho0T[0],0,t)
rho_end=rho[:,-1]
#print(rho_end)
#print(np.shape(rho_end))
#print(rho_inf)
print('n_exp')
n_exp=n(0,rho_end,basis,leftVacuum)
#print('n_time evolved: ',n_exp[-1])


#n_inf=n(0,rho_inf,basis,leftVacuum)
print('n_exp',n_exp)
#print('n_inf',n_inf)
print('n_analytic',2*Gamma2[0]/(Gamma1[0]+Gamma2[0]))



# plt.figure()
# plt.title('time dependence of n0')
# plt.plot(t,n_exp)




















