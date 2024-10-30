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
L=3# system size
center=int(np.floor(L/2))
T=(np.sqrt(1.0)*np.ones(L)+0j)*0.5#hopping right
U=2
eps=-U/2 #onsite energy
#eps=0
Gamma1=np.ones(L)*0.5
Gamma1[center]=0
#Gamma1=[0,0,0.1]
Gamma2=np.ones(L)*0.5
Gamma2[center]=0
#Gamma2=[0.1,0,0]
#total lenght is N=4L, due to spin up and down
basis = spinful_fermion_basis_1d(2*L)

# %%
def defineH_super(L,T,U,eps,basis):
    if np.isscalar(T):
        T=np.ones(L)*T
    if np.isscalar(eps):
        eps=np.ones(L)*eps
        
    T_c=np.conjugate(T)
    #i_middle=int(np.floor(L/2))
    #odd
    if L%2:
        i_middle=L
    #even
    else:
        i_middle=L+1
    
    #Hamiltonian acting on Fockspace (Fockspace are the odd sites)  
    hop_left = [[T[int((i-1)/2)],i,(i+2)] for i in range(1,2*L-2,2)] # hopping to the right
    hop_right = [[T_c[int((i-1)/2)],(i+2),i] for i in range(1,2*L-2,2)] # hopping to the left
    int_list = [[U,i_middle,i_middle]] # onsite interaction
    eps_list=[[eps[int((i-1)/2)],i] for i in range(1,2*L,2)]
    
    #Hamiltonian acting on augmented Fockspace  
    hop_left_a = [[-1*T[int((i)/2)],i,(i+2)] for i in range(0,2*L-2,2)] # hopping to the right
    hop_right_a = [[-1*T_c[int((i)/2)],(i+2),i] for i in range(0,2*L-2,2)] # hopping to the left
    int_list_a = [[-1*U,i_middle-1,i_middle-1]] # onsite interaction
    eps_list_a=[[-1*eps[int((i)/2)],i] for i in range(0,2*L,2)]
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
    #mixed Fock and augmented
    part1 = [[-2j*Gamma1[int(i/2)],(i+1),i] for i in range(0,2*L,2)]
    #Fock
    part2 = [[-1*Gamma1[int((i-1)/2)],i,i] for i in range(1,2*L,2)]
    #augmented
    part3 = [[-1*Gamma1[int((i)/2)],i,i] for i in range(0,2*L,2)]
    
    
    
    
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
    #mixed
    part1 = [[-2j*Gamma2[int(i/2)],(i+1),(i)] for i in range(0,2*L,2)]
    #fock
    part2 = [[-1*Gamma2[int((i-1)/2)],i,i] for i in range(1,2*L,2)]
    #augmented
    part3 = [[-1*Gamma2[int((i)/2)],i,i] for i in range(0,2*L,2)]
    
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
    #define a mask for even and odd states       
    mask_even=int('01'*(4*L//2),2)
    mask_odd=mask_even << 1 
    for state in states:
        #print(bin(mask_odd))
        even_bits=state&mask_even
        odd_bits=(state&mask_odd) >> 1
            
        if(even_bits==odd_bits):
            #shifting 2*L to the right => only the 2*L leftmost remain
            spin_up=state >> 2*L
            #masking the left part of the state
            spin_down=state & 2**(2*L)-1
            k=basis.index(format(spin_up,'0{}b'.format(2*L)),\
                          format(spin_down,'0{}b'.format(2*L)))
            n=int(int(state).bit_count()/2)
            data.append((-1j)**n)
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
        data_2=v2[i_1].data
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
            print('data_1')
            print(format(2**(4*L)-i_1-1,'0{}b'.format(4*L)), num)
            print(f"{data_1:.3}")
            print('')
            
        if abs(data_2)>1e-10:
            print('data_2')
            print(format(2**(4*L)-i_1-1,'0{}b'.format(4*L)), num)
            print(f"{data_2:.3}")
            print('')
    
    print(false)     
    print(correct)      
    print()  
        
def expectationValue(i,operator,rho,basis,leftVacuum):
    if operator=='n':
        operator=n(i,rho,basis,leftVacuum)
    operator_csr=operator.tocsr()
    
    rho_norm=leftVacuum.H@rho
    rho=rho/rho_norm
    
    n_rho=operator_csr@rho
    n_exp=leftVacuum.H@n_rho
    return n_exp[0]
    
def n(i,rho,basis,leftVacuum):
    #i counts from the middle
    if L%2:
        i_middle=L
    #even
    else:
        i_middle=L+1

    index=i_middle+2*i
    n_list=[[1+0j,index]]
    
    static=[
        ["n|", n_list],
        ["|n", n_list],
        ]
    dynamic = []
    
    n_operator=hamiltonian(static,dynamic,dtype=np.complex128,basis=basis,check_herm=False,check_symm=False)
    return n_operator

def a(i,rho,basis,leftVacuum):
    #i counts from the middle
    if L%2:
        i_middle=L
    #even
    else:
        i_middle=L+1

    index=i_middle+2*i
    a_list=[[1+0j,index]]
    
    static=[
        ["-|", a_list],
        ["|-", a_list],
        ]
    dynamic = []
    
    a_operator=hamiltonian(static,dynamic,dtype=np.complex128,basis=basis,check_herm=False,check_symm=False)
    return a_operator

def a_dag(i,rho,basis,leftVacuum):
    #i counts from the middle
    if L%2:
        i_middle=L
    #even
    else:
        i_middle=L+1

    index=i_middle+2*i
    adag_list=[[1+0j,index]]
    
    static=[
        ["+|", adag_list],
        ["|+", adag_list],
        ]
    dynamic = []
    
    adag_operator=hamiltonian(static,dynamic,dtype=np.complex128,basis=basis,check_herm=False,check_symm=False)
    return adag_operator

# %%
#calculate the Lindbladoperator
L_static=get_Lindbladoperator(L,T,U,eps,Gamma1,Gamma2,basis)
print('define left vacuum')
leftVacuum=get_leftVacuum(L_static)

print('calculate EV')
#vl0,rho_inf=exact_Diagonalization(L_static)
rho_inf=lowestEV(L_static)

# %%

def G_r(tf,Tf,basis,leftVacuum,Lindblatt,i=0,dt=0.1):
    t=np.linspace(0,tf,int(tf/dt)+1)
    rho=L_static.evolve(rho0T[0],0,t)
    #rho_end=rho[:,-1]
    rho_end=rho_inf[:,0]
    a_op=a(i,rho,basis,leftVacuum)
    adag_op=a_dag(i,rho,basis,leftVacuum)
    
    rho_a=a_op.dot(rho_end)
    rho_adag=adag_op.dot(rho_end)
    
    Tau=np.linspace(0,Tf,int(Tf/dt)+1)
    
    rhoTau_a=L_static.evolve(rho_a,0,Tau)
    rhoTau_adag=L_static.evolve(rho_adag,0,Tau)
    
    G1=leftVacuum.H@(adag_op.dot(rhoTau_a))
    G2=leftVacuum.H@(a_op.dot(rhoTau_adag))
    
    return Tau, -1j*(np.conj(G1)+G2)*np.heaviside(Tau,0.5)
    
#initial rho
rho0=leftVacuum


#calculate timeevolution of rho
t0=0
tf=1e2
dt=0.1

t=np.linspace(t0,tf,int(tf/dt)+1)

rho0T=rho0.T.toarray()
#check that inital is not multiplied by -i
rho=L_static.evolve(rho0T[0],0,t)
rho_end=rho[:,-1]
#n_exp=n(0,rho_end,basis,leftVacuum)
n_exp=expectationValue(0,'n',rho_end,basis,leftVacuum)

n_inf=expectationValue(0,'n',rho_inf,basis,leftVacuum)

print('n_time evolved: ',n_exp)
print('n_inf',n_inf)
print('n_analytic',2*Gamma2[0]/(Gamma1[0]+Gamma2[0]))

n_exp=expectationValue(0,'n',rho,basis,leftVacuum)

#plt.figure()
#plt.title('time dependence of n0')
#plt.plot(t,n_exp)


#calculate retarted Green's function
Tau,GR_Tau=G_r(1,3e2,basis,leftVacuum,L_static,i=0,dt=0.01)

plt.figure()
plt.title('GR')
plt.plot(Tau,GR_Tau[0])


N_Tau=len(Tau)
T=abs(Tau[-1]-Tau[0])
norm=T/N_Tau#/np.sqrt(2*np.pi)

#FT of retarted Green's function
omegas=np.fft.fftfreq(N_Tau,Tau[1]-Tau[0])*2*np.pi
omegas=np.fft.fftshift(omegas)
G_r=np.fft.ifft(GR_Tau[0],norm='forward')*norm
G_r=np.fft.fftshift(G_r)


spectrum=(-1)/np.pi*np.imag(G_r)
#%%
N_start=0
plt.figure()
plt.title('spectral function')
print(len(omegas))
start=14000
end=16000
#plt.plot(Tau[N_start:],np.real(G_r[N_start:]),label='real')
#plt.plot(omegas,np.imag(G_r[N_start:]),label='imaginary')
print(spectrum[-1])
plt.plot(omegas[start:end],spectrum[start:end],label='spectralfunction')
plt.plot(omegas[start:end],0*omegas[start:end])
#plt.axvline(1,0,1)
plt.legend()
print('norm',np.trapz(spectrum,x=omegas))
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


#for testing Greensfunction, we use fixed t for now













