#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:57:37 2024

@author: theresa
"""

from Lindblatt import createLindblad
from augmented_basis import augmented_basis
import numpy as np
import scipy
from quspin.operators import hamiltonian
import matplotlib.pyplot as plt

parameters = {"sites": 3,
              "epsilon": 0,
              "hopping": 0.5,
              "drive": 1,
              "interaction":0,
              "frequency":1,
              "coupling_empty":[0.5,0,0.1],
              "coupling_full":[0.1,0,0.5],
              "spin_symmetric":True,
}

L=parameters['sites']
center=int(np.floor(L/2))
T=(np.sqrt(1.0)*np.ones(L)+0j)*0.0#hopping right
U=1
V=1
Om=1
#eps=-U/2 #onsite energy
#eps=[0,-U/2,0]
eps=[1,1,1]
Gamma1=np.ones(L)*0.1
Gamma1[center]=0
#Gamma1=[0,0,0.1]
Gamma2=np.ones(L)*0.2
Gamma2[center]=0
#Gamma2=[0.1,0,0]


basis=augmented_basis(L,'extended')



def drive(t,Om):
    return np.cos(Om*t)

def defineH_super(L,T,U,eps,basis,V=0,Om=0):
    if np.isscalar(T):
        T=np.ones(L)*T
    if np.isscalar(eps):
        eps=np.ones(L)*eps
    
    if len(T) == L:
        T=np.concatenate((T,T))
    if len(eps) == L:
        eps=np.concatenate((eps,eps))
    T_c=np.conjugate(T)
    #i_middle=int(np.floor(L/2))
    #odd
    if L%2:
        i_middle=L
    #even
    else:
        i_middle=L+1
    
    
    #Hamiltonian acting on Fockspace (Fockspace are the odd sites)  
    hop_left_up = [[T[int((i-1)/2)],i,(i+2)] for i in range(1,2*L-2,2)] 
    hop_left_down = [[T[int((i-1)/2)],i,(i+2)] for i in range(2*L+1,4*L-2,2)]
    #hop_left=np.concatenate((hop_left_up,hop_left_down))
    hop_left=hop_left_up+hop_left_down

    # hopping to the right
    hop_right_up = [[T_c[int((i-1)/2)],(i+2),i] for i in range(1,2*L-2,2)]  # hopping to the left
    hop_right_down= [[T_c[int((i-1)/2)],(i+2),i] for i in range(2*L+1,4*L-2,2)]
    #hop_right=np.concatenate((hop_right_up,hop_right_down))
    hop_right=hop_right_up+hop_right_down
    
    int_list = [[U,i_middle,i_middle+2*L]] # onsite interaction
    eps_list=[[eps[int((i-1)/2)],i] for i in range(1,4*L,2)]
    
    
    #Hamiltonian acting on augmented Fockspace  
    hop_left_up_a = [[-1*T[int((i)/2)],i,(i+2)] for i in range(0,2*L-2,2)] # hopping to the right
    hop_left_down_a = [[-1*T[int((i)/2)],i,(i+2)] for i in range(2*L,4*L-2,2)]
    hop_left_a=hop_left_up_a+hop_left_down_a

    hop_right_up_a = [[-1*T_c[int((i)/2)],(i+2),i] for i in range(0,2*L-2,2)] # hopping to the left
    hop_right_down_a = [[-1*T_c[int((i)/2)],(i+2),i] for i in range(2*L,4*L-2,2)]
    hop_right_a=hop_right_up_a+hop_right_down_a

    
    
    int_list_a = [[-1*U,i_middle-1,i_middle-1+2*L]] # onsite interaction
    eps_list_a=[[-1*eps[int((i)/2)],i] for i in range(0,4*L,2)]
    # static and dynamic lists
    static= [	
            #Hamiltonian acting on Fockspace
            
    		["+-", hop_left], # up hop left
    		["+-", hop_right], # up hop right
    		["nn", int_list], # onsite interaction
            ["n", eps_list], # onsite energy
            
            #Hamiltonian acting on augmented Fockspace
    		["+-", hop_left_a], # up hop left
    		["+-", hop_right_a], # up hop right
    		["nn", int_list_a], # onsite interaction
            ["n", eps_list_a], # onsite energy
    		]
    #print(static)
    if V:
        drive_args=[Om]
        V_list = [[V,i_middle],[V,i_middle+2*L]]
        V_list_a = [[-1*V,i_middle-1],[-1*V,i_middle-1+2*L]]
        dynamic= [
            ["n", V_list,drive,drive_args], # onsite energy
            ["n", V_list_a,drive,drive_args], # onsite energy
            ]
            
        
    else:
        dynamic=[]
    ###### construct Hamiltonian
    return hamiltonian(static,dynamic,dtype=np.complex128,basis=basis,check_pcon=False,check_symm=False,check_herm=False)

#removes particels from the system
def define_Dissipators1(L,Gamma1,basis):
    if len(Gamma1) == L:
        Gamma1=np.concatenate((Gamma1,Gamma1))
    #mixed Fock and augmented
    part1 = [[-2j*Gamma1[int(i/2)],(i+1),i] for i in range(0,4*L,2)]
    #Fock
    part2 = [[-1*Gamma1[int((i-1)/2)],i,i] for i in range(1,4*L,2)]
    #augmented
    part3 = [[-1*Gamma1[int((i)/2)],i,i] for i in range(0,4*L,2)]
    
    
    
    
    static=[
        ["--",part1],
        ["+-",part2],
        ["+-",part3],  
        ]
    dynamic=[]
    return hamiltonian(static,dynamic,dtype=np.complex128,basis=basis,check_herm=False,check_pcon=False,check_symm=False)

#add particels to the system
def define_Dissipators2(L,Gamma2,basis):
    if len(Gamma2) == L:
        Gamma2=np.concatenate((Gamma2,Gamma2))
    #mixed
    part1 = [[-2j*Gamma2[int(i/2)],(i+1),(i)] for i in range(0,4*L,2)]
    #fock
    part2 = [[-1*Gamma2[int((i-1)/2)],i,i] for i in range(1,4*L,2)]
    #augmented
    part3 = [[-1*Gamma2[int((i)/2)],i,i] for i in range(0,4*L,2)]
    
    static=[
        ["++",part1],
        ["-+",part2],
        ["-+",part3],
        ]
    
    dynamic=[]

    return hamiltonian(static,dynamic,dtype=np.complex128,basis=basis,check_herm=False,check_pcon=False,check_symm=False)

def get_staticLindblad(L,T,U,eps,Gamma1,Gamma2,basis=0):
    H_super=defineH_super(L,T,U,eps,basis)
    D1=define_Dissipators1(L,Gamma1,basis)
    D2=define_Dissipators2(L,Gamma2,basis)

    #return -1j*H_super+D1+D2
    return H_super+1j*(D1+D2)  # multiply by -i due to .evolve structure

def get_dynamicLindblad(L,T,U,eps,V,Om,Gamma1,Gamma2,basis=0):
    #print(basis)
    H_super=defineH_super(L,T,U,eps,basis,V,Om)
    D1=define_Dissipators1(L,Gamma1,basis)
    D2=define_Dissipators2(L,Gamma2,basis)

    #return -1j*H_super+D1+D2
    return H_super+1j*(D1+D2)



def get_leftVacuum(L_static):
    basis=L_static.basis
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
            #print(format(state,'0{}b'.format(4*L)))
            #shifting 2*L to the right => only the 2*L leftmost remain
            #spin_up=state >> np.uint64(2*L)
            #masking the left part of the state
            #spin_down=state & np.uint64(2**(2*L)-1)
            #k=basis.index(format(spin_up,'0{}b'.format(2*L)),format(spin_down,'0{}b'.format(2*L)))
            k=basis.index(format(state,'0{}b'.format(4*L)))
            n=int(int(state).bit_count()/2)
            data.append((-1j)**n)
            row_ind.append(k)
    
    col_ind=np.zeros(len(data),dtype=int)
    data=np.array(data)
    #create csr matrix
    leftVacuum=scipy.sparse.csr_matrix((data,(row_ind,col_ind)),shape=(basis.Ns,1))
    norm=np.sqrt(leftVacuum.H@leftVacuum)
    leftVacuum=leftVacuum/norm[0,0]
    return leftVacuum


def expectationValue(i,operator,rho,basis,leftVacuum):
    if operator=='n':
        operator=n(i,basis)
    operator_csr=operator.tocsr()
    
    rho_norm=leftVacuum.H@rho
    rho=rho/rho_norm
    
    n_rho=operator_csr@rho
    n_exp=leftVacuum.H@n_rho
    return n_exp[0]
    
def n(i,basis):
    #i counts from the middle
    if L%2:
        i_middle=L
    #even
    else:
        i_middle=L+1

    index=i_middle+2*i
    n_list=[[1+0j,index],[1+0j,index+2*L]]
    n_list=[[1+0j,index+2*L]]
    
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

    index=i_middle+2*i
    a_list=[[1+0j,index],[1+0j,index+2*L]]
    a_list=[[1+0j,index+2*L]]
    
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

    index=i_middle+2*i
    adag_list=[[1+0j,index],[1+0j,index+2*L]]
    adag_list=[[1+0j,index+2*L]]
    
    static=[
        ["+", adag_list],
        ]
    dynamic = []
    
    adag_operator=hamiltonian(static,dynamic,dtype=np.complex128,basis=basis,check_herm=False,check_symm=False,check_pcon=False)
    return adag_operator

def G_r(tf,Tf,basis,leftVacuum,Lindblatt,i=0,dt=0.1):
    t=np.linspace(0,tf,int(tf/dt)+1)
    rho=Lindblatt.evolve(rho0T[0],0,t)
    #print(rho)
    rho_end=rho[:,-1]
    #rho_end=rho_inf[:,0]
    
    a_op=a(i,basis)
    adag_op=a_dag(i,basis)
    
    rho_a=a_op.dot(rho_end)
    rho_adag=adag_op.dot(rho_end)
    
    Tau=np.linspace(0,Tf,int(Tf/dt)+1)
    
    rhoTau_a=Lindblatt.evolve(rho_a,0,Tau)
    rhoTau_adag=Lindblatt.evolve(rho_adag,0,Tau)
    
    G1=leftVacuum.H@(adag_op.dot(rhoTau_a))
    G2=leftVacuum.H@(a_op.dot(rhoTau_adag))
    Gr=-1j*(np.conj(G1)+G2)*np.heaviside(Tau,0.5)
    return Tau, Gr[0]
    
def Gr_floquet1(tf,Tf,basis,leftVacuum,Lindblatt,i=0,dt=0.1):
    if Om:
        period=np.pi*2/Om
    else:
        period=1
    t=np.linspace(tf-period,tf,int(np.ceil(period/dt)))#period/dt)
    rhos=Lindblatt.evolve(rho0T[0],0,t)
    
    #plot equal time
    n_exp=expectationValue(i,'n',rhos,basis,leftVacuum)
    plt.plot(t,np.real(n_exp))
    plt.plot(t,np.imag(n_exp))
    
    a_op=a(i,basis)
    adag_op=a_dag(i,basis)
    Tau=np.linspace(0,Tf,int(Tf/dt)+1)
    print(len(rhos[0,:]))
    Gr=np.zeros((len(t),len(Tau)))*0j
    percent=0
    for j in range(len(rhos[0,:])):  
        #print(j)
        if j/len(t)>=percent:
            print(np.ceil(j/len(t)*100),'%')
            percent+=0.01
        rho=rhos[:,j]
        rho_a=a_op.dot(rho)
        rho_adag=adag_op.dot(rho)
        rhoTau_a=Lindblatt.evolve(rho_a,t[j],t[j]+Tau)
        rhoTau_adag=Lindblatt.evolve(rho_adag,t[j],t[j]+Tau)
        G1=leftVacuum.H@(adag_op.dot(rhoTau_a))
        G2=leftVacuum.H@(a_op.dot(rhoTau_adag))
        Gr[j]=-1j*(np.conj(G1)+G2)*np.heaviside(Tau,0.5)
        print(np.sum(abs(G2.imag)))
        print(np.sum(abs(G1.imag)))
    

    
    return Tau, np.trapz(Gr,t,axis=0)/period

#include change in integral borders
def Gr_floquet(tf,Tf,basis,leftVacuum,Lindblatt,i=0,dt=0.1):
    if Om:
        period=np.pi*2/Om
    else:
        period=1
    
    a_op=a(i,basis)
    adag_op=a_dag(i,basis)
    Tau=np.linspace(0,Tf,int(Tf/dt)+1)
    percent=0
    i+=0
    N_Tau=int(Tf/dt)+1
    #N_Tauhalf=int(np.ceil(N_Tau/2))
    N_period=int(np.ceil(period/dt)+1)
    t=np.linspace(tf-period/2-Tf/2,tf+period/2,N_period+N_Tau)#period/dt)
    rhos=Lindblatt.evolve(rho0T[0],0,t)
    print(len(rhos[0,:]))
    Gr=np.zeros(len(Tau))*0j
    
    rhos=Lindblatt.evolve(rho0T[0],0,t)
    for i in range(len(Tau)):
        if i/len(Tau)>=percent:
            print(np.ceil(i/len(Tau)*100),'%')
            percent+=0.01
        Gr_t=np.zeros(N_period)*0j   
        m=0
        for j in range(N_Tau-i,N_period+N_Tau-i):
            rho=rhos[:,j]
            rho_a=a_op.dot(rho)
            rho_adag=adag_op.dot(rho)
            rhoTau_a=Lindblatt.evolve(rho_a,t[j],t[j]+Tau[i])
            rhoTau_adag=Lindblatt.evolve(rho_adag,t[j],t[j]+Tau[i])
            G1=leftVacuum.H@(adag_op.dot(rhoTau_a))
            G2=leftVacuum.H@(a_op.dot(rhoTau_adag))
            Gr_t[m]=-1j*(np.conj(G1[0])+G2[0])*np.heaviside(Tau[i],0.5)
            m+=1
        #print(np.shape(Gr_t))
        #print(len(t),i,N_Tau-i,np.shape(t[N_Tau-i:N_period+N_Tau-i]))
        print(np.sum(abs(G2.imag)))
        print(np.sum(abs(G1.imag)))
        Gr[i]=np.trapz(Gr_t,t[N_Tau-i:N_period+N_Tau-i],axis=0)/period

    

    
    return Tau, Gr

#L_static=get_staticLindblad(L,T,U,eps,Gamma1,Gamma2,basis)
#L_dynamic=get_dynamicLindblad(L,T,U,eps,V,Om,Gamma1,Gamma2,basis)
Lindblad=createLindblad(basis,parameters)
Lindblad.displayParameters()
Lindblad_op=Lindblad.operator.toarray(time=1)
L_dynamic=Lindblad.operator
Lind_test=get_dynamicLindblad(L,T,U,eps,V,Om,Gamma1,Gamma2,basis).toarray(time=1)
#print(Lind_test)
#print(Lindblad_op)
print('diff test Hamiltonian')
print(np.sum(abs(Lindblad_op-Lind_test)))

print(L_dynamic.Ns)
print('define left vacuum')
leftVacuum=get_leftVacuum(L_dynamic)
#print(leftVacuum)
rho_inf=Lindblad.lowestEV()
#initial rho
rho0=leftVacuum
#print(L_static.dot(leftVacuum))

#calculate timeevolution of rho
t0=0
tf=1e2
dt=0.1

t=np.linspace(t0,tf,int(tf/dt)+1)

rho0T=rho0.T.toarray()
#check that inital is not multiplied by -i
rho=L_dynamic.evolve(rho0T[0],0,t)
rho_end=rho[:,-1]
#n_exp=n(0,rho_end,basis,leftVacuum)
#rho_inf[rho_inf < 0.0001] = 0
#print(scipy.sparse.csr_array(rho_inf))
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
#Tau,GR_Tau=Gr_floquet1(3e2,3e2,basis,leftVacuum,L_dynamic,i=0,dt=0.05)
Tau,GR_Tau=Gr_floquet1(3e2,5e2,basis,leftVacuum,L_dynamic,i=0,dt=0.05)
#Tau,GR_Tau=G_r(3e2,3e2,basis,leftVacuum,L_dynamic,i=0,dt=0.05)


plt.figure()
plt.title('GR')
plt.plot(Tau,np.real(GR_Tau))
plt.plot(Tau,np.imag(GR_Tau))


N_Tau=len(Tau)
Period=abs(Tau[-1]-Tau[0])
norm=Period/N_Tau#/np.sqrt(2*np.pi)

#FT of retarted Green's function
omegas=np.fft.fftfreq(N_Tau,Tau[1]-Tau[0])*2*np.pi
omegas=np.fft.fftshift(omegas)
Gr=np.fft.ifft(GR_Tau,norm='forward')*norm
Gr=np.fft.fftshift(Gr)

spectrum=(-1)/np.pi*np.imag(Gr)
#%%
N_start=0
plt.figure()
plt.title('spectral function')
txt='V='+str(V)+'  Om='+str(Om)+'  U='+str(U)+'  T='+str(T[0])+'  G='+str(Gamma1[0])
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
print('len omegas:',len(omegas))
start=4500
end=5500
#plt.plot(Tau[N_start:],np.real(G_r[N_start:]),label='real')
#plt.plot(omegas,np.imag(G_r[N_start:]),label='imaginary')
print(spectrum[-1])
plt.plot(omegas[start:end],spectrum[start:end],label='spectralfunction')
plt.plot(omegas[start:end],0*omegas[start:end])
#plt.axvline(1,0,1)
plt.legend()
print('norm',np.trapz(spectrum,x=omegas))


