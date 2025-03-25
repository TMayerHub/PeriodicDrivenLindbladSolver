#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 08:02:54 2024

@author: theresa
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = ("True")  # uncomment this line if omp error occurs on OSX for python 3
os.environ["OMP_NUM_THREADS"] = str(4)# set number of OpenMP threads to run in parallel
os.environ["MKL_NUM_THREADS"] = str(4)# set number of MKL threads to run in parallel
import json

#os.environ["MKL_NUM_THREADS"] = str(4)
#os.environ['OMP_NUM_THREADS'] = '4'

from Lindblatt import createLindblad
from augmented_basis import augmented_basis

from quspin.operators import hamiltonian

import scipy
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit, prange
from joblib import Parallel, delayed
import threading
import gc
#from pathos.multiprocessing import Pool

import warnings
from joblib import Memory

memory = Memory(location="cache_dir", verbose=0)
progress_lock = threading.Lock()
progress_count = 0


def process_j(j,t,Tau,rhos_a,rhos_adag,plus_lV,minus_lV):
    # Evolution for this particular `j`
    rhoTau_a = np.array(LindbladM.operator.evolve(rhos_a[:, j], t[j], t[j] + Tau))
    #print(rhoTau_a.type)
    rhoTau_adag = np.array(LindbladP.operator.evolve(rhos_adag[:, j], t[j], t[j] + Tau))
            
    # Calculate Lesser and Greater contributions for this `j`
    Lesser_j = plus_lV.T.conjugate() @ rhoTau_a
    Greater_j = minus_lV.T.conjugate() @ rhoTau_adag
    
    return rhoTau_a[:, -1], rhoTau_adag[:, -1], Lesser_j, Greater_j

class calculateGreensFunction:
    def __init__(self,parameters,sites,spin):
        self.parameters=parameters
        self.spin_sym=parameters['spin_symmetric']
        self.sites=sites
        self.L=parameters['length']
        self.center=self.findCenter()
        self.spin=spin

        print('start define basis')
        self.basis0=augmented_basis(self.L,'restricted',[0,0],spin_sym=self.spin_sym)
        self.basisE=augmented_basis(self.L,spin_sym=self.spin_sym)
        
        self.basisM=self._basisM()
        self.basisP=self._basisP()
        print('end define basis')
        print(self.basis0.Ns)
        print(self.basisE.Ns)
        print(self.basisM.Ns)
        print(self.basisP.Ns)
        
        self.leftVacuum=self._leftVacuum()
        #self.plus_lV=self.plus_leftVacuum()
        #self.minus_lV=self.minus_leftVacuum()
        
        self.greater=None
        self.lesser=None
        self.t_period=None
        self.rhos_period=None
        self.Tau=None
        
    
    def findCenter(self):
        if self.L%2:
            return self.L
        #even
        else:
            return self.L+1
        
    @staticmethod
    def get_extendedIndices(basis,extended_basis):
        #find index location of first occurrence of each value of interest
        #sorter = np.argsort(x)
        #sorter[np.searchsorted(x, vals, sorter=sorter)]
        
        sorter = np.argsort(extended_basis[:])
        ind_extended = sorter[np.searchsorted(extended_basis[:], basis[:], sorter=sorter)]
        return ind_extended
    
    @staticmethod
    def to_extended_vector(state_vector,basis,extended_basis):
    
        v_shape=state_vector.shape
        if len(v_shape)==2:
            rows = v_shape[0]
            cols = v_shape[1]
            
            if rows == 1:
                extended_vector=np.zeros(extended_basis.Ns)+0j
                extended_vector[calculateGreensFunction.get_extendedIndices(basis,extended_basis)]=state_vector[0]
                return extended_vector.reshape((1,extended_basis.Ns))
            
            if cols == 1:
                extended_vector=np.zeros(extended_basis.Ns)+0j
                extended_vector[calculateGreensFunction.get_extendedIndices(basis,extended_basis)]=state_vector[:,0]
                return extended_vector.reshape((extended_basis.Ns,1))
        
        if len(v_shape) ==1:
            extended_vector=np.zeros(extended_basis.Ns)+0j
            extended_vector[calculateGreensFunction.get_extendedIndices(basis,extended_basis)]=state_vector
            return extended_vector
        
    @staticmethod
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
                reduced_vector=state_vector[calculateGreensFunction.get_extendedIndices(basis,extended_basis)]
                return reduced_vector.reshape((1,basis.Ns))
            
            if cols == 1:
                state_vector=state_vector[:,0]
                reduced_vector=state_vector[calculateGreensFunction.get_extendedIndices(basis,extended_basis)]
                return reduced_vector.reshape((basis.Ns,1))
        
        if len(v_shape) ==1:
            reduced_vector=state_vector[calculateGreensFunction.get_extendedIndices(basis,extended_basis)]
            return reduced_vector
        
    def _basisM(self):
        if self.spin == 'updown':
            diff_sector = [2,0,0,2]
            
        if self.spin == 'up':
            diff_sector = [2,0]
            
        if self.spin == 'down':
            diff_sector = [0,2]
            
        return augmented_basis(self.L,'restricted',diff_sector,spin_sym=self.spin_sym)
    
    def _basisP(self):
        if self.spin == 'updown':
            diff_sector = [1,0,0,1]
            
        if self.spin == 'up':
            diff_sector = [1,0]
            
        if self.spin == 'down':
            diff_sector = [0,1]
            
        return augmented_basis(self.L,'restricted',diff_sector,spin_sym=self.spin_sym)
        

    def action_n(self,state,site):
        index=self.center+2*site
        if self.spin == 'updown':
            n_list=[[1+0j,index],[1+0j,index+2*self.L]]
        if self.spin == 'up':
            if self.spin_sym:
                warnings.warn('Using operator which is not spin symmetric with symmetric basis',
                              category=UserWarning)
            n_list=[[1+0j,index]]
        if self.spin == 'down':
            if self.spin_sym:
                warnings.warn('Using operator which is not spin symmetric with symmetric basis',
                              category=UserWarning)
            n_list=[[1+0j,index+2*self.L]]
            
        static=[
            ["n", n_list],
            ]
        dynamic = []
        #print(static)
        n_op=hamiltonian(static,dynamic,dtype=np.complex128,basis=self.basis0,check_herm=False,check_symm=False,check_pcon=False)
        return n_op.dot(state)
    
    def action_a(self,state_vector,site):
       
        index=self.center+2*site
        if self.spin == 'updown':
            a_list=[[1+0j,index],[1+0j,index+2*self.L]]
        if self.spin == 'up':
            if self.spin_sym:
                warnings.warn('Using operator which is not spin symmetric with symmetric basis',
                              category=UserWarning)

            a_list=[[1+0j,index]]
        if self.spin == 'down':
            if self.spin_sym:
                warnings.warn('Using operator which is not spin symmetric with symmetric basis',
                              category=UserWarning)

            a_list=[[1+0j,index+2*self.L]]
            
        static=[
            ["-", a_list],
            ]
        dynamic = []
        #print(a_list)
        a_op=hamiltonian(static,dynamic,dtype=np.complex128,basis=self.basisE,check_herm=False,check_symm=False,check_pcon=False)
        extended_vector=self.to_extended_vector(state_vector,self.basis0,self.basisE)
        a_vector_extended=a_op.dot(extended_vector)

        a_vector=self.to_reduced_vector(a_vector_extended,self.basisM,self.basisE)

        return a_vector
    
    def action_adag(self,state_vector,site):
        
        index=self.center+2*site
        if self.spin == 'updown':
            a_list=[[1+0j,index],[1+0j,index+2*self.L]]
            
        if self.spin == 'up':
            if self.spin_sym:
                warnings.warn('Using operator which is not spin symmetric with symmetric basis',
                              category=UserWarning)

            a_list=[[1+0j,index]]
            
        if self.spin == 'down':
            if self.spin_sym:
                warnings.warn('Using operator which is not spin symmetric with symmetric basis',
                              category=UserWarning)

            a_list=[[1+0j,index+2*self.L]]
            
        static=[
            ["+", a_list],
            ]
        dynamic = []
        
        adag_op=hamiltonian(static,dynamic,dtype=np.complex128,basis=self.basisE,check_herm=False,check_symm=False,check_pcon=False)
        extended_vector=self.to_extended_vector(state_vector,self.basis0,self.basisE)
        adag_vector_extended=adag_op.dot(extended_vector)
        
        adag_vector=self.to_reduced_vector(adag_vector_extended,self.basisP,self.basisE)
        return adag_vector
    
    
    
    def _leftVacuum(self):
        L=self.L
        
        states=self.basis0.states
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
                k=self.basis0.index(format(state,'0{}b'.format(4*L)))
                n=int(int(state).bit_count()/2)
                #data.append((-1j)**n)
                row_ind.append(k)
                if self.spin_sym:
                    L2=L*2
                    s_down = state & np.uint64(2**(L2)-1)
                    s_up = state >>np.uint64(L2)
                    
                    if state==(s_up| s_down << np.uint64(L2)):
                        data.append((-1j)**n)
                        
                    else:
                        data.append(np.sqrt(2)*(-1j)**n)
                        
                else:
                    data.append((-1j)**n)
                    
            
        col_ind=np.zeros(len(data),dtype=int)
        data=np.array(data)

        #create csr matrix
        leftVacuum=scipy.sparse.csr_matrix((data,(row_ind,col_ind)),shape=(self.basis0.Ns,1))
        norm=np.sqrt(leftVacuum.T.conjugate()@leftVacuum)
        leftVacuum=leftVacuum/norm[0,0]
        return leftVacuum
        
    
    def plus_leftVacuum(self,site):
        '''adag acting from the right <I|a+, returns a|I>'''
        lV=self.leftVacuum.toarray()
        plV=self.action_a(lV,site)
        return scipy.sparse.csr_array(plV)
    
    def minus_leftVacuum(self,site):
        '''a acting from the right <I|a, , returns a+|I>'''
        lV=self.leftVacuum.toarray()
        mlV=self.action_adag(lV,site)
        return scipy.sparse.csr_array(mlV)
    
    def plot_n(self,site,tf,dt=0.1):
        basis0=augmented_basis(self.L,'restricted',[0,0],spin_sym=self.spin_sym)

        Lindblad0=createLindblad(basis0,self.parameters,spin_sym=self.spin_sym)
        rho0=self.leftVacuum
        rho0T=rho0.T.toarray()
        
        if Lindblad0.Om:
            period=np.pi*2/Lindblad0.Om
        else:
            period=1
            
        t=np.linspace(0,tf,int(np.ceil(tf/dt)))#period/dt)
        rhos=Lindblad0.operator.evolve(rho0T[0],0,t)
        #plt.figure()
        n_exps=[]
        for i in range(len(rhos[0])):
            rho_n=self.action_n(rhos[:,i],site)
            n_exp=self.leftVacuum.T.conjugate()@rho_n
            n_exps.append(n_exp)
            
        #plt.plot(t,np.real(n_exps))
        #plt.plot(t,np.imag(n_exps))
        
        #plt.figure()
        #t_period=np.linspace(tf-period,tf-5dt,int(np.floor(period/dt)))#period/dt)
        #rhos=Lindblad0.operator.evolve(rho0T[0],0,t_period)
        
        #plot equal time
        N_period=int(np.floor(period/dt))
        n_exps=[]
        for i in range(len(rhos[0])-N_period,len(rhos[0])):
            rho_n=self.action_n(rhos[:,i],site)
            n_exp=self.leftVacuum.T.conjugate()@rho_n
            n_exps.append(n_exp)
            
        #plt.plot(t[len(rhos[0])-N_period:],np.real(n_exps))
        print(n_exps[-1])
        return n_exps[-1]
        #plt.plot(t_period,np.imag(n_exps))
        
        
    def _GreaterLesserPlotFT(self,sites,dt=0.05,eps=1e-12,max_iter=1000,av_periods=5,
                       tf=5e2,t_step=1e2,av_Tau=10):
        
        
        if self.parameters['frequency']:
            period=np.pi*2/self.parameters['frequency']
        else:
            period=1


        if self.rhos_period==None:
            self.rhoOf_t(dt=dt,eps=eps,max_iter=max_iter,av_periods=av_periods,
                         tf=tf,t_step=t_step,return_all=False)
        
        t=self.t_period
        rhos=self.rhos_period
        Tau=np.linspace(0,tf,int(tf/dt)+1)


        global LindbladM
        LindbladM=createLindblad(self.basisM,self.parameters,spin_sym=self.spin_sym)
        global LindbladP
        LindbladP=createLindblad(self.basisP,self.parameters,spin_sym=self.spin_sym)
        
        Greater=np.zeros((len(t),len(Tau)))*0j
        Lesser=np.zeros((len(t),len(Tau)))*0j
        
        rhos_a=np.zeros((self.basisM.Ns,len(rhos[0,:])))+0j
        rhos_adag=np.zeros((self.basisP.Ns,len(rhos[0,:])))+0j
        
        for j in range(len(rhos[0,:])):  
            rhos_a[:,j]=self.action_a(rhos[:,j],sites[1])
            rhos_adag[:,j]=self.action_adag(rhos[:,j],sites[1])
        
        self.Tau=None
        self.a_adag=None
        self.adag_a=None
        
        Tau_last=0
        diff=1
        i=0
        while diff>eps:
            if i>max_iter:
                raise RuntimeError(f"Max iterations reached ({max_iter}) without convergence. Last epsilon: {diff}")
            print(diff)
            Tau, Lesser, Greater,rhos_a,rhos_adag,diff=self.stepsGreaterLesser(
                                  sites,Tau_last,dt,tf,t_step,av_Tau,rhos_a,rhos_adag,
                                  )#LindbladM,LindbladP)
            Tau_last=Tau[-1]

            if self.Tau is None:
                self.Tau=Tau
                self.adag_a=Lesser
                self.a_adag=Greater
                
            else:
                self.Tau=np.concatenate((self.Tau,Tau),axis=0)
                #print(self.Tau)
                self.adag_a=np.concatenate((self.adag_a,Lesser),axis=1)
                self.a_adag=np.concatenate((self.a_adag,Greater),axis=1)
            i+=1
            
        
        Gr=-1j*(np.conj(self.adag_a)+self.a_adag)*np.heaviside(self.Tau,0.5)
        
        
        Tau_minus=np.flip(self.Tau[1:])*(-1)
        
        lesser_plus=+1j*np.conj(self.adag_a)
        greater_plus=-1j*self.a_adag
        lesser_minus=+1j*np.flip(self.adag_a[:,1:],axis=1)
        greater_minus=-1j*np.flip(np.conj(self.a_adag[:,1:]),axis=1)
        
        Tau_total=np.concatenate((Tau_minus,self.Tau))
        lesser_total=np.concatenate((lesser_minus,lesser_plus),axis=1)
        greater_total=np.concatenate((greater_minus,greater_plus),axis=1)
        
        
        #print(Tau_minus)
        #Gk_pos=1j*np.heaviside(self.Tau,0.5)*(-1*self.greater + np.conj(self.lesser))
        
        
        #Gk_min=np.heaviside(-1*Tau_minus,0.5)*(-1j*np.conj(greater_minus)+1j*lesser_minus)
        #Gk=np.concatenate((Gk_min,Gk_pos),axis=1)
        Gk=lesser_total+greater_total
        G_advanced = -1*np.heaviside(-Tau_total,0.5)*(greater_total-lesser_total)
        
        
        
        print('imaginary part')
        print(np.sum(abs(self.adag_a.imag)))
        print(np.sum(abs(self.a_adag.imag)))
        
        print('real part')
        print(np.sum(abs(self.adag_a.real)))
        print(np.sum(abs(self.a_adag.real)))

        lesser_tau=np.trapz(lesser_total,t,axis=0)/period
        greater_tau=np.trapz(greater_total,t,axis=0)/period
        adag_a_tau=np.trapz(self.adag_a,t,axis=0)/period
        a_adag_tau=np.trapz(self.a_adag,t,axis=0)/period
        Gr_tau=np.trapz(Gr,t,axis=0)/period
        Gk_tau=np.trapz(Gk,t,axis=0)/period
        Ga_tau=np.trapz(G_advanced,t,axis=0)/period
        
        plt.figure()
        plt.title('lesser, greater imaginary')
        plt.plot(Tau_total,lesser_tau.imag,label='lesser')
        plt.plot(Tau_total,greater_tau.imag,label='greater')
        plt.legend()
        plt.xlabel('Tau')
        plt.figure()
        plt.title('lesser, greater real')
        plt.plot(Tau_total,lesser_tau.real,label='lesser')
        plt.plot(Tau_total,greater_tau.real,label='greater')
        plt.legend()
        plt.xlabel('Tau')
        
        plt.figure()
        plt.title('retarted,keldysh imaginary')
        plt.plot(self.Tau,Gr_tau.imag,label='retarted')
        plt.plot(Tau_total,Gk_tau.imag,label='keldysh')
        plt.legend()
        plt.xlabel('Tau')

        plt.figure()
        plt.title('retarted real')
        plt.plot(self.Tau,Gr_tau.real,label='retarted')
        plt.plot(Tau_total,Gk_tau.real,label='keldysh')
        plt.legend()
        
        plt.figure()
        plt.title('expectation values imaginary')
        plt.plot(self.Tau,a_adag_tau.imag,label='a adag')
        plt.plot(self.Tau,adag_a_tau.imag,label='adag a')
        plt.legend()
        plt.xlabel('Tau')
        plt.figure()
        plt.title('expectation values real')
        plt.plot(self.Tau,a_adag_tau.real,label='a adag')
        plt.plot(self.Tau,adag_a_tau.real,label='adag a')
        plt.xlabel('Tau')
        plt.legend()
        
        N_Tau=len(Tau_total)
        Period=abs(Tau_total[-1]-Tau_total[0])
        norm=Period/N_Tau#/np.sqrt(2*np.pi)
        
        omegas=np.fft.fftfreq(N_Tau,Tau[1]-Tau[0])*2*np.pi
        omegas=np.fft.fftshift(omegas)

        lesser_om=np.fft.ifftshift(lesser_tau)
        #lesser_om=lesser_tau
        lesser_om=np.fft.ifft(lesser_om,norm='forward')*norm
        lesser_om=np.fft.fftshift(lesser_om)
        
        greater_om=np.fft.ifftshift(greater_tau)
        #greater_om=greater_tau
        greater_om=np.fft.ifft(greater_om,norm='forward')*norm
        greater_om=np.fft.fftshift(greater_om)
        
        #plt.title('spectral function')
        #txt='V='+str(V)+'  Om='+str(Om)+'  U='+str(U)+'  T='+str(T[0])+'  G='+str(Gamma1[0])
        
        #txt=''
        #plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
        print('len omegas:',len(omegas))
        #start=8000
        #end=12000
        #plt.plot(Tau[N_start:],np.real(G_r[N_start:]),label='real')
        #plt.plot(omegas,np.imag(G_r[N_start:]),label='imaginary')

        
        plt.figure()
        plt.title('real part')
        plt.plot(omegas[0:-1],lesser_om.real[0:-1],label='lesser')
        plt.plot(omegas[0:-1],greater_om.real[0:-1],label='greater')
        plt.xlabel('w')
        plt.legend()
        
        plt.figure()
        plt.title('imaginary part')
        plt.plot(omegas[0:-1],lesser_om.imag[0:-1],label='lesser')
        plt.plot(omegas[0:-1],greater_om.imag[0:-1],label='greater')
        plt.xlabel('w')
        plt.legend()
        plt.show()
        
        
        return self.Tau, Tau_total, np.trapz(Gr,t,axis=0)/period, Ga_tau, np.trapz(Gk,t,axis=0)/period,lesser_om,greater_om
    
    def _GreaterLesser(self,sites,dt=0.05,eps=1e-12,max_iter=1000,av_periods=5,
                       tf=5e2,t_step=1e2,av_Tau=10):
        
        
        if self.parameters['frequency']:
            period=np.pi*2/self.parameters['frequency']
        else:
            period=1


        if self.rhos_period is None:
            self.rhoOf_t(dt=dt,eps=eps,max_iter=max_iter,av_periods=av_periods,
                         tf=tf,t_step=t_step,return_all=False)
        
        t=self.t_period
        rhos=self.rhos_period
        Tau=np.linspace(0,tf,int(tf/dt)+1)

        global LindbladM
        LindbladM=createLindblad(self.basisM,self.parameters,spin_sym=self.spin_sym)
        global LindbladP
        LindbladP=createLindblad(self.basisP,self.parameters,spin_sym=self.spin_sym)
        
        Greater=np.zeros((len(t),len(Tau)))*0j
        Lesser=np.zeros((len(t),len(Tau)))*0j
        
        rhos_a=np.zeros((self.basisM.Ns,len(rhos[0,:])))+0j
        rhos_adag=np.zeros((self.basisP.Ns,len(rhos[0,:])))+0j
        
        for j in range(len(rhos[0,:])):  
            rhos_a[:,j]=self.action_a(rhos[:,j],sites[1])
            rhos_adag[:,j]=self.action_adag(rhos[:,j],sites[1])
        
        self.Tau=None
        self.a_adag=None
        self.adag_a=None
        
        Tau_last=0
        diff=1
        i=0
        while diff>eps:
            if i>max_iter:
                raise RuntimeError(f"Max iterations reached ({max_iter}) without convergence. Last epsilon: {diff}")
                
            Tau, Lesser, Greater,rhos_a,rhos_adag,diff=self.stepsGreaterLesser(
                                  sites,Tau_last,dt,tf,t_step,av_Tau,rhos_a,rhos_adag,
                                  )#LindbladM,LindbladP)
            print('diff',diff)
            Tau_last=Tau[-1]

            if self.Tau is None:
                self.Tau=Tau
                self.adag_a=Lesser
                self.a_adag=Greater
                
            else:
                self.Tau=np.concatenate((self.Tau,Tau),axis=0)
                self.adag_a=np.concatenate((self.adag_a,Lesser),axis=1)
                self.a_adag=np.concatenate((self.a_adag,Greater),axis=1)
            i+=1
            print(Tau[-1])
            print(diff)
        return t,self.Tau,self.adag_a,self.a_adag 
    
    #@njit(parallel=True)
    #@memory.cache
    def evolve_single_step(self,rhos_a, rhos_adag, t, Tau, Tau_last,j,plus_lV,minus_lV):
        """
        Evolve a single step for rhos_a and rhos_adag, then update Lesser, Greater, and the evolution matrices.
        """
        # Evolve the system for rhos_a and rhos_adag
        rhoTau_a = LindbladM.operator.evolve(rhos_a[:, j], t[j] +Tau_last , t[j] + Tau)
        #rhoTau_a = np.array(list(rhoTau_a)).T
        rhoTau_adag = LindbladP.operator.evolve(rhos_adag[:, j], t[j] +Tau_last , t[j] + Tau)
        #rhoTau_adag = np.array(list(rhoTau_adag)).T
    
        # Update matrices for rhos_Tau_a and rhos_Tau_adag
        #rhos_Tau_a[:, j] = rhoTau_a[:, -1]
        #rhos_Tau_adag[:, j] = rhoTau_adag[:, -1]
    
        # Calculate Lesser and Greater 
        Lesser_j = (plus_lV.T.conjugate() @ rhoTau_a) [0]
        Greater_j = (minus_lV.T.conjugate() @ rhoTau_adag) [0]
        rhoTau_adag=rhoTau_adag[:, -1]
        rhoTau_a=rhoTau_a[:, -1]
        
        #del rhoTau_a
        #del rhoTau_adag
        gc.collect()
        with progress_lock:  # Ensure thread safety
            global progress_count
            progress_count += 1
            #if progress_count % 10 == 0 or progress_count == len(rhos_a[0, :]):  # Print every 10%
                #print(f"Progress: {progress_count}/{len(rhos_a[0, :])} ({progress_count/len(rhos_a[0, :])*100:.1f}%)")

        return j, rhoTau_a, rhoTau_adag, Lesser_j, Greater_j

    def stepsGreaterLesser(self,sites,Tau_last,dt,tf,t_step,av_Tau,rhos_a,rhos_adag,
                           ):
        plus_lV=self.plus_leftVacuum(sites[0])
        minus_lV=self.minus_leftVacuum(sites[0])
        t=self.t_period
        if Tau_last==0:
            Tau=np.linspace(0,tf,int(tf/dt)+1)+Tau_last
        else:
            Tau=np.linspace(dt,t_step,int(t_step/dt)+1)+Tau_last
            
        Greater=np.zeros((len(t),len(Tau)))+0j
        Lesser=np.zeros((len(t),len(Tau)))+0j
        #percent=0
        rhos_Tau_a=np.zeros(rhos_a.shape)+0j
        rhos_Tau_adag=np.zeros(rhos_a.shape)+0j
        times=Tau[:, np.newaxis] + t
        print('trying t')
        print(np.shape(rhos_a),np.shape(np.conj([t])),np.shape(times))
        
        #print('parallel')
        col_int=32
        for jstart in range(0,len(rhos_a[0,:]),col_int):
            print(jstart)
            if jstart +col_int > len(rhos_a[0,:]):
                jend=len(rhos_a[0,:])
            else:
                jend=jstart +col_int
                
            results = Parallel(n_jobs=8, backend="threading", prefer="threads",batch_size=8,require='sharedmem')(
            delayed(self.evolve_single_step)(rhos_a, rhos_adag, t, Tau, Tau_last,j,plus_lV,minus_lV)
            for j in range(jstart,jend)
            )
            results.sort(key=lambda x: x[0])
            _,rhos_Tau_a_j, rhos_Tau_adag_j, Lesser_j, Greater_j = zip(*results)
            
            Lesser[jstart:jend]=np.array(Lesser_j)
            Greater[jstart:jend]=np.array(Greater_j)
            rhos_Tau_a[:,jstart:jend]=np.column_stack(rhos_Tau_a_j)
            rhos_Tau_adag[:,jstart:jend]=np.column_stack(rhos_Tau_adag_j)
            print('sleeping')
            time.sleep(0.5)  # Pause execution for 2 seconds to clear swap

            # Force garbage collection 
            gc.collect()


        N_av=int(np.ceil(av_Tau/dt))
        Lesser_av_t=np.trapz(abs(Lesser),t,axis=0)/(t[-1]-t[0])
        Lesser_av=np.trapz(Lesser_av_t[-N_av:],Tau[-N_av:])/av_Tau
        Greater_av_t=np.trapz(abs(Greater),t,axis=0)/(t[-1]-t[0])
        Greater_av=np.trapz(Greater_av_t[-N_av:],Tau[-N_av:])/av_Tau
        
        diff=Lesser_av+Greater_av
        return Tau, Lesser, Greater,rhos_Tau_a,rhos_Tau_adag,diff
        

    def _GreaterLesserSites(self,sites,dt=0.05,eps=1e-12,max_iter=1000,
                        av_periods=5,tf=5e2,t_step=1e2,av_Tau=10,writeFile=False,
                        dirName=None):
        
        res_sites=[]
        t_res=[]
        Tau_res=[]
        for i in range(len(sites)):
            print(sites[i])
            t,Tau,adag_a,a_adag=self._GreaterLesser(sites[i],dt,eps,max_iter,
                                av_periods,tf,t_step,av_Tau)
            if len(t_res)==0:
                t_res=t.tolist()
            if len(Tau_res)<len(Tau):
                Tau_res=Tau.tolist()
                
            res_sites.append({'sites': str(sites[i][0])+' '+str(sites[i][1]),
                              'a+_a real': adag_a.real.tolist(),
                              'a+_a imag': adag_a.imag.tolist(),
                              'a_a+ real': a_adag.real.tolist(),
                              'a_a+ imag': a_adag.imag.tolist(),

                              })
            
        for i in range(len(sites)):
            if len(res_sites[i]['a+_a real'][0])<len(Tau_res):
                zeros=np.zeros(len(Tau_res)-len(res_sites[i]['a+_a real'][0])).tolist()
                res_sites[i]['a+_a real'] = res_sites[i]['a+_a real'].append(zeros)
                res_sites[i]['a+_a imag'] = res_sites[i]['a+_a imag'].append(zeros)
                res_sites[i]['a_a+ real'] = res_sites[i]['a_a+ real'].append(zeros)
                res_sites[i]['a_a+ imag'] = res_sites[i]['a_a+ imag'].append(zeros)
                
        
               
        if writeFile:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            _input = {
                'parameters':self.parameters,
                'spin':self.spin,
                'sites':sites,
                'time_param': {'dt':dt,
                               'eps':eps,
                               'max_iter':max_iter,
                               'av_periods':av_periods,
                               'tf':tf,
                               't_step':t_step,
                               'av_Tau':av_Tau,

                               }
                }
            
            _output={
                       't':t_res,
                       'Tau':Tau_res,
                       'results':res_sites,
                       }
            
            json_File={
                       'input':_input,
                       'output':_output,
                       }
            
            filename = ('U'+str(self.parameters['interaction']) +
            'V'+str(self.parameters['drive'])+
            'Om'+str(self.parameters['frequency'])+
             '_'+timestr+'.json')
            
            filepath=os.path.join(dirName, filename )
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(json_File, f, ensure_ascii=False)
                
        return t,Tau,res_sites
            
            

    def rhoOf_t(self,dt=0.05,eps=1e-12,max_iter=1000,av_periods=5,tf=5e2,t_step=1e2,return_all=False):
        Lindblad0=createLindblad(self.basis0,self.parameters,spin_sym=self.spin_sym)
        rho0=self.leftVacuum
        rho0T=rho0.T.toarray()
        
        if Lindblad0.Om:
            period=np.pi*2/Lindblad0.Om
        else:
            period=1
            
        #calculate the number of period in tf
        num_periods_tf=int(np.ceil(tf/period))
        #the required points per period given dt
        N_period=int(period/dt)
        #the timepoints for one period
        t_period=np.linspace(0,period-dt,N_period)
        #the complete timevector
        t = np.concatenate([t_period + n * period for n in range(num_periods_tf)])
        
        rhos=Lindblad0.operator.evolve(rho0T[0],0,t,iterate=True)
        rhos = np.array(list(rhos)).T
        
        #calulate the differnce per period
        diff_period=0
        for j in range(1,av_periods):
            if (av_periods+1)*N_period>len(t)+1:
                diff_period=1
                break
            t_period_diff=t[-(1+j)*N_period-1:-j*N_period]
            print(len(t_period_diff))
            diff_period+=np.sum(np.trapz(abs(rhos[:,-N_period-1:]-rhos[:,-(j+1)*N_period-1:-j*N_period]),t_period_diff,axis=1))/period/len(rhos)/av_periods
        num_periods_step=int(np.ceil(t_step/period))

        t_total=t.copy()
        rhos_total=rhos.copy()
        i=0
        while diff_period > eps:
            t_last=t[-1]
            t = np.concatenate([t_period + n * period for n in range(num_periods_step)])+t_last+dt
            rhos=Lindblad0.operator.evolve(rhos[:,-1],t_last,t)
            
            t_total=np.append(t_total,t)
            rhos_total=np.append(rhos_total,rhos,axis=1)
            print(t_total.shape)
            print(rhos_total.shape)
                
            if i>max_iter:
                if return_all:
                    return t_total,rhos_total
                else:
                    return t,rhos
                raise RuntimeError(f"Max iterations reached ({max_iter}) without convergence. Last epsilon: {diff_period}")

            diff_period=0
            for j in range(1,av_periods):
                if (av_periods+1)*N_period>len(t_total):
                    diff_period=1
                    break
                t_period_diff=t_total[-(1+j)*N_period-1:-j*N_period]
                diff_period+=np.sum(np.trapz(abs(rhos_total[:,-N_period-1:]-rhos_total[:,-(j+1)*N_period-1:-j*N_period]),t_period_diff,axis=1))/period/len(rhos_total)/av_periods
            print(t_total[-1])
            print(diff_period)
                
                
        self.t_period=t_total[-N_period-1:]
        self.rhos_period=rhos_total[:,-N_period-1:]
        
        if return_all:
            return t_total,rhos_total
        
        else:
            return t_total[-2*N_period-1:],rhos_total[:,-2*N_period-1:]
            
        
        
        
        
    def testConvergence_t(self,tf,Tf,dt=0.1):
        
        Lindblad0=createLindblad(self.basis0,self.parameters,spin_sym=self.spin_sym)
        rho0=self.leftVacuum
        rho0T=rho0.T.toarray()
        
        if Lindblad0.Om:
            period=np.pi*2/Lindblad0.Om
        else:
            period=1
        
        num_periods=int(np.ceil(tf/period))
        N_period=int(period/dt)
        t=np.linspace(0,period-dt,N_period)
        t = np.concatenate([t + n * period for n in range(num_periods)])
        print(t)
        
        
        rhos=Lindblad0.operator.evolve(rho0T[0],0,t)
        
        diff_period=0
        plt.figure()
        t_test=np.zeros(5)
        t_test[0]=t[-(N_period+1)]
        for j in range(1,5):
            diff_time=np.sum(abs(rhos[:,-(N_period+1):]-rhos[:,-(j+1)*N_period-1:-j*N_period]),axis=0)/len(rhos)
            t_period=t[-(1+j)*N_period-1:-j*N_period]
            t_test[j]=t[-(j+1)*N_period-1]
            print(t_test[j-1]-t_test[j]-2*np.pi)
            diff_state=np.trapz(abs(rhos[:,-N_period-1:]-rhos[:,-(j+1)*N_period-1:-j*N_period]),t_period,axis=1)/period
            
            
            diff_period+=np.sum(np.trapz(abs(rhos[:,-N_period-1:]-rhos[:,-(j+1)*N_period-1:-j*N_period]),t_period,axis=1))
            plt.plot(diff_time,label=j)
        plt.legend()
        
        diff_period=0
        plt.figure()
        for j in range(1,5):
            diff_time=np.sum(abs(rhos[:,-N_period:]-rhos[:,-(j+1)*N_period:-j*N_period]),axis=0)
            diff_state=np.trapz(abs(rhos[:,-N_period-1:]-rhos[:,-(j+1)*N_period-1:-j*N_period]),t_period,axis=1)/period
            diff_period+=np.sum(np.trapz(abs(rhos[:,-N_period-1:]-rhos[:,-(j+1)*N_period-1:-j*N_period]),t_period,axis=1))/period/len(rhos)
            plt.plot(diff_state,label=j)
        plt.legend()
        print('diff_period: ',diff_period)
            
            
        print(len(t))
        start=len(t)-5000
        i=0
        print(self.basis0.Ns)
        for rho in rhos:
            if np.max(abs(rho[start:].real))>1e-8 and i>205:
                print(i)
                print(np.max(abs(rho[start:])))
                plt.figure()
                plt.plot(t[start:],rho[start:].real,label='real')
                plt.title(self.basis0.int_to_state(self.basis0[i]))
                plt.axvline(x=t[-1]-period,color='r', linestyle='--', linewidth=0.6)
                plt.axvline(x=t[-1]-2*period,color='r', linestyle='--', linewidth=0.6)
                plt.plot(t[-(1+1)*N_period-1:-1*N_period],rhos[i,-(1+1)*N_period-1:-1*N_period].real,color='red')
                plt.plot(t[-(2+1)*N_period-1:-2*N_period],rhos[i,-(2+1)*N_period-1:-2*N_period].real,color='orange')
                plt.plot(t[-(3+1)*N_period-1:-3*N_period],rhos[i,-(3+1)*N_period-1:-3*N_period].real,color='yellow')
                plt.legend()
            
            if np.max(abs(rho[start:].imag))>1e-8 and i>205:
                print(i)
                print(np.max(abs(rho[start:])))
                plt.figure()
                plt.plot(t[start:],rho[start:].imag,label='imag')
                plt.axvline(x=t[-1]-period,color='r', linestyle='--', linewidth=0.6)
                plt.axvline(x=t[-1]-2*period,color='r', linestyle='--', linewidth=0.6)
                plt.title(self.basis0.int_to_state(self.basis0[i]))
                plt.plot(t[-(1+1)*N_period:-1*N_period],rhos[i,-(1+1)*N_period:-1*N_period].imag,color='red')
                plt.plot(t[-(2+1)*N_period:-2*N_period],rhos[i,-(2+1)*N_period:-2*N_period].imag,color='orange')
                plt.plot(t[-(3+1)*N_period:-3*N_period],rhos[i,-(3+1)*N_period:-3*N_period].imag,color='yellow')
                plt.legend()
                
            i+=1
            
            
        rhos=rhos[:,len(t)-N_period+1:]
        t=t[len(t)-N_period+1:]
        
        print(t[-1]-t[0])
        
        
    def _GreaterLesser3(self,tf,Tf,dt=0.1):
        
        Lindblad0=createLindblad(self.basis0,self.parameters,spin_sym=self.spin_sym)
        rho0=self.leftVacuum
        rho0T=rho0.T.toarray()
        
        if Lindblad0.Om:
            period=np.pi*2/Lindblad0.Om
        else:
            period=1
        
        num_periods=int(np.ceil(tf/period))
        N_period=int(period/dt)
        t=np.linspace(0,period-dt,N_period)
        t = np.concatenate([t + n * period for n in range(num_periods)])
        print(t)
        
        
        rhos=Lindblad0.operator.evolve(rho0T[0],0,t)


        plt.figure()
        for rho in rhos:
            plt.plot(t,rho)
            
            
        t=t[len(t)-N_period+1:]
        rhos=rhos[:,len(t)-N_period+1:]
        print(t[-1]-t[0])

        
        
        
        Tau=np.linspace(0,Tf,int(Tf/dt)+1)

        Greater=np.zeros((len(t),len(Tau)))*0j
        Lesser=np.zeros((len(t),len(Tau)))*0j
        percent=0
        
        
        LindbladM=createLindblad(self.basisM,self.parameters,spin_sym=self.spin_sym)
        LindbladP=createLindblad(self.basisP,self.parameters,spin_sym=self.spin_sym)
        print(rhos.shape)
        for j in range(len(rhos[0,:])):  
            #print(j)
            if j/len(t)>=percent:
                print(np.ceil(j/len(t)*100),'%')
                percent+=0.01
            rho=rhos[:,j]
            print(rho.shape)
            rho_a=self.action_a(rho)
            rho_adag=self.action_adag(rho)

            rhoTau_a=LindbladM.operator.evolve(rho_a,t[j],t[j]+Tau)
            rhoTau_adag=LindbladP.operator.evolve(rho_adag,t[j],t[j]+Tau)
            
            Lesser[j]=self.plus_lV.T.conjugate()@rhoTau_a
            Greater[j]=self.minus_lV.T.conjugate()@rhoTau_adag
        
        self.greater=Greater
        self.lesser=Lesser
        self.t=t
        self.Tau=Tau
        
    def G_r(self,tf,Tf,dt=0.1):
        t=np.linspace(0,tf,int(tf/dt)+1)
        basis0=augmented_basis(self.L,'restricted',[0,0])

        Lindblad0=createLindblad(basis0,self.parameters,spin_sym=self.spin_sym)
        rho0=self.leftVacuum
        rho0T=rho0.T.toarray()
        
        rho=Lindblad0.operator.evolve(rho0T[0],0,t)
        rho_end=rho[:,-1]


        basisM,rho_a=self.action_a(basis0,rho_end)
        LindbladM=createLindblad(basisM,self.parameters,spin_sym=self.spin_sym)
        basisP,rho_adag=self.action_adag(basis0,rho_end)
        LindbladP=createLindblad(basisP,self.parameters,spin_sym=self.spin_sym)
        
        Tau=np.linspace(0,Tf,int(Tf/dt)+1)
        
        
        rhoTau_a=LindbladM.operator.evolve(rho_a,0,Tau)
        rhoTau_adag=LindbladP.operator.evolve(rho_adag,0,Tau)
        
        G1=self.plus_lV.T.conjugate()@rhoTau_a
        G2=self.minus_lV.T.conjugate()@rhoTau_adag
        Gr=-1j*(np.conj(G1)+G2)*np.heaviside(Tau,0.5)
        
        
        return Tau, Gr[0]

    
    def Gr_floquet1(self,tf,Tf,dt=0.1):

        Lindblad0=createLindblad(self.basis0,self.parameters,spin_sym=self.spin_sym)
        rho0=self.leftVacuum
        rho0T=rho0.T.toarray()
        
        if Lindblad0.Om:
            period=np.pi*2/Lindblad0.Om
        else:
            period=1
        
        t=np.linspace(0,tf,int(np.ceil(tf/dt)))#period/dt)
        rhos=Lindblad0.operator.evolve(rho0T[0],0,t)
        print(t.shape)
        print(rhos.shape)
        N_period=int(np.floor(period/dt))+1
        rhos=rhos[:,len(t)-N_period:]
        t=t[len(t)-N_period:]

        print(len(t))
        print(rhos.shape)
        #t=np.linspace(tf-period,tf,int(np.ceil(period/dt)))#period/dt)
        #print(t)
        #rhos=Lindblad0.operator.evolve(rho0T[0],0,t)
        
        #plot equal time
        n_exps=[]
        for i in range(len(rhos[0])):
            rho_n=self.action_n(rhos[:,i])
            n_exp=self.leftVacuum.T.conjugate()@rho_n
            n_exps.append(n_exp)
            
        #plt.plot(t,np.real(n_exps))
        #plt.plot(t,np.imag(n_exps))
        

        Tau=np.linspace(0,Tf,int(Tf/dt)+1)
        #print(len(rhos[0,:]))
        Gr=np.zeros((len(t),len(Tau)))*0j
        percent=0
        
        
        LindbladM=createLindblad(self.basisM,self.parameters,spin_sym=self.spin_sym)
        LindbladP=createLindblad(self.basisP,self.parameters,spin_sym=self.spin_sym)
        print(rhos.shape)
        for j in range(len(rhos[0,:])):  
            #print(j)
            if j/len(t)>=percent:
                print(np.ceil(j/len(t)*100),'%')
                percent+=0.01
            rho=rhos[:,j]
            print(rho.shape)
            rho_a=self.action_a(rho)
            rho_adag=self.action_adag(rho)

            rhoTau_a=LindbladM.operator.evolve(rho_a,t[j],t[j]+Tau)
            rhoTau_adag=LindbladP.operator.evolve(rho_adag,t[j],t[j]+Tau)
            
            G1=self.plus_lV.T.conjugate()@rhoTau_a
            G2=self.minus_lV.T.conjugate()@rhoTau_adag
            Gr[j]=-1j*(np.conj(G1)+G2)*np.heaviside(Tau,0.5)
        

        
        return Tau, np.trapz(Gr,t,axis=0)/period

    #include change in integral borders
    def Gr_floquet2(self,tf,Tf,dt=0.1):
        basis0=augmented_basis(self.L,'restricted',[0,0])

        Lindblad0=createLindblad(basis0,self.parameters,spin_sym=self.spin_sym)
        rho0=self.leftVacuum
        rho0T=rho0.T.toarray()
        
        if Lindblad0.Om:
            period=np.pi*2/Lindblad0.Om
        else:
            period=1
            
        basisM,rho_a=self.action_a(basis0,rho0.toarray())
        LindbladM=createLindblad(basisM,self.parameters,spin_sym=self.spin_sym)
        basisP,rho_adag=self.action_adag(basis0,rho0.toarray())
        LindbladP=createLindblad(basisP,self.parameters,spin_sym=self.spin_sym)
        
        
        Tau=np.linspace(0,Tf,int(Tf/dt)+1)
        percent=0


        N_Tau=int(Tf/dt)+1
        #N_Tauhalf=int(np.ceil(N_Tau/2))
        N_period=int(np.ceil(period/dt)+1)
        t=np.linspace(tf-period/2-Tf/2,tf+period/2,N_period+N_Tau)#period/dt)
        
        rhos=Lindblad0.operator.evolve(rho0T[0],0,t)
        
        
        #print(len(rhos[0,:]))
        Gr=np.zeros(len(Tau))*0j
        for i in range(len(Tau)):
            if i/len(Tau)>=percent:
                print(np.ceil(i/len(Tau)*100),'%')
                percent+=0.01
            Gr_t=np.zeros(N_period)*0j   
            m=0
            for j in range(N_Tau-i,N_period+N_Tau-i):
                rho=rhos[:,j]

                basisM,rho_a=self.action_a(basis0,rho,basisM)
                basisP,rho_adag=self.action_adag(basis0,rho,basisP)

                rhoTau_a=LindbladM.operator.evolve(rho_a,t[j],t[j]+Tau[i])
                rhoTau_adag=LindbladP.operator.evolve(rho_adag,t[j],t[j]+Tau[i])
                    
                G1=self.plus_lV.T.conjugate()@rhoTau_a
                G2=self.minus_lV.T.conjugate()@rhoTau_adag

                Gr_t[m]=-1j*(np.conj(G1[0])+G2[0])*np.heaviside(Tau[i],0.5)
                m+=1

            Gr[i]=np.trapz(Gr_t,t[N_Tau-i:N_period+N_Tau-i],axis=0)/period

        return Tau, Gr

parameters_sym = {"sites": 3,
              "epsilon": [0,0,0],
              "hopping": 0.5,
              "interaction":0,
              "drive": 1,
              "frequency":1,
              "coupling_empty":[0.5,0,0.5],
              "coupling_full":[0.5,0,0.5],
              "spin_symmetric":True,
}


#start=time.time()
#GF0_sym=calculateGreensFunction(parameters_sym,0,'updown')
#Tau,Gr_Tau=GF0_sym._GreaterLesser()   
#end=time.time()
#print(end-start)
#print(LindbladM)
    
    
    