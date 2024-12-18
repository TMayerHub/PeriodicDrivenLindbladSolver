#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 08:02:54 2024

@author: theresa
"""

from Lindblatt import createLindblad
from augmented_basis import augmented_basis

from quspin.operators import hamiltonian

import scipy
import numpy as np
import matplotlib.pyplot as plt
import warnings

class calculateGreensFunction:
    def __init__(self,parameters,site,spin):
        self.parameters=parameters
        self.spin_sym=parameters['spin_symmetric']
        self.site=site
        self.L=parameters['sites']
        self.center=self.findCenter()
        self.spin=spin

        print('start define basis')
        self.basis0=augmented_basis(self.L,'restricted',[0,0],spin_sym=self.spin_sym)
        self.basisE=augmented_basis(self.L,spin_sym=self.spin_sym)
        self.basisP=self._basisP()
        self.basisM=self._basisM()
        print('end define basis')
        
        self.leftVacuum=self._leftVacuum()
        self.plus_lV=self.plus_leftVacuum()
        self.minus_lV=self.minus_leftVacuum()
        
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
            diff_sector = [-1,0,0,-1]
            
        if self.spin == 'up':
            diff_sector = [-1,0]
            
        if self.spin == 'down':
            diff_sector = [0,-1]
            
        return augmented_basis(self.L,'restricted',diff_sector,spin_sym=self.spin_sym)
    
    def _basisP(self):
        if self.spin == 'updown':
            diff_sector = [1,0,0,1]
            
        if self.spin == 'up':
            diff_sector = [1,0]
            
        if self.spin == 'down':
            diff_sector = [0,1]
            
        return augmented_basis(self.L,'restricted',diff_sector,spin_sym=self.spin_sym)
        

    def action_n(self,state):
        index=self.center+2*self.site
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
    
    def action_a(self,state_vector):
        
        index=self.center+2*self.site
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
    
    def action_adag(self,state_vector):
        
        index=self.center+2*self.site
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
        
    
    def plus_leftVacuum(self):
        '''adag acting from the right <I|a+, returns a|I>'''
        lV=self.leftVacuum.toarray()
        plV=self.action_a(lV)
        return scipy.sparse.csr_array(plV)
    
    def minus_leftVacuum(self):
        '''a acting from the right <I|a, , returns a+|I>'''
        lV=self.leftVacuum.toarray()
        mlV=self.action_adag(lV)
        return scipy.sparse.csr_array(mlV)
    
    def plot_n(self,tf,dt=0.1):
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
        plt.figure()
        n_exps=[]
        for i in range(len(rhos[0])):
            rho_n=self.action_n(rhos[:,i])
            n_exp=self.leftVacuum.T.conjugate()@rho_n
            n_exps.append(n_exp)
            
        plt.plot(t,np.real(n_exps))
        #plt.plot(t,np.imag(n_exps))
        
        plt.figure()
        #t_period=np.linspace(tf-period,tf-5dt,int(np.floor(period/dt)))#period/dt)
        #rhos=Lindblad0.operator.evolve(rho0T[0],0,t_period)
        
        #plot equal time
        N_period=int(np.floor(period/dt))
        n_exps=[]
        for i in range(len(rhos[0])-N_period,len(rhos[0])):
            rho_n=self.action_n(rhos[:,i])
            n_exp=self.leftVacuum.T.conjugate()@rho_n
            n_exps.append(n_exp)
            
        plt.plot(t[len(rhos[0])-N_period:],np.real(n_exps))
        print(n_exps[-1])
        return n_exps[-1]
        #plt.plot(t_period,np.imag(n_exps))
        
        
    def _GreaterLesser(self,dt=0.05,eps=1e-12,max_iter=1000,av_periods=5,
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


        LindbladM=createLindblad(self.basisM,self.parameters,spin_sym=self.spin_sym)
        LindbladP=createLindblad(self.basisP,self.parameters,spin_sym=self.spin_sym)
        
        Greater=np.zeros((len(t),len(Tau)))*0j
        Lesser=np.zeros((len(t),len(Tau)))*0j
        
        rhos_a=np.zeros((self.basisM.Ns,len(rhos[0,:])))+0j
        rhos_adag=np.zeros((self.basisP.Ns,len(rhos[0,:])))+0j
        
        for j in range(len(rhos[0,:])):  
            rhos_a[:,j]=self.action_a(rhos[:,j])
            rhos_adag[:,j]=self.action_adag(rhos[:,j])
        
        self.Tau=None
        self.Greater=None
        self.Lesser=None
        
        Tau_last=0
        diff=1
        i=0
        while diff>eps:
            if i>max_iter:
                raise RuntimeError(f"Max iterations reached ({max_iter}) without convergence. Last epsilon: {diff}")
                
            Tau, Lesser, Greater,rhos_a,rhos_adag,diff=self.stepsGreaterLesser(
                                  Tau_last,dt,tf,t_step,av_Tau,rhos_a,rhos_adag,
                                  LindbladM,LindbladP)
            Tau_last=Tau[-1]

            if self.Tau is None:
                self.Tau=Tau
                self.lesser=Lesser
                self.greater=Greater
                
            else:
                self.Tau=np.concatenate((self.Tau,Tau),axis=0)
                self.lesser=np.concatenate((self.lesser,Lesser),axis=1)
                self.greater=np.concatenate((self.greater,Greater),axis=1)
            i+=1
            
        
        Gr=-1j*(np.conj(self.lesser)+self.greater)*np.heaviside(self.Tau,0.5)
        
        return self.Tau, np.trapz(Gr,t,axis=0)/period
    
    def stepsGreaterLesser(self,Tau_last,dt,tf,t_step,av_Tau,rhos_a,rhos_adag,
                           LindbladM,LindbladP):
        t=self.t_period
        if Tau_last==0:
            Tau=np.linspace(0,tf,int(tf/dt)+1)+Tau_last
        else:
            Tau=np.linspace(dt,t_step,int(t_step/dt))+Tau_last
            
        Greater=np.zeros((len(t),len(Tau)))*0j
        Lesser=np.zeros((len(t),len(Tau)))*0j
        percent=0
        rhos_Tau_a=np.zeros(rhos_a.shape)+0j
        rhos_Tau_adag=np.zeros(rhos_a.shape)+0j
        for j in range(len(rhos_a[0,:])):  
            #print(j)
            if j/len(t)>=percent:
                print(np.ceil(j/len(t)*100),'%')
                percent+=0.01
            
            rhoTau_a=LindbladM.operator.evolve(rhos_a[:,j],t[j],t[j]+Tau)
            rhoTau_adag=LindbladP.operator.evolve(rhos_adag[:,j],t[j],t[j]+Tau)
            
            rhos_Tau_a[:,j]=rhoTau_a[:,-1]
            rhos_Tau_adag[:,j]=rhoTau_adag[:,-1]
            
            Lesser[j]=self.plus_lV.T.conjugate()@rhoTau_a
            Greater[j]=self.minus_lV.T.conjugate()@rhoTau_adag
        
        N_av=int(np.ceil(av_Tau/dt))
        Lesser_av_t=np.trapz(abs(Lesser),t,axis=0)/(t[-1]-t[0])
        Lesser_av=np.trapz(Lesser_av_t[-N_av:],Tau[-N_av:])/av_Tau
        Greater_av_t=np.trapz(abs(Greater),t,axis=0)/(t[-1]-t[0])
        Greater_av=np.trapz(Greater_av_t[-N_av:],Tau[-N_av:])/av_Tau
        
        diff=Lesser_av+Greater_av
        return Tau, Lesser, Greater,rhos_Tau_a,rhos_Tau_adag,diff
        
        
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
        
        rhos=Lindblad0.operator.evolve(rho0T[0],0,t)
        
        #calulate the differnce per period
        diff_period=0
        for j in range(1,av_periods):
            t_period_diff=t[-(1+j)*N_period-1:-j*N_period]
            diff_period+=np.sum(np.trapz(abs(rhos[:,-N_period-1:]-rhos[:,-(j+1)*N_period-1:-j*N_period]),t_period_diff,axis=1))/period/len(rhos)/av_periods
        
        num_periods_step=int(np.ceil(t_step/period))

        t_total=t.copy()
        rhos_total=rhos.copy()
        i=0
        while diff_period > eps:
            t_last=t[-1]
            t = np.concatenate([t_period + n * period for n in range(num_periods_step)])+t_last
            rhos=Lindblad0.operator.evolve(rhos[:,-1],t_last,t)
            
            if return_all:
                t_total.append(t)
                rhos_total.append(rhos,axis=1)
                
            if i>max_iter:
                if return_all:
                    return t_total,rhos_total
                else:
                    return t,rhos
                raise RuntimeError(f"Max iterations reached ({max_iter}) without convergence. Last epsilon: {diff_period}")
            
            diff_period=0
            for j in range(1,av_periods):
                t_period_diff=t[-(1+j)*N_period-1:-j*N_period]
                diff_period+=np.sum(np.trapz(abs(rhos[:,-N_period-1:]-rhos[:,-(j+1)*N_period-1:-j*N_period]),t_period_diff,axis=1))/period/len(rhos)/av_periods
                
                
                
        self.t_period=t[-N_period-1:]
        self.rhos_period=rhos[:,-N_period-1:]
        
        if return_all:
            return t_total,rhos_total
        
        else:
            return t,rhos
            
        
        
        
        
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
    
    
    
    
    
    
    