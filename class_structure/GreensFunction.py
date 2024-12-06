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
        self.leftVacuum=self._leftVacuum()
        self.plus_lV=self.plus_leftVacuum()
        self.minus_lV=self.minus_leftVacuum()
        
        self.a_adag=None
        self.adag_a=None
        self.equal_time=None
        
    
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
        
        
    def action_n(self,basis,state):
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
        n_op=hamiltonian(static,dynamic,dtype=np.complex128,basis=basis,check_herm=False,check_symm=False,check_pcon=False)
        return basis,n_op.dot(state)
    
    def action_a(self,basis,state_vector,basis_a=None):
        extended_basis=augmented_basis(self.L,spin_sym=self.spin_sym)
        
        index=self.center+2*self.site
        if self.spin == 'updown':
            diff_sector = [-1,0,0,-1]
            a_list=[[1+0j,index],[1+0j,index+2*self.L]]
        if self.spin == 'up':
            if self.spin_sym:
                warnings.warn('Using operator which is not spin symmetric with symmetric basis',
                              category=UserWarning)
            diff_sector = [-1,0]
            a_list=[[1+0j,index]]
        if self.spin == 'down':
            if self.spin_sym:
                warnings.warn('Using operator which is not spin symmetric with symmetric basis',
                              category=UserWarning)
            diff_sector = [0,-1]
            a_list=[[1+0j,index+2*self.L]]
            
        static=[
            ["-", a_list],
            ]
        dynamic = []
        #print(a_list)
        a_op=hamiltonian(static,dynamic,dtype=np.complex128,basis=extended_basis,check_herm=False,check_symm=False,check_pcon=False)
        
        extended_vector=self.to_extended_vector(state_vector,basis,extended_basis)

        a_vector_extended=a_op.dot(extended_vector)

        #print(a_vector_extended)
        if basis_a==None:
            new_basis=augmented_basis(self.L,'restricted',diff_sector,spin_sym=self.spin_sym)
        else:
            new_basis=basis_a
        a_vector=self.to_reduced_vector(a_vector_extended,new_basis,extended_basis)
        #print(a_vector)
        return new_basis,a_vector
    
    def action_adag(self,basis,state_vector,basis_adag=None):
        extended_basis=augmented_basis(self.L,spin_sym=self.spin_sym)
        
        index=self.center+2*self.site
        if self.spin == 'updown':
            diff_sector = [1,0,0,1]
            a_list=[[1+0j,index],[1+0j,index+2*self.L]]
        if self.spin == 'up':
            if self.spin_sym:
                warnings.warn('Using operator which is not spin symmetric with symmetric basis',
                              category=UserWarning)
            diff_sector = [1,0]
            a_list=[[1+0j,index]]
        if self.spin == 'down':
            if self.spin_sym:
                warnings.warn('Using operator which is not spin symmetric with symmetric basis',
                              category=UserWarning)
            diff_sector = [0,1]
            a_list=[[1+0j,index+2*self.L]]
            
        static=[
            ["+", a_list],
            ]
        dynamic = []
        
        adag_op=hamiltonian(static,dynamic,dtype=np.complex128,basis=extended_basis,check_herm=False,check_symm=False,check_pcon=False)
        
        extended_vector=self.to_extended_vector(state_vector,basis,extended_basis)
        
        if basis_adag==None:
            new_basis=augmented_basis(self.L,'restricted',diff_sector,spin_sym=self.spin_sym)
            
        else: 
            new_basis=basis_adag

        #new_basis=augmented_basis(self.L,'restricted',diff_sector)
        adag_vector_extended=adag_op.dot(extended_vector)
        
        adag_vector=self.to_reduced_vector(adag_vector_extended,new_basis,extended_basis)
        return new_basis,adag_vector
    
    
    
    def _leftVacuum(self):
        L=self.L
        basis=augmented_basis(self.L,'restricted',[0,0],spin_sym=self.spin_sym)
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
                data.append((-1j)**n)
                row_ind.append(k)
            
        col_ind=np.zeros(len(data),dtype=int)
        data=np.array(data)
        #create csr matrix
        leftVacuum=scipy.sparse.csr_matrix((data,(row_ind,col_ind)),shape=(basis.Ns,1))
        norm=np.sqrt(leftVacuum.T.conjugate()@leftVacuum)
        leftVacuum=leftVacuum/norm[0,0]
        return leftVacuum
        
    
    def plus_leftVacuum(self):
        '''adag acting from the right <I|a+, returns a|I>'''
        basis=augmented_basis(self.L,'restricted',[0,0],spin_sym=self.spin_sym)
        lV=self.leftVacuum.toarray()
        basis,plV=self.action_a(basis,lV)
        return scipy.sparse.csr_array(plV)
    
    def minus_leftVacuum(self):
        '''a acting from the right <I|a, , returns a+|I>'''
        basis=augmented_basis(self.L,'restricted',[0,0],spin_sym=self.spin_sym)
        lV=self.leftVacuum.toarray()
        basis,mlV=self.action_adag(basis,lV)
        return scipy.sparse.csr_array(mlV)
        
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
        basis0=augmented_basis(self.L,'restricted',[0,0],spin_sym=self.spin_sym)

        Lindblad0=createLindblad(basis0,self.parameters,spin_sym=self.spin_sym)
        rho0=self.leftVacuum
        rho0T=rho0.T.toarray()
        
        if Lindblad0.Om:
            period=np.pi*2/Lindblad0.Om
        else:
            period=1
        
        t=np.linspace(tf-period,tf,int(np.ceil(period/dt)))#period/dt)
        rhos=Lindblad0.operator.evolve(rho0T[0],0,t)
        
        #plot equal time
        #n_exp=expectationValue(i,'n',rhos,basis,leftVacuum)
        #plt.plot(t,np.real(n_exp))
        #plt.plot(t,np.imag(n_exp))
        

        Tau=np.linspace(0,Tf,int(Tf/dt)+1)
        #print(len(rhos[0,:]))
        Gr=np.zeros((len(t),len(Tau)))*0j
        percent=0
        
        basisM,rho_a=self.action_a(basis0,rhos[:,-1])
        LindbladM=createLindblad(basisM,self.parameters,spin_sym=self.spin_sym)
        basisP,rho_adag=self.action_adag(basis0,rhos[:,-1])
        LindbladP=createLindblad(basisP,self.parameters,spin_sym=self.spin_sym)
        
        for j in range(len(rhos[0,:])):  
            #print(j)
            if j/len(t)>=percent:
                print(np.ceil(j/len(t)*100),'%')
                percent+=0.01
            rho=rhos[:,j]
            basisM,rho_a=self.action_a(basis0,rho,basisM)
            #rho_a=a_op.dot(rho)
            basisP,rho_adag=self.action_adag(basis0,rho)
            #rho_adag=adag_op.dot(rho)
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
    
    
    
    
    
    
    