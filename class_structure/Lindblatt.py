#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:48:02 2024

@author: theresa
"""

import warnings

import numpy as np

from quspin.operators import hamiltonian

class Lindblad:
    def __init__(self,basis,parameters,spin_sym=False):
        self.parameters = parameters
        self.basis = basis
        
        self.L = parameters["sites"]
        self.eps = parameters["epsilon"]
        self.T = parameters["hopping"]
        self.U = parameters["interaction"]
        self.V = parameters["drive"]
        self.Om= parameters["frequency"]
        self.Gamma1 = parameters["coupling_empty"]
        self.Gamma2= parameters["coupling_full"]
        self.spin_sym =spin_sym
        
        self.center = self.findCenter()
        self.check_parameters()
        
        self.Hamiltonian = None
        
        

    def findCenter(self):
        if self.L%2:
            self.center=self.L
        #even
        else:
            self.center=self.L+1
        
        
        
    def check_parameters(self):
        if not(self.L == self.basis.N):
            raise ValueError(f'the basis passed has {self.basis.N} not {self.L}')
        
        if np.isscalar(self.eps):
            if self.L == 1:
                self.eps = [self.eps]
            
            else:
                self.eps = np.ones(self.L*2)*self.eps
                warnings.warn("A scalar was passed for epsilon, same epsilon on every site",category=UserWarning)
        
        if np.isscalar(self.T):
            if self.L == 1:
                self.T = [self.T]
            
            else:
                self.T = np.ones(self.L*2)*self.T
                warnings.warn("A scalar was passed for hopping, same hopping on every site",category=UserWarning)
        
        if np.isscalar(self.Gamma1):
            if self.L == 1:
                self.Gamma1 = [self.Gamma1]
            
            else:
                self.Gamma1 = np.ones(self.L*2)*self.Gamma1
                self.Gamma1[self.center]=0
                self.Gamma1[2*self.L+self.center]=0
                warnings.warn("""A scalar was passed for coupling to the empty bath, 
                              same coupling on every site (except for the central site
                               where gamma1 = 0)"""
                              ,category=UserWarning)
        
        if np.isscalar(self.Gamma2):
            if self.L == 1:
                self.Gamma2 = [self.Gamma2]
            
            else:
                self.Gamma2 = np.ones(self.L*2)*self.Gamma2
                self.Gamma2[self.center]=0
                self.Gamma2[2*self.L+self.center]=0
                warnings.warn("""A scalar was passed for coupling to the full bath, 
                              same coupling on every site (except for the central site
                               where gamma1 = 0)"""
                              ,category=UserWarning)
        
        sym = True
        for parameter,name in zip([self.eps,self.T,self.Gamma1,self.Gamma2],
                                  'epsilon','hopping','coupling_empty','couplfing_full'):
            if not(len(parameter)==self.L or len(parameter)==2*self.L):
                   raise ValueError(f''''the lenght of the parameterlist for
                                    {name} has invalid length it should be a scalar
                                    L or 2L long''')
                                    
            elif len(parameter)==self.L:
                parameter=np.concatenate((parameter,parameter))
            
            elif len(parameter)==2*self.L:
                sym = False
        
        if sym and not(self.spin_sym):
            warnings.warn("""The Lindbladoperator is spin symmetric, consider
                          using a spin symmetric basis"""
                          ,category=UserWarning)
            
        if not(sym) and self.spin_sym:
                raise ValueError("""The Lindblad is not spin symmetric, use
                              complete spin basis instead""")
        
                
    @staticmethod()            
    def drive(t,Om):
        return np.cos(Om*t)
    
    def defineH_super(self):
        L=self.L
        T_c=np.conjugate(self.T)
       #Hamiltonian acting on Fockspace (Fockspace are the odd sites)  
        hop_left_up = [[self.T[int((i-1)/2)],i,(i+2)] for i in range(1,2*L-2,2)] 
        hop_left_down = [[self.T[int((i-1)/2)],i,(i+2)] for i in range(2*L+1,4*L-2,2)]
        #hop_left=np.concatenate((hop_left_up,hop_left_down))
        hop_left=hop_left_up+hop_left_down
    
        # hopping to the right
        hop_right_up = [[T_c[int((i-1)/2)],(i+2),i] for i in range(1,2*L-2,2)]  # hopping to the left
        hop_right_down= [[T_c[int((i-1)/2)],(i+2),i] for i in range(2*L+1,4*L-2,2)]
        #hop_right=np.concatenate((hop_right_up,hop_right_down))
        hop_right=hop_right_up+hop_right_down
        
        int_list = [[self.U,self.center,self.center+2*L]] # onsite interaction
        eps_list=[[self.eps[int((i-1)/2)],i] for i in range(1,4*L,2)]
        
        
        #Hamiltonian acting on augmented Fockspace  
        hop_left_up_a = [[-1*self.T[int((i)/2)],i,(i+2)] for i in range(0,2*L-2,2)] # hopping to the right
        hop_left_down_a = [[-1*self.T[int((i)/2)],i,(i+2)] for i in range(2*L,4*L-2,2)]
        hop_left_a=hop_left_up_a+hop_left_down_a
    
        hop_right_up_a = [[-1*T_c[int((i)/2)],(i+2),i] for i in range(0,2*L-2,2)] # hopping to the left
        hop_right_down_a = [[-1*T_c[int((i)/2)],(i+2),i] for i in range(2*L,4*L-2,2)]
        hop_right_a=hop_right_up_a+hop_right_down_a
    
        
        
        int_list_a = [[-1*self.U,self.center-1,self.center-1+2*L]] # onsite interaction
        eps_list_a=[[-1*self.eps[int((i)/2)],i] for i in range(0,4*L,2)]
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
        if self.V:
            drive_args=[self.Om]
            V_list = [[self.V,self.center],[self.V,self.center+2*L]]
            V_list_a = [[-1*self.V,self.center-1],[-1*self.V,self.center-1+2*L]]
            dynamic= [
                ["n", V_list,self.drive,drive_args], # onsite energy
                ["n", V_list_a,self.drive,drive_args], # onsite energy
                ]
                
            
        else:
            dynamic=[]
        ###### construct Hamiltonian
        return hamiltonian(static,dynamic,dtype=np.complex128,basis=self.basis,check_pcon=False,check_symm=False,check_herm=False)
    
    #removes particels from the system
    def define_Dissipators1(self):
        L=self.L
        #mixed Fock and augmented
        part1 = [[-2j*self.Gamma1[int(i/2)],(i+1),i] for i in range(0,4*L,2)]
        #Fock
        part2 = [[-1*self.Gamma1[int((i-1)/2)],i,i] for i in range(1,4*L,2)]
        #augmented
        part3 = [[-1*self.Gamma1[int((i)/2)],i,i] for i in range(0,4*L,2)]

        static=[
            ["--",part1],
            ["+-",part2],
            ["+-",part3],  
            ]
        dynamic=[]
        return hamiltonian(static,dynamic,dtype=np.complex128,basis=self.basis,check_herm=False,check_pcon=False,check_symm=False)
    
    #add particels to the system
    def define_Dissipators2(self):
        L = self.L
        #mixed
        part1 = [[-2j*self.Gamma2[int(i/2)],(i+1),(i)] for i in range(0,4*L,2)]
        #fock
        part2 = [[-1*self.Gamma2[int((i-1)/2)],i,i] for i in range(1,4*L,2)]
        #augmented
        part3 = [[-1*self.Gamma2[int((i)/2)],i,i] for i in range(0,4*L,2)]
        
        static=[
            ["++",part1],
            ["-+",part2],
            ["-+",part3],
            ]
        
        dynamic=[]
    
        return hamiltonian(static,dynamic,dtype=np.complex128,basis=self.basis,check_herm=False,check_pcon=False,check_symm=False)
    
    def get_staticLindblad(self):

        H_super=self.defineH_super(self)
        D1=self.define_Dissipators1(self)
        D2=self.define_Dissipators2(self)
    
        #return -1j*H_super+D1+D2
        return H_super+1j*(D1+D2)  # multiply by -i due to .evolve structure
    
    def get_dynamicLindblad(self):
        H_super=self.defineH_super(self)
        D1=self.define_Dissipators1(self)
        D2=self.define_Dissipators2(self)
    
        #return -1j*H_super+D1+D2
        return H_super+1j*(D1+D2)
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
        