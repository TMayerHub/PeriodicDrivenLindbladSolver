#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:48:02 2024

@author: theresa
"""

import warnings
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = ("True")  # uncomment this line if omp error occurs on OSX for python 3
os.environ["OMP_NUM_THREADS"] = str(4)# set number of OpenMP threads to run in parallel
os.environ["MKL_NUM_THREADS"] = str(4)# set number of MKL threads to run in parallel
#os.environ['OMP_NUM_THREADS']='4' # set number of OpenMP threads to run in parallel
import numpy as np
import scipy

from quspin.operators import hamiltonian

def custom_formatwarning(message, category, filename, lineno, line=None):
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"

# Apply the custom formatter
warnings.formatwarning = custom_formatwarning

class createLindblad:
    def __init__(self,basis,parameters,spin_sym=False):
        self.parameters = parameters
        self.basis = basis
        
        self.L = self.require_parameter('length')
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
        
        self.diff_sector = basis.diff_sector
        self.operator = self.get_dynamicLindblad()
    
    def require_parameter(self, key):
        """
        Retrieve a required parameter. Raise KeyError if missing.
        """
        if key not in self.parameters:
            raise KeyError(f"Missing required parameter: '{key}'")
        return self.parameters[key]

    def optional_parameter(self, key):
        """
        Checks optional parameters and gives a warning if it is not set.
        """
        if key not in self.parameters:
            warnings.warn(f'''Parameter {key} not set. Assumed to be 0 for all
                          sites. ''')
        return self.parameters[key]
        

    def findCenter(self):
        if self.L%2:
            return self.L
        #even
        else:
            return self.L+1
        
        
        
    def check_parameters(self):
        if not(self.L == self.basis.N/4):
            raise ValueError(f'the basis passed has {self.basis.N/4} not {self.L} sites')
        
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
                self.Gamma1 = np.array([[self.Gamma1]])
            
            else:
                self.Gamma1 = np.ones(self.L)*self.Gamma1
                ind0=int(np.floor(self.L/2))
                self.Gamma1[ind0]=0
                self.Gamma1=np.concatenate((self.Gamma1,self.Gamma1))
                warnings.warn("""A scalar was passed for coupling to the empty bath, 
                              same coupling on every site (except for the central site
                               where gamma1 = 0)"""
                              ,category=UserWarning)
        
        if np.isscalar(self.Gamma2):
            if self.L == 1:
                self.Gamma2 = np.array([[self.Gamma2]])
            
            else:
                self.Gamma2 = np.ones(self.L)*self.Gamma2
                ind0=int(np.floor(self.L/2))
                self.Gamma2[ind0]=0
                self.Gamma2=np.concatenate((self.Gamma2,self.Gamma2))
                warnings.warn("""A scalar was passed for coupling to the full bath, 
                              same coupling on every site (except for the central site
                               where gamma1 = 0)"""
                              ,category=UserWarning)
        #print('gamma shape',len(np.shape(self.Gamma1)))
        if len(np.shape(self.Gamma1))==1:
            self.Gamma1=np.diag(self.Gamma1)
        #print(self.Gamma1)
        if len(np.shape(self.Gamma2))==1:
            self.Gamma2=np.diag(self.Gamma2)

        sym = True
        for parameter_name,name in zip(['eps', 'T', 'Gamma1', 'Gamma2'],
                                       ['epsilon', 'hopping', 'coupling_empty', 'coupling_full']):
            parameter = getattr(self, parameter_name)
            #print(name)
            if not(len(parameter)==self.L or len(parameter)==2*self.L):
                   raise ValueError(f''''the lenght of the parameterlist for
                                    {name} has invalid length it should be a scalar
                                    L or 2L long''')
                                    
            if len(parameter)==self.L:
                if parameter_name=='Gamma1' or parameter_name=='Gamma2':
                    N=parameter.shape[0]
                    parameter_new=np.zeros((2*N, 2*N), dtype=parameter.dtype)
                    parameter_new[:N, :N] = parameter
                    parameter_new[N:2*N, N:2*N] = parameter
                else:
                    parameter_new=np.concatenate((parameter,parameter))
                
                setattr(self, parameter_name, parameter_new)
            
            if len(parameter)==2*self.L and not((parameter[:self.L]==parameter[self.L:]).all()):
                sym = False
        
        if sym and not(self.spin_sym):
            warnings.warn("""The Lindbladoperator is spin symmetric, consider
                          using a spin symmetric basis"""
                          ,category=UserWarning)
            
        if not(sym) and self.spin_sym:
                raise ValueError("""The Lindbladoperator is not spin symmetric, use
                              complete spin basis instead""")
        
    
    def displayParameters(self):
        print('parameters')
        print('sites: ',self.L)
        print()
        print('0..L-1 act on spin up, L...2L-1 act on spin down')
        print('epsilon:        ',self.eps)
        print('hopping:        ',self.T)
        print('coupling_empty: ',self.Gamma1)
        print('coupling_full:  ',self.Gamma2)
        print()
        print('apply to central site only:')
        print('interaction: ',self.U)
        print('drive:       ',self.V)
        print('frequency:   ',self.Om)

    @staticmethod      
    def drive_fun(t,Om):
        return np.cos(Om*t)
    
    def defineH_super(self):
        L=self.L
        self.T=self.T[:-1]
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
                ["n", V_list,self.drive_fun,drive_args], # onsite energy
                ["n", V_list_a,self.drive_fun,drive_args], # onsite energy
                ]
                
            
        else:
            dynamic=[]
        ###### construct Hamiltonian
        return hamiltonian(static,dynamic,dtype=np.complex128,basis=self.basis,check_pcon=False,check_symm=False,check_herm=False)
    
    #removes particels from the system
    def define_Dissipators1(self):
        L=self.L
        
        #mixed Fock and augmented
        part1 = []
        #print(self.Gamma1)
        for i in range(0, 4*L, 2):
            for j in range(1, 4*L, 2):
                gamma_value = self.Gamma1[int(i/2), int((j-1)/2)]
                #print(gamma_value, i, j)  # This will print the values
                part1.append([-2j * gamma_value, j, i])  # Adding to part1 list
        #part1 = [[-2j*self.Gamma1[int(i/2),int((j-1)/2)],(j),(i)] for i in range(0,4*L,2) for j in range(1,4*L,2)]
        #Fock
        part2 = [[-1*self.Gamma1[int((i-1)/2),int((j-1)/2)],i,j] for i in range(1,4*L,2) for j in range(1,4*L,2)]
        #augmented
        part3 = [[-1*self.Gamma1[int((i)/2),int((j)/2)],j,i] for i in range(0,4*L,2) for j in range(0,4*L,2)]

        static=[
            ["--",part1],
            ["+-",part2],
            ["+-",part3],  
            ]
        
        dynamic=[]
        h1=hamiltonian([["--",part1]],dynamic,dtype=np.complex128,basis=self.basis,check_herm=False,check_pcon=False,check_symm=False)
        #print(h1.basis)
        #print('h1')
        #print(h1)
        h2=hamiltonian([["+-",part2]],dynamic,dtype=np.complex128,basis=self.basis,check_herm=False,check_pcon=False,check_symm=False)
        #print('h2')
        #print(h2)
        h3=hamiltonian([["+-",part3]],dynamic,dtype=np.complex128,basis=self.basis,check_herm=False,check_pcon=False,check_symm=False)
        #print('h3')
        #print(h3)
        return hamiltonian(static,dynamic,dtype=np.complex128,basis=self.basis,check_herm=False,check_pcon=False,check_symm=False)
    
    #add particels to the system
    def define_Dissipators2(self):
        L = self.L
        #mixed
        part1 = [[-2j*self.Gamma2[int(i/2),int((j-1)/2)],(j),(i)] for i in range(0,4*L,2) for j in range(1,4*L,2)]
        #fock
        part2 = [[-1*self.Gamma2[int((i-1)/2),int((j-1)/2)],i,j] for i in range(1,4*L,2) for j in range(1,4*L,2)]
        #augmented
        part3 = [[-1*self.Gamma2[int((i)/2),int((j)/2)],j,i] for i in range(0,4*L,2) for j in range(0,4*L,2)]
        
        static=[
            ["++",part1],
            ["-+",part2],
            ["-+",part3],
            ]
        
        dynamic=[]
    
        return hamiltonian(static,dynamic,dtype=np.complex128,basis=self.basis,check_herm=False,check_pcon=False,check_symm=False)
    
    def define_Dissipators(self):
        L = self.L
        #mixed
        part1_empty = [[-2j*self.Gamma1[int(i/2),int((j-1)/2)],(j),(i)] for i in range(0,4*L,2) for j in range(1,4*L,2)]
        #part1_empty = [[-2j*self.Gamma1[int((j-1)/2),int(i/2)],(j),(i)] for i in range(0,4*L,2) for j in range(1,4*L,2)]
        #Fock
        part2_empty = [[-1*self.Gamma1[int((i-1)/2),int((j-1)/2)],i,j] for i in range(1,4*L,2) for j in range(1,4*L,2)]
        #part2_empty = [[-1*self.Gamma1[int((j-1)/2),int((i-1)/2)],i,j] for i in range(1,4*L,2) for j in range(1,4*L,2)]
        #augmented
        part3_empty = [[-1*self.Gamma1[int((i)/2),int((j)/2)],j,i] for i in range(0,4*L,2) for j in range(0,4*L,2)]
        #part3_empty = [[-1*self.Gamma1[int((j)/2),int((i)/2)],j,i] for i in range(0,4*L,2) for j in range(0,4*L,2)]

        #mixed
        #part1_full = [[-2j*self.Gamma2[int(i/2),int((j-1)/2)],(j),(i)] for i in range(0,4*L,2) for j in range(1,4*L,2)]
        part1_full = [[-2j*self.Gamma2[int((j-1)/2),int(i/2)],(j),(i)] for i in range(0,4*L,2) for j in range(1,4*L,2)]
        #fock
        #part2_full = [[-1*self.Gamma2[int((i-1)/2),int((j-1)/2)],i,j] for i in range(1,4*L,2) for j in range(1,4*L,2)]
        part2_full = [[-1*self.Gamma2[int((j-1)/2),int((i-1)/2)],i,j] for i in range(1,4*L,2) for j in range(1,4*L,2)]
        #augmented
        #part3_full = [[-1*self.Gamma2[int((i)/2),int((j)/2)],j,i] for i in range(0,4*L,2) for j in range(0,4*L,2)]
        part3_full = [[-1*self.Gamma2[int((j)/2),int((i)/2)],j,i] for i in range(0,4*L,2) for j in range(0,4*L,2)]
        
        static=[
            ["--",part1_empty],
            ["+-",part2_empty],
            ["+-",part3_empty],  
            ["++",part1_full],
            ["-+",part2_full],
            ["-+",part3_full],
            ]

        dynamic=[]
        return hamiltonian(static,dynamic,dtype=np.complex128,basis=self.basis,check_herm=False,check_pcon=False,check_symm=False)
    
    def get_staticLindblad(self):

        H_super=self.defineH_super(self)
        #D1=self.define_Dissipators1(self)
        #D2=self.define_Dissipators2(self)
        D=self.define_Dissipators(self)
        #return -1j*H_super+D1+D2
        #return H_super+1j*(D1+D2)  # multiply by -i due to .evolve structure
        return H_super+1j*(D)
    
    def get_dynamicLindblad(self):
        H_super=self.defineH_super()
        #D1=self.define_Dissipators1()
        #D2=self.define_Dissipators2()
        D=self.define_Dissipators()
    
        #return -1j*H_super+D1+D2
        #return H_super+1j*(D1+D2)
        return H_super+1j*(D)
    
    def exact_Diagonalization(self):
        L_static_csr=self.operator.as_sparse_format()
        w,vl,vr=scipy.linalg.eig(L_static_csr.toarray(),left=True,right=True)
        w_min=np.argmin(abs(w))

        rho_inf=np.array([vr[:,w_min]]).T

        #column vector
        vl0=scipy.sparse.csc_array(np.transpose([vl[:,w_min]]))
        
        norm=vl0@rho_inf
        rho_inf=rho_inf/norm
        
        return vl0,rho_inf

    def lowestEV(self):
        w,rho_inf=self.operator.eigsh(k=1,sigma=0)
        print('w rho inf: ', w)
        #print(leftVacuum.shape,rho_inf.shape)
        #print(leftVacuum)
        #print(rho_inf)
        
        #norm=leftVacuum.H@rho_inf
        
        return rho_inf#/norm
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
        