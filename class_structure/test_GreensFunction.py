#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 08:01:21 2024

@author: theresa
"""

from GreensFunction import calculateGreensFunction
from augmented_basis import augmented_basis
import numpy as np
import matplotlib.pyplot as plt
import time

#
parameters = {"sites": 3,
              "epsilon": [0,-1,0],
              "hopping": 0.5,
              "drive": 0,
              "interaction":2,
              "frequency":0,
              "coupling_empty":0.5,
              "coupling_full":0.5,
              "spin_symmetric":False,
}

parameters_sym = {"sites": 3,
              "epsilon": [0,-1,0],
              "hopping": 0.5,
              "drive": 0,
              "interaction":2,
              "frequency":0,
              "coupling_empty":0.5,
              "coupling_full":0.5,
              "spin_symmetric":True,
}
L=parameters['sites']

def checkOperators(site,spin):
    user_basis=augmented_basis(L,'restricted',[0,0])

    GF=calculateGreensFunction(parameters,site,spin)
    for i in range(user_basis.Ns):
        state_vector=np.zeros((user_basis.Ns,1))
        state=user_basis[i]
        state_vector[i]=1
        print(user_basis.int_to_state(state))
        basis,n_vector=GF.action_n(user_basis,state_vector)
        indices = np.nonzero(n_vector)
        print('n')
        if len(indices[0])==0:
            print(0)
        else:
            rows=indices[0]
            for row in rows:
                print(n_vector[row,0],basis.int_to_state(basis[row]))
        
        state_vector=np.zeros((user_basis.Ns,1))
        state_vector[i]=1
        basis,a_vector=GF.action_a(user_basis,state_vector)
        #print(basis)
        indices = np.nonzero(a_vector)
        print('a')
        if len(indices[0])==0:
            print(0)
        else:
            rows=indices[0]
            for row in rows:
                print(a_vector[row,0],basis.int_to_state(basis[row]))
        
        state_vector=np.zeros((user_basis.Ns,1))
        state_vector[i]=1
        basis,adag_vector=GF.action_adag(user_basis,state_vector)
        indices = np.nonzero(adag_vector)
        print('adag')
        if len(indices[0])==0:
            print(0)
        else:
            rows=indices[0]
            for row in rows:
                print(adag_vector[row,0],basis.int_to_state(basis[row]))
        print()
    
user_basis=augmented_basis(L,'restricted',[0,0])

GF=calculateGreensFunction(parameters,0,'updown')
#Tau,GR_Tau=GF.G_r(3e2,3e2,dt=0.05)
start=time.time()
Tau,GR_Tau=GF.Gr_floquet1(3e2,3e2,dt=0.05)
end=time.time()
time_nosym=end-start


GF_sym=calculateGreensFunction(parameters_sym,0,'updown')
Tau,GR_Tau=GF.G_r(3e2,3e2,dt=0.05)
start=time.time()
Tau,GR_Tau=GF_sym.Gr_floquet1(3e2,3e2,dt=0.05)
end=time.time()

time_sym=end-start
print('sym: ',time_sym)
print('no sym: ', time_nosym)

print('speed up: ',time_nosym/time_sym)

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
#txt='V='+str(V)+'  Om='+str(Om)+'  U='+str(U)+'  T='+str(T[0])+'  G='+str(Gamma1[0])
txt=''
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
print('len omegas:',len(omegas))
start=1500
end=6500
#plt.plot(Tau[N_start:],np.real(G_r[N_start:]),label='real')
#plt.plot(omegas,np.imag(G_r[N_start:]),label='imaginary')
print(spectrum[-1])
plt.plot(omegas[start:end],spectrum[start:end],label='spectralfunction')
plt.plot(omegas[start:end],0*omegas[start:end])
#plt.axvline(1,0,1)
plt.legend()
print('norm',np.trapz(spectrum,x=omegas))























