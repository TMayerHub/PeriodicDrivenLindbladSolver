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
parameters = {"length": 1,
              "epsilon": 0,
              "hopping": 0,
              "interaction":0,
              "drive": 0,
              "frequency":0,
              "coupling_empty":0,
              "coupling_full":0,
              "spin_symmetric":False,
}

parameters3 = {"length": 3,
              "epsilon": [1,-5,-1],
              "hopping": 0.5,
              "interaction":10,
              "drive": 0,
              "frequency":0,
              "coupling_empty":[0.5,0,0.5],
              "coupling_full":[0.5,0,0.5],
              "spin_symmetric":False,
}

parameters1 = {"length": 1,
              "epsilon": 0,
              "hopping": 0,
              "interaction":0,
              "drive": 0,
              "frequency":0,
              "coupling_empty":1,
              "coupling_full":0,
              "spin_symmetric":False,
}

parameters_sym = {"length": 1,
              "epsilon": 0,
              "hopping": 0,
              "interaction":0,
              "drive": 0,
              "frequency":1,
              "coupling_empty":1,
              "coupling_full":0.0,
              "spin_symmetric":True,
}




L=parameters['length']

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
        
def plotSpectrum(Tau,Gr_Tau,Ga_Tau,Gk_Tau,start,end):
    N_Tau=len(Tau)
    Period=abs(Tau[-1]-Tau[0])
    norm=Period/N_Tau#/np.sqrt(2*np.pi)

    #FT of retarted Green's function
    omegas=np.fft.fftfreq(N_Tau,Tau[1]-Tau[0])*2*np.pi
    omegas=np.fft.fftshift(omegas)

    Gr=np.fft.ifft(Gr_Tau,norm='forward')*norm
    Gr=np.fft.fftshift(Gr)
    
    omegas_k=np.fft.fftfreq(N_Tau*2-1,Tau[1]-Tau[0])*2*np.pi
    omegas_k=np.fft.fftshift(omegas_k)
    Gk=np.fft.ifftshift(Gk_Tau)
    Gk=np.fft.ifft(Gk,norm='forward')*norm
    Gk=np.fft.fftshift(Gk)
    
    Ga=np.fft.ifftshift(Ga_Tau)
    Ga=np.fft.ifft(Ga,norm='forward')*norm
    Ga=np.fft.fftshift(Ga)
    
    #plt.title('spectral function')
    #txt='V='+str(V)+'  Om='+str(Om)+'  U='+str(U)+'  T='+str(T[0])+'  G='+str(Gamma1[0])
    #txt=''
    #plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    #print('len omegas:',len(omegas))
    #start=8000
    #end=12000
    #plt.plot(Tau[N_start:],np.real(G_r[N_start:]),label='real')
    #plt.plot(omegas,np.imag(G_r[N_start:]),label='imaginary')
    spectrum=(-1)/np.pi*np.imag(Gr)
    print(spectrum[-1])
    plt.figure()
    plt.title('spectral function')
    plt.plot(omegas[start//2:end//2],spectrum[start//2:end//2],label='site 0')
    plt.xlabel('w')
    
    plt.figure()
    plt.title('retarted')
    plt.plot(omegas[start//2:end//2],Gr.imag[start//2:end//2],label='imag')
    plt.plot(omegas[start//2:end//2],Gr.real[start//2:end//2],label='real')
    plt.xlabel('w')
    plt.legend()
    
    plt.figure()
    plt.title('advanced')
    plt.plot(omegas_k[start:end],Ga.imag[start:end],label='imag')
    plt.plot(omegas_k[start:end],Ga.real[start:end],label='real')
    plt.xlabel('w')
    plt.legend()
    
    plt.figure()
    plt.title('Keldysh')
    plt.plot(omegas_k[start:end],Gk.imag[start:end],label='imag')
    plt.plot(omegas_k[start:end],Gk.real[start:end],label='real')
    plt.xlabel('w')
    plt.legend()
    
    return omegas,omegas_k,Gr,Gk

#user_basis=augmented_basis(L,'restricted',[0,0])

#GF=calculateGreensFunction(parameters,0,'updown')
#Tau,GR_Tau=GF.G_r(3e2,3e2,dt=0.05)
start=time.time()
#Tau,GR_Tau=GF.Gr_floquet1(3e2,3e2,dt=0.05)
end=time.time()
time_nosym=end-start


#GF0_sym=calculateGreensFunction(parameters_sym,0,'updown')
#Tau,GR0_Tau=GF0_sym.Gr_floquet1(1e3,1e3,dt=0.05)

#plotSpectrum(Tau,GR0_Tau,8000,12000)


#print(GF0_sym.plus_lV)

#%%
GF0_sym=calculateGreensFunction(parameters_sym,0,'updown')
GF0_sym._GreaterLesser(tf=1e2)



tau_end=1000

Tau_grid, T= np.meshgrid(GF0_sym.Tau[:tau_end], GF0_sym.t)


# Plotting the contour plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(T, Tau_grid, GF0_sym.greater.imag[:,:tau_end], levels=50, cmap='viridis')
plt.colorbar(contour, label='G')

plt.figure(figsize=(8, 6))
contour = plt.contourf(T, Tau_grid, GF0_sym.greater.real[:,:tau_end], levels=50, cmap='viridis')
plt.colorbar(contour, label='G')

# Labels and title
plt.xlabel('t')
plt.ylabel('Tau')
plt.title('Contour Plot of G')



#%%
#Tau,GR_Tau=GF.G_r(3e2,3e2,dt=0.05)
start=time.time()
n0=GF0_sym.plot_n(1e3,dt=0.05)

GF1_sym=calculateGreensFunction(parameters_sym,1,'updown')
n1=GF1_sym.plot_n(1e3,dt=0.05)
GFm1_sym=calculateGreensFunction(parameters_sym,-1,'updown')
nm1=GFm1_sym.plot_n(1e3,dt=0.05)
#Tau,GR_Tau=GF.G_r(3e2,3e2,dt=0.05)
start=time.time()
#GF2_sym.plot_n(2e2,dt=0.05)


#%%
Tau,GR0_Tau=GF0_sym._GreaterLesser()
Tau,GR1_Tau=GF1_sym.Gr_floquet1(1e3,1e3,dt=0.05)
Tau,GRm1_Tau=GFm1_sym.Gr_floquet1(1e3,1e3,dt=0.05)
end=time.time()

time_sym=end-start
print('sym: ',time_sym)
print('no sym: ', time_nosym)

print('speed up: ',time_nosym/time_sym)

plt.figure()
plt.title('GR')
plt.plot(Tau,np.real(GR0_Tau))
plt.plot(Tau,np.imag(GR0_Tau))


N_Tau=len(Tau)
Period=abs(Tau[-1]-Tau[0])
norm=Period/N_Tau#/np.sqrt(2*np.pi)

#FT of retarted Green's function
omegas=np.fft.fftfreq(N_Tau,Tau[1]-Tau[0])*2*np.pi
omegas=np.fft.fftshift(omegas)

Gr0=np.fft.ifft(GR0_Tau,norm='forward')*norm
Gr0=np.fft.fftshift(Gr0)
Gr1=np.fft.ifft(GR1_Tau,norm='forward')*norm
Gr1=np.fft.fftshift(Gr1)
Grm1=np.fft.ifft(GRm1_Tau,norm='forward')*norm
Grm1=np.fft.fftshift(Grm1)

spectrum0=(-1)/np.pi*np.imag(Gr0)
spectrum1=(-1)/np.pi*np.imag(Gr1)
spectrumm1=(-1)/np.pi*np.imag(Grm1)
#%%
GF0_sym=calculateGreensFunction(parameters1,0,'up')

#print((GF0_sym.plus_lV.T.conjugate()@a))
n=GF0_sym.plot_n(500)

Tau,Tau_total,Gr_Tau,Ga_Tau,Gk_Tau,lesser,greater=GF0_sym._GreaterLesser(tf=100,eps=1e-8)
N_Gk=len(Gk_Tau)
Gk_tau0=Gk_Tau[int(np.floor(N_Gk/2))]
print('calculated n: ', n)
n_keldysh=-1j/2*Gk_tau0+1/2
#%%
print('calculated n: ', n)
print('from keldysh n: ',n_keldysh)
start=1000
end=-1000
omegas,omegas_k,Gr,Gk=plotSpectrum(Tau,Gr_Tau,Ga_Tau,Gk_Tau,start,end)
plt.figure()
plt.plot(omegas,2*Gr.imag)
plt.plot(omegas_k,(greater-lesser).imag)

#plt.figure()
#plt.plot(Tau,Gr_Tau.real)
#plt.plot(Tau,Gr_Tau.imag)
#%%
N_half=int(np.floor(len(Tau_total)/2))
#Tau_total[:N_half]=Tau_total[:N_half]*(-1)
plt.figure()
plt.plot(Tau_total,Gk_Tau.real)
plt.figure()
plt.plot(Tau_total,Gk_Tau.imag)

plt.figure()
N_Tau=len(Tau_total)
Period=abs(Tau_total[-1]-Tau_total[0])
norm=Period/N_Tau#/np.sqrt(2*np.pi)

    #FT of retarted Green's function
omegas=np.fft.fftfreq(N_Tau,Tau_total[1]-Tau_total[0])*2*np.pi
omegas=np.fft.fftshift(omegas)

#Gk_Tau=np.fft.fftshift(Gk_Tau)
Gk=np.fft.ifft(Gk_Tau,norm='forward')*norm
Gk=np.fft.fftshift(Gk)
    
plt.title('keldysh')
    #txt='V='+str(V)+'  Om='+str(Om)+'  U='+str(U)+'  T='+str(T[0])+'  G='+str(Gamma1[0])
txt=''
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
print('len omegas:',len(omegas))
start=0
end=-1
plt.plot(omegas[start:end],Gk.imag[start:end],label='site 0')
plt.plot(omegas[start:end],Gk.real[start:end],label='site 0')
#%%
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

cumulative_integral = cumulative_trapezoid(spectrum0+spectrum1+spectrumm1, omegas, initial=0)
print(Gr0+Gr1+Grm1)
print(cumulative_integral)
plt.figure()
plt.plot(omegas,cumulative_integral)
# Interpolate to find the x-value where the integral equals n
interp = interp1d(cumulative_integral, omegas, bounds_error=False, fill_value="extrapolate")
w_n = interp(n0+n1+nm1)
print(n0+n1+nm1)
N_start=0
plt.figure()
plt.title('spectral function')
#txt='V='+str(V)+'  Om='+str(Om)+'  U='+str(U)+'  T='+str(T[0])+'  G='+str(Gamma1[0])
txt=''
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
print('len omegas:',len(omegas))
start=8000
end=12000
#plt.plot(Tau[N_start:],np.real(G_r[N_start:]),label='real')
#plt.plot(omegas,np.imag(G_r[N_start:]),label='imaginary')
print(spectrum0[-1])
plt.plot(omegas[start:end],spectrum0[start:end],label='site 0')
plt.plot(omegas[start:end],spectrum1[start:end],label='site 1')
plt.plot(omegas[start:end],spectrumm1[start:end],label='site -1')
plt.axvline(x=w_n, color='r', linestyle='--', linewidth=0.6,label='n',)
plt.plot(omegas[start:end],0*omegas[start:end])
#plt.axvline(1,0,1)
plt.legend()
print('norm',np.trapz(spectrum0,x=omegas))






















