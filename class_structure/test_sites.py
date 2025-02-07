#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 08:29:12 2025

@author: theresa
"""
from GreensFunction_sites import calculateGreensFunction
from augmented_basis import augmented_basis
import numpy as np
import matplotlib.pyplot as plt
import time


parameters3 = {"length": 5,
              "epsilon": [-3,-2,-1,2,3],
              "hopping": 0.3,
              "interaction":2,
              "drive": 1,
              "frequency":1,
              "coupling_empty":[0.3,0.4,0,0.6,0.7],
              "coupling_full":[0.7,0.6,0,0.4,0.3],
              "spin_symmetric":False,
}

parameters1 = {"length": 1,
              "epsilon": 0,
              "hopping": 0,
              "interaction":0,
              "drive": 1,
              "frequency":1,
              "coupling_empty":0.06,
              "coupling_full":0.04,
              "spin_symmetric":False,
}

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


GF0=calculateGreensFunction(parameters3,[[2,2]],'up')

#print((GF0_sym.plus_lV.T.conjugate()@a))
n=GF0.plot_n(0,500)
GF0._GreaterLesserSites([[0,0]],dt=0.05,eps=1e-8,max_iter=1000,
                    av_periods=5,tf=2e2,t_step=1e2,av_Tau=5,writeFile=True,
                    dirName='results')
#GF0._GreaterLesserPlotFT([[0,0],[-1,-1],[1,1],[0,1],[1,0],[0,-1],[-1,0]],dt=0.05,eps=1e-12,max_iter=1000,
                    #av_periods=5,tf=2e2,t_step=1e2,av_Tau=10)
#Tau,Tau_total,Gr_Tau,Ga_Tau,Gk_Tau,lesser,greater=GF0._GreaterLesser([0,0],tf=100,eps=1e-8)
N_Gk=len(Gk_Tau)
Gk_tau0=Gk_Tau[int(np.floor(N_Gk/2))]
print('calculated n: ', n)
n_keldysh=-1j/2*Gk_tau0+1/2

print('calculated n: ', n)
print('from keldysh n: ',n_keldysh)
start=1000
end=-1000
omegas,omegas_k,Gr,Gk=plotSpectrum(Tau,Gr_Tau,Ga_Tau,Gk_Tau,start,end)
plt.figure()
plt.plot(omegas,2*Gr.imag)
plt.plot(omegas_k,(greater-lesser).imag)

