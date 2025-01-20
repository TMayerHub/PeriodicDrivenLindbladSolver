#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 10:17:51 2024

@author: theresa
"""
def plotSpectrum(Tau,Gr_Tau,start,end):
    N_Tau=len(Tau)
    Period=abs(Tau[-1]-Tau[0])
    norm=Period/N_Tau#/np.sqrt(2*np.pi)

    #FT of retarted Green's function
    omegas=np.fft.fftfreq(N_Tau,Tau[1]-Tau[0])*2*np.pi
    omegas=np.fft.fftshift(omegas)

    Gr=np.fft.ifft(Gr_Tau,norm='forward')*norm
    Gr=np.fft.fftshift(Gr)
    
    plt.title('spectral function')
    #txt='V='+str(V)+'  Om='+str(Om)+'  U='+str(U)+'  T='+str(T[0])+'  G='+str(Gamma1[0])
    txt=''
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    print('len omegas:',len(omegas))
    #start=8000
    #end=12000
    #plt.plot(Tau[N_start:],np.real(G_r[N_start:]),label='real')
    #plt.plot(omegas,np.imag(G_r[N_start:]),label='imaginary')
    spectrum=(-1)/np.pi*np.imag(Gr)
    print(spectrum[-1])
    plt.plot(omegas[start:end],spectrum[start:end],label='site 0')
    
if __name__=='__main__':
    from GreensFunction import calculateGreensFunction
    from augmented_basis import augmented_basis
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    #
    parameters = {"sites": 1,
                  "epsilon": [0],
                  "hopping": 0.5,
                  "interaction":0,
                  "drive": 0,
                  "frequency":0,
                  "coupling_empty":0.2,
                  "coupling_full":0.2,
                  "spin_symmetric":False,
    }

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




    L=parameters['sites']

    GF0_sym=calculateGreensFunction(parameters_sym,0,'updown')
    Tau,Gr_Tau=GF0_sym._GreaterLesser(tf=3e1,av_periods=3,t_step=3e1)
    #%%
    start=0
    end=-1
    plotSpectrum(Tau,Gr_Tau,start,end)
    plt.figure()
    plt.plot(Tau,Gr_Tau.real)
    plt.plot(Tau,Gr_Tau.imag)