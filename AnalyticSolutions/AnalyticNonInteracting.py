#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:23:19 2025

@author: theresa
"""
import numpy as np
import matplotlib.pyplot as plt

coupling1=[0.6,0,0.4]
coupling2=[0.4,0,0.6]

eps=[-1,0,1]

hopping=[0.2,0.2]
hopping_right=np.conj(hopping)

gamma1=np.diag(coupling1,0)
gamma2=np.diag(coupling2,0)
E=np.diag(eps,0)+np.diag(hopping,1)+np.diag(hopping_right,-1)
Gamma_plus=gamma1+gamma2
Gamma_minus=gamma2-gamma1


def calcGreen(component):
    i0=component[0]+int(np.floor(len(eps)/2))
    i1=component[1]+int(np.floor(len(eps)/2))
    omegas=np.linspace(-1,1,1000)
    Gr=np.zeros(len(omegas))*1j
    Gk=np.zeros(len(omegas))*1j
    for i in range(len(omegas)):
        omega=np.diag(omegas[i]*np.ones(len(eps)))
        Gr_matrix=np.linalg.inv(omega-E+1j*Gamma_plus)
        Gr[i]=Gr_matrix[i0,i1]
        Gk_matrix=2j*Gr_matrix@Gamma_minus@(np.conj(Gr_matrix))
        Gk[i]=Gk_matrix[i0,i1]
    return omegas,Gr,Gk
    
omegas,Gr10,Gk10=calcGreen([0,-1])
omegas,Gr01,Gk01=calcGreen([-1,0])
omegas,Gr00,Gk00=calcGreen([0,0])

plt.figure()
plt.title('keldysh')
plt.plot(omegas,np.imag(Gk00),label='imag')
plt.plot(omegas,np.real(Gk00),label='real')
plt.legend()

plt.figure()
plt.title('retarted')
plt.plot(omegas,np.imag(Gr10),label='imag')
plt.plot(omegas,np.real(Gr10),label='real')
plt.plot(omegas,np.imag(Gr01),label='imag',linestyle='--')
plt.plot(omegas,np.real(Gr01),label='real',linestyle='--')
plt.legend()

plt.figure()
plt.title('keldysh')
plt.plot(omegas,np.imag(Gk10),label='imag')
plt.plot(omegas,np.real(Gk10),label='real')
plt.plot(omegas,np.imag(Gk01),label='imag')
plt.plot(omegas,np.real(Gk01),label='real')
plt.legend()

plt.figure()
plt.title('retarted')
plt.plot(omegas,np.imag(Gr10)-np.imag(Gr01),label='imag')
plt.plot(omegas,np.real(Gr10)-np.real(Gr01),label='real')
plt.legend()

plt.figure()
plt.title('keldysh')
plt.plot(omegas,np.imag(Gk10)-np.imag(Gk01),label='imag')
plt.plot(omegas,np.real(Gk10)-np.real(Gk01),label='real')
plt.legend()