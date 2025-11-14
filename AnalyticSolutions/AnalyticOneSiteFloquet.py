# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np  # generic math functions
import matplotlib.pyplot as plt
from matplotlib import cm

Om=0.5
V=0
x=np.linspace(-np.pi, np.pi,100)
y=x
eta=1e-3
gamma=0.01
period=2*np.pi/Om

#'analytic calculation of fft no gamma#
###############################################################################
def fx(x,l):
    return np.exp(-1j*(l+1/Om*V*np.sin(x)))/(2*np.pi)

def fy(y,l):
    return np.exp(1j*(l+1/Om*V*np.sin(x)))/(2*np.pi)

l=[-3,-2,-1,0,1,2,3]
l=np.linspace(int(-4/Om),int(4/Om),int(4/Om)*2+1)
omegas=np.linspace(-4, 4,20001)
Gr_l=np.zeros((len(l),len(omegas)))+0j
#plt.figure()
for i in range(len(Gr_l)):
    #print(l[i])
    Gr_l[i]=1/(omegas+l[i]*Om+1j*eta)*np.trapz(fx(x,l[i]),x)*np.trapz(fy(y,l[i]),y)

#sum over l
Gr=np.sum(Gr_l,axis=0)
#calculate spectrum
spectrum=(-1)/np.pi*np.imag(Gr)


#plot results
plt.figure()
plt.title('analytic calculation, no coupling')
plt.plot(omegas,0*omegas)
plt.plot(omegas,spectrum,label='spectrum')

#%%
print('start')
#use definition of heaviside, no gamma#
###############################################################################
omegas=np.linspace(-10,10,1000)
t_abs=np.linspace(1000-period/2,1000+period/2,100)
T=np.linspace(-500,500,10000+1)
w=np.linspace(-1000,1000,100+1)
G_T_t=np.zeros((len(t_abs),len(T)))*0j
heavy_fft=1/(w+1j*eta)
heavy=np.zeros(len(T))*0j
#for i in range(len(T)):
#    heavy[i]=np.trapz(np.exp(-1j*w*T[i])*heavy_fft,w)/(2*np.pi*1j)*(-1)
#heavy=heavy-np.max(heavy)+1  
#plt.figure()
#plt.title('heavy')
#plt.plot(T,np.imag(heavy))
#plt.plot(T,np.real(heavy))
for j in range(len(T)):
    t=t_abs-T[j]/2
    #t=t_abs
    for i in range(len(t)):
        G_T_t[i,j]=(-1j)*np.heaviside(T[j],0.5)*np.exp(-1j*(V/Om)*(np.sin(Om*(t[i]+T[j]))-np.sin(Om*t[i]))-gamma*T[j])
    #if T[j]==0:
        #print(np.heaviside(T[j],0.5))
        #plt.figure()
        #plt.plot(t,np.real(G_T_t[:,j]))
        #plt.plot(t,np.imag(G_T_t[:,j]))
print('finished loop')
GT=np.trapz(G_T_t,t,axis=0)/period   
#GT=(-1j)*np.heaviside(T,0.5)*np.exp(-1j*(V/Om)*(np.sin(Om*(500+T))-np.sin(Om*500))-gamma*T)
#print(GT[-1])
plt.ion()
start=int(len((T)-1)/2)-100
end=start+int(np.ceil(start/2))+100
X, Y = np.meshgrid(T[start:end], np.linspace(-period/2,period/2,len(t_abs))) 

levels =  np.linspace(-1,1,100+1)
fig, ax = plt.subplots(2)
surf1 = ax[0].contourf(X, Y, np.real(G_T_t[:,start:end]), cmap='PiYG',levels=levels) 
fig.colorbar(surf1) 
ax[0].set_title('real')
ax[0].set_xlabel('T')
ax[0].set_ylabel('t')


surf2 = ax[1].contourf(X, Y, np.imag(G_T_t[:,start:end]), cmap='PiYG',levels=levels)   
fig.colorbar(surf2)
ax[1].set_title('imaginary')
ax[1].set_xlabel('T')
ax[1].set_ylabel('t')
fig.tight_layout()
plt.show()
fig.savefig("debug_plot.png")
print('meshgrid')
print(X.shape, Y.shape, np.real(G_T_t[:, start:end]).shape)
plt.figure()
plt.title('GT')
plt.plot(T,np.imag(GT))
plt.plot(T,np.real(GT))

GT_2=np.array([GT]).transpose()
T_2=np.array([T]).transpose()

#print(np.shape(np.array([omegas])),np.shape(T_2))
#print(np.shape(np.exp(1j*T_2@np.array([omegas]))),np.shape(GT_2))

#calculate fft with trapz
Gr_2=np.trapz(np.exp(1j*T_2@np.array([omegas]))*GT_2,T,axis=0)

#calculate spectrum
spectrum=(-1)/np.pi*np.imag(Gr_2)
plt.figure()
plt.title('fft with trapz, with coupling')
plt.plot(omegas,0*omegas)
plt.plot(omegas,spectrum)



#for i in range(len(t)):
    #for j in range(len(T)):
        #G[i,j]=
#%%
#use trapz to calculate fft, with gamma
#############################################################################
omegas=np.linspace(-4, 4,2001)
Tau=np.linspace(-1000,1000,5000+1)

#define Green's function in time according to equation of motion
G_Tau=-1j*np.heaviside(Tau,0.5)*np.exp(-1j*V/Om*np.sin(Om*Tau)-gamma*abs(Tau))

#plot function in time domain
plt.figure()
plt.title('time dependence')
plt.plot(Tau,np.real(G_Tau))
plt.plot(Tau,np.imag(G_Tau))

#prepare for matrix multiplication
G_Tau_2=np.array([G_Tau]).transpose()
Tau_2=np.array([Tau]).transpose()

#print(np.shape(np.array([omegas])),np.shape(Tau_2))
#print(np.shape(np.exp(1j*Tau_2@np.array([omegas]))),np.shape(G_Tau_2))

#calculate fft with trapz
Gr_2=np.trapz(np.exp(1j*Tau_2@np.array([omegas]))*G_Tau_2,Tau,axis=0)

#calculate spectrum
spectrum=(-1)/np.pi*np.imag(Gr_2)
plt.figure()
plt.title('fft with trapz, with coupling')
plt.plot(omegas,0*omegas)
plt.plot(omegas,spectrum)


#use np.fft to calculate fft, with gamma
###############################################################################
G_r=np.fft.ifft(np.fft.ifftshift(G_Tau))
omegas=np.fft.fftfreq(len(Tau),Tau[1]-Tau[0])*2*np.pi
#ishift order
omegas=np.fft.ifftshift(omegas)
G_r=np.fft.ifftshift(G_r)

start=1300
end=3800
spectrum=(-1)/np.pi*np.imag(G_r)
plt.figure()
plt.title('fft with np.fft, with coupling')
plt.plot(omegas[start:end],0*omegas[start:end])
plt.plot(omegas[start:end],spectrum[start:end])


#include the t dependence
#############################################################################
#%%
period=2*np.pi/Om
t=np.array([np.linspace(0,period,100)]).transpose()
omegas=np.linspace(-4, 4,2001)
tau=np.array([np.linspace(-1000,1000,500+1)])
times=t@tau
#print(times.shape)
Tau=np.ones(np.shape(t))@tau
#define Green's function in time according to equation of motion
Gt_Tau=-1j*np.heaviside(Tau,0.5)*np.exp(-1j*V/Om*np.sin(Om*times)-gamma*abs(Tau))
#print(np.shape(Gt_Tau))
GTau=np.trapz(Gt_Tau,t[:,0],axis=0)/period
#print(GTau.shape)
#GTau=GTau[0]

#plot function in time domain
plt.figure()
plt.title('time dependence')
plt.plot(tau[0],np.real(GTau))
plt.plot(tau[0],np.imag(GTau))

#prepare for matrix multiplication
G_Tau_2=np.array([GTau]).transpose()
Tau_2=tau.transpose()

#print(np.shape(np.array([omegas])),np.shape(Tau_2))
#print(np.shape(np.exp(1j*Tau_2@np.array([omegas]))),np.shape(G_Tau_2))

#calculate fft with trapz
Gr_2=np.trapz(np.exp(1j*Tau_2@np.array([omegas]))*G_Tau_2,tau[0],axis=0)

#calculate spectrum
spectrum=(-1)/np.pi*np.imag(Gr_2)
plt.figure()
plt.title('fft with trapz, with coupling')
plt.plot(omegas,0*omegas)
plt.plot(omegas,spectrum)

#include the t dependence, use the correct 
#############################################################################
#%%
V=1
period=2*np.pi/Om
omegas=np.linspace(-4, 4,2001)
Taus=np.linspace(-10000,10000,5000+1)
G_Tau=np.zeros(np.shape(Taus))*0j
i = 0
t=np.linspace(-period/2+1000,period/2+1000,1000)
#t=np.linspace(0,period,100)
norm=0.5/(-1j*np.heaviside(0,0.5)*np.exp(-1j*V/Om*np.sin(Om*(t))))
#print(norm)
for tau in Taus:
    t=np.linspace(-period/2+1000-tau/2,period/2+1000-tau/2,1000)
    if tau>=0:
        Gt_Tau=-1j*np.heaviside(tau,0.5)*np.exp(-1j*V/Om*np.sin(Om*(t+tau))-gamma*abs(tau))*norm
    else:
        Gt_Tau=-1j*np.heaviside(tau,0.5)*np.exp(-1j*V/Om*np.sin(Om*(t+tau))-gamma*abs(tau))

    G_Tau[i]=np.trapz(Gt_Tau,t)
    i+=1

#plot function in time domain
plt.figure()
plt.title('time dependence')
plt.plot(Taus,np.real(G_Tau))
plt.plot(Taus,np.imag(G_Tau))

#prepare for matrix multiplication
G_Tau_2=np.array([G_Tau]).transpose()
Tau_2=np.array([Taus]).transpose()

#print(np.shape(np.array([omegas])),np.shape(Tau_2))
#print(np.shape(np.exp(1j*Tau_2@np.array([omegas]))),np.shape(G_Tau_2))

#calculate fft with trapz
Gr_2=np.trapz(np.exp(1j*Tau_2@np.array([omegas]))*G_Tau_2,Taus,axis=0)

#calculate spectrum
spectrum=(-1)/np.pi*np.imag(Gr_2)
plt.figure()
plt.title('fft with trapz, with coupling')
plt.plot(omegas,0*omegas)
plt.plot(omegas,spectrum)


