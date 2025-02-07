#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:56:13 2025

@author: theresa
"""

import json
import numpy as np
import matplotlib.pyplot as plt

def loadTimeJson(filePath):
    with open(filePath, "r") as f:
        loaded_data = json.load(f)
        _input = loaded_data['input']
        _output = loaded_data['output']
        for el in _output['results']:
            el['a+_a']=np.array(el['a+_a real']) + 1j * np.array(el['a+_a imag'])
            el['a_a+']=np.array(el['a_a+ real']) + 1j * np.array(el['a_a+ imag'])
            
        return _input,_output

def calculateWigner_t(ex_val,mode,Tau,t):
    Om=_input['parameters']['frequency']
    if Om==0:
        period=1
    else:
        period=np.pi*2/Om
    times=t[:, np.newaxis] + Tau/2
    integrand=np.exp(1j*mode*Om*times)*ex_val
    #integrand=ex_val
    return np.trapz(integrand,t,axis=0)/period
    
 
def RetartedWigner(mode_max,site,t,Tau_total,adag_a,a_adag):
    
    Om=_input['parameters']['frequency']
    if Om==0:
        Om=1    
    
    norm=abs(Tau_total[-1]-Tau_total[0])/len(Tau_total)
    wigner_retarted=np.empty((mode_max+1), dtype=object)
    Tau=Tau_total[int(np.floor(len(Tau_total))/2):]
    
    for m in range(0,mode_max+1):
        a_adagWigner0=np.concatenate((np.zeros(len(Tau)-1),calculateWigner_t(a_adag[0],m,Tau,t)))
        adag_a_conjWigner0=np.concatenate((np.zeros(len(Tau)-1),calculateWigner_t(np.conj(adag_a[0]),m,Tau,t)))


        integrand=np.heaviside(Tau_total,0.5)*(-1j)*(a_adagWigner0+adag_a_conjWigner0)
            
        wigner=np.fft.ifft(np.fft.ifftshift(integrand),norm='forward')*norm
        wigner=np.fft.fftshift(wigner)
        wigner_retarted[m]=wigner
    return np.array(wigner_retarted)

def LesserWigner(mode_max,site,t,Tau_total,adag_a):
    Om=_input['parameters']['frequency']
    if Om==0:
        Om=1    
    
    norm=abs(Tau_total[-1]-Tau_total[0])/len(Tau_total)
    wigner_lesser=np.empty((mode_max+1), dtype=object)
    
    for m in range(0,mode_max+1):
        
        positive=+1j*np.conj(adag_a[0])
        negative=+1j*np.flip(adag_a[1][:,1:],axis=1)
        lesser=np.concatenate((negative,positive),axis=1)
        integrand=calculateWigner_t(lesser, m, abs(Tau_total), t)
        #integrand=calculateWigner_t(np.concatenate((positve,negative,axis=1)),m,abs(Tau_total),t)
            
        wigner=np.fft.ifft(np.fft.ifftshift(integrand),norm='forward')*norm
        wigner=np.fft.fftshift(wigner)
        wigner_lesser[m]=wigner
    return np.array(wigner_lesser)

def GreaterWigner(mode_max,site,t,Tau_total,a_adag):
    Om=_input['parameters']['frequency']
    if Om==0:
        Om=1    
    
    norm=abs(Tau_total[-1]-Tau_total[0])/len(Tau_total)
    wigner_greater=np.empty((mode_max+1), dtype=object)
    for m in range(0,mode_max+1):
        
        positive=-1j*a_adag[0]
        negative=-1j*np.flip(np.conj(a_adag[1][:,1:]),axis=1)
        greater=np.concatenate((negative,positive),axis=1)
        integrand=calculateWigner_t(greater, m, abs(Tau_total), t)
            
        wigner=np.fft.ifft(np.fft.ifftshift(integrand),norm='forward')*norm
        wigner=np.fft.fftshift(wigner)
        wigner_greater[m]=wigner
    return np.array(wigner_greater)


def KeldyshWigner(mode_max,site,t,Tau_total,adag_a,a_adag):
    wigner_lesser=LesserWigner(mode_max,site,t,Tau_total,adag_a)
    wigner_greater=GreaterWigner(mode_max,site,t,Tau_total,a_adag)
    return wigner_lesser+wigner_greater
    
    
    
def RetartedFloq(modes,site,t,Tau,adag_a,a_adag):
    Om=_input['parameters']['frequency']
    if Om==0:
        Om=1
    
    Tau_minus=np.flip(Tau[1:])*(-1)
    Tau_total=np.concatenate((Tau_minus,Tau))
    
    norm=abs(Tau_total[-1]-Tau_total[0])/len(Tau_total)
    floquet_retarted=np.empty((2*modes[0]+1, 2*modes[1]+1), dtype=object)
    plt.figure()
    
    omegas=np.fft.fftfreq(len(Tau_total),Tau[1]-Tau[0])*2*np.pi
    omegas=np.fft.fftshift(omegas)
    
    for m in range(-modes[0],modes[0]+1):
        for n in range(-modes[1],modes[1]+1):
            a_adagWigner0=np.concatenate((np.zeros(len(Tau)-1),calculateWigner_t(a_adag[0],m-n,Tau,t)))
            adag_a_conjWigner0=np.concatenate((np.zeros(len(Tau)-1),calculateWigner_t(np.conj(adag_a[0]),m-n,Tau,t)))

            #shift in frequency due to Floquet structure
            frequ_shift=np.exp(1j*(m+n)/2*Om*Tau_total/2)
            integrand=np.heaviside(Tau_total,0.5)*(-1j)*(a_adagWigner0+adag_a_conjWigner0)*frequ_shift
            
            integrand=np.fft.ifftshift(integrand)
            floq=np.fft.ifft(integrand,norm='forward')*norm
            floq=np.fft.fftshift(floq)
            
            floquet_retarted[m+modes[0],n+modes[1]]=floq
    plt.legend()
    return floquet_retarted
            
def calculateCurrent(sites): 
    current=[]
    for site in sites:
        if site=='-1 -1' or site=='1 1':
            print('end')
        else:
            T=_input['parameters']['hopping']*np.ones(2)
            if site[0]=='-':
                i=int(site[0:2])
                j=int(site[3])
            else:
                i=int(site[0])
                j=int(site[2])
            if not(abs(i-j)==1):
                raise ValueError('not neighbours')
            
            T_ij=T[int(len(T)/2)+i]
            for site1 in _output['results']:
                if site1['sites'] == str(i)+' '+str(j):
                    adag_aij=np.array(site1['a+_a'])[:,0]
                    
            for site2 in _output['results']:
                if site2['sites'] == str(j)+' '+str(i):
                    adag_aji=np.array(site2['a+_a'])[:,0]
            
            current.append(-1j*(T_ij*adag_aij-np.conj(T_ij)*adag_aji))
    return sites,current
            
    
    
def calculateWigner(mode_max,components,sites):
    wigner_dic={}
    for site in sites:
        wigner_dic[site]={}
    t=np.array(_output['t'])
    Tau=np.array(_output['Tau'])
    
    Tau_minus=np.flip(Tau[1:])*(-1)
    Tau_total=np.concatenate((Tau_minus,Tau))
    omegas=np.fft.fftfreq(len(Tau_total),Tau[1]-Tau[0])*2*np.pi
    omegas=np.fft.fftshift(omegas)
    for site in _output['results']:
        if site['sites'] in sites:
            site0 = site['sites'][0]
            site1 = site['sites'][2]
            adag_a0=np.array(site['a+_a'])
            a_adag0=np.array(site['a_a+'])
            if site0 == site1:
                adag_a1=adag_a0
                a_adag1=a_adag0
            else:
                for site2 in _output['results']:
                    if site2['sites'] == str(site1)+' '+str(site0):
                        adag_a1=np.array(site2['a+_a'])
                        a_adag1=np.array(site2['a_a+'])
            adag_a=[adag_a0,adag_a1]
            a_adag=[a_adag0,a_adag1]
            
            for comp in components:
                if comp=='retarted':
                    wigner_dic[site['sites']]['retarted']=RetartedWigner(mode_max,site,t,Tau_total,adag_a,a_adag)
                
                if comp=='lesser':
                    wigner_dic[site['sites']]['lesser']=LesserWigner(mode_max,site,t,Tau_total,adag_a)
                    
                if comp=='greater':
                    wigner_dic[site['sites']]['greater']=GreaterWigner(mode_max,site,t,Tau_total,a_adag)
                    
                if comp=='keldysh':
                    wigner_dic[site['sites']]['keldysh']=KeldyshWigner(mode_max,site,t,Tau_total,adag_a,a_adag)
        return sites,omegas,wigner_dic
        

               
def calcGreen(component):
    eps=_input['parameters']['epsilon']
    gamma1=np.diag(_input['parameters']['coupling_empty'],0)
    gamma2=np.diag(_input['parameters']['coupling_full'],0)
    hopping=_input['parameters']['hopping']*np.ones(len(eps)-1)
    hopping_left=np.conj(hopping)
    
    E=np.diag(eps,0)+np.diag(hopping,1)+np.diag(hopping_left,-1)
    Gamma_plus=gamma1+gamma2
    Gamma_minus=gamma2-gamma1
    i0=component[0]+int(np.floor(len(eps)/2))
    i1=component[1]+int(np.floor(len(eps)/2))
    omegas=np.linspace(-5,5,1000)
    Gr=np.zeros(len(omegas))*1j
    Gk=np.zeros(len(omegas))*1j
    for i in range(len(omegas)):
        omega=np.diag(omegas[i]*np.ones(len(eps)))
        Gr_matrix=np.linalg.inv(omega-E+1j*Gamma_plus)
        Gr[i]=Gr_matrix[i0,i1]
        Gk_matrix=2j*Gr_matrix@Gamma_minus@(np.conj(Gr_matrix))
        Gk[i]=Gk_matrix[i0,i1]
    return omegas,Gr,Gk
    
_input,_output=loadTimeJson('results/U2V1Om1_20250206-222144.json')
print(_input)
#calculateFloqet([0,0],['retarted'])
adag_a=_output['results'][0]['a+_a']
sites,omegas,wigner_dic=calculateWigner(0,['retarted','lesser','greater','keldysh'],['0 0'])
#print(wigner_dic)


#for site in sites:
    #G_site=wigner_dic[site]

    #for key in G_site:
        #plt.figure()
        #plt.title('site'+site+'  '+key)
        #plt.plot(omegas,G_site[key][0].real,label='real')
        #plt.plot(omegas,G_site[key][0].imag,label='imag')
        #plt.legend()
        
n=_output['results'][0]['a+_a'][:,0]
t=_output['t']
print(len(t))
print(len(n))
print(wigner_dic)
omegas_an00,Gr_an00,Gk_an00=calcGreen([0,0])
start=np.where(omegas>-5)[0][0]
end=np.where(omegas>5)[0][0]
print(start)
Gr=wigner_dic['0 0']['retarted'][0]
Gk=wigner_dic['0 0']['keldysh'][0]
spectral=(-1)/np.pi*np.imag(Gr)
spectral_ann=(-1)/np.pi*np.imag(Gr_an00)
print(np.trapz(wigner_dic['0 0']['keldysh'][0].imag,omegas)/(4*np.pi)+1/2)
print(np.trapz(spectral,omegas))
print(np.trapz(spectral_ann,omegas_an00))
plt.figure()
plt.title('retarted, 00')
plt.plot(omegas[start:end],wigner_dic['0 0']['retarted'][0].real[start:end],label='real')
#plt.plot(omegas_an00,Gr_an00.real,label='analytic',linestyle=(0,(5, 5)),color='k')
plt.plot(omegas[start:end],wigner_dic['0 0']['retarted'][0].imag[start:end],label='imag')
#plt.plot(omegas_an00,Gr_an00.imag,linestyle=(0,(5, 5)),color='k')
plt.xlabel('w')
plt.legend()

plt.figure()
plt.title('keldysh, 00')
plt.plot(omegas[start:end],wigner_dic['0 0']['keldysh'][0].real[start:end],label='real')
plt.plot(omegas[start:end],wigner_dic['0 0']['keldysh'][0].imag[start:end],label='imag')
#plt.plot(omegas_an00,Gk_an00.real,label='analytic',linestyle=(0,(5, 5)),color='k')
#plt.plot(omegas_an00,Gk_an00.imag,linestyle=(0,(5, 5)),color='k')
plt.xlabel('w')
plt.legend()

plt.figure()
plt.title('greater lesser, 00')
plt.plot(omegas[start:end],wigner_dic['0 0']['greater'][0].real[start:end],label='greater real')
plt.plot(omegas[start:end],wigner_dic['0 0']['greater'][0].imag[start:end],label='greater imag')
plt.plot(omegas[start:end],wigner_dic['0 0']['lesser'][0].real[start:end],label='lesser real')
plt.plot(omegas[start:end],wigner_dic['0 0']['lesser'][0].imag[start:end],label='lesser imag')
plt.xlabel('w')
plt.legend()

sites,j=calculateCurrent(['-1 0','0 1'])
fig, ax1 = plt.subplots()
ax1.set_title('current')
ax1.plot(t, j[0], label='-10')
print(np.min(j[0]))
print(np.max(j[0]))
ax1.plot(t, j[1], label='01',color='cyan',linestyle='dashed')

ax1.set_xlabel("t")
ax1.set_ylabel("j")
ax1.set_ylim(-2e-3, -1e-3)

# Create a secondary x-axis for n
ax2 = ax1.twinx()
#ax1.plot(t, np.gradient(n,t), color='k', label='grad n')
#ax1.plot(t, j[1]-j[0], linestyle='dashed', label='01-(-10)',color='green')
ax2.set_ylabel("n")
ax2.plot(t, n, color='magenta', label='n')
ax2.set_ylim(0.45, 0.55)
# Show legends and grid
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.show()

start=np.where(omegas>-2.5)[0][0]
end=np.where(omegas>2.5)[0][0]
F=0.5*(1-Gk.imag/(2*Gr.imag))
plt.figure()
plt.title('distribution function')
plt.plot(omegas[start:end],F[start:end])

