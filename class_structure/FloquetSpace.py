#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:56:13 2025

@author: theresa
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import os

#print(multiprocessing.cpu_count())
#print(os.cpu_count())
#print(len(os.sched_getaffinity(0)))

def loadTimeJson(filePath):
    with open(filePath, "r") as f:
        loaded_data = json.load(f)
        _input = loaded_data['input']
        _output = loaded_data['output']
        for el in _output['results']:
            el['a+_a']=np.array(el['a+_a real']) + 1j * np.array(el['a+_a imag'])
            el['a_a+']=np.array(el['a_a+ real']) + 1j * np.array(el['a_a+ imag'])
            
        return _input,_output

def plots(_output,omegas,wigner_dic):
    n=_output['results'][0]['a+_a'][:,0]
    t=_output['t']
    print(len(t))
    print(len(n))
    print(wigner_dic)
    #omegas_an00,Gr_an00,Gk_an00=calcGreen([0,0])
    start=np.where(omegas>-5)[0][0]
    end=np.where(omegas>5)[0][0]
    print(start)
    Gr=wigner_dic['0 0']['retarted'][0]
    Gk=wigner_dic['0 0']['keldysh'][0]
    spectral=(-1)/np.pi*np.imag(Gr)
    #spectral_ann=(-1)/np.pi*np.imag(Gr_an00)
    print(np.trapz(wigner_dic['0 0']['keldysh'][0].imag,omegas)/(4*np.pi)+1/2)
    print(np.trapz(spectral,omegas))
    #print(np.trapz(spectral_ann,omegas_an00))
    plt.figure()
    plt.title('retarted, 00')
    plt.plot(omegas[start:end],wigner_dic['0 0']['retarted'][0].real[start:end],label='real')
    #plt.plot(omegas_an00,Gr_an00.real,label='analytic',linestyle=(0,(5, 5)),color='k')
    plt.plot(omegas[start:end],wigner_dic['0 0']['retarted'][0].imag[start:end],label='imag')
    #plt.plot(omegas_an00,Gr_an00.imag,linestyle=(0,(5, 5)),color='k')
    plt.xlabel('w')
    plt.legend()

    plt.show(block=False)

    plt.figure()
    plt.title('keldysh, 00')
    plt.plot(omegas[start:end],wigner_dic['0 0']['keldysh'][0].real[start:end],label='real')
    plt.plot(omegas[start:end],wigner_dic['0 0']['keldysh'][0].imag[start:end],label='imag')
    #plt.plot(omegas_an00,Gk_an00.real,label='analytic',linestyle=(0,(5, 5)),color='k')
    #plt.plot(omegas_an00,Gk_an00.imag,linestyle=(0,(5, 5)),color='k')
    plt.xlabel('w')
    plt.legend()
    print('reached break')
    plt.show(block=False)

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

def calculateWigner_t(_input,ex_val,mode,Tau,t):
    Om=_input['parameters']['frequency']
    if Om==0:
        period=1
    else:
        period=np.pi*2/Om
    times=t[:, np.newaxis] + Tau/2
    integrand=np.exp(1j*mode*Om*times)*ex_val
    #integrand=ex_val
    return np.trapezoid(integrand,t,axis=0)/period
    
 
def RetartedWigner(_input,mode_max,site,t,Tau_total,adag_a,a_adag):
    
    Om=_input['parameters']['frequency']
    if Om==0:
        Om=1    
    
    norm=abs(Tau_total[-1]-Tau_total[0])/len(Tau_total)
    wigner_retarted=np.empty((mode_max*2+1), dtype=object)
    Tau=Tau_total[int(np.floor(len(Tau_total))/2):]
    #plt.figure()
    #plt.title('test retarded')
    for m in range(-mode_max,mode_max+1):
        #print('modes in retarded:',m,m+mode_max)
        a_adagWigner0=np.concatenate((np.zeros(len(Tau)-1),calculateWigner_t(_input,a_adag[0],m,Tau,t)))
        adag_a_conjWigner0=np.concatenate((np.zeros(len(Tau)-1),calculateWigner_t(_input,np.conj(adag_a[0]),m,Tau,t)))


        integrand=np.heaviside(Tau_total,0.5)*(-1j)*(a_adagWigner0+adag_a_conjWigner0)
        #plt.figure()
        #plt.title('retarted')
        #plt.plot(Tau_total,integrand)
        wigner=np.fft.ifft(np.fft.ifftshift(integrand),norm='forward')*norm
        wigner=np.fft.fftshift(wigner)
        wigner_retarted[m+mode_max]=wigner

        #plt.plot(wigner_retarted[m+mode_max].imag,label=m)
    
    #plt.legend()

    #wigner_retarted=np.array(wigner_retarted)
    #plt.plot(wigner_retarted[2])
    #plt.show()
    return np.array(wigner_retarted)

def LesserWigner(_input,mode_max,site,t,Tau_total,adag_a):
    Om=_input['parameters']['frequency']
    if Om==0:
        Om=1    
    
    norm=abs(Tau_total[-1]-Tau_total[0])/len(Tau_total)
    wigner_lesser=np.empty((2*mode_max+1), dtype=object)
    
    for m in range(-mode_max,mode_max+1):
        
        positive=+1j*np.conj(adag_a[0])
        negative=+1j*np.flip(adag_a[1][:,1:],axis=1)
        lesser=np.concatenate((negative,positive),axis=1)
        integrand=calculateWigner_t(_input,lesser, m, abs(Tau_total), t)
        #integrand=calculateWigner_t(np.concatenate((positve,negative,axis=1)),m,abs(Tau_total),t)
        #plt.figure()
        #plt.plot(Tau_total,integrand)
        wigner=np.fft.ifft(np.fft.ifftshift(integrand),norm='forward')*norm
        wigner=np.fft.fftshift(wigner)
        wigner_lesser[m+mode_max]=wigner
    return np.array(wigner_lesser)

def GreaterWigner(_input,mode_max,site,t,Tau_total,a_adag):
    Om=_input['parameters']['frequency']
    if Om==0:
        Om=1    
    
    norm=abs(Tau_total[-1]-Tau_total[0])/len(Tau_total)
    wigner_greater=np.empty((2*mode_max+1), dtype=object)
    for m in range(-mode_max,mode_max+1):
        
        positive=-1j*a_adag[0]
        negative=-1j*np.flip(np.conj(a_adag[1][:,1:]),axis=1)
        greater=np.concatenate((negative,positive),axis=1)
        integrand=calculateWigner_t(_input,greater, m, abs(Tau_total), t)
        #plt.figure()
        #plt.plot(Tau_total,integrand)  
        wigner=np.fft.ifft(np.fft.ifftshift(integrand),norm='forward')*norm
        wigner=np.fft.fftshift(wigner)
        wigner_greater[m+mode_max]=wigner
    return np.array(wigner_greater)


def KeldyshWigner(_input,mode_max,site,t,Tau_total,adag_a,a_adag):
    wigner_lesser=LesserWigner(_input,mode_max,site,t,Tau_total,adag_a)
    wigner_greater=GreaterWigner(_input,mode_max,site,t,Tau_total,a_adag)
    return wigner_lesser+wigner_greater
    
    
    
def RetardedFloquet(_input,modes,t,Tau_total,adag_a,a_adag):
    Om=_input['parameters']['frequency']
    if Om==0:
        Om=1
    
    Tau=Tau_total[int(np.floor(len(Tau_total))/2):]
    Tau_minus=np.flip(Tau[1:])*(-1)
    norm=abs(Tau_total[-1]-Tau_total[0])/len(Tau_total)
    #floquet_retarted=np.empty((2*modes[0]+1, 2*modes[1]+1), dtype=object)
    #plt.figure()
    
    omegas=np.fft.fftfreq(len(Tau_total),Tau_total[1]-Tau_total[0])*2*np.pi
    omegas=np.fft.fftshift(omegas)
    valid_floquet_ind=np.where((omegas > -Om/2) & (omegas <= Om/2))[0]
    omegas_center=omegas[valid_floquet_ind[0]:valid_floquet_ind[-1]+1]
    l_om=len(omegas_center)
    floquet_retarted=np.zeros((l_om,2*modes[0]+1, 2*modes[1]+1), dtype=complex)
    #fig, axs = plt.subplots(2, figsize=(10, 6))
    for m in range(-modes[0],modes[0]+1):
        for n in range(-modes[1],modes[1]+1):
            a_adagWigner0=np.concatenate((np.zeros(len(Tau)-1),calculateWigner_t(_input,a_adag[0],m-n,Tau,t)))
            adag_a_conjWigner0=np.concatenate((np.zeros(len(Tau)-1),calculateWigner_t(_input,np.conj(adag_a[0]),m-n,Tau,t)))

            #shift in frequency due to Floquet structure
            frequ_shift=np.exp(1j*(m+n)/2*Om*Tau_total)
            integrand=np.heaviside(Tau_total,0.5)*(-1j)*(a_adagWigner0+adag_a_conjWigner0)*frequ_shift

            integrand=np.fft.ifftshift(integrand)

            floq=np.fft.ifft(integrand,norm='forward')*norm
            floq=np.fft.fftshift(floq)

            floq=floq[valid_floquet_ind[0]:valid_floquet_ind[-1]+1]
            omegas_nm=omegas_center+(m+n)/2*Om
            for i in range(l_om):
                #create the matrix for each omega value between -Om/2 & Om/2
                floquet_retarted[i,m+modes[0],n+modes[1]]=floq[i]
            #if n==m:
            #    axs[0].plot(omegas_nm,floquet_retarted[:,m+modes[0],n+modes[1]].real,label=n)
            #    axs[1].plot(omegas_nm,floquet_retarted[:,m+modes[0],n+modes[1]].imag,label=n)

    #plt.legend()
    #plt.show()
    #print(omegas_center.shape)
    #print(floquet_retarted.shape)
    return floquet_retarted

def LesserFloquet(_input,modes,t,Tau_total,adag_a):
    Om=_input['parameters']['frequency']
    if Om==0:
        Om=1    
    
    norm=abs(Tau_total[-1]-Tau_total[0])/len(Tau_total)

    #calculate relevant omegas and there indices
    omegas=np.fft.fftfreq(len(Tau_total),Tau_total[1]-Tau_total[0])*2*np.pi
    omegas=np.fft.fftshift(omegas)
    valid_floquet_ind=np.where((omegas > -Om/2) & (omegas <= Om/2))[0]
    omegas_center=omegas[valid_floquet_ind[0]:valid_floquet_ind[-1]+1]
    l_om=len(omegas_center)

    #define storage for matrix elements
    floquet_lesser=np.zeros((l_om,2*modes[0]+1, 2*modes[1]+1), dtype=complex)
    #fig, axs = plt.subplots(2, figsize=(10, 6))
    for m in range(-modes[0],modes[0]+1):
        for n in range(-modes[1],modes[1]+1):
        
            positive=+1j*np.conj(adag_a[0])
            negative=+1j*np.flip(adag_a[1][:,1:],axis=1)
            lesser=np.concatenate((negative,positive),axis=1)

            integrand=calculateWigner_t(_input,lesser, m-n, abs(Tau_total), t)
            frequ_shift=np.exp(1j*(m+n)/2*Om*Tau_total)
            integrand=integrand*frequ_shift

            floq=np.fft.ifft(np.fft.ifftshift(integrand),norm='forward')*norm
            floq=np.fft.fftshift(floq)
            floq=floq[valid_floquet_ind[0]:valid_floquet_ind[-1]+1]
            for i in range(l_om):
                floquet_lesser[i,m+modes[0],n+modes[1]]=floq[i]
            #omegas_nm=omegas_center+(m+n)/2*Om
            #if n==m:
            #    axs[0].plot(omegas_nm,floquet_lesser[:,m+modes[0],n+modes[1]].real,label=n)
            #    axs[1].plot(omegas_nm,floquet_lesser[:,m+modes[0],n+modes[1]].imag,label=n)

    #plt.legend()
    #plt.show()

    return np.array(floquet_lesser)

def GreaterFloquet(_input,modes,t,Tau_total,a_adag):
    Om=_input['parameters']['frequency']
    if Om==0:
        Om=1    
    
    norm=abs(Tau_total[-1]-Tau_total[0])/len(Tau_total)

    #calculate relevant omegas and there indices
    omegas=np.fft.fftfreq(len(Tau_total),Tau_total[1]-Tau_total[0])*2*np.pi
    omegas=np.fft.fftshift(omegas)
    valid_floquet_ind=np.where((omegas > -Om/2) & (omegas <= Om/2))[0]
    omegas_center=omegas[valid_floquet_ind[0]:valid_floquet_ind[-1]+1]
    l_om=len(omegas_center)

    #define storage for matrix elements
    floquet_greater=np.zeros((l_om,2*modes[0]+1, 2*modes[1]+1), dtype=complex)
    #fig, axs = plt.subplots(2, figsize=(10, 6))
    for m in range(-modes[0],modes[0]+1):
        for n in range(-modes[1],modes[1]+1):
        
            positive=-1j*a_adag[0]
            negative=-1j*np.flip(np.conj(a_adag[1][:,1:]),axis=1)
            greater=np.concatenate((negative,positive),axis=1)
            integrand=calculateWigner_t(_input,greater, m-n, abs(Tau_total), t)
            frequ_shift=np.exp(1j*(m+n)/2*Om*Tau_total)
            integrand=integrand*frequ_shift

            floq=np.fft.ifft(np.fft.ifftshift(integrand),norm='forward')*norm
            floq=np.fft.fftshift(floq)
            floq=floq[valid_floquet_ind[0]:valid_floquet_ind[-1]+1]
            for i in range(l_om):
                floquet_greater[i,m+modes[0],n+modes[1]]=floq[i]
            #omegas_nm=omegas_center+(m+n)/2*Om
            #if n==m:
            #    axs[0].plot(omegas_nm,floquet_greater[:,m+modes[0],n+modes[1]].real,label=n)
            #    axs[1].plot(omegas_nm,floquet_greater[:,m+modes[0],n+modes[1]].imag,label=n)

    #plt.legend()
    #plt.show()
    return np.array(floquet_greater)
            
def KeldyshFloquet(_input,modes,t,Tau_total,adag_a,a_adag):
    Om=_input['parameters']['frequency']
    if Om==0:
        Om=1   

    floquet_lesser=LesserFloquet(_input,modes,t,Tau_total,adag_a)
    floquet_greater=GreaterFloquet(_input,modes,t,Tau_total,a_adag)
    floquet_keldysh=floquet_lesser+floquet_greater

    
    #omegas=np.fft.fftfreq(len(Tau_total),Tau_total[1]-Tau_total[0])*2*np.pi
    #omegas=np.fft.fftshift(omegas)
    #valid_floquet_ind=np.where((omegas > -Om/2) & (omegas <= Om/2))[0]
    #omegas_center=omegas[valid_floquet_ind[0]:valid_floquet_ind[-1]+1]
    
    #plt.figure()
    #for l in range(-modes[0],modes[0]+1):
    #    plt.plot(omegas_center+l*Om,floquet_keldysh[:,l+modes[0],l+modes[0]].imag,label=l)
    #plt.legend()
    #plt.show()

    return floquet_lesser+floquet_greater


def calculateCurrent(_input,_output,sites): 
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
                print(site1['sites'])
                if site1['sites'] == str(i)+' '+str(j):
                    print('site1 found')
                    adag_aij=np.array(site1['a+_a'])[:,0]
                    
            for site2 in _output['results']:
                if site2['sites'] == str(j)+' '+str(i):
                    print('site2 found')
                    adag_aji=np.array(site2['a+_a'])[:,0]
            
            current.append(1j*(T_ij*adag_aij-np.conj(T_ij)*adag_aji))
    return sites,np.array(current)
            
    
    
def calculateWigner(_output,mode_max,components,sites):
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
        

               
def calcGreen(_input,component):
    eps=_input['parameters']['epsilon']
    #gamma1=np.diag(_input['parameters']['coupling_empty'],0)
    #gamma2=np.diag(_input['parameters']['coupling_full'],0)
    gamma1=np.array(_input['parameters']['coupling_emptyReal'])+1j*np.array(_input['parameters']['coupling_emptyImag'])
    gamma2=np.array(_input['parameters']['coupling_fullReal'])+1j*np.array(_input['parameters']['coupling_fullImag'])
    #hopping=_input['parameters']['hopping']*np.ones(len(eps)-1)
    hopping=_input['parameters']['hopping'][:-1]
    hopping_left=np.conj(hopping)
    
    E=np.diag(eps,0)+np.diag(hopping,1)+np.diag(hopping_left,-1)
    Gamma_plus=gamma1+gamma2
    Gamma_minus=gamma2-gamma1
    i0=component[0]+int(np.floor(len(eps)/2))
    i1=component[1]+int(np.floor(len(eps)/2))
    omegas=np.linspace(-10,10,1000)
    Gr=np.zeros(len(omegas))*1j
    Gk=np.zeros(len(omegas))*1j
    for i in range(len(omegas)):
        omega=np.diag(omegas[i]*np.ones(len(eps)))
        Gr_matrix=np.linalg.inv(omega-E+1j*Gamma_plus)
        Gr[i]=Gr_matrix[i0,i1]
        Gk_matrix=2j*Gr_matrix@Gamma_minus@(np.conj(Gr_matrix).T)
        Gk[i]=Gk_matrix[i0,i1]
    return omegas,Gr,Gk

def calculateWignerFromFile(file,mode_max,components,sites):
    
    _input,_output=loadTimeJson(file)
    #print(_input)
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
                if comp=='retarded':
                    wigner_dic[site['sites']]['retarded']=RetartedWigner(_input,mode_max,site,t,Tau_total,adag_a,a_adag)
                if comp=='lesser':
                    wigner_dic[site['sites']]['lesser']=LesserWigner(_input,mode_max,site,t,Tau_total,adag_a)
                if comp=='greater':
                    wigner_dic[site['sites']]['greater']=GreaterWigner(_input,mode_max,site,t,Tau_total,a_adag)
                if comp=='keldysh':
                    wigner_dic[site['sites']]['keldysh']=KeldyshWigner(_input,mode_max,site,t,Tau_total,adag_a,a_adag)
    return sites,omegas,wigner_dic

def calculateFloquetFromFile(file,modes,components,sites):
    
    _input,_output=loadTimeJson(file)
    #print(_input)
    Om=_input['parameters']['frequency']
    if Om==0:
        Om=1 

    floquet_dic={}
    for site in sites:
        floquet_dic[site]={}
    t=np.array(_output['t'])
    Tau=np.array(_output['Tau'])
    
    Tau_minus=np.flip(Tau[1:])*(-1)
    Tau_total=np.concatenate((Tau_minus,Tau))

    omegas=np.fft.fftfreq(len(Tau_total),Tau_total[1]-Tau_total[0])*2*np.pi
    omegas=np.fft.fftshift(omegas)
    valid_floquet_ind=np.where((omegas > -Om/2) & (omegas <= Om/2))[0]
    omegas_center=omegas[valid_floquet_ind[0]:valid_floquet_ind[-1]+1]
    #print(omegas_center.shape)
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
                if comp=='retarded':
                    floquet_dic[site['sites']]['retarded']=RetardedFloquet(_input,modes,t,Tau_total,adag_a,a_adag)
                
                if comp=='lesser':
                    floquet_dic[site['sites']]['lesser']=LesserFloquet(_input,modes,t,Tau_total,adag_a)
                    
                if comp=='greater':
                    floquet_dic[site['sites']]['greater']=GreaterFloquet(_input,modes,t,Tau_total,a_adag)
                    
                if comp=='keldysh':
                    floquet_dic[site['sites']]['keldysh']=KeldyshFloquet(_input,modes,t,Tau_total,adag_a,a_adag)
        return sites,omegas_center,floquet_dic

def calculateCurrentFromFile(file,sites):
    _input,_output=loadTimeJson(file)
    Om=_input['parameters']['frequency']
    if Om==0:
        period=1
    else:
        period=np.pi*2/Om
    print(_input)
    t=np.array(_output['t'])
    sites,current=calculateCurrent(_input,_output,sites)

    print('from equal time')
    print('sites: ',sites)
    print('cureent: ',np.trapezoid(current,t)/period)
    return sites,current

def calculateCurrentFromKeldysh(file,sites):
    _input,_output=loadTimeJson(file)
    #print(_input)
    calculateWignerFromFile(file,0,['keldysh'],sites)
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
            sites_dic,omegas,dic_ij=calculateWignerFromFile(file,0,['keldysh'],[str(i)+' '+str(j)])
            sites_dic,omegas,dic_ji=calculateWignerFromFile(file,0,['keldysh'],[str(j)+' '+str(i)])
            kel_ij=dic_ij[str(i)+' '+str(j)]['keldysh'][0]
            kel_ji=dic_ji[str(j)+' '+str(i)]['keldysh'][0]
            
            current.append(np.trapezoid(-T_ij*kel_ij+np.conj(T_ij)*kel_ji,omegas)/(2*np.pi)/2)
        print('from keldysh')
        print('sites: ',sites)
        print('current: ',current)
        return sites, current

# file='class_structure/results/U0V0Om0_20250408-134137.json'
# _input,_output=loadTimeJson(file)

# sites,omegas_wigner,wigner_dic=calculateWignerFromFile(file,0,['retarded','keldysh','greater','lesser'],['0 0'])
# omegas,Gr,Gk=calcGreen(_input,[0,0])
# print(_input)
# print(len(omegas_wigner))
# om_start=10
# mask=(omegas_wigner > -om_start) & (omegas_wigner < om_start)
# plt.figure()
# plt.title('retarded')
# plt.plot(omegas,Gr.imag)
# plt.plot(omegas_wigner[mask],wigner_dic['0 0']['retarded'][0][mask].imag,linestyle='dashed')
# plt.plot(omegas,Gr.real)
# plt.plot(omegas_wigner[mask],wigner_dic['0 0']['retarded'][0][mask].real,linestyle='dashed')
# #plt.show(block=False)
# plt.show(block=False)

# plt.figure()
# plt.title('keldysh')
# plt.plot(omegas,Gk.imag)
# plt.plot(omegas_wigner[mask],wigner_dic['0 0']['greater'][0][mask].imag)
# plt.plot(omegas_wigner[mask],wigner_dic['0 0']['lesser'][0][mask].imag)
# plt.plot(omegas_wigner[mask],wigner_dic['0 0']['keldysh'][0][mask].imag,linestyle='dashed')
# plt.plot(omegas_wigner[mask],wigner_dic['0 0']['greater'][0][mask].real)
# plt.show()

#calculateCurrentFromFile(file,['0 1'])
#calculateCurrentFromKeldysh(file,['0 1'])

#calculateFloquetFromFile(file,[2,2],['greater'],['0 0'])

#_input,_output=loadTimeJson('results/U0V1Om1_20250212-092450.json')
#print(_input)
#calculateFloqet([0,0],['retarted'])
#adag_a=_output['results'][0]['a+_a']
#sites,omegas,wigner_dic=calculateWigner(0,['retarted','lesser','greater','keldysh'],['0 0'])
#print(wigner_dic)


#for site in sites:
    #G_site=wigner_dic[site]

    #for key in G_site:
        #plt.figure()
        #plt.title('site'+site+'  '+key)
        #plt.plot(omegas,G_site[key][0].real,label='real')
        #plt.plot(omegas,G_site[key][0].imag,label='imag')
        #plt.legend()
        

