import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from class_structure.FloquetSpace import calculateWignerFromFile
import os

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Navigate up one level to the project_directory
#valid_indices = np.where((omegasF >= omegas[0]) & (omegasF <= omegas[-1]))[0]
            
omegas=np.linspace(-4,4,250)


#axs[1,v].plot(omegasF,np.imag(GkFile),linewidth=1,color=colors_n[order],label=f'numeric {order}',linestyle='dashed')
plt.show(block=False)
V=[1,2]
Om=[1,1]
tl=0.3
tr=0.3
eps0=0
epsl=1
epsr=-1
Gamma=1.5
Delta=0.2
Gammar=1.5
Gammal=1.5
Deltar=-0.3
Deltal=0.3
Delta=0.02
eta=0


#file1='class_structure/results/U0V1Om1_20250212-092450.json'
#small file
file1 = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'class_structure', 'results', 'U0V1Om1_20250318-072625.json')
file2 = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'class_structure', 'results', 'U0V2Om1_20250318-071004.json')
#large file
#file1 = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'class_structure', 'results', 'U0V1Om1_20250317-225539.json')
#file2 = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'class_structure', 'results', 'U0V2Om1_20250317-152846.json')
#print('printing filename')
#print(file1)
files=[file1,file2]
#sites,omegasF,wigner_dic=calculateWignerFromFile(file1,0,['retarted','lesser','greater','keldysh'],['0 0'])

#valid_indices = np.where((omegasF >= omegas[0]) & (omegasF <= omegas[-1]))[0]
eps=0
order_max=1
l_max=21

print('test start')
g0_til_inv=(omegas - eps0+1j*eta)
Gr=1/(omegas - epsr + 1j*Gammar)
Gl=1/(omegas - epsl + 1j*Gammal)
#G_test = (g0_til/(1-g0_til*(tr**2*Gr+tl**2*Gl)))
G_test=(g0_til_inv-tr**2*Gr-tl**2*Gl)**(-1)
plt.figure()
plt.plot(omegas,G_test.imag)
print('test end')


def calcKeldysh_a(omegas,l_max,Delta,Gamma,V,Om,m,n):
    Gk=np.zeros(len(omegas), dtype=complex)
    Gk_l=np.zeros((len(omegas),2*l_max+1), dtype=complex)
    for k, om in enumerate(omegas):
        temp_sum1 = 0  
        for l in range(-l_max, l_max+1, 1):
            Gr=calcRetardedFromInv_a([om],l_max*2,Gamma,V,Om,m,l)[0]
            Ga=np.conj(calcRetardedFromInv_a([om],l_max*2,Gamma,V,Om,n,l))[0]
            Gleft=1j*2*(Deltal)/((om-epsl+l*Om)**2+Gammal**2)
            Gright=1j*2*(Deltar)/((om-epsr+l*Om)**2+Gammar**2)
            Gk_l[k,l]=Gr *  Ga* (tl**2*Gleft+tr**2*Gright)
            temp_sum1 += Gk_l[k,l]
        #print(temp_sum1)
            
        Gk[k] += temp_sum1

    return Gk


def calcRetarded_a(omegas,l_max,Gamma,V,Om,m,n):
    Gret = np.zeros(len(omegas), dtype=complex)
    for i, om in enumerate(omegas):
        temp_sum = 0  
        for l in range(-l_max, l_max+1, 1):
            g0_til_inv=(om + l*Om - eps0)
            Gright=1/(om + l*Om - epsr + 1j*Gammar)#(om+ l*Om-epsl)
            Gleft=1/(om + l*Om - epsl + 1j*Gammal)#(om+ l*Om-epsr)
            temp_sum += (g0_til_inv-tr**2*Gright-tl**2*Gleft)**(-1) * jv(m-l, V/Om) * jv(n-l, V/Om)
        Gret[i] += temp_sum
    return Gret

def calcRetardedFromInv_a(omegas,l_max,Gamma,V,Om,m,n):
    #print(om-n*Om)
    Gret = np.zeros(len(omegas), dtype=complex)
    for i, om in enumerate(omegas):
        Gr_inv_diag=[]
        #p=int(np.ceil((om-Om/2)/Om))
        for j in range(-l_max,l_max+1,1):
            #om-n*Om used to shift Om into the valid region
            Gr_inv_diag.append(om + j*Om -eps0-tl**2/(om +j*Om -epsl+1j*Gammal)-tr**2/(om +j*Om -epsr+1j*Gammar))
            
        Gr_inv_diag=np.array(Gr_inv_diag)
        Gr_inv_off=np.ones(len(Gr_inv_diag)-1)*V/2
        Gr_inv=np.diag(Gr_inv_diag,k=0)+np.diag(Gr_inv_off,k=1)+np.diag(Gr_inv_off,k=-1)

        #print(Gr_inv.shape)
        Gr_mat=np.linalg.inv(Gr_inv)
        Gret[i] += Gr_mat[m+l_max,n+l_max]
        
    #plt.figure()
    #plt.title('from inverse')
    #plt.plot(omegas,Gret.real)
    #plt.plot(omegas,Gret.imag)
    #plt.show()
    return Gret*(-1)**(m+n)

#calcRetardedFromInv_a(omegas,20,Gamma,V[0],Om[0],10,-10)
#print(bla)
fig, axs = plt.subplots(2,len(V), figsize=(4.1*len(V), 4),sharex=True, sharey='row')
colors=['#a65628','#f781bf','#a6cee3','#984ea3','#c97b4a']
colors_a=['#a6cee3','#f781bf','#c97b4a','#64b5cd']
colors_n=['navy','#984ea3','#a65628','#468c9e']
sites,omegasF1,wigner_dic=calculateWignerFromFile(file1,order_max,['retarted','lesser','greater','keldysh'],['0 0'])
sites,omegasF2,wigner_dic=calculateWignerFromFile(file2,0,['retarted','lesser','greater','keldysh'],['0 0'])
#print(omegasF[0],omegasF[-1])
valid_indices1 = np.where((omegasF1 >= omegas[0]) & (omegasF1 <= omegas[-1]))[0]
omegasF1=omegasF1[valid_indices1[0]:valid_indices1[-1]]
valid_indices2 = np.where((omegasF2 >= omegas[0]) & (omegasF2 <= omegas[-1]))[0]
omegasF2=omegasF2[valid_indices2[0]:valid_indices2[-1]]
omegas=[omegasF1,omegasF2]
print('om shape: ',len(omegas[0]),len(omegas[1]))
Gr_error=np.zeros(len(V))
Gk_error=np.zeros(len(V))

for v in range(len(V)):
    sites,omegasF,wigner_dic=calculateWignerFromFile(files[v],order_max,['retarted','lesser','greater','keldysh'],['0 0'])
    valid_indices = np.where((omegasF >= omegas[v][0]) & (omegasF <= omegas[v][-1]))[0]
    omegasF=omegasF[valid_indices[0]:valid_indices[-1]+1]
    #print(bla)
    print('shape',wigner_dic['0 0']['retarted'].shape)
    for order in range(-order_max,order_max+1):
        if order%2:
            print('if order:',order)
            m=int((order+1)/2)
            n=-int((order-1)/2)
            print('mn',m,n)
            Gret=calcRetardedFromInv_a((omegas[v]-Om[v]/2),l_max,Gamma,V[v],Om[v],m,n)
            Gk=calcKeldysh_a((omegas[v]-Om[v]/2),l_max,Delta,Gamma,V[v],Om[v],m,n)
        else:
            m=int((order)/2)
            n=-int((order)/2)
            print('lmax: ',l_max)
            Gret=calcRetardedFromInv_a(omegas[v],l_max,Gamma,V[v],Om[v],m,n)
            Gk=calcKeldysh_a(omegas[v],l_max,Delta,Gamma,V[v],Om[v],m,n)

        if order > 0:
            l_style='dashed'
        else:
            l_style='-'
        
        #axs[0,v].plot(omegas,Gr.real,label=f'order={order}',color=colors[order+order_max-1],linestyle=l_style,linewidth=2)
        
        #axs[0,v].plot(omegas[v],Gret.imag,label=f'analytic {order}',color=colors_a[order],linewidth=1)
        #axs[1,v].plot(omegas[v],Gk.imag,label=f'analytic {order}',color=colors_a[order],linewidth=1)
        #axs[2,v].plot(omegas[v],Gret.real,label=f'analytic {order}',color=colors_a[order],linewidth=1)
        if order>=-5:
            
            #valid_indices = np.where((omegasF >= omegas[0]) & (omegasF <= omegas[-1]))[0]
            
            GrFile=wigner_dic['0 0']['retarted'][order+order_max]
            GkFile=wigner_dic['0 0']['keldysh'][order+order_max]

            GrFile=GrFile[valid_indices[0]:valid_indices[-1]+1]
            GkFile=GkFile[valid_indices[0]:valid_indices[-1]+1]
            print(len(omegasF),len(GrFile))
            axs[0,v].plot(omegasF,np.imag(GrFile),linewidth=1,color=colors_n[order],label=f'numeric {order}',linestyle='dashed')
            axs[1,v].plot(omegasF,np.imag(GkFile),linewidth=1,color=colors_n[order],label=f'numeric {order}',linestyle='dashed')
            #axs[2,v].plot(omegasF,(GrFile*np.conj(GrFile)-Gret*np.conj(Gret))/(Gret*np.conj(Gret)),linewidth=1,color=colors_n[order],label=f'numeric {order}',linestyle='dashed')
            #axs[2,v].plot(omegasF,np.imag(GkFile)-Gk.imag,linewidth=1,color=colors_a[order],label=f'numeric {order}')
            if order == 0:
                axs[1,v].label='numeric'

            print('order: ',order)
            print(sum(abs(omegas[v]-omegasF)))
            #print(len(omegas))
            print('retarded imag: ',np.sqrt(np.trapz(abs((Gret.imag-GrFile.imag)**2),omegas[v]))/np.trapz(abs(Gret.imag),omegas[v]))
            Gr_error[v]+=np.sqrt(np.trapz(abs((Gret.imag-GrFile.imag)**2),omegas[v]))/np.trapz(abs(Gret.imag),omegas[v])
            print('retarded real: ',np.sqrt(np.trapz(abs((Gret.real-GrFile.real)**2),omegas[v]))/np.trapz(abs(Gret.real),omegas[v]))
            
            print('keldysh: ',np.sqrt(np.trapz(abs((Gk.imag-GkFile.imag)**2),omegas[v]))/np.trapz(abs(Gk.imag),omegas[v]))
            Gk_error[v]+=np.sqrt(np.trapz(abs((Gk.imag-GkFile.imag)**2),omegas[v]))/np.trapz(abs(Gk.imag),omegas[v])
    #axs[v,0].legend()
    #axs[v,1].legend()
    #axs[v,2].legend()
    print('finished',v)
print(Gr_error)
print(Gk_error)
axs[0,0].set_ylabel('$Im(G_r)$',fontsize=8)
axs[1,0].set_ylabel('$Im(G_k)$',fontsize=8)
axs[1,0].set_xlabel('$\omega$',fontsize=8)
axs[1,1].set_xlabel('$\omega$',fontsize=8)
#axs[2,2].set_xlabel('$\omega$')
#plt.xticks(fontsize=8)
axs[1, 0].tick_params(axis='x', labelsize=8)
axs[1, 1].tick_params(axis='x', labelsize=8)
axs[0, 0].tick_params(axis='y', labelsize=8)
axs[1, 0].tick_params(axis='y', labelsize=8)

#plt.yticks(fontsize=14)
handles, labels = axs[1,0].get_legend_handles_labels()
print(handles)
#fig.subplots_adjust(bottom=0.5) 
fig.legend(handles,labels,loc="center", bbox_to_anchor=(0.55, 0.1),ncol=3,fontsize=8)
#fig.subplots_adjust(bottom=-2)

plt.tight_layout()
plt.show()
