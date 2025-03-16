import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from class_structure.FloquetSpace import calculateWignerFromFile
import os

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Navigate up one level to the project_directory
print('current place')
# Print the current working directory for debugging
print("Current directory:", os.getcwd())
file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'class_structure', 'results', 'U0V1Om1_20250206-111652.json')
sites,omegasF,wigner_dic=calculateWignerFromFile(file,0,['retarted','lesser','greater','keldysh'],['0 0'])
#valid_indices = np.where((omegasF >= omegas[0]) & (omegasF <= omegas[-1]))[0]
            
omegas=np.linspace(-5,5,800)

plt.figure()
GrFile=wigner_dic['0 0']['retarted'][0]
GkFile=wigner_dic['0 0']['keldysh'][0]
valid_indices = np.where((omegasF >= omegas[0]) & (omegasF <= omegas[-1]))[0]
omegasF=omegasF[valid_indices[0]:valid_indices[-1]]
GrFile=GrFile[valid_indices[0]:valid_indices[-1]]
GkFile=GkFile[valid_indices[0]:valid_indices[-1]]
plt.plot(omegasF,np.imag(GrFile),linewidth=1,color='k',label=f'numeric 0')
#axs[1,v].plot(omegasF,np.imag(GkFile),linewidth=1,color=colors_n[order],label=f'numeric {order}',linestyle='dashed')
plt.show(block=False)
V=[1,2]
Om=[1,1]
tl=0.2
tr=0.2
eps0=0
epsl=-1
epsr=1
Gamma=1
Delta=0.2
Gammar=1
Gammal=1
Deltar=0.2
Deltal=-0.2
Delta=0.02
eta=0


#file1='class_structure/results/U0V1Om1_20250212-092450.json'
file1 = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'class_structure', 'results', 'U0V1Om1_20250206-111652.json')
file2 = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'class_structure', 'results', 'U0V2Om1_20250312-163713.json')
#file1 = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'class_structure', 'results', 'U0V1Om1_20250313-113654.json')
#file2 = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'class_structure', 'results', 'U0V2Om1_20250313-111413.json')
print('printing filename')
print(file1)
files=[file1,file1]
#sites,omegasF,wigner_dic=calculateWignerFromFile(file1,0,['retarted','lesser','greater','keldysh'],['0 0'])

#valid_indices = np.where((omegasF >= omegas[0]) & (omegasF <= omegas[-1]))[0]
eps=0
order_max=1
l_max=3

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
        temp_sum=0
        temp_sum1 = 0  
        for l in range(-l_max, l_max+1, 1):
            Gr=calcRetarded_a([om],l_max,Gamma,V,Om,m,l)[0]
            Ga=np.conj(calcRetarded_a([om],l_max,Gamma,V,Om,n,l))[0]
            Gleft=1j*2*(Deltal)/((om-epsl-l*Om)**2+Gammal**2)
            Gright=1j*2*(Deltar)/((om-epsr-l*Om)**2+Gammar**2)
            Gk_l[k,l]=Gr *  Ga* (tl**2*Gleft+tr**2*Gright)
            temp_sum1 += Gk_l[k,l]
        #print(temp_sum1)
            
        Gk[k] += temp_sum1
    plt.figure()
    plt.title('keldysh')
    for l in range(2*l_max+1):
        plt.plot(omegas,Gk_l[:,l].imag,label='l')
    plt.show(block=False)
    return Gk

def calcRetarded_a(omegas,l_max,Gamma,V,Om,m,n):
    Gret = np.zeros(len(omegas), dtype=complex)
    for i, om in enumerate(omegas):
        temp_sum = 0  
        for l in range(-l_max, l_max+1, 1):
            g0_til_inv=(om + l*Om - eps0+1j*eta)
            Gr=(om+ l*Om-epsl)/(om + l*Om - epsr + 1j*Gammar)
            Gl=(om+ l*Om-epsr)/(om + l*Om - epsl + 1j*Gammal)
            temp_sum += (g0_til_inv-tr**2*Gr-tl**2*Gl)**(-1) * jv(m-l, V/Om) * jv(n-l, V/Om)
        Gret[i] += temp_sum
    return Gret


fig, axs = plt.subplots(2,len(V), figsize=(2.1*len(V), 1*2),sharex=True, sharey='row')
colors=['#a65628','#f781bf','#a6cee3','#984ea3','#c97b4a']
colors_a=['#a6cee3','#f781bf','#c97b4a']
colors_n=['navy','#984ea3','#a65628']
sites,omegasF,wigner_dic=calculateWignerFromFile(file1,0,['retarted','lesser','greater','keldysh'],['0 0'])
print(omegasF[0],omegasF[-1])
valid_indices = np.where((omegasF >= omegas[0]) & (omegasF <= omegas[-1]))[0]
omegasF=omegasF[valid_indices[0]:valid_indices[-1]]
omegas=omegasF.copy()
print(len(omegas))
Gr_error=np.zeros(len(V))
Gk_error=np.zeros(len(V))
for v in range(len(V)):
    for order in range(0,order_max):
        if order%2:
            m=int((order+1)/2)
            n=-int((order-1)/2)
            Gr=calcRetarded_a(omegas-Om[v]/2,l_max,Gamma,V[v],Om[v],m,n)
            Gk=calcKeldysh_a(omegas-Om[v]/2,l_max,Delta,Gamma,V[v],Om[v],m,n)
        else:
            m=int((order)/2)
            n=-int((order)/2)
            Gr=calcRetarded_a(omegas,l_max,Gamma,V[v],Om[v],m,n)
            Gk=calcKeldysh_a(omegas,l_max,Delta,Gamma,V[v],Om[v],m,n)

        if order > 0:
            l_style='dashed'
        else:
            l_style='-'
        
        #axs[0,v].plot(omegas,Gr.real,label=f'order={order}',color=colors[order+order_max-1],linestyle=l_style,linewidth=2)
        axs[0,v].plot(omegas,Gr.imag,label=f'analytic {order}',color=colors_a[order],linewidth=1)
        axs[1,v].plot(omegas,Gk.imag,label=f'analytic {order}',color=colors_a[order],linewidth=1)
        if order>=0:
            sites,omegasF,wigner_dic=calculateWignerFromFile(files[v],order,['retarted','lesser','greater','keldysh'],['0 0'])
            #valid_indices = np.where((omegasF >= omegas[0]) & (omegasF <= omegas[-1]))[0]
            
            GrFile=wigner_dic['0 0']['retarted'][-1]
            GkFile=wigner_dic['0 0']['keldysh'][-1]
            omegasF=omegasF[valid_indices[0]:valid_indices[-1]]
            GrFile=GrFile[valid_indices[0]:valid_indices[-1]]
            GkFile=GkFile[valid_indices[0]:valid_indices[-1]]
            print(len(omegasF),len(GrFile))
            axs[0,v].plot(omegasF,np.imag(GrFile),linewidth=1,color=colors_n[order],label=f'numeric {order}',linestyle='dashed')
            axs[1,v].plot(omegasF,np.imag(GkFile),linewidth=1,color=colors_n[order],label=f'numeric {order}',linestyle='dashed')
            if order == 0:
                axs[1,v].label='numeric'

            print('order: ',order)
            print(sum(abs(omegas-omegasF)))
            print(len(omegas))
            print('retarded: ',np.sqrt(np.trapz(abs((Gr.imag-GrFile.imag)**2),omegas))/np.trapz(abs(Gr.imag)))
            Gr_error[v]+=np.sqrt(np.trapz(abs((Gr.imag-GrFile.imag)**2),omegas))
            print('keldysh: ',np.sqrt(np.trapz(abs((Gk.imag-GkFile.imag)**2),omegas))/np.trapz(abs(Gk.imag)))
            Gk_error[v]+=np.sqrt(np.trapz(abs((Gk.imag-GkFile.imag)**2),omegas))
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
fig.subplots_adjust(bottom=-2)

plt.tight_layout()
plt.show()
