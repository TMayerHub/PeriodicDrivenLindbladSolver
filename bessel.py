import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from class_structure.FloquetSpace import calculateWignerFromFile
import os

# Set the working directory to the parent directory where 'class_structure' exists
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Navigate up one level to the project_directory
print('current place')
# Print the current working directory for debugging
print("Current directory:", os.getcwd())

# Define the orders of Bessel functions to plot
orders = np.arange(0, 8)  # n = 0 to 9
zs = [0, 0.5, 1, 2, 3, 6]  # Six different z values
colors = ['mediumblue', 'mediumblue', 'mediumblue', 'mediumblue', 'mediumblue', 'mediumblue']  # Unique colors

fig, axes = plt.subplots(2, 3, figsize=(6.9, 3.4), sharex=True, sharey=True)  # 2 rows, 3 columns

for i, (ax, z, c) in enumerate(zip(axes.flatten(), zs, colors)):
    J_values = [jv(n, z) for n in orders]
    
    stems = ax.stem(orders, J_values, linefmt=c, markerfmt='o', basefmt=" ")

    # Set the marker color separately
    stems[0].set_markerfacecolor(c)  # Set the marker (tip) color
    stems[0].set_markeredgecolor(c)  # Set the marker border color (optional)

    # Set the line color (the stems)
    stems[1].set_color(c)  # Set the stem (line) color
    # Set x-label only for the bottom-most axes
    if i // axes.shape[1] == axes.shape[0] - 1:
        ax.set_xlabel("Order $n$")

    # Set y-label only for the left-most axes
    if i % axes.shape[1] == 0:
        ax.set_ylabel(r"$J_n(z)$")
    
    # Set x-ticks to be integers
    ax.set_xticks(np.arange(min(orders), max(orders) + 1, 1))  # Integer ticks for x-axis
    ax.grid(True)
    from matplotlib.lines import Line2D
    empty_handle = Line2D([0], [0], color='none')
    ax.legend([empty_handle], [r'$J_{n}(' + str(z) + ')$'], frameon=False)

# Adjust layout for better spacing
plt.tight_layout()  # Leaves space for the main title
plt.show(block=False)
plt.savefig('bessel.pdf')

Om=1
V=1
eps=0
Gamma=0.1
omegas=np.linspace(-2.5,2.5,500)

Gr = np.zeros(len(omegas), dtype=complex)+0j

plt.figure()
plt.title('Gr')
plt.grid()

# Loop over n and store valid omega segments
for n in range(-4, 5):
    valid_omegas = []
    valid_Gr_imag = []

    for i, om in enumerate(omegas):
        if (-Om/2 + n*Om) <= om <= (Om/2 + n*Om):  # Check condition
            temp_sum = 0  
            for l in range(-3,4,1):
                temp_sum += 1/(om -n*Om + l*Om + eps + 1j*Gamma) * jv(l-n, V/Om) * jv(n-l, V/Om)*np.exp(-1j*np.pi*(l-n))
            Gr[i] += temp_sum

            # Store values for plotting
            valid_omegas.append(om)
            valid_Gr_imag.append(Gr[i].imag)
    
    # Plot the continuous segments where the condition holds
    if valid_omegas:
        plt.plot(valid_omegas, valid_Gr_imag, label=f'n={n}')  

plt.xlabel('Frequency (ω)')
plt.ylabel('Imaginary part of Gr')
plt.legend()
plt.show(block=False)

# Create figure and subplots
colors = [
     "navy", "navy", "#a65628", "#a6cee3",
    "#984ea3", "#f781bf", "#999999", "navy", "navy",
]
l_values = list(range(-4, 5))  # [-5, -4, ..., 5]

# Create figure and subplots (2 rows × 2 columns)
fig, axs = plt.subplots(2, 2, figsize=(6.8, 3.4), sharex='col', sharey='row')

# First subplot (top-right): Total sum of Gr (Real part)
Gr = np.zeros(len(omegas), dtype=complex)
for i, om in enumerate(omegas):
    temp_sum = 0  
    for l in range(-10, 11, 1):
        temp_sum += 1/(om + l*Om + eps + 1j*Gamma) * jv(-l, V/Om) * jv(-l, V/Om)
    Gr[i] += temp_sum

axs[0, 1].plot(omegas, Gr.real, label="Total $Re(G_r)$", color="navy", linewidth=1)
axs[0, 1].legend(loc='lower right',fontsize=6)
axs[0, 1].grid()

# Third subplot (bottom-right): Total sum of Gr (Imaginary part)
axs[1, 1].plot(omegas, Gr.imag, label="Total $Im(G_r)$", color="navy", linewidth=1)
axs[1, 1].set_xlabel('$\omega$')
axs[1, 1].legend(fontsize=6,loc='lower right')
axs[1, 1].grid()

# Second subplot (top-left): Individual contributions (Real part)
for i, l in enumerate(l_values):
    Gr = np.zeros(len(omegas), dtype=complex)
    for j, om in enumerate(omegas):
        Gr[j] = 1/(om + l*Om + eps + 1j*Gamma) * jv(-l, V/Om) * jv(-l, V/Om)
    axs[0, 0].plot(omegas, Gr.real, label=f"$l={l}$", color=colors[i])

axs[0, 0].grid()
axs[0, 0].set_ylabel('$Re(G_r)$')

# Fourth subplot (bottom-left): Individual contributions (Imaginary part)
for i, l in enumerate(l_values):
    Gr = np.zeros(len(omegas), dtype=complex)
    for j, om in enumerate(omegas):
        Gr[j] = 1/(om + l*Om + eps + 1j*Gamma) * jv(-l, V/Om) * jv(-l, V/Om)
    axs[1, 0].plot(omegas, Gr.imag, label=f"$l={l}$", color=colors[i])

axs[1, 0].set_xlabel('$\omega$')
axs[1, 0].set_ylabel('$Im(G_r)$')
axs[1, 0].grid()
axs[1, 0].legend(fontsize=6, loc='lower right', ncol=2)

# Adjust layout
plt.tight_layout()
plt.show(block=False)

#Keldysh component G00

l_max=5
i_max=4
Delta=-0.02
Gk=0
n=0
m=0
colors = [
     "navy", "navy",  "#999999","#a6cee3",
    "#984ea3", "#f781bf", "#a65628", "navy", "navy",
]
fig, axs = plt.subplots(1,2, figsize=(6.8, 2.4))
for i in range(-i_max, i_max+1, 1):
    Gk_i=np.zeros(len(omegas), dtype=complex)
    for k, om in enumerate(omegas):
        temp_sum=0
        temp_sum1 = 0  
        for l in range(-l_max, l_max+1, 1):
            temp_sum1 += 1/(om + l*Om + eps + 1j*Gamma) * jv(m-l, V/Om) * jv(i-l, V/Om)
        temp_sum2 = 0  
        for l in range(-l_max, l_max+1, 1):
            temp_sum2 += 1/(om + l*Om + eps + 1j*Gamma) * jv(n-l, V/Om) * jv(i-l, V/Om)
        temp_sum+=temp_sum1*np.conj(temp_sum2)
        #plt.plot(-2*1j*Delta*temp_sum.real)
        Gk_i[k] += 2*1j*Delta*temp_sum
    Gk+=Gk_i
    axs[0].plot(omegas,Gk_i.imag,color=colors[i+i_max],label=f"$i={i}$")

axs[1].plot(omegas,Gk.imag,color='navy',label="Total $Im(G_k)$")

axs[1].set_xlabel('$\omega$')
axs[0].set_xlabel('$\omega$')
axs[0].set_ylabel('$Im(G_k)$')
axs[0].legend(fontsize=6, loc='lower right', ncol=2)
axs[1].legend(fontsize=6,loc='lower right')
axs[0].grid()
axs[1].grid()
plt.tight_layout()
plt.show(block=False)

def calcKeldysh_a(omegas,l_max,Delta,Gamma,V,Om,m,n):
    Gk=0
    i_max=2*l_max
    for i in range(-i_max, i_max+1, 1):
        Gk_i=np.zeros(len(omegas), dtype=complex)
        for k, om in enumerate(omegas):
            temp_sum=0
            temp_sum1 = 0  
            for l in range(-l_max, l_max+1, 1):
                temp_sum1 += 1/(om + l*Om + eps + 1j*Gamma) * jv(m-l, V/Om) * jv(i-l, V/Om)
            temp_sum2 = 0  
            for l in range(-l_max, l_max+1, 1):
                temp_sum2 += 1/(om + l*Om + eps + 1j*Gamma) * jv(n-l, V/Om) * jv(i-l, V/Om)
            temp_sum+=temp_sum1*np.conj(temp_sum2)
            Gk_i[k] += 2*1j*Delta*temp_sum
        Gk+=Gk_i
    return Gk

def calcRetarded_a(omegas,l_max,Gamma,V,Om,m,n):
    Gr = np.zeros(len(omegas), dtype=complex)
    for i, om in enumerate(omegas):
        temp_sum = 0  
        for l in range(-l_max, l_max+1, 1):
            temp_sum += 1/(om + l*Om + eps + 1j*Gamma) * jv(m-l, V/Om) * jv(n-l, V/Om)
        Gr[i] += temp_sum
    return Gr

V=[1,2]
Om=[1,1]
#file1='class_structure/results/U0V1Om1_20250212-092450.json'
file1 = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'class_structure', 'results', 'U0V1Om1_20250312-163032.json')
file2 = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'class_structure', 'results', 'U0V2Om1_20250312-163713.json')
#file1 = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'class_structure', 'results', 'U0V1Om1_20250313-113654.json')
#file2 = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'class_structure', 'results', 'U0V2Om1_20250313-111413.json')
print('printing filename')
print(file1)
files=[file1,file2]
#sites,omegasF,wigner_dic=calculateWignerFromFile(file1,0,['retarted','lesser','greater','keldysh'],['0 0'])
omegas=np.linspace(-3,3,800)
#valid_indices = np.where((omegasF >= omegas[0]) & (omegasF <= omegas[-1]))[0]
Gamma=0.1
Delta=0.02
order_max=3
l_max=8
fig, axs = plt.subplots(2,len(V), figsize=(2.1*len(V), 1*2),sharex=True, sharey='row')
colors=['#a65628','#f781bf','#a6cee3','#984ea3','#c97b4a']
colors_a=['#a6cee3','#f781bf','#c97b4a']
colors_n=['navy','#984ea3','#a65628']
sites,omegasF,wigner_dic=calculateWignerFromFile(file2,0,['retarted','lesser','greater','keldysh'],['0 0'])
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
