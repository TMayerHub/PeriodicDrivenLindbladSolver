import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

omega_max=2.5
omega_min=-2.5

Om=1
V=1
eps=0
Gamma=0.5
n_max=20

om=omega_min
dom=0.01
omegas=[]
Gr=[]
n=-n_max
om_count=0
while om<omega_max:
    if not(n==(int(np.ceil((om-Om/2)/Om)))):
        n=int(np.ceil((om-Om/2)/Om))
        print('n',n)
    #if om >= Om*n+Om/2:
        #n+=1
        #print(n)

    Gr_inv_diag=[]
    #print(om-n*Om)
    for i in range(-n_max,n_max+1):
        #om-n*Om used to shift Om into the valid region
        Gr_inv_diag.append(om-n*Om +i*Om +eps+1j*Gamma)
        #Gr_inv_diag.append(om +i*Om +eps+1j*Gamma)
    
    Gr_inv_diag=np.array(Gr_inv_diag)
    Gr_inv_off=np.ones(len(Gr_inv_diag)-1)*V/2
    Gr_inv=np.diag(Gr_inv_diag,k=0)+np.diag(Gr_inv_off,k=1)+np.diag(Gr_inv_off,k=-1)

    #print(Gr_inv.shape)
    Gr_mat=np.linalg.inv(Gr_inv)

    #if om==0.5:
        #print(om)
        #df = pd.DataFrame(Gr_mat)
        #print(df)
    
    j=n+n_max
    Gr.append(Gr_mat[j,j])
    #Gr.append(Gr_mat[n_max,n_max])
    omegas.append(om)
    om_count+=1
    om=omega_min+om_count*dom
Gr=np.array(Gr)
plt.figure()
plt.plot(omegas,Gr.imag,marker='x')
plt.show()