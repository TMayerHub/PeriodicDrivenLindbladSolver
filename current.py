import Fit.simple_fit_wrapper as fit
import numpy as np
import os
from Fit.utils import KK
import matplotlib.pyplot as plt
import time
import json
from class_structure.GreensFunction_sites import  calculateGreensFunction as gf_solver
from class_structure.FloquetSpace import  calculateWignerFromFile,calculateFloquetFromFile

#matrices,chi,del_aux, del_phys=fit.get_parameters(1/20,1.5,-1.5,20,1,fit.flat_delta,5,plot=True,return_phys=True)
#print('chi',chi)
def Wigner0ToMatrix(om,omegas,wigner0,Om):
    #-1 to ensure that all regions are included completely
    i_om=np.abs(omegas - om).argmin()
    om_num=omegas[i_om]
    if Om==0:
        Om=1
    n_max=int(omegas[-1]/Om-1)

    if -Om/2 >= omegas[i_om] or omegas[i_om] > Om/2:
        raise ValueError(f"""The nearest omega to {om}, {omegas[i_om]} is outside the 
                         allowed intervall for the floquet matrix ({-Om/2},{Om/2}].""")
    
    diag_floq=np.zeros(2*n_max+1,dtype=complex)
    for n in range(-n_max,n_max+1):
        i_om=np.abs(omegas - (om_num+n*Om)).argmin()
        #print(om_num+n*Om)
        #print(omegas[i_om])
        diag_floq[n+n_max]=wigner0[i_om]
    
    return om_num,np.diag(diag_floq)

def InvNonIntImpurity(om,n_max,V,Om,eps):
    Gr_inv_diag=np.zeros(2*n_max+1,dtype=complex)
    for n in range(-n_max,n_max+1):
        Gr_inv_diag[n+n_max]=om+n*Om -eps

    Gr_inv_off=np.ones(len(Gr_inv_diag)-1)*V/2
    Gr_inv=np.diag(Gr_inv_diag,k=0)+np.diag(Gr_inv_off,k=1)+np.diag(Gr_inv_off,k=-1)

    return Gr_inv

def G0FromDyson(del_aux,V,Om,eps=0):
    omegas=del_aux[:,0]
    if Om==0:
        Om=1
    omegas_floq=omegas[np.logical_and(-Om/2<omegas,omegas <=Om/2)]
    _,Del_shape=Wigner0ToMatrix(0,omegas,del_aux[:,2],Om)
    n_max=int((Del_shape.shape[0]-1)/2)
    Delr_imag=del_aux[:,1]
    Delr,_=KK(del_aux[:,0],Delr_imag)

    n = np.arange(-n_max,n_max+1,1)
    omegas_order=[]
    Gr0_wigner0=[]
    Gk0_wigner0=[]
    for om in omegas_floq:
        #om1,Delrfloq_om=Wigner0ToMatrix(om,del_aux[:,0],Delr_real+1j*Delr_imag,Om)
        om1,Delrfloq_om=Wigner0ToMatrix(om,del_aux[:,0],Delr,Om)
        om1,Delkfloqu_om=Wigner0ToMatrix(om,del_aux[:,0],1j*del_aux[:,2],Om)

        Gr0_inv=InvNonIntImpurity(om,n_max,V,Om,eps)-Delrfloq_om
        Gr0_floq = np.linalg.inv(Gr0_inv)
        Gk0_floq = Gr0_floq@Delkfloqu_om@(Gr0_floq.T.conj())

        omegas_order.extend(om+n*Om)
        Gr0_wigner0.extend(np.diag(Gr0_floq))
        Gk0_wigner0.extend(np.diag(Gk0_floq))
        
    return np.array(omegas_order),np.array(Gr0_wigner0),np.array(Gk0_wigner0)
    #return np.array(omegas_order),np.array(Delrfloq_om),np.array(Gk0_wigner0)

def gamma(D: float = 10, beta_fict: float = 2, w= np.linspace(-10, 10, 1001),Gamma=1):
    return (1 - fit.fermi(w, -D, beta_fict)) * fit.fermi(w, D, beta_fict)*Gamma

def calcGreen(eps,hopping,gamma1,gamma2):
    hopping_left=np.conj(hopping)
    E=np.diag(eps,0)+np.diag(hopping,1)+np.diag(hopping_left,-1)
    Gamma_plus=gamma1+gamma2
    Gamma_minus=gamma2-gamma1
    i0=int(np.floor(len(eps)/2))
    i1=int(np.floor(len(eps)/2))
    omegas=np.linspace(-20,20,10001)
    Gr=np.zeros(len(omegas))*1j
    Gk=np.zeros(len(omegas))*1j
    for i in range(len(omegas)):
        omega=np.diag(omegas[i]*np.ones(len(eps)))
        Gr_matrix=np.linalg.inv(omega-E+1j*Gamma_plus)
        Gr[i]=Gr_matrix[i0,i1]
        Gk_matrix=2j*Gr_matrix@Gamma_minus@(np.conj(Gr_matrix).T)
        Gk[i]=Gk_matrix[i0,i1]
    return omegas,Gr,Gk

def current(w,Gr,T=1/20,muL=0,muR=0,D=10,T_fict=0.5,Gamma=1):
    g=gamma(D, 1/T_fict, w= w,Gamma=Gamma)
    return 1j*np.trapezoid((fit.fermi(w,muL,1/T)-fit.fermi(w,muR,1/T))*g*(Gr-np.conj(Gr)),w)/(2*np.pi)

#omegas_order,Gr0_wigner0_phys,Gk0_wigner0_phys=G0FromDyson(del_phys,2,4)

def G0FromSolverSpinSym(Lindblad_params,V,Om,U,fit_params,solver_params=None):
    H = Lindblad_params['hopping matrix']
    Gamma1 = Lindblad_params['ReG1'] + 1j*Lindblad_params['ImG1'] 
    Gamma2 = Lindblad_params['ReG2'] + 1j*Lindblad_params['ImG2']

    epsilon = np.diag(H)
    #append zero since the last site only has one hopping
    T = np.append(np.diag(H,k=1),0)
    parameters = {"length": len(epsilon),
              "epsilon": epsilon,
              "hopping": T,
              "interaction":U,
              "drive": V,
              "frequency":Om,
              "coupling_empty":Gamma1,
              "coupling_full":Gamma2,
              "spin_symmetric":False,
              }
    #print(parameters)
    #solver_params['dirName'] = os.path.join(solver_params['dirName'] , f'V{V}Om{Om}U{U}')
    solver_params['fit_parameters']=fit_params

    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename=f"phi{fit_params['phi']}-{timestr}.json"
    solver_params['file_name']=filename
    GF0=gf_solver(parameters,[[0,0]],'up')
    GF0._GreaterLesserSites([[0,0]],**solver_params)
    
    filepath=os.path.join(solver_params['dirName'],filename)
    sites,omegas_wigner,wigner_dic=calculateWignerFromFile(filepath,0,['retarded','keldysh'],['0 0'])

    return filename,omegas_wigner,wigner_dic


def calcCurrentsTest(T=1/20,t=1,D=15,T_fict=1,V=3,Om=3,sites=3,phis=[0]):

    Gamma=t*np.pi/(2*D)

    chis=[]
    Es_Gr=[]
    Es_Gk=[]
    currents_phys=[]
    currents_aux=[]
    currents_solver=[]
    for phi in phis:
        
        muL=phi/2
        muR=-muL
        fit_params = {'phi':phi,'T':T,'D':D,'T_fict':T_fict,'sites':sites}
        #get_parameters(T=1/20,muL=0,muR=0,D=10,T_fict=0.5,delta=flat_delta,Gamma=1,sites=5,
                    #return_phys=False,plot=False)
        print(Gamma)
        matrices,chi,del_aux, del_phys=fit.get_parameters(T,muL,muR,D,T_fict,fit.flat_delta,Gamma=Gamma,sites=sites,plot=False,return_phys=True)
        print(matrices)
        H = matrices['hopping matrix']
        gamma1 = matrices['ReG1'] + 1j*matrices['ImG1']
        gamma2 = matrices['ReG2'] + 1j*matrices['ImG2']
        eps = np.diag(H)
        #append zero since the last site only has one hopping
        hopping = np.diag(H,k=1)
        omanal,Gr_analytic,Gk_analytic=calcGreen(eps,hopping,gamma1,gamma2)

        omegas_phys,Gr0_wigner0_phys,Gk0_wigner0_phys=G0FromDyson(del_phys,V,Om)
        omegas_aux,Gr0_wigner0_aux,Gk0_wigner0_aux=G0FromDyson(del_aux,V,Om)


        Delr_imag=del_aux[:,1]
        Delr_real,_=KK(del_aux[:,0],Delr_imag)
        Delr=Delr_real+1j*Delr_imag

        sorted_phys = np.argsort(omegas_phys)
        omegas_phys = omegas_phys[sorted_phys]
        Gr0_wigner0_phys = Gr0_wigner0_phys[sorted_phys]
        Gk0_wigner0_phys = Gk0_wigner0_phys[sorted_phys]

        sorted_aux = np.argsort(omegas_aux)
        omegas_aux = omegas_aux[sorted_aux]
        Gr0_wigner0_aux = Gr0_wigner0_aux[sorted_aux]
        Gk0_wigner0_aux = Gk0_wigner0_aux[sorted_aux]
        tag,omegas_wigner,wigner_dic=G0FromSolverSpinSym(matrices,V,Om,0,fit_params,solver_params)

        plt.figure()
        om_reg=15
        plt.title('retarded phi='+str(phi))
        plt.plot(omegas_phys,Gr0_wigner0_phys.imag,color='navy',label=r'$\mathrm{Im}(G_0^R)\ \mathrm{from}\ \Delta_{\mathrm{phys}}$')
        #plt.plot(omegas_phys,Gk0_wigner0_phys.imag,color='blue',label='keldysh')
        plt.plot(omegas_aux,Gr0_wigner0_aux.imag,color='#f781bf',label=r'$\mathrm{Im}(G_0^R)\ \mathrm{from}\ \Delta_{\mathrm{aux}}$')
        #plt.plot(omanal,Gr_analytic.real,color='green',label='analytic from fit',linestyle='dashed')
        ret_wig_imag=wigner_dic['0 0']['retarded'][0].imag
        plt.plot(omegas_wigner[(omegas_wigner>-om_reg )& (omegas_wigner<om_reg)],ret_wig_imag[(omegas_wigner>-om_reg) & (omegas_wigner<om_reg)],color='#984ea3',
                 linestyle='dashed',label=r'$\mathrm{Im}(G_0^R)\ \mathrm{from solver}\ $')
        #plt.show(block=False)
        #plt.plot(omegas_aux,Gk0_wigner0_aux.imag,color='green',label='aux keldysh',linestyle='dashed')
        plt.legend()

        #plt.show(block=False)
        plt.figure()
        plt.title('keldysh phi='+str(phi))
        plt.plot(omegas_phys,Gk0_wigner0_phys.imag,color='navy',label=r'$\mathrm{Im}(G_0^K)\ \mathrm{from}\ \Delta_{\mathrm{phys}}$')
        plt.plot(omegas_aux,Gk0_wigner0_aux.imag,color='#f781bf',label=r'$\mathrm{Im}(G_0^K)\ \mathrm{from}\ \Delta_{\mathrm{aux}}$')
        #plt.plot(omanal,Gk_analytic.imag,color='green',label='analytic from fit',linestyle='dashed')
        kel_wig_imag=wigner_dic['0 0']['keldysh'][0].imag
        plt.plot(omegas_wigner[(omegas_wigner>-om_reg )& (omegas_wigner<om_reg)],kel_wig_imag[(omegas_wigner>-om_reg) & (omegas_wigner<om_reg)],color='#984ea3',
                 linestyle='dashed',label=r'$\mathrm{Im}(G_0^K)\ \mathrm{from solver}\ $')
        #plt.show(block=False)
        #plt.plot(omegas_aux,Gk0_wigner0_aux.imag,color='green',label='aux keldysh',linestyle='dashed')
        plt.legend()

        plt.show(block=False)

        current_aux=current(omegas_aux,Gr0_wigner0_aux,T=T,muL=muL,muR=muR,D=D,T_fict=1,Gamma=Gamma)
        current_phys=current(omegas_phys,Gr0_wigner0_phys,T=T,muL=muL,muR=muR,D=D,T_fict=1,Gamma=Gamma)
        current_solver=current(omegas_wigner,wigner_dic['0 0']['retarded'][0],T=T,muL=muL,muR=muR,D=D,T_fict=1,Gamma=Gamma)
        currents_phys.append(current_phys)
        currents_aux.append(current_aux)
        currents_solver.append(current_solver)

        E_Gr=np.sqrt(np.trapezoid((abs(Gr0_wigner0_phys.imag-Gr0_wigner0_aux.imag))**2,omegas_phys))/np.sqrt(np.trapezoid(abs(Gr0_wigner0_phys.imag)**2,omegas_phys))
        E_Gk=np.sqrt(np.trapezoid((abs(Gk0_wigner0_phys.imag-Gk0_wigner0_aux.imag))**2,omegas_phys))/np.sqrt(np.trapezoid(abs(Gk0_wigner0_phys.imag)**2,omegas_phys))
        chis.append(chi)
        Es_Gr.append(E_Gr)
        Es_Gk.append(E_Gk)
        #chi=chi/Gamma
        print(chi)
        print(E_Gr)
        print(E_Gk)
    currents_phys=np.array(currents_phys)
    currents_aux=np.array(currents_aux)
    currents_solver=np.array(currents_solver)
    fig, ax1 = plt.subplots()
    #plt.show()
    #Plot Gr and Gk on the left axis
    ax1.set_xlabel('phis')
    ax1.set_ylabel('Gr / Gk')
    ax1.plot(phis, Es_Gr, label='Gr',color='orange')
    ax1.plot(phis, Es_Gk, label='Gk',color='blue')
    ax1.plot(phis,abs(currents_phys-currents_aux),label='current')
    ax1.tick_params(axis='y')
    ax1.legend(loc="center left")
    # Create a second y-axis for chi
    ax2 = ax1.twinx()
    ax2.set_ylabel('chi')
    ax2.plot(phis, chis, label='chi',color='red')
    ax2.tick_params(axis='y')
    #plt.legend()
    ax2.legend(loc="center right")
    # Set title
    plt.title('Errors')

    # Show plot
    plt.show(block=False)

    plt.figure()
    plt.title('currents')
    plt.plot(phis,currents_phys.real,label='current phys')
    plt.plot(phis,currents_aux.real,label='current aux')
    plt.plot(phis,currents_solver.real,label='current solver')
    #plt.plot(phis,currents_phys.imag,label='current phys imag')
    #plt.plot(phis,currents_aux.imag,label='current aux imag')
    plt.legend()
    plt.show()

def save_all(filename, phis,currents_aux, currents_phys, currents_solver, E_Gr, E_Gk,chi,tags):
    with open(filename, 'w') as f:
        json.dump({
            "phis": phis,
            "currents_aux": currents_aux,
            "currents_phys": currents_phys,
            "currents_solver": currents_solver,
            "E_Gr": E_Gr,
            "E_Gk": E_Gk,
            "chis": chi,
            "tags": tags
        }, f)

def load_all(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return (data['phis'],data["currents_aux"], data["currents_phys"], data["currents_solver"], 
            data['E_Gr'],data['E_Gk'],data['chis'],data["tags"])
    else:
        return [],[],[],[],[],[],[],[]
    

def calcCurrents(phis=[0,1],V=3,Om=3,U=0,sites=3,T=1/20,t=1,D=15,T_fict=1,solver_params=None):
    dirName = os.path.join(solver_params['dirName'] , f'V{V}Om{Om}U{U}')
    solver_params['dirName']=dirName
    timestr = time.strftime("%Y%m%d-%H%M%S")
    #filename=f"currents-{timestr}.json"
    filename=f"currents.json"
    filepath=os.path.join(dirName,filename)
    print(filepath)
    Gamma=t*np.pi/(2*D)
    
    phis_file,currents_aux, currents_phys, currents_solver, Es_Gr, Es_Gk,chis,tags = load_all(filepath)
    for phi in phis:
        
        muL=phi/2
        muR=-muL
        fit_params = {'phi':phi,'T':T,'D':D,'T_fict':T_fict,'sites':sites}
        #print(Gamma)
        matrices,chi,del_aux, del_phys=fit.get_parameters(T,muL,muR,D,T_fict,fit.flat_delta,Gamma=Gamma,sites=sites,plot=False,return_phys=True)
        #print(matrices)

        omegas_phys,Gr0_wigner0_phys,Gk0_wigner0_phys=G0FromDyson(del_phys,V,Om)
        omegas_aux,Gr0_wigner0_aux,Gk0_wigner0_aux=G0FromDyson(del_aux,V,Om)

        sorted_phys = np.argsort(omegas_phys)
        omegas_phys = omegas_phys[sorted_phys]
        Gr0_wigner0_phys = Gr0_wigner0_phys[sorted_phys]
        Gk0_wigner0_phys = Gk0_wigner0_phys[sorted_phys]

        sorted_aux = np.argsort(omegas_aux)
        omegas_aux = omegas_aux[sorted_aux]
        Gr0_wigner0_aux = Gr0_wigner0_aux[sorted_aux]
        Gk0_wigner0_aux = Gk0_wigner0_aux[sorted_aux]

        tag,omegas_wigner,wigner_dic=G0FromSolverSpinSym(matrices,V,Om,U,fit_params,solver_params)

        current_aux=current(omegas_aux,Gr0_wigner0_aux,T=T,muL=muL,muR=muR,D=D,T_fict=1,Gamma=Gamma)
        current_phys=current(omegas_phys,Gr0_wigner0_phys,T=T,muL=muL,muR=muR,D=D,T_fict=1,Gamma=Gamma)
        current_solver=current(omegas_wigner,wigner_dic['0 0']['retarded'][0],T=T,muL=muL,muR=muR,D=D,T_fict=1,Gamma=Gamma)
        currents_phys.append(current_phys.real)
        currents_aux.append(current_aux.real)
        currents_solver.append(current_solver.real)
        tags.append(tag)

        E_Gr=np.sqrt(np.trapezoid((abs(Gr0_wigner0_phys-Gr0_wigner0_aux))**2,omegas_phys))/np.sqrt(np.trapezoid(abs(Gr0_wigner0_phys)**2,omegas_phys))
        E_Gk=np.sqrt(np.trapezoid((abs(Gk0_wigner0_phys.imag-Gk0_wigner0_aux.imag))**2,omegas_phys))/np.sqrt(np.trapezoid(abs(Gk0_wigner0_phys.imag)**2,omegas_phys))
        chis.append(float(chi))
        Es_Gr.append(E_Gr)
        Es_Gk.append(E_Gk)
        phis_file.append(phi)
        save_all(filepath, phis_file,currents_aux, currents_phys, currents_solver, Es_Gr, Es_Gk,chis,tags)
    return filepath


solver_params={'dt':0.05,'eps':1e-8,'max_iter': 100,
                'av_periods':4,'tf':1.5e1,'t_step':1.5e1,'av_Tau':5,'writeFile':True,
                'dirName':'current_results'}

filepath=calcCurrents(phis=[1.5,6,7],V=3,Om=3,U=0,sites=3,T=1/20,t=1,D=15,T_fict=1,solver_params=solver_params)
#calcCurrentsTest(T=1/20,t=1,D=15,T_fict=1,V=3,Om=3,sites=3,phis=[0])
calcCurrentsTest(T=1/20,t=1,D=15,T_fict=1,V=3,Om=3,sites=3,phis=[0])
phis,currents_aux, currents_phys, currents_solver, Es_Gr, Es_Gk,chis,tags = load_all(filepath)

plt.figure()
plt.plot(phis,currents_phys,color='navy',label=r'$\Delta_{\mathrm{phys}}$')
plt.plot(phis,currents_aux,color='#f781bf',label=r'$\Delta_{\mathrm{aux}}$')
plt.plot(phis,currents_aux,color='#984ea3',linestyle='dashed',label=r'$\mathrm{solver}$')
plt.label()

plt.figure()
plt.plot(phis,chis,label='chi',color='navy')
plt.plot(phis,Es_Gr,label='Error Gr',color='#984ea3')
plt.plot(phis,Es_Gk,label='Error Gk',color='#a65628')
plt.label()

plt.show()