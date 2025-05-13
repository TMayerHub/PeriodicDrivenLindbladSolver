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
def fermi(w: np.ndarray, mu: float, beta: float) -> np.ndarray:
    """This version avoids exponential overflow/underflow"""

    if beta > 1e10:
        return np.heaviside(mu - w, 0.5)

    energy = w - mu
    #mask for troubling region
    pos_exp = beta * energy > 35
    neg_exp = beta * energy < -35

    f = np.zeros_like(energy)

    f[pos_exp] = np.exp(-energy[pos_exp] * beta)
    f[neg_exp] = 1.0
    f[~(pos_exp | neg_exp)] = 1 / (np.exp(energy[~(pos_exp | neg_exp)] * beta) + 1)

    return f

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

def MatrixToWigner0(om_floq,matrix,Om):
    if Om==0:
        Om=1
    n_max=int((matrix.shape[1]-1)/2)

    n = np.arange(-n_max,n_max+1,1)
    omegas_order=[]
    wig=[]
    for i,om in enumerate(om_floq):

        omegas_order.extend(om+n*Om)
        wig.extend(np.diag(matrix[i]))
    #print(omegas_order)
    omegas_order=np.array(omegas_order)
    wig=np.array(wig)
    sorted = np.argsort(omegas_order)
    omegas_order = omegas_order[sorted]
    wig = wig[sorted]
    return np.array(omegas_order),np.array(wig)

def InvNonIntImpurity(om,n_max,V,Om,eps):
    Gr_inv_diag=np.zeros(2*n_max+1,dtype=complex)
    for n in range(-n_max,n_max+1):
        Gr_inv_diag[n+n_max]=om+n*Om -eps

    Gr_inv_off=np.ones(len(Gr_inv_diag)-1)*V/2
    Gr_inv=np.diag(Gr_inv_diag,k=0)+np.diag(Gr_inv_off,k=1)+np.diag(Gr_inv_off,k=-1)

    return Gr_inv
def DelAux(omegas,eps,hopping,gamma1,gamma2):
    omegas,Gr,Gk=calcGreen(eps,hopping,gamma1,gamma2,omegas)
    DeltaR=omegas-1/Gr
    DeltaK=1/Gr*Gk*1/np.conj(Gr)
    return DeltaR,DeltaK

def GetGphysFloq(omegas_floq,GauxR,GauxK,eps,hopping,gamma1,gamma2,V,Om,fit_params):
    T_fict=fit_params['T_fict']
    D=fit_params['D']
    T=fit_params['T']
    phi=fit_params['phi']
    Gamma=1/(2*D)
    #+1 to make sure Wigner0 to Floq returns the right size
    n_max=int((GauxR.shape[1]-1)/2)+1
    #print(GauxR.shape)
    n=np.linspace(-n_max,n_max,2*n_max+1)
    #print(n)
    omegas = (omegas_floq[:, None] + n * Om).flatten()
    #print(omegas)
    DeltaR,DeltaK=DelAux(omegas,eps,hopping,gamma1,gamma2)
    del_aux=np.array([omegas,DeltaR.imag,DeltaK.imag]).T
    del_physr = - (1 - fermi(omegas, -D, 1/T_fict)) * fermi(omegas, D, 1/T_fict)*Gamma
    del_physk = del_physr * (1 - 2 * fermi(omegas, phi/2, 1/T)+1 - 2 * fermi(omegas, -phi/2, 1/T))
    del_phys=np.array([omegas,del_physr,del_physk]).T
    omegas_floq,invG0R_aux,invG0K_aux=invG0FloqFromDyson(del_aux,V,Om)
    omegas_floq,invG0R_phys,invG0K_phys=invG0FloqFromDyson(del_phys,V,Om)
    Gr_phys_floq=[]
    Gk_phys_floq=[]
    for i,om in enumerate(omegas_floq):
        GauxR_inv=np.linalg.inv(GauxR[i])
        SigmaR=invG0R_aux[i]-GauxR_inv
        SigmaK=invG0K_aux[i] + GauxR_inv@GauxK[i]@np.conj(GauxR_inv.T)
        invGR_phys=invG0R_phys[i]-SigmaR
        invGK_phys=invG0K_phys[i]-SigmaK
        
        Gr_phys_floq.append(np.linalg.inv(invGR_phys))
        Gk_phys_floq.append(-Gr_phys_floq[i]@invGK_phys@np.conj(Gr_phys_floq[i].T))

    #print(invG0R_aux)
    #G0_aux=G0FromDyson(del_aux,V,Om,eps=0)
    #G0_phys=G0FromDyson(del_aux,V,Om,eps=0)
    
    return np.array(Gr_phys_floq),np.array(Gk_phys_floq)

def invG0FloqFromDyson(del_aux,V,Om,eps=0):
    omegas=del_aux[:,0]
    #print(omegas)
    if Om==0:
        Om=1
    omegas_floq=omegas[np.logical_and(-Om/2<omegas,omegas <=Om/2)]
    #print(omegas_floq)
    _,Del_shape=Wigner0ToMatrix(0,omegas,del_aux[:,2],Om)
    n_max=int((Del_shape.shape[0]-1)/2)
    Delr_imag=del_aux[:,1]
    Delr,_=KK(del_aux[:,0],Delr_imag)

    n = np.arange(-n_max,n_max+1,1)
    Gr0_floq_inv=[]
    Gk0_floq_inv=[]
    for om in omegas_floq:
        #print(om)
        #om1,Delrfloq_om=Wigner0ToMatrix(om,del_aux[:,0],Delr_real+1j*Delr_imag,Om)
        om1,Delrfloq_om=Wigner0ToMatrix(om,del_aux[:,0],Delr,Om)
        om1,Delkfloqu_om=Wigner0ToMatrix(om,del_aux[:,0],1j*del_aux[:,2],Om)

        Gr0_floq_inv.append(InvNonIntImpurity(om,n_max,V,Om,eps)-Delrfloq_om)
        Gk0_floq_inv.append(-Delkfloqu_om)

    return np.array(omegas_floq),np.array(Gr0_floq_inv),np.array(Gk0_floq_inv)

def G0FloqFromDyson(del_aux,V,Om):
    omegas_floq,Gr0_floq_inv,Gk0_floq_inv=invG0FloqFromDyson(del_aux,V,Om)
    Gr0_floq=[]
    Gk0_floq=[]
    for i,om in enumerate(omegas_floq):
        om1,Delkfloqu_om=Wigner0ToMatrix(om,del_aux[:,0],1j*del_aux[:,2],Om)
        Gr0_floq.append(np.linalg.inv(Gr0_floq_inv[i]))
        Gk0_floq.append(Gr0_floq[i]@Delkfloqu_om@(Gr0_floq[i].T.conj()))
    return np.array(Gr0_floq),np.array(Gk0_floq)

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
    
    omegas_order=np.array(omegas_order)
    Gr0_wigner0=np.array(Gr0_wigner0)
    Gk0_wigner0=np.array(Gk0_wigner0)
    sorted = np.argsort(omegas_order)
    omegas_order = omegas_order[sorted]
    Gr0_wigner0 = Gr0_wigner0[sorted]
    Gk0_wigner0 = Gk0_wigner0[sorted]
    return omegas_order,Gr0_wigner0,Gk0_wigner0
    #return np.array(omegas_order),np.array(Delrfloq_om),np.array(Gk0_wigner0)

def gamma(D: float = 10, beta_fict: float = 2, w= np.linspace(-10, 10, 1001),Gamma=1):
    return (1 - fit.fermi(w, -D, beta_fict)) * fit.fermi(w, D, beta_fict)*Gamma

def calcGreen(eps,hopping,gamma1,gamma2,omegas=np.linspace(-20,20,10001)):
    hopping_left=np.conj(hopping)
    E=np.diag(eps,0)+np.diag(hopping,1)+np.diag(hopping_left,-1)
    Gamma_plus=gamma1+gamma2
    Gamma_minus=gamma2-gamma1
    i0=int(np.floor(len(eps)/2))
    i1=int(np.floor(len(eps)/2))
    #omegas=np.linspace(-20,20,10001)
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
    H = np.array(Lindblad_params['hopping matrix'])
    Gamma1 = np.array(Lindblad_params['ReG1']) + 1j*np.array(Lindblad_params['ImG1'])
    Gamma2 = np.array(Lindblad_params['ReG2']) + 1j*np.array(Lindblad_params['ImG2'])

    epsilon = np.diag(H).copy()
    #print(np.dtype(epsilon))
    center=int(np.floor(len(epsilon)/2))
    epsilon[center]=-U/2
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

def deltaTest(T=1/20,t=1,D=15,T_fict=1/2,sites=3,phis=[0]):
    Om=0
    V=0
    Gamma=t*np.pi/(2*D)
    for phi in phis:
        print(phi)
        muL=phi/2
        muR=-muL
        matrices,chi,del_aux, del_phys=fit.get_parameters(T,muL,muR,D,T_fict,fit.flat_delta,Gamma=Gamma,sites=sites,plot=False,return_phys=True)
        del_auxfit_r,_=KK(del_aux[:,0],del_aux[:,1])
        del_auxfit_k=1j*del_aux[:,2]
        del_physfit_r,_=KK(del_phys[:,0],del_phys[:,1])
        del_physfit_k=1j*del_phys[:,2]
        H = matrices['hopping matrix']
        gamma1 = matrices['ReG1'] + 1j*matrices['ImG1']
        gamma2 = matrices['ReG2'] + 1j*matrices['ImG2']
        eps = np.diag(H)
        hopping = np.diag(H,k=1)
        omegas_del=np.linspace(-20,20,500)
        del_auxr,del_auxk=DelAux(omegas_del,eps,hopping,gamma1,gamma2)
        del_physr = - (1 - fermi(omegas_del, -D, 1/T_fict)) * fermi(omegas_del, D, 1/T_fict)*Gamma*1j
        del_physk = del_physr * (1 - 2 * fermi(omegas_del, muL, 1/T)+1 - 2 * fermi(omegas_del, muR, 1/T))
        #print(Gamma)

        #GetGphys(omegas_del,0,eps,hopping,gamma1,gamma2,V,Om)
        plt.figure()
        plt.title('auxilary')
        plt.plot(del_aux[:,0],del_auxfit_r.imag,label='retarded fit')
        plt.plot(del_aux[:,0],del_auxfit_k.imag,label='keldysh fit')
        plt.plot(omegas_del,del_auxr.imag,linestyle='dashed',label='retarded')
        plt.plot(omegas_del,del_auxk.imag,linestyle='dashed',label='keldysh')
        plt.legend()
        plt.show(block=False)

        plt.figure()
        plt.title('physikal')
        plt.plot(del_phys[:,0],del_physfit_r.imag,label='retarded fit')
        plt.plot(del_phys[:,0],del_physfit_k.imag,label='keldysh fit')
        plt.plot(omegas_del,del_physr.imag,linestyle='dashed',label='retarded')
        #plt.plot(omegas_del,-fermi(-omegas_del, D, 1/T_fict)*(1 - fermi(omegas_del, -D, 1/T_fict)),label='fermi')
        plt.plot(omegas_del,del_physk.imag,linestyle='dashed',label='keldysh')
        plt.legend()
        plt.show()

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

def save_all(filename, phis,currents_aux, currents_phys, currents_solver_aux, currents_solver_phys,E_Gr, E_Gk,chi,tags):
    with open(filename, 'w') as f:
        json.dump({
            "phis": phis,
            "currents_aux": currents_aux,
            "currents_phys": currents_phys,
            "currents_solver_aux": currents_solver_aux,
            "currents_solver_phys": currents_solver_phys,
            "E_Gr": E_Gr,
            "E_Gk": E_Gk,
            "chis": chi,
            "tags": tags
        }, f)

def load_all(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return (data['phis'],data["currents_aux"], data["currents_phys"], data["currents_solver_aux"], 
            data["currents_solver_phys"],data['E_Gr'],data['E_Gk'],data['chis'],data["tags"])
    else:
        return [],[],[],[],[],[],[],[],[]
    
def GphysFromFileTest(filepath):

    with open(filepath, 'r') as file:
        data = json.load(file)
    fit_params=data['input']['fit_parameters']
    parameters=data['input']['parameters']
    eps=np.array(parameters['epsilon'])
    hopping=np.array(parameters['hopping'][:-1])
    gamma1=np.array(parameters['coupling_emptyReal'])+1j*np.array(parameters['coupling_emptyImag'])
    gamma2=np.array(parameters['coupling_fullReal'])+1j*np.array(parameters['coupling_fullImag'])
    V=parameters['drive']
    Om=parameters['frequency']
    T_fict=fit_params['T_fict']
    D=fit_params['D']
    T=fit_params['T']
    phi=fit_params['phi']
    Gamma=1/(2*D)
    n_max=20
    sites,omegas_floq,floq_dic=calculateFloquetFromFile(filepath,[n_max,n_max],['retarded','keldysh'],['0 0'])
    sites,omegas_wig,wig_dic=calculateWignerFromFile(filepath,0,['retarded','keldysh'],['0 0'])
    print('finished floq from file')
    GauxR=floq_dic['0 0']['retarded']
    print(omegas_floq.shape)
    print(GauxR.shape)
    GauxK=floq_dic['0 0']['keldysh']
    GauxR_wigFile=wig_dic['0 0']['retarded'][0]
    GauxK_wigFile=wig_dic['0 0']['keldysh'][0]
    print(GauxR)
    Gr_phys_floq,Gk_phys_floq=GetGphysFloq(omegas_floq,GauxR,GauxK,eps,hopping,gamma1,gamma2,V,Om,fit_params)
    omegas,Gr_wig=MatrixToWigner0(omegas_floq,Gr_phys_floq,Om)
    omegas,Gk_wig=MatrixToWigner0(omegas_floq,Gk_phys_floq,Om)

    omegas_aux,Gr_wig0aux=MatrixToWigner0(omegas_floq,GauxR,Om)
    omegas_aux,Gk_wig0aux=MatrixToWigner0(omegas_floq,GauxK,Om)


    del_physr = - (1 - fermi(omegas, -D, 1/T_fict)) * fermi(omegas, D, 1/T_fict)*Gamma*1j
    del_physk = del_physr * (1 - 2 * fermi(omegas, fit_params['phi']/2, 1/T)+1 - 2 * fermi(omegas, -fit_params['phi']/2, 1/T))
    del_phys=np.array([omegas,del_physr.imag,del_physk.imag]).T
    #omegas_phys,Gr_phys0,Gk_phys0=G0FromDyson(del_phys,V,Om,eps=0)
    Gr_phys0,Gk_phys0=G0FloqFromDyson(del_phys,V,Om)
    omegas_phys,Gr_phys0=MatrixToWigner0(omegas_floq,Gr_phys0,Om)
    omegas_phys,Gk_phys0=MatrixToWigner0(omegas_floq,Gk_phys0,Om)

    start=np.where(omegas < -5)[0][-1]
    end=np.where(omegas > 5)[0][0]
    start_aux=np.where(omegas_aux < -5)[0][-1]
    end_aux=np.where(omegas_aux > 5)[0][0]
    start_phys=np.where(omegas_phys < -5)[0][-1]
    end_phys=np.where(omegas_phys > 5)[0][0]
    start_file=np.where(omegas_wig < -5)[0][-1]
    end_file=np.where(omegas_wig > 5)[0][0]
    print('om diff',omegas[1]-omegas[0])
    plt.figure()
    plt.plot(omegas[start:end],Gr_wig.imag[start:end],label='Gr')
    plt.plot(omegas[start:end],Gk_wig.imag[start:end],label='Gk')
    #plt.plot(omegas_floq,GauxR[:,n_max,n_max].imag,label='Grfloq')
    #plt.plot(omegas_aux[start_aux:end_aux],Gr_wig0aux.imag[start_aux:end_aux],linestyle='dashed',label='floq from file')
    #plt.plot(omegas_aux[start_aux:end_aux],Gk_wig0aux.imag[start_aux:end_aux],linestyle='dashed',label='floq from file')
    plt.plot(omegas_phys[start_phys:end_phys],Gr_phys0.imag[start_phys:end_phys],linestyle='dashed',label='G0phys')
    plt.plot(omegas_phys[start_phys:end_phys],Gk_phys0.imag[start_phys:end_phys],linestyle='dashed',label='G0phys')
    plt.plot(omegas_wig[start_file:end_file],GauxR_wigFile.imag[start_file:end_file],linestyle='dotted',label='wig from file')
    plt.plot(omegas_wig[start_file:end_file],GauxK_wigFile.imag[start_file:end_file],linestyle='dotted',label='wig from file')
    plt.legend()
    plt.show()

def GphysFromFile(filepath):

    with open(filepath, 'r') as file:
        data = json.load(file)
    fit_params=data['input']['fit_parameters']
    parameters=data['input']['parameters']
    eps=np.array(parameters['epsilon'])
    hopping=np.array(parameters['hopping'][:-1])
    gamma1=np.array(parameters['coupling_emptyReal'])+1j*np.array(parameters['coupling_emptyImag'])
    gamma2=np.array(parameters['coupling_fullReal'])+1j*np.array(parameters['coupling_fullImag'])
    V=parameters['drive']
    Om=parameters['frequency']

    n_max=20
    sites,omegas_floq,floq_dic=calculateFloquetFromFile(filepath,[n_max,n_max],['retarded','keldysh'],['0 0'])
    GauxR=floq_dic['0 0']['retarded']
    GauxK=floq_dic['0 0']['keldysh']
    Gr_phys_floq,Gk_phys_floq=GetGphysFloq(omegas_floq,GauxR,GauxK,eps,hopping,gamma1,gamma2,V,Om,fit_params)
    omegas,Gr_wig=MatrixToWigner0(omegas_floq,Gr_phys_floq,Om)
    omegas,Gk_wig=MatrixToWigner0(omegas_floq,Gk_phys_floq,Om)

    start=np.where(omegas < -5)[0][-1]
    end=np.where(omegas > 5)[0][0]
    #plt.figure()
    #plt.plot(omegas[start:end],Gr_wig.imag[start:end],label='Gr')
    #plt.plot(omegas[start:end],Gk_wig.imag[start:end],label='Gk')
    #plt.legend()
    #plt.show()
    return omegas,Gr_wig,Gk_wig

def createFit(filename,phis=[0,1],change_phis=[],V=3,Om=3,sites=3,T=1/10,t=1,D=15,T_fict=1,plot=False):
    Gamma=t*np.pi/(2*D)

    if os.path.exists(filename):
        with open(filename, "r") as f:
            fits_dict = json.load(f)
    else:
        fits_dict = {}
    for key in change_phis:
        if str(key) in fits_dict:
            del fits_dict[str(key)]
    for phi in phis:
        print(phi)
        muL=phi/2
        muR=-muL

        if str(phi) in fits_dict:
            print(f"Warning: Key '{phi}' already exists.")
            continue
        
        if phi>0.6:
            w_L1=muL-0.4
            w_L2=muL+0.3
            w_R1=muR-0.3
            w_R2=muR+0.4
            weights=f"2,{w_L1},{w_L2},{w_R1},{w_R2}"
        else:
            weights=f"50,-0.5,0.5"
        
        print(weights)
        fit_params = {'phi':phi,'T':T,'D':D,'T_fict':T_fict,'sites':sites,'weights':weights}
        #print(Gamma)
        matrices,chi,del_aux, del_phys=fit.get_parameters(T,muL,muR,D,T_fict,fit.flat_delta,Gamma=Gamma,sites=sites,plot=False,return_phys=True,weights=weights)

        matrices_lists = {key: value.tolist() if isinstance(value, np.ndarray) else value
             for key, value in matrices.items()}

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

        if plot:
            print('plotting')
            #print(omegas_aux)
            start_aux=np.where(omegas_aux < -15)[0][-1]
            end_aux=np.where(omegas_aux > 15)[0][0]
            start_phys=np.where(omegas_phys < -15)[0][-1]
            end_phys=np.where(omegas_phys > 15)[0][0]
            plt.figure()
            plt.title(f'non interacting phi = {phi}')
            plt.plot(omegas_phys[start_phys:end_phys],Gr0_wigner0_phys.imag[start_phys:end_phys],label='retarded phys', color = 'navy')
            plt.plot(omegas_phys[start_phys:end_phys],Gk0_wigner0_phys.imag[start_phys:end_phys],label='keldysh phys', color = '#a65628')
            plt.plot(omegas_aux[start_aux:end_aux],Gr0_wigner0_aux.imag[start_aux:end_aux],label='retarded aux',color='#a6cee3',linestyle='dashed')
            plt.plot(omegas_aux[start_aux:end_aux],Gk0_wigner0_aux.imag[start_aux:end_aux],label='keldysh aux',color = '#f78843',linestyle='dashed')
            plt.legend()
            plt.show(block=False)

            plt.figure()
            plt.title('Delta')
            plt.plot(del_phys[:,0],del_phys[:,1],label='retarded phys', color = 'navy')
            plt.plot(del_phys[:,0],del_phys[:,2],label='keldysh phys', color = '#a65628')
            plt.plot(del_aux[:,0],del_aux[:,1],label='retarded aux',color='#a6cee3',linestyle='dashed')
            plt.plot(del_aux[:,0],del_aux[:,2],label='keldysh aux',color = '#f78843',linestyle='dashed')
            plt.legend()
            plt.show()
        
        E_Gr=float(np.sqrt(np.trapezoid((abs(Gr0_wigner0_phys-Gr0_wigner0_aux))**2,omegas_phys))/np.sqrt(np.trapezoid(abs(Gr0_wigner0_phys)**2,omegas_phys)))
        E_Gk=float(np.sqrt(np.trapezoid((abs(Gk0_wigner0_phys.imag-Gk0_wigner0_aux.imag))**2,omegas_phys))/np.sqrt(np.trapezoid(abs(Gk0_wigner0_phys.imag)**2,omegas_phys)))
        Gk_int=float(np.sqrt(np.trapezoid(abs(Gk0_wigner0_phys.imag)**2,omegas_phys)))
        chi=float(chi)
        fit_results={'matrices':matrices_lists,'chi':chi,'E_Gr':E_Gr,'E_Gk':E_Gk,'Gk_int':Gk_int}
        fit_dict = {'fit_parameters': fit_params,'fit_results':fit_results}
        
        print(chi)
        print(type(chi))
        print(E_Gr)
        print(type(E_Gr))
        fits_dict[phi]=fit_dict
        with open(filename, "w") as f:
            json.dump(fits_dict, f, indent=2)
    
    fits_dict = {float(k): v for k, v in fits_dict.items()}

    phi_values = sorted(fits_dict.keys())
    e_gr_values = [fits_dict[phi]["fit_results"]["E_Gr"] for phi in phi_values]
    e_gk_values = [fits_dict[phi]["fit_results"]["E_Gk"] for phi in phi_values]
    chi_values = [fits_dict[phi]["fit_results"]["chi"] for phi in phi_values]
    int_gk_values = [fits_dict[phi]["fit_results"]["Gk_int"] for phi in phi_values]

    plt.plot(phi_values, e_gr_values, marker='o', label='E_Gr',color='#984ea3')
    plt.plot(phi_values, e_gk_values, marker='o', label='E_Gk',color='#a65628')
    plt.plot(phi_values, chi_values, marker='o', label='Chi',color='navy')
    plt.plot(phi_values, int_gk_values, marker='o', label='in gk',color='k')
    plt.xlabel("phi")
    plt.ylabel("Error")
    plt.title("V3Om3")
    plt.grid(True)
    plt.legend()
    plt.show()

def checkFit(filename,phis,V,Om,plot=True):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            fits_dict = json.load(f)
    else:
        print('no such file')
    E_Grs=[]
    E_Gks=[]
    Chis=[]
    Gk_ints=[]
    for phi, fit_dict in fits_dict.items():
        phi=float(phi)
        print(phi)
        if not(phi in phis):
            continue

        fit_params=fit_dict['fit_parameters']
        fit_results=fit_dict['fit_results']
        matrices=fit_results['matrices']
        chi=fit_results['chi']

        T_fict=fit_params['T_fict']
        D=fit_params['D']
        T=fit_params['T']
        muL=phi/2
        muR=-muL
        Gamma=np.pi/(2*D)
        H = np.array(matrices['hopping matrix'])
        gamma1 = np.array(matrices['ReG1']) + 1j*np.array(matrices['ImG1'])
        gamma2 = np.array(matrices['ReG2']) + 1j*np.array(matrices['ImG2'])
        eps = np.diag(H)
        hopping = np.diag(H,k=1)
        omegas_del=np.linspace(-20,20,500)
        del_auxr,del_auxk=DelAux(omegas_del,eps,hopping,gamma1,gamma2)
        del_physr = - (1 - fermi(omegas_del, -D, 1/T_fict)) * fermi(omegas_del, D, 1/T_fict)*Gamma*1j
        del_physk = del_physr * (1 - 2 * fermi(omegas_del, muL, 1/T)+1 - 2 * fermi(omegas_del, muR, 1/T))
        
        del_aux=np.array([omegas_del,del_auxr.imag,del_auxk.imag]).T
        del_phys=np.array([omegas_del,del_physr.imag,del_physk.imag]).T

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

        if plot:
            start_aux=np.where(omegas_aux < -10)[0][-1]
            end_aux=np.where(omegas_aux > 10)[0][0]
            start_phys=np.where(omegas_phys < -10)[0][-1]
            end_phys=np.where(omegas_phys > 10)[0][0]
            plt.figure()
            plt.title(f'non interacting phi = {phi}')
            plt.plot(omegas_phys[start_phys:end_phys],Gr0_wigner0_phys.imag[start_phys:end_phys],label='retarded phys', color = 'navy')
            plt.plot(omegas_phys[start_phys:end_phys],Gk0_wigner0_phys.imag[start_phys:end_phys],label='keldysh phys', color = '#a65628')
            plt.plot(omegas_aux[start_aux:end_aux],Gr0_wigner0_aux.imag[start_aux:end_aux],label='retarded aux',color='#a6cee3',linestyle='dashed')
            plt.plot(omegas_aux[start_aux:end_aux],Gk0_wigner0_aux.imag[start_aux:end_aux],label='keldysh aux',color = '#f78843',linestyle='dashed')
            plt.legend()
            plt.show(block=False)

            plt.figure()
            plt.title('Delta')
            plt.plot(del_phys[:,0],del_phys[:,1],label='retarded phys', color = 'navy')
            plt.plot(del_phys[:,0],del_phys[:,2],label='keldysh phys', color = '#a65628')
            plt.plot(del_aux[:,0],del_aux[:,1],label='retarded aux',color='#a6cee3',linestyle='dashed')
            plt.plot(del_aux[:,0],del_aux[:,2],label='keldysh aux',color = '#f78843',linestyle='dashed')
            plt.legend()
            plt.show()
        
        E_Gr=float(np.sqrt(np.trapezoid((abs(Gr0_wigner0_phys-Gr0_wigner0_aux))**2,omegas_phys))/np.sqrt(np.trapezoid(abs(Gr0_wigner0_phys)**2,omegas_phys)))
        E_Gk=float(np.sqrt(np.trapezoid((abs(Gk0_wigner0_phys.imag-Gk0_wigner0_aux.imag))**2,omegas_phys))/np.sqrt(np.trapezoid(abs(Gk0_wigner0_phys.imag)**2,omegas_phys)))
        Gk_int=float(np.sqrt(np.trapezoid(abs(Gk0_wigner0_phys.imag)**2,omegas_phys)))
        chi=float(chi)

        E_Grs.append(E_Gr)
        E_Gks.append(E_Gk)
        Chis.append(chi)
        Gk_ints.append(Gk_int)
        
    print(phis)
    print(E_Grs)
    print(E_Gks)
    print(Chis)
    print(Gk_ints)
    combined = list(zip(
        phis,
        E_Grs,
        E_Gks,
        Chis,
        Gk_ints
    ))

    # Sort based on the first item in each tuple (phi value)
    combined_sorted = sorted(combined, key=lambda x: x[0])

    # Unzip everything back into separate lists
    (phis,
        E_Grs,
        E_Gks,
        Chis,
        Gk_ints
    ) = zip(*combined_sorted)

    plt.figure()
    plt.plot(phis,Chis,label='chi',color='navy')
    plt.plot(phis,E_Grs,label='Error Gr',color='#984ea3')
    plt.plot(phis,E_Gks,label='Error Gk',color='#a65628')
    plt.plot(phis,Gk_ints,label='Error Gk',color='k')
    plt.xlabel('voltage')
    plt.ylabel('errors')
    plt.legend()
    plt.grid(True)

    plt.show()


def calcCurrents(phis=[0,1],V=3,Om=3,U=0,sites=3,T=1/20,t=1,D=15,T_fict=1,solver_params=None,plot=False):
    dirName = os.path.join(solver_params['dirName'] , f'V{V}Om{Om}U{U}')
    solver_params['dirName']=dirName
    timestr = time.strftime("%Y%m%d-%H%M%S")
    #filename=f"currents-{timestr}.json"
    filename=f"currents.json"
    filepath=os.path.join(dirName,filename)
    #print(filepath)
    Gamma=t*np.pi/(2*D)


    
    phis_file,currents_aux, currents_phys, currents_solver_aux, currents_solver_phys,Es_Gr, Es_Gk,chis,tags = load_all(filepath)
    for phi in phis:
        print(phi)
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

        start_time = time.time()
        tag,omegas_wigner,wigner_dic=G0FromSolverSpinSym(matrices,V,Om,U,fit_params,solver_params)
        omegas_sol_phys,Gr_solver_phys,Gk_solver_phys=GphysFromFile(filepath=os.path.join(dirName,tag))
        end_time = time.time()
        time_elapsed=end_time-start_time
        print(f"Solver time: {time_elapsed/60:.2f} min")

        current_aux=current(omegas_aux,Gr0_wigner0_aux,T=T,muL=muL,muR=muR,D=D,T_fict=1,Gamma=Gamma)
        current_phys=current(omegas_phys,Gr0_wigner0_phys,T=T,muL=muL,muR=muR,D=D,T_fict=1,Gamma=Gamma)
        current_solver_aux=current(omegas_wigner,wigner_dic['0 0']['retarded'][0],T=T,muL=muL,muR=muR,D=D,T_fict=1,Gamma=Gamma)
        current_solver_phys=current(omegas_sol_phys,Gr_solver_phys,T=T,muL=muL,muR=muR,D=D,T_fict=1,Gamma=Gamma)

        if plot:
            start_aux=np.where(omegas_wigner < -10)[0][-1]
            end_aux=np.where(omegas_wigner > 10)[0][0]
            start_phys=np.where(omegas_sol_phys < -10)[0][-1]
            end_phys=np.where(omegas_sol_phys > 10)[0][0]
            plt.figure()
            plt.title(f'solver phi = {phi}')
            gr_plot=wigner_dic['0 0']['retarded'][0]
            gk_plot=wigner_dic['0 0']['keldysh'][0]
            plt.plot(omegas_sol_phys[start_phys:end_phys],Gr_solver_phys.imag[start_phys:end_phys],label='retarded phys',color='navy')
            plt.plot(omegas_sol_phys[start_phys:end_phys],Gk_solver_phys.imag[start_phys:end_phys],label='keldysh phys', color = '#a65628')
            plt.plot(omegas_wigner[start_aux:end_aux],gr_plot.imag[start_aux:end_aux],label='retarded aux',color='#a6cee3',linestyle='dashed')
            plt.plot(omegas_wigner[start_aux:end_aux],gk_plot.imag[start_aux:end_aux],label='keldysh aux',color = '#f78843',linestyle='dashed')

            plt.legend()
            plt.show(block=False)

            start_aux=np.where(omegas_aux < -10)[0][-1]
            end_aux=np.where(omegas_aux > 10)[0][0]
            start_phys=np.where(omegas_phys < -10)[0][-1]
            end_phys=np.where(omegas_phys > 10)[0][0]

            plt.figure()
            plt.title(f'non interacting phi = {phi}')
            plt.plot(omegas_phys[start_phys:end_phys],Gr0_wigner0_phys.imag[start_phys:end_phys],label='retarded phys', color = 'navy')
            plt.plot(omegas_phys[start_phys:end_phys],Gk0_wigner0_phys.imag[start_phys:end_phys],label='keldysh phys', color = '#a65628')
            plt.plot(omegas_aux[start_aux:end_aux],Gr0_wigner0_aux.imag[start_aux:end_aux],label='retarded aux',color='#a6cee3',linestyle='dashed')
            plt.plot(omegas_aux[start_aux:end_aux],Gk0_wigner0_aux.imag[start_aux:end_aux],label='keldysh aux',color = '#f78843',linestyle='dashed')
            plt.legend()
            plt.show(block=False)

            plt.figure()
            plt.title('Delta')
            plt.plot(del_phys[:,0],del_phys[:,1],label='retarded phys', color = 'navy')
            plt.plot(del_phys[:,0],del_phys[:,2],label='keldysh phys', color = '#a65628')
            plt.plot(del_aux[:,0],del_aux[:,1],label='retarded aux',color='#a6cee3',linestyle='dashed')
            plt.plot(del_aux[:,0],del_aux[:,2],label='keldysh aux',color = '#f78843',linestyle='dashed')
            plt.legend()
            plt.show(block=False)

        currents_phys.append(current_phys.real)
        currents_aux.append(current_aux.real)
        currents_solver_aux.append(current_solver_aux.real)
        currents_solver_phys.append(current_solver_phys.real)
        tags.append(tag)

        E_Gr=np.sqrt(np.trapezoid((abs(Gr0_wigner0_phys-Gr0_wigner0_aux))**2,omegas_phys))/np.sqrt(np.trapezoid(abs(Gr0_wigner0_phys)**2,omegas_phys))
        E_Gk=np.sqrt(np.trapezoid((abs(Gk0_wigner0_phys.imag-Gk0_wigner0_aux.imag))**2,omegas_phys))/np.sqrt(np.trapezoid(abs(Gk0_wigner0_phys.imag)**2,omegas_phys))
        chis.append(float(chi))
        Es_Gr.append(E_Gr)
        Es_Gk.append(E_Gk)
        phis_file.append(phi)
        save_all(filepath, phis_file,currents_aux, currents_phys, currents_solver_aux, currents_solver_phys,Es_Gr, Es_Gk,chis,tags)
    return filepath

def runSolverWithFitFile(phis,V,Om,U,filename,solver_params=None):
    dirName = os.path.join(solver_params['dirName'] , f'V{V}Om{Om}U{U}')
    solver_params['dirName']=dirName
    if os.path.exists(filename):
        with open(filename, "r") as f:
            fits_dict = json.load(f)
    else:
        print('no such file')
    for phi, fit_dict in fits_dict.items():
        phi=float(phi)
        print(phi)
        if not(phi in phis):
            continue

        fit_params=fit_dict['fit_parameters']
        fit_results=fit_dict['fit_results']
        matrices=fit_results['matrices']
        tag,omegas_wigner,wigner_dic=G0FromSolverSpinSym(matrices,V,Om,U,fit_params,solver_params)
        #test plot
        start_aux=np.where(omegas_wigner < -10)[0][-1]
        end_aux=np.where(omegas_wigner > 10)[0][0]
        #plt.figure()
        #plt.title(f'solver phi = {phi}')
        #gr_plot=wigner_dic['0 0']['retarded'][0]
        #plt.plot(omegas_wigner[start_aux:end_aux],gr_plot.imag[start_aux:end_aux],label='retarded aux',color='#a6cee3',linestyle='dashed')
        #plt.plot(omegas_wigner[start_aux:end_aux],gk_plot.imag[start_aux:end_aux],label='keldysh aux',color = '#f78843',linestyle='dashed')
        #plt.legend()
        # plt.show(block=False)



#deltaTest()
#filepath='current_results/V3Om3U0/phi0-20250409-124047.json'
#GphysFromFile(filepath)

solver_params={'dt':0.05,'eps':1e-8,'max_iter': 100,
                'av_periods':4,'tf':1e1,'t_step':1e1,'av_Tau':5,'writeFile':True,
                'dirName':'current_results5sites'}

#20 from 7 onwoards, around 7 the keldysh is small, so the relatvie error is large for T=1/20

#createFit('Fit_constantDOS.json',plot=True,change_phis=[14],phis=[14],T=1/10,sites=5)
#phis_check=np.arange(0,15.5,0.5)
#checkFit('Fit_constantDOS.json',phis_check,6,3,plot=True)
phis=[7]
#phis=np.arange(2,13,0.5)
filepath='Fit_constantDOS.json'
V=6
Om=3
U=6
runSolverWithFitFile(phis,V,Om,U,filepath,solver_params)
a

filepath=calcCurrents(phis=[0],V=3,Om=3,U=3,sites=5,T=1/25,t=1,D=15,T_fict=1,solver_params=solver_params,plot=True)
#calcCurrentsTest(T=1/20,t=1,D=15,T_fict=1,V=3,Om=3,sites=3,phis=[0])
#calcCurrentsTest(T=1/20,t=1,D=15,T_fict=1,V=3,Om=3,sites=3,phis=[0])
phis,currents_aux, currents_phys, currents_solver_aux, currents_solver_phys,Es_Gr, Es_Gk,chis,tags = load_all(filepath)
# Zip everything together
combined = list(zip(
    phis,
    currents_aux,
    currents_phys,
    currents_solver_aux,
    currents_solver_phys,
    Es_Gr,
    Es_Gk,
    chis,
    tags
))

# Sort based on the first item in each tuple (phi value)
combined_sorted = sorted(combined, key=lambda x: x[0])

# Unzip everything back into separate lists
(
    phis_sorted,
    currents_aux_sorted,
    currents_phys_sorted,
    currents_solver_aux_sorted,
    currents_solver_phys_sorted,
    Es_Gr_sorted,
    Es_Gk_sorted,
    chis_sorted,
    tags_sorted
) = zip(*combined_sorted)

phis = list(phis_sorted)
currents_aux = list(currents_aux_sorted)
currents_phys = list(currents_phys_sorted)
currents_solver_aux = list(currents_solver_aux_sorted)
currents_solver_phys = list(currents_solver_phys_sorted)
Es_Gr = list(Es_Gr_sorted)
Es_Gk = list(Es_Gk_sorted)
chis = list(chis_sorted)
tags = list(tags_sorted)

plt.figure()
#plt.plot(phis,currents_phys,color='navy',label=r'$\Delta_{\mathrm{phys}}$')
#plt.plot(phis,currents_aux,color='#f781bf',label=r'$\Delta_{\mathrm{aux}}$')
plt.plot(phis,currents_solver_phys,color='navy',label=r'$\mathrm{solver phys}$')
plt.plot(phis,currents_solver_aux,color='#64b5cd',label=r'$\mathrm{solver aux}$',linestyle='dashed')
plt.xlabel('voltage')
plt.ylabel('current')
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(phis,chis,label='chi',color='navy')
plt.plot(phis,Es_Gr,label='Error Gr',color='#984ea3')
plt.plot(phis,Es_Gk,label='Error Gk',color='#a65628')
plt.xlabel('voltage')
plt.ylabel('errors phys, aux')
plt.legend()
plt.grid(True)

plt.show()