import Fit.simple_fit_wrapper as fit
import numpy as np
import os
from Fit.utils import KK
import matplotlib.pyplot as plt
import time
import json
import glob
import re
from pathlib import Path
from periodicSolver.GreensFunction_sites import  calculateGreensFunction as gf_solver
from periodicSolver.FloquetSpace import  calculateWignerFromFile,calculateFloquetFromFile

#matrices,chi,del_aux, del_phys=fit.get_parameters(1/20,1.5,-1.5,20,1,fit.flat_delta,5,plot=True,return_phys=True)
#print('chi',chi)
script_dir = Path(__file__).parent 

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
    """
    Maps the zero Wigner mode onto a diagonal Matrix

    Parameters:
        Om (float): Frequency of the system
        om (float): omega at which the matrix should be given.
        omegas (np.array): A 1d array of values at which the wigner0 is given.
        wigner0 (np.array): A 1d array of values of the zero wigner mode.

    Returns:
        np.array: A 2d array, of the diagonal floquet matrix

    """
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
        diag_floq[n+n_max]=wigner0[i_om]
    
    return om_num,np.diag(diag_floq)

def MatrixToWigner0(om_floq,matrix,Om):
    """
    Maps the diagonal of the Floquet matrix to the zero wigner mode

    Parameters:
        om_floq (np.array): 1d array Frequencies of the floquet matrices given
        matrix (np.array): 3d array, where each entry is a 2d floquet matrix at a specific frequency
        omegas (np.array): A 1d array of values at which the wigner0 is given.
        wigner0 (np.array): A 1d array of values of the zero wigner mode.

    Returns:
        np.array: A 2d array, of the diagonal floquet matrix

    """
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
    '''
    Calculates the inverse Floquet matrix of a non-interacting System (of the central site)
    Parameters:
        om(float): frequeny to calculate the floquet matrix at
        n_max (int): dimensions of the floquet matrix
        V,Om,eps (float): parameters of the central site
    Returns: 
        np.array 2d inverse Floquet matrix
    '''
    Gr_inv_diag=np.zeros(2*n_max+1,dtype=complex)
    for n in range(-n_max,n_max+1):
        Gr_inv_diag[n+n_max]=om+n*Om -eps

    Gr_inv_off=np.ones(len(Gr_inv_diag)-1)*V/2
    Gr_inv=np.diag(Gr_inv_diag,k=0)+np.diag(Gr_inv_off,k=1)+np.diag(Gr_inv_off,k=-1)

    return Gr_inv

def DelAux(omegas,eps,hopping,gamma1,gamma2):
    '''
    Function to recalculate the auxilary hybridization function, given all bath parameters
        omegas (np.array): frequenies to calculate hybridization function
        eps,hopping,gamma1/2 (np.array): all are np.arrays describing all parameters of the system
    Returns: 
        np.array 1d of the retareded component of the hybritization function
        np.array 1d of the keldysh component of the hybritization function
    '''
    omegas,Gr,Gk=calcGreen(eps,hopping,gamma1,gamma2,omegas)
    DeltaR=omegas-1/Gr
    DeltaK=1/Gr*Gk*1/np.conj(Gr)
    return DeltaR,DeltaK

def GetGphysFloq(omegas_floq,GauxR,GauxK,eps,hopping,gamma1,gamma2,V,Om,fit_params,t=1/np.sqrt(2)):
    '''
    Function to calculate the physical floqeut green's function, starting with the auxiliary result from the solver
        omegas (np.array): frequenies the auxiliary function was calculated at
        eps,hopping,gamma1/2 (np.array): all are np.arrays describing all parameters of the system
        fit_params (dictionary): containing all parameters necessary for the fitting of the hybritization function
    Returns: 
        np.array 1d of the retareded component of the physical floqeut green's function
        np.array 1d of the keldysh component of the physical floqeut green's function
    '''
    #t=1
    T_fict=fit_params['T_fict']
    D=fit_params['D']
    T=fit_params['T']
    phi=fit_params['phi']
    print(t)
    #t=1/np.sqrt(2)
    #D=10
    #T=0.1
    #T_fict=0.5
    Gamma=np.pi*t**2/D
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
        invGR_phys=invG0R_phys[i]#-SigmaR
        invGK_phys=invG0K_phys[i]#-SigmaK
        
        Gr_phys_floq.append(np.linalg.inv(invGR_phys))
        Gk_phys_floq.append(-Gr_phys_floq[i]@invGK_phys@np.conj(Gr_phys_floq[i].T))

    #print(invG0R_aux)
    #G0_aux=G0FromDyson(del_aux,V,Om,eps=0)
    #G0_phys=G0FromDyson(del_aux,V,Om,eps=0)
    
    return np.array(Gr_phys_floq),np.array(Gk_phys_floq)
    

def invG0FloqFromDyson(del_aux,V,Om,eps=0):
    '''
    Function to calculate the non-interacting floqeut green's function, 
    of the auxiliary system
        del_aux (np.array): with frequenies in the first column
            and retared and keldysh component of the auxiliary function in the second and third column 
        V,om,eps (float): parameters of the central site
    Returns: 
        np.array 1d frequency the floquet matrix was calculated at
        np.array 3d of the retareded component of the inverse non-interacting floqeut green's function 
        (first axis is the frequency)
        np.array 3d of the keldysh component of the inverse non-interacting floqeut green's function
    '''
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
    '''
    Function to calculate the non-interacting floqeut green's function, 
    of the auxiliary system
        del_aux (np.array): with frequenies in the first column
            and retared and keldysh component of the auxiliary function in the second and third column 
        V,Om (float): parameters of the central site
    Returns: 
        np.array 3d of the retareded component of the non-interacting floqeut green's function 
        (first axis is the frequency)
        np.array 3d of the keldysh component of the non-interacting floqeut green's function
    '''
    omegas_floq,Gr0_floq_inv,Gk0_floq_inv=invG0FloqFromDyson(del_aux,V,Om)
    Gr0_floq=[]
    Gk0_floq=[]
    for i,om in enumerate(omegas_floq):
        om1,Delkfloqu_om=Wigner0ToMatrix(om,del_aux[:,0],1j*del_aux[:,2],Om)
        Gr0_floq.append(np.linalg.inv(Gr0_floq_inv[i]))
        Gk0_floq.append(Gr0_floq[i]@Delkfloqu_om@(Gr0_floq[i].T.conj()))
    return np.array(Gr0_floq),np.array(Gk0_floq)

def G0FromDyson(del_aux,V,Om,eps=0):
    '''
    Function to calculate the non-interacting wigner green's function, 
    of the auxiliary system
        del_aux (np.array): with frequenies in the first column
            and retared and keldysh component of the auxiliary function in the second and third column 
        V,Om (float): parameters of the central site
    Returns: 
        np.array 1d frequency the Green's funciton was calculated at
        np.array 3d of the retareded component of the non-interacting wigner green's function 
        (first axis is the frequency)
        np.array 3d of the keldysh component of the non-interacting wigner green's function
    '''
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
    '''Function to calculate the phyiscal hybritization function'''
    return (1 - fit.fermi(w, -D, beta_fict)) * fit.fermi(w, D, beta_fict)*Gamma

def calcGreen(eps,hopping,gamma1,gamma2,omegas=np.linspace(-20,20,10001)):
    '''
    Function to calculate the the non-interacting, non-driven Greens  function of the 
    central site, given all bath parameters
        omegas (np.array): frequenies to calculate hybridization function
        eps,hopping,gamma1/2 (np.array): all are np.arrays describing all parameters of the system
    Returns: 
        np.array 1d frequencies at which the GF was calculated
        np.array 1d of the retareded component of the GF
        np.array 1d of the keldysh component of the GF
    '''
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
    '''
    Function to calculate the current according to Meir-Wingreen form the retared Green's function
    (see thesis) 
        w (np.array): frequenies at which to calculate the current
        T,muL,muR,D,T_fict,Gamma (float): all are floats describing parameters of the fit
    Returns: 
        np.array 1d current depending on the frequency
    '''
    g=gamma(D, 1/T_fict, w= w,Gamma=Gamma)
    return 1j*np.trapezoid((fit.fermi(w,muL,1/T)-fit.fermi(w,muR,1/T))*g*(Gr-np.conj(Gr)),w)/(2*np.pi)

#omegas_order,Gr0_wigner0_phys,Gk0_wigner0_phys=G0FromDyson(del_phys,2,4)

def G0FromSolverSpinSym(Lindblad_params,V,Om,U,fit_params,solver_params=None):
    '''
        Getting the zero Wignermode directly from the solver, for a spinsymmetric set up
        Parameters:
            Lindblad_params (dict): dictionary containing all the parameters 
                necessary for the Lindblad equation
            V, Om, U (float): parameters of the central site
            fit_params (dict): containing all parameters necessary for the fitting of the hybritization function
            solver_params (dict): containing possible additional parameters for the solver
        Resturns:
            String: path to where the results of the solver are stored at
            np.array: frequencies at which the zero mode was calculated
            dict: dictionary containing the retared and keldysh component of the zero Wigner mode
    '''
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

def deltaTest(T=1/20,t=1/np.sqrt(2),D=15,T_fict=1/2,sites=3,phis=[0]):
    '''A function to check the calculation of the hybirtization function 
    see, createFit and checkFit for details'''
    Om=0
    V=0
    Gamma=t**2*np.pi/D
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

def calcCurrentsTest(T=1/20,t=1/np.sqrt(2),D=15,T_fict=1,V=3,Om=3,sites=3,phis=[0]):
    '''
        see calcCurrents, the function here just has some additonal plotting to check
    '''
    Gamma=t**2*np.pi/D

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
    '''
        helper function to save all the parameter to a file in json format
    '''
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
    '''
        helper function to load the quantities
        phis,currents_aux, currents_phys, currents_solver_aux, currents_solver_phys,E_Gr, E_Gk,chi,tags
        from file
    '''
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return (data['phis'],data["currents_aux"], data["currents_phys"], data["currents_solver_aux"], 
            data["currents_solver_phys"],data['E_Gr'],data['E_Gk'],data['chis'],data["tags"])
    else:
        return [],[],[],[],[],[],[],[],[]
    
def GphysFromFileTest(filepath):
    '''
        see GphysFromFile, simply contains some additional plotting function, etc. to check that 
        everything works correctlx
    '''
    t=1/np.sqrt(2)
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
    Gamma=t**2*np.pi/D
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

def GphysFromFile(filepath,rep='wig'):
    '''
        Calculates the physical Green's function (either the zero Wigner mode or the Floquetmatrix) 
        from the file containing the results from the solver
        Parameters:
            filepath: path to a file, storing the results from the solver 
            rep: Wether to calculate the floquet matrix or the zero wigner mode
        Returns:
            (np.array): 1d array containing the frequencies the floquet matrix was calculated at
            (np.array): 3d array, with all the Floquet matrices of the retareded component
            (np.array): 3d array, with all the Floquet matrices of the keldysh component
                or:
            (np.array): 1d array containing the frequencies the floquet matrix was calculated at
            (np.array): 1d array, with the zero wigner mode of the retareded component
            (np.array): 1d array, with the zero wigner mode  of the keldysh component
    '''
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
    if rep=='floq':
        return omegas_floq,Gr_phys_floq,Gk_phys_floq
    
    if rep=='wig':
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

def createFit(filename,phis=[0,1],change_phis=[],V=3,Om=3,sites=3,T=1/10,t=1/np.sqrt(2),D=15,T_fict=1,plot=False):
    '''
        Function to create a file containing all fit parameters for the bath sites, for a constant 
        DOS and different potentials as well as calculateing the error in the hybratization functiong 
        and the Greens function of the central site
        Parameters:
            filename (string): the filepath, where the results should be stored
            phis (list): list of potentials, for which the hybritization function should be fitted
            change_phis (list): in case the file already exists, but a specific phi values should be 
                refitted
            V,Om (float): parameters of the central site (not relevant for the fitting procedure, 
                but for calculating the error of the non interacting Green's function)
            sites (int): number of bath sites
            T,t,D,T_fict (float): parameters of the hybritization function
            plot (boolean): wether to plot the hybritization and Green's function for each phi, 
                recommended to ensure, that the relevant features are captured by the fit, 
                weights might have to be adjusted (sofar this has to be done manually in this function)
    '''
    #Gamma=t*np.pi/(2*D)
    Gamma=t**2*np.pi/D

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
        #fit_params = {'phi':phi,'T':T,'D':D,'T_fict':T_fict,'sites':sites,'weights':'None'}
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

def checkFit(filename,phis,V,Om,t=1/np.sqrt(2),plot=True):
    '''
        Function to check the fit from a file containing all fit parameters for the bath sites, for a constant 
        DOS and different potentials as well as calculateing the error in the hybratization functiong 
        and the Greens function of the central site. 
        Parameters:
            filename (string): the filepath, where the results should be stored
            phis (list): list of potentials, for which the hybritization function should be checked
            V,Om (float): parameters of the central site (not relevant for the fitting procedure, 
                but for calculating the error of the non interacting Green's function)
            t (float): parameters of the hybritization function
            plot (boolean): wether to plot the hybritization and Green's function for each phi, 
                recommended to ensure, that the relevant features are captured by the fit, 
                weights might have to be adjusted (sofar this has to be done manually in this function)
    '''
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
        Gamma=np.pi*t**2/D
        H = np.array(matrices['hopping matrix'])
        gamma1 = np.array(matrices['ReG1']) + 1j*np.array(matrices['ImG1'])
        gamma2 = np.array(matrices['ReG2']) + 1j*np.array(matrices['ImG2'])
        eps = np.diag(H)
        hopping = np.diag(H,k=1)
        omegas_del=np.linspace(-25,25,500)
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
            start_aux=np.where(omegas_aux < -20)[0][-1]
            end_aux=np.where(omegas_aux > 20)[0][0]
            start_phys=np.where(omegas_phys < -20)[0][-1]
            end_phys=np.where(omegas_phys > 20)[0][0]

            #start_aux=0
            #end_aux=-1
            #start_phys=0
            #end_phys=-1

            plt.figure(figsize=(5, 3))
            #plt.title(f'non interacting phi = {phi}')
            #plt.plot(omegas_phys[start_phys:end_phys],Gr0_wigner0_phys.imag[start_phys:end_phys],label=r'$G_{0,\text{phys}}^r$', color = 'navy')
            plt.plot(omegas_phys[start_phys:end_phys],Gk0_wigner0_phys.imag[start_phys:end_phys],label=r'$G_{0,\text{phys}}^k$', color = '#a65628')
            #plt.plot(omegas_aux[start_aux:end_aux],Gr0_wigner0_aux.imag[start_aux:end_aux],label=r'$G_{0,\text{aux}}^r$',color='#a6cee3',linestyle='dashed')
            plt.plot(omegas_aux[start_aux:end_aux],Gk0_wigner0_aux.imag[start_aux:end_aux],label=r'$G_{0,\text{aux}}^k$',color = '#f78843',linestyle='dashed')
            plt.xlabel(r'$\omega/t^{\prime}$')
            plt.ylabel(r'$\text{Im}(G_0^x)/t^{\prime-1}$')
            plt.text(-18, -0.3, 'b)', #transform=plt.get_yaxis_transform(),
            va='top', ha='left', fontsize=10)
            plt.legend()
            plt.show(block=False)

            plt.figure(figsize=(5, 3))
            #plt.title('Delta')
            start=np.where(del_phys[:,0] < -20)[0][-1]
            end=np.where(del_phys[:,0] > 20)[0][0]
            plt.plot(del_phys[start:end,0],del_phys[start:end,1],label=r'$\Delta_{\text{phys}}^r$', color = 'navy')
            plt.plot(del_phys[start:end,0],del_phys[start:end,2],label=r'$\Delta_{\text{phys}}^k$', color = '#a65628')
            plt.plot(del_aux[start:end,0],del_aux[start:end,1],label=r'$\Delta_{\text{aux}}^r$',color='#a6cee3',linestyle='dashed')
            plt.plot(del_aux[start:end,0],del_aux[start:end,2],label=r'$\Delta_{\text{aux}}^k$',color = '#f78843',linestyle='dashed')
            plt.xlabel(r'$\omega/t^{\prime}$')
            plt.ylabel(r'$\text{Im}(\Delta^x)/t^{\prime}$')
            plt.text(-18, 0.2, 'a)', #transform=plt.get_yaxis_transform(),
            va='top', ha='left', fontsize=10)
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

    fig, ax1 = plt.subplots(figsize=(5,2.5))
    #ax2 = ax1.twinx()
    ax1.plot(phis,Chis,label=r'$\chi$',color='navy',marker='o',markersize=2)
    ax1.plot(phis,E_Grs,label=r'$ \delta G_{0}^r$',color='#984ea3',marker='o',markersize=2)
    ax1.plot(phis,E_Gks,label=r'$ \delta G_{0}^k$',color='#a65628',marker='o',markersize=2)
    #ax1.set_ylim(0,0.6)
    #plt.plot(phis,Gk_ints,label='Error Gk',color='k')
    ax1.set_xlabel(r'$\phi/t^{\prime}$')
    ax1.set_ylabel('Errors')
    
    plt.grid(True)
    ax1.legend(fontsize=10)

    plt.show()


def calcCurrents(phis=[0,1],V=3,Om=3,U=0,sites=3,T=1/20,t=1/np.sqrt(2),D=15,T_fict=1,solver_params=None,plot=False):
    '''
        Recommend calcCurrentsFromFolder instead, since here the fit cannot be checked first

        calculating the current according to Meir-Wingreen, where the solver is run, withing this 
        function accoring to the fit parameters
        where the directory of the results from the solver are stored in 
        the following folder (solver_params['dirName'] , f'V{V}Om{Om}U{U}')
        calculates the current for different potentials and saves the results to a file currents.json
        Parameters:
            phis: potentials to calculate the current at (results of the solver must be in the folder)
            V,Om,U: parameters of the central site, to find the correct folder
            sites: number of bath sites
            T,t,D,T_fict (float): parameters of the fit, should correspond to the ones used by the solver
            solver_params (dict): additional solver parameters, must contain the key 'dirName'
            plot: wether to plot the auxilary and physical Greens'function for each phi
    '''
    dirName = os.path.join(solver_params['dirName'] , f'V{V}Om{Om}U{U}')
    solver_params['dirName']=dirName
    timestr = time.strftime("%Y%m%d-%H%M%S")
    #filename=f"currents-{timestr}.json"
    filename=f"currents.json"
    filepath=os.path.join(dirName,filename)
    #print(filepath)
    Gamma=t**2*np.pi/D


    
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

def calcCurrentsFromFolder(dirName,phis=[0,1]):
    '''
        same as calcCurrents, however it expects that the solver already calculated all relevant 
        Green's functions. The results are saved to the file currents.json
        dirName (string): folder all the solver results are stored in, for which the current should 
            be calculated (for the results to make sense, the only difference should be phi)
        phis (list): values at which the current should be calculated, a corresponding file from 
        the solver must exist, phi has to be recognisible from the filename
    '''
    t=1/np.sqrt(2)
    current_filename=f"currents.json"
    current_filepath=os.path.join(dirName,current_filename)
    print(current_filepath)
    phis_file,currents_aux, currents_phys, currents_solver_aux, currents_solver_phys,Es_Gr, Es_Gk,chis,tags = load_all(current_filepath)
    #print(filepath)
    

    pattern = re.compile(r'phi([0-9]*\.?[0-9]+)-')
    json_files = glob.glob(os.path.join(dirName, '*.json'))
    
    for json_file in json_files:
        if os.path.basename(json_file) == 'currents.json':
            continue  # Skip this file
        match = pattern.search(os.path.basename(json_file))
        if match:
            phi_value = float(match.group(1))

            if not(phi_value in phis):continue
        
        else: continue

        if phi_value in phis_file: continue

        with open(json_file, 'r') as f:
            data = json.load(f)
            print(f"Loaded {os.path.basename(json_file)}")
        
        input=data['input']
        fit_params=input['fit_parameters']
        T_fict=fit_params['T_fict']
        D=fit_params['D']
        T=fit_params['T']
        muL=phi_value/2
        muR=-muL
        Gamma=t**2*np.pi/D

        parameters=input['parameters']
        eps=np.array(parameters['epsilon'])
        hopping=np.array(parameters['hopping'][:-1])
        gamma1=np.array(parameters['coupling_emptyReal'])+1j*np.array(parameters['coupling_emptyImag'])
        gamma2=np.array(parameters['coupling_fullReal'])+1j*np.array(parameters['coupling_fullImag'])
        V=parameters['drive']
        Om=parameters['frequency']
        phis_file.append(phi_value)

        #solver green's functions
        omegas_sol_phys,Gr_solver_phys,Gk_solver_phys=GphysFromFile(json_file,rep='wig')
        sites,omegas_sol_aux,wigner_dic_aux=calculateWignerFromFile(json_file,0,['retarded','keldysh'],['0 0'])
        Gr_solver_aux=wigner_dic_aux['0 0']['retarded'][0]
        Gk_solver_aux=wigner_dic_aux['0 0']['keldysh'][0]

        #non-interacting green's functions
        omegas_del=omegas_sol_aux
        del_auxr,del_auxk=DelAux(omegas_del,eps,hopping,gamma1,gamma2)
        del_physr = - (1 - fermi(omegas_del, -D, 1/T_fict)) * fermi(omegas_del, D, 1/T_fict)*Gamma*1j
        del_physk = del_physr * (1 - 2 * fermi(omegas_del, muL, 1/T)+1 - 2 * fermi(omegas_del, muR, 1/T))
        
        del_aux=np.array([omegas_del,del_auxr.imag,del_auxk.imag]).T
        del_phys=np.array([omegas_del,del_physr.imag,del_physk.imag]).T

        omegas_phys,Gr0_wigner0_phys,Gk0_wigner0_phys=G0FromDyson(del_phys,V,Om)
        omegas_aux,Gr0_wigner0_aux,Gk0_wigner0_aux=G0FromDyson(del_aux,V,Om)

        #current calculations
        current_aux=current(omegas_aux,Gr0_wigner0_aux,T=T,muL=muL,muR=muR,D=D,T_fict=1,Gamma=Gamma)
        current_phys=current(omegas_phys,Gr0_wigner0_phys,T=T,muL=muL,muR=muR,D=D,T_fict=1,Gamma=Gamma)
        current_solver_aux=current(omegas_sol_aux,Gr_solver_aux,T=T,muL=muL,muR=muR,D=D,T_fict=1,Gamma=Gamma)
        current_solver_phys=current(omegas_sol_phys,Gr_solver_phys,T=T,muL=muL,muR=muR,D=D,T_fict=1,Gamma=Gamma)
        
        currents_phys.append(current_phys.real)
        currents_aux.append(current_aux.real)
        currents_solver_aux.append(current_solver_aux.real)
        currents_solver_phys.append(current_solver_phys.real)

        save_all(current_filepath, phis_file,currents_aux, currents_phys, currents_solver_aux, currents_solver_phys,Es_Gr, Es_Gk,chis,tags)
    return current_filepath


def runSolverWithFitFile(phis,V,Om,U,filename,solver_params=None):
    '''
        running the solver directly from a file, which contains already the
        parameters of the bathsites, coming from the fit of the physical hybritization function
        The results are saved to the file given by the following path: 
            solver_params['dirName'] , f'V{V}Om{Om}U{U}'
        Parameters:
            phis (list): list of voltages, the solver should be run for, each fee must have its own entry in 
            the zip file
            V,Om,U (float): parameters of the central site
            filename (string): the path to a file containing the fit parameters 
                (see createFit, for the necessary structure of the file)
            solver_params: dictionary containing any further information passed to the solver
    '''
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

def plotDOS_phi(filepaths,functions):
    '''
    plotting the spectral A, distribution F and occupation function N
    Parameters: 
        filepaths (list): list of strings, pointing to files created by the solver 
        function (list): list of strings, depending on what functions should be plotted
    '''
    #fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3), sharey=True)
    right_axes = [ax.twinx() for ax in axes]
    for ax in right_axes:
        #ax.sharey(right_axes[0]) 
        ax.set_ylim(0, 1)
    colors=['navy','#984ea3','#a65628']
    #colors=['#984ea3','#a65628']
    colors_F = ['#a6cee3','#f781bf', '#c97b4a']
    for i,filepath in enumerate(filepaths):
        with open(filepath, 'r') as file:
            data = json.load(file)
        fit_params=data['input']['fit_parameters']
        phi=fit_params['phi']
        parameters=data['input']['parameters']
        V=parameters['drive']
        Om=parameters['frequency']
        U=parameters['interaction']

        print('wigner')
        omegas_wig,Gr_wig,Gk_wig=GphysFromFile(filepath)
        print('floq')
        omegas_floq,Gr_floq,Gk_floq=GphysFromFile(filepath,rep='floq')
        
        start=np.where(omegas_wig < -10)[0][-1]
        end=np.where(omegas_wig > 10)[0][0]
        
        if "F" in functions:
            print('F')
            F_floq=1/2*(1-Gk_floq.imag@np.linalg.inv(Gr_floq).imag/2)
            omegas_wig,F_wig=MatrixToWigner0(omegas_floq,F_floq,Om)
            frac=Gk_wig.imag/Gr_wig.imag

            Gr_wig_check=Gr_wig.imag[start:end]
            omegas_F=omegas_wig[start:end]
            omegas_F=omegas_F[abs(Gr_wig_check)>1e-3]
            F2_wig=1/2*(1-frac/2)
            F2_wig=F2_wig[start:end]
            F2_wig=F2_wig[abs(Gr_wig_check)>1e-3]
            #F2_wig[abs(Gk_wig.imag)<1e-5]=0
            #plt.plot(omegas_wig[start:end],Gk_wig[start:end].imag,label='Gk')
            #plt.plot(omegas_wig[start:end],F_wig[start:end],label='F',linestyle='dashed')
            #plt.plot(omegas_wig[start:end],F2_wig[start:end],label=fr'$F2(\omega), \phi={phi}$',color=colors[i], linewidth=0.8)
            #axes[i].scatter(omegas_F, F2_wig,color=colors_F[i],label='F',s=0.5)
            #ax_right = axes[i].twinx()
            axes[i].plot([0], [0],color=colors_F[i],label='F')
            right_axes[i].plot(omegas_F, F2_wig,color=colors_F[i])
            
        if "A" in functions:
            A_wig=-Gr_wig.imag/np.pi
            #plt.plot(omegas_wig[start:end],A_wig[start:end],label=r'$A(\omega)$',color=colors[i], linewidth=1)
            if i==2:
                axes[i].plot(omegas_wig[start:end],A_wig[start:end],label=r'DOS',color=colors[i], linewidth=0.5)
            else:
                axes[i].plot(omegas_wig[start:end],A_wig[start:end],color=colors[i], linewidth=0.5,label=r'DOS')
            print(np.trapezoid(A_wig,omegas_wig))
        
        if "N" in functions:
            N2_wig=(1/2*Gk_wig-1j*Gr_wig.imag)/(2*np.pi)

            N2=np.trapezoid(N2_wig,omegas_wig)
            N2_lower=np.trapezoid(N2_wig[omegas_wig<0],omegas_wig[omegas_wig<0])
            print(f'U{U},V{V}')
            N2_upper=N2-N2_lower
            print('total',N2.imag)
            print('lower',N2_lower.imag)
            print('upper',N2_upper.imag)
            #plt.plot(omegas_wig[start:end],N2_wig[start:end].imag,color=colors[i], linewidth=0.8,linestyle='dashed',label=r'$N(\omega)$')
            #axes[i].plot(omegas_wig[start:end],N2_wig[start:end].imag,color=colors[i], linewidth=0.8,linestyle='dashed',label=r'$N(\omega)$')
            #plt.fill_between(omegas_wig[start:end],N2_wig[start:end].imag, 0, alpha=0.3,color=colors[i],label=r'$N(\omega)$')
            if i==2:
                axes[i].fill_between(omegas_wig[start:end],N2_wig[start:end].imag, 0, alpha=0.3,color=colors[i],label=r'filling')
            else:
                axes[i].fill_between(omegas_wig[start:end],N2_wig[start:end].imag, 0, alpha=0.3,color=colors[i],label=r'filling')
        
        #drawing the chemical potential
        ylim = axes[i].get_ylim()
        #right_axes[i].axvline(x=phi/2, color='black', linestyle='--', linewidth=1,zorder=10)
        #axes[i].text(phi/2+0.2, ylim[1]/2, r'$\mu_r$', #transform=axes[i].get_yaxis_transform(),
        #va='top', ha='left', fontsize=8)
        #right_axes[i].axvline(x=-phi/2, color='black', linestyle='--', linewidth=1,zorder=10)
        #axes[i].text(-phi/2-0.2, ylim[1]/2, r'$\mu_l$', #transform=axes[i].get_yaxis_transform(),
        #va='top', ha='right', fontsize=8)

        #text=f'U={U}\nV={V}'
        #text=rf'$\phi={phi}$'
        #axes[i].text(0.05, 0.85, text, transform=axes[i].transAxes,
        #    fontsize=10, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
        axes[i].set_xlabel(r'$\omega/t^{\prime}$')
        axes[i].legend(fontsize=8)
    axes[0].set_ylabel(r'$A(\omega)/t^{\prime -1}$')

    if 'F' in functions:
        right_axes[-1].set_ylabel(r'$F(\omega)/t^{\prime -1}$')
    print('done')

    for i, ax in enumerate(right_axes):
        if i != len(right_axes) - 1:
            print('setting ticks')
            #print(i)
            #ax.set_yticklabels([])
    #right_axes[0].set_yticklabels([])
    right_axes[0].tick_params(axis='y', labelright=False)
    plt.subplots_adjust(wspace=0.1) 

    plt.tight_layout()
    plt.show()

def plotJ_om(filepaths):
    '''plotting the integrand of the current
    filepaths (list): list of strings, pointing to files created by the solver 
    '''
    t=1/np.sqrt(2)
    #fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    
    colors=['navy','#984ea3','#a65628']
    colors_F = ['#a6cee3','#f781bf', '#c97b4a']

    for i,filepath in enumerate(filepaths):
        with open(filepath, 'r') as file:
            data = json.load(file)
        fit_params=data['input']['fit_parameters']
        phi=fit_params['phi']


        parameters=data['input']['parameters']
        T_fict=fit_params['T_fict']
        D=fit_params['D']
        T=fit_params['T']
        muL=phi/2
        muR=-muL
        Gamma=np.pi*t**2/D
        V=parameters['drive']
        Om=parameters['frequency']
        U=parameters['interaction']

        print('wigner')
        omegas_wig,Gr_wig,Gk_wig=GphysFromFile(filepath)
        print('floq')
        #omegas_floq,Gr_floq,Gk_floq=GphysFromFile(filepath,rep='floq')

        start=np.where(omegas_wig < -10)[0][-1]
        end=np.where(omegas_wig > 10)[0][0]
        omegas=omegas_wig[start:end]
        gamma=- (1 - fermi(omegas, -D, 1/T_fict)) * fermi(omegas, D, 1/T_fict)*Gamma
        f=fermi(omegas, muR, 1/T)-fermi(omegas, muL, 1/T)
        A=-1/np.pi*np.imag(Gr_wig)[start:end]

        axes[i].plot(omegas,A,color=colors[i],label='DOS')
        axes[i].plot(omegas,gamma*f,color=colors[i],linestyle='dashed',label=r'$n_\text{bath}(\omega)$')
        axes[i].fill_between(omegas,gamma*f*A, 0, alpha=0.3,color=colors[i],label=r'$j(\omega)$')
        print(np.trapezoid(gamma*f*A,omegas))
        axes[i].legend(fontsize=8)


        ylim = axes[i].get_ylim()
        xlim = axes[i].get_xlim()
        axes[i].axvline(x=phi/2, color='black', linestyle='--', linewidth=1,zorder=10)
        axes[i].text(phi/2+0.2, ylim[1]/2, r'$\mu_r$', #transform=axes[i].get_yaxis_transform(),
        va='top', ha='left', fontsize=8)
        axes[i].axvline(x=-phi/2, color='black', linestyle='--', linewidth=1,zorder=10)
        axes[i].text(-phi/2-0.2, ylim[1]/2, r'$\mu_l$', #transform=axes[i].get_yaxis_transform(),
        va='top', ha='right', fontsize=8)
        axes[i].set_xlabel(r'$\omega/t^\prime$')

        axes[i].text(xlim[0]+1, ylim[1]-0.1, fr'$V={V}$', #transform=axes[i].get_yaxis_transform(),
        va='top', ha='left', fontsize=8,bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
        
    axes[0].set_ylabel(r'$A(\omega)/t^{\prime-1}$')
    plt.tight_layout()
    plt.show()

def plotCurrent(dirpaths,U0=False,fig=None,ax=None):
    '''plotting the total current over phi
    dirpaths (list): list of strings, pointing to folders containing 
    files created by the solver at different phis, but same parameters of the central site
    '''
    if fig==None:
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    colors=['navy','#984ea3','#a65628','k']
    alpha=1
    for i,dirpath in enumerate(dirpaths):
        print(i)
        dirname = os.path.basename(dirpath)
        params = [int(n) for n in re.findall(r'\d+', dirname)]
        V=params[0]
        Om=params[1]
        U=params[2]

        filepath=os.path.join(dirpath,'currents.json')
        print(filepath)
        phis,currents_aux, currents_phys, currents_solver_aux, currents_solver_phys,Es_Gr, Es_Gk,chis,tags = load_all(filepath)
        print(phis)
        # Zip everything together
        combined = list(zip(
            phis,
            currents_aux,
            currents_phys,
            currents_solver_aux,
            currents_solver_phys,
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
        ) = zip(*combined_sorted)

        phis = np.array(list(phis_sorted))
        currents_aux = list(currents_aux_sorted)
        currents_phys = list(currents_phys_sorted)
        currents_solver_aux = list(currents_solver_aux_sorted)
        currents_solver_phys = list(currents_solver_phys_sorted)

        #if i==0:
        #    plt.plot(phis/2,currents_phys,color=colors[i],label=fr'$V=1$')
        if i==3:
            alpha=0.5
        if U0:
            ax.plot(phis/2,currents_phys,color=colors[i],label=fr'$V={V}$',alpha=alpha)
        else:
            ax.plot(phis/2,currents_solver_phys,color=colors[i],label=fr'$V={V}$',alpha=alpha)
    if U0:
        ax.text(0.05, 0.9, fr'$U=0$', transform=ax.transAxes,
            fontsize=10, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
    else:
        ax.text(0.05, 0.9, fr'$U={U}$', transform=ax.transAxes,
            fontsize=10, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
    ax.set_xlabel(r'$\frac{\phi}{2}/t^{\prime}$')
    if U0:
        ax.set_ylabel(r'$j/t^{\prime}$')
        ax.legend()
    ax.grid(True)
    
    #fig.show()

#####################################################################################
# this section shows, how to execute the function, to obtain the relevant physical 
# Greens functions and plotting them, some parts (especially filepaths) might 
# have to be adjusted, read the comments carefully

#deltaTest()
#filepath='current_results/V3Om3U0/phi0-20250409-124047.json'
#GphysFromFile(filepath)
#createFit("Fit_constantDOS_Gamma.json",sites=5)
###################################################################################################
# here we define the parameters for the solver, including where to store the results
solver_params={'dt':0.05,'eps':1e-8,'max_iter': 100,
                'av_periods':4,'tf':1e1,'t_step':1e1,'av_Tau':5,'writeFile':True,
                'dirName':'current_results5sites'}

####################################################################################################
#here we create the fitting parameters according to a given DOS, 
# for different 'phis' they can than be checked
# seperatly by visual inspecting all of them including a driving, if the fit is bad it might make 
# sense to adjust the weights, such that the most relevant features are covered

phis_check=[14.5]
#createFit('Fit_constantDOS_noWeights.json',plot=True,change_phis=[],phis=[3],T=1/10,sites=5,t=1/np.sqrt(2))
#phis_check=np.arange(0,15,0.5)
#phis_check=[3]
#checkFit('Fit_constantDOS.json',phis_check,6,3,plot=True)
#phis=[0]

#####################################################################################################
# in this section, one runs the solver for the given phis and with the fitparameters read from a file
phis=np.arange(9.5,13,0.5)
filepath='Fit_constantDOS.json'
V=0
Om=15
U=6
#runSolverWithFitFile(phis,V,Om,U,filepath,solver_params)

###################################################################################################
#for the following sections, the files below must be adjusted, such that they fit a file which 
# contains the results from the solver, the solver must have been run first, so files actually exist
#u0
file1=script_dir /'current_results5sites/V1Om3U0/phi0-20250514-102059.json'
file2=script_dir /'current_results5sites/V3Om3U0/phi0-20250513-171043.json'
file3=script_dir /'current_results5sites/V6Om3U0/phi0-20250514-084826.json'
#u3
#file1='current_results5sites/V1Om3U3/phi0-20250514-111148.json'
#file2='current_results5sites/V3Om3U3/phi0-20250510-190839.json'
#file3='current_results5sites/V6Om3U3/phi0-20250511-231059.json'
#u6
#file1='current_results5sites/V1Om3U6/phi0-20250514-113453.json'
#file2='current_results5sites/V3Om3U6/phi0-20250511-210219.json'
#file3='current_results5sites/V6Om3U6/phi0-20250512-142907.json'
plotDOS_phi([file1,file2],["A","N"])

#file1='current_results5sites/V0Om3U6/phi0-20250520-161546.json'
#file2='current_results5sites/V0Om3U6/phi6-20250521-104833.json'
#file2='current_results5sites/V3Om3U3/phi6-20250511-012007.json'

#file1='current_results5sites/V3Om3U6/phi0-20250511-210219.json'
#file2='current_results5sites/V3Om3U6/phi12-20250511-163413.json'

#file1='current_results5sites/V0Om3U6/phi6-20250521-104833.json'

#file1='current_results5sites/V1Om3U6/phi6-20250515-133805.json'
#file2='current_results5sites/V3Om3U6/phi6-20250511-113146.json'
#file3='current_results5sites/V6Om3U6/phi6-20250512-195337.json'

#file1='current_results5sites/V1Om3U6/phi12-20250516-021811.json'
#file2='current_results5sites/V3Om3U6/phi12-20250511-163413.json'
#file3='current_results5sites/V6Om3U6/phi12-20250516-181121.json'
#file2='current_results5sites/V3Om3U3/phi6-20250511-012007.json'
#file3='current_results5sites/V3Om3U6/phi6-20250511-113146.json'
filepaths=[file1,file2,file3]
plotJ_om(filepaths)

##############################################################################
#here we calculate and plot the current(phi) from each folder, which stores 
#several files with the solver results
#phis=[0]
phis=np.arange(0,13,0.5)
filepath=calcCurrentsFromFolder('current_results5sites/V0Om10U3',phis)
dirV0U3=script_dir /'current_results5sites/V0Om10U3'
dirV1U3=script_dir /'current_results5sites/V1Om3U3'
dirV3U3=script_dir /'current_results5sites/V3Om3U3'
dirV6U3=script_dir /'current_results5sites/V6Om3U3'

dirV0U6=script_dir /'current_results5sites/V0Om15U6'
dirV1U6=script_dir /'current_results5sites/V1Om3U6'
dirV3U6=script_dir /'current_results5sites/V3Om3U6'
dirV6U6=script_dir /'current_results5sites/V6Om3U6'

# calcCurrentsFromFolder must be executed for each folder, before plotting the current
fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
plotCurrent([dirV1U3,dirV3U3,dirV6U3,dirV0U3],True,fig,axes[0])
plotCurrent([dirV1U3,dirV3U3,dirV6U3,dirV0U3],False,fig,axes[1])
plotCurrent([dirV1U6,dirV3U6,dirV6U6,dirV0U6],False,fig,axes[2])
plt.tight_layout
plt.show()





###############################################################################################
# this is an old part of the program, comparing the results of the current for 
# the auxilariy and the physical Green's function
# it requires a file already containing all the variable listed below

#filepath=calcCurrents(phis=[0],V=3,Om=3,U=3,sites=5,T=1/25,t=1,D=15,T_fict=1,solver_params=solver_params,plot=True)
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
    #Es_Gr,
    #Es_Gk,
    #chis,
    #tags
))
print(combined)
# Sort based on the first item in each tuple (phi value)
combined_sorted = sorted(combined, key=lambda x: x[0])

# Unzip everything back into separate lists
(
    phis_sorted,
    currents_aux_sorted,
    currents_phys_sorted,
    currents_solver_aux_sorted,
    currents_solver_phys_sorted,
    #Es_Gr_sorted,
    #Es_Gk_sorted,
    #chis_sorted,
    #tags_sorted
) = zip(*combined_sorted)

phis = list(phis_sorted)
currents_aux = list(currents_aux_sorted)
currents_phys = list(currents_phys_sorted)
currents_solver_aux = list(currents_solver_aux_sorted)
currents_solver_phys = list(currents_solver_phys_sorted)
#Es_Gr = list(Es_Gr_sorted)
#Es_Gk = list(Es_Gk_sorted)
#chis = list(chis_sorted)
#tags = list(tags_sorted)

plt.figure()
#plt.plot(phis,currents_phys,color='navy',label=r'$\Delta_{\mathrm{phys}}$')
#plt.plot(phis,currents_aux,color='#f781bf',label=r'$\Delta_{\mathrm{aux}}$')
plt.plot(phis/2,currents_solver_phys,color='navy',label=r'$\mathrm{solver phys}$')
#plt.plot(phis,currents_solver_aux,color='#64b5cd',label=r'$\mathrm{solver aux}$',linestyle='dashed')
plt.xlabel(r'$\frac{\phi}{2}/t^{\prime}$')
plt.ylabel(r'$j/t^{\prime}$')
plt.grid(True)
plt.legend()

#plt.figure()
#plt.plot(phis,chis,label='chi',color='navy')
#plt.plot(phis,Es_Gr,label='Error Gr',color='#984ea3')
#plt.plot(phis,Es_Gk,label='Error Gk',color='#a65628')
#plt.xlabel('voltage')
#plt.ylabel('errors phys, aux')
#plt.legend()
#plt.grid(True)

plt.show()