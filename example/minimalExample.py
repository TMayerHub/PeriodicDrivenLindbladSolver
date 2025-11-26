import os
from periodicSolver.GreensFunction_sites import calculateGreensFunction
from periodicSolver.FloquetSpace import calculateWignerFromFile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
script_dir = Path(__file__).parent 
# different system parameters set ups, with different system sizes
parameters5 = {"length": 5,
              "epsilon": [-3,-2,-1,2,3],
              "hopping": 0.3,
              "interaction":2,
              "drive": 1,
              "frequency":1,
              "coupling_empty":[0.3,0.4,0,0.6,0.7],
              "coupling_full":[0.7,0.6,0,0.4,0.3],
              "spin_symmetric":False,
}

parameters3 = {"length": 3,
              "epsilon": [-1,0,1],
              "hopping": [0.3,0.3,0],
              "interaction":0,
              "drive": 1,
              "frequency":1,
              "coupling_empty":np.array([[0.9,0,0],[0,0,0],[0,0,0.9]]),
              "coupling_full":np.array([[0.6,0,0.0],[0,0,0],[0.0,0,0.6]]),
              "spin_symmetric":False,
}

parameters1 = {"length": 1,
              "epsilon": 0,
              "hopping": 0,
              "interaction":0,
              "drive": 1,
              "frequency":1,
              "coupling_empty":0.04,
              "coupling_full":0.06,
              "spin_symmetric":False,
}

#defining the GF object according to the system parameters, the GF sites and spin of the GF
GF0=calculateGreensFunction(parameters3,[[0,0]],'up')
print(script_dir.parent/'results/minimalResults')
#calculateing the GF of the central site in time, results are stored in a json file
GF0._GreaterLesserSites([[0,0]],dt=0.05,eps=1e-8,max_iter=1000,
                   av_periods=4,tf=1.5e1,t_step=1.5e1,av_Tau=5,writeFile=True,
                    dirName=script_dir.parent/'results/minimalResults',file_name='3sites')

# calculating the zero wigner mode of the retareded and keldysh from a file for the
#  on site Green's function of the central site
sites,omegas_wigner,wigner_dic=calculateWignerFromFile(script_dir.parent/'results/minimalResults/3sites',0,['retarded','keldysh'],['0 0'])

# extracting the entries in the wigner dictionary of the 0 mode, central site
wigner_ret0=wigner_dic['0 0']['retarded'][0]
wigner_kel0=wigner_dic['0 0']['keldysh'][0]

#plotting the results
om_start=5
mask=(omegas_wigner > -om_start) & (omegas_wigner < om_start)

fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

axes[0].plot(omegas_wigner[mask], wigner_ret0[mask].imag, label='imaginary')
axes[0].plot(omegas_wigner[mask], wigner_ret0[mask].real, label='real')
axes[0].set_title('retarded')
axes[0].legend()

axes[1].plot(omegas_wigner[mask], wigner_kel0[mask].imag, label='imaginary')
axes[1].plot(omegas_wigner[mask], wigner_kel0[mask].real, label='real')
axes[1].set_title('keldysh')
axes[1].legend()

plt.tight_layout()
plt.show()
