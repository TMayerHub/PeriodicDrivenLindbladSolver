
import numpy as np  # generic math functions
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
#matplotlib.use('QtAgg')
Om=1
V=1.2
x=np.linspace(-np.pi, np.pi,100)
y=x
eta=1e-2
gamma=0.015
period=2*np.pi/Om

t_abs=np.linspace(1000-period,1000+period,100)
T=np.linspace(-500,500,10000+1)
w=np.linspace(-1000,1000,100+1)
G_T_t=np.zeros((len(t_abs),len(T)))*0j


for j in range(len(T)):
    t=t_abs-T[j]/2
    #t=t_abs
    for i in range(len(t)):
        G_T_t[i,j]=(-1j)*np.heaviside(T[j],0.5)*np.exp(-1j*(V/Om)*(np.sin(Om*(t[i]+T[j]))-np.sin(Om*t[i]))-gamma*T[j])
plt.ion()
start=int(len((T)-1)/2)
end=start+int(np.ceil(start/4))
X, Y = np.meshgrid(T[start:end], np.linspace(0,2*period,len(t_abs))) 
levels =  np.linspace(-1,1,100+1)
fig, ax = plt.subplots(2,1,figsize=(4.5,3.5),sharex=True)

surf2 = ax[0].contourf(X, Y, np.real(G_T_t[:,start:end]), cmap='PuOr',levels=levels)   
tick_values = np.arange(-0.8, 0.81, 0.4)  # or whatever range and step you want
cbar = fig.colorbar(surf2, ticks=tick_values)
cbar.set_label(fr'Re$(G^r)$')
ax[0].axhline(y=2*np.pi, color='black', linewidth=1)
ax[0].text(x=X.max(), y=2*np.pi, s=r'$P = 2\pi$', va='bottom', ha='right', fontsize=10, color='black')
#ax.set_title(fr'$f(t+\tau,\tau)$')
#ax[0].set_xlabel(fr'$\tau$')
ax[0].set_ylabel('$t$')

surf2 = ax[1].contourf(X, Y, np.imag(G_T_t[:,start:end]), cmap='PuOr',levels=levels)   
tick_values = np.arange(-0.8, 0.81, 0.4)  # or whatever range and step you want
cbar = fig.colorbar(surf2, ticks=tick_values)
cbar.set_label(fr'Im$(G^r)$')
ax[1].axhline(y=2*np.pi, color='black', linewidth=1)
ax[1].text(x=X.max(), y=2*np.pi, s=r'$P = 2\pi$', va='bottom', ha='right', fontsize=10, color='black')
#ax.set_title(fr'$f(t+\tau,\tau)$')
ax[1].set_xlabel(fr'$\tau$')
ax[1].set_ylabel('$t$')
fig.tight_layout()
plt.show()
fig.savefig("Gr_time.pdf")
print('meshgrid')
#print(X.shape, Y.shape, np.real(G_T_t[:, start:end]).shape)
