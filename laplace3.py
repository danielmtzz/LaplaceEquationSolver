import numpy as _np
import scipy.sparse as _sparse
import scipy.sparse.linalg as _sparselinalg
from scipy.linalg import solve_banded as _solve_banded
from matplotlib import pyplot as _plt
import scipy.special as _sp
_plt.rcParams["font.family"] = 'Times New Roman'
_plt.rcParams['mathtext.fontset'] = 'stix'
bbox_props = dict(boxstyle="square",fc="white")

rho = 18.  # in milimeters
L = 50. # in milimeters
rho0bar = 1
z0bar = 0.849
Lbar = L/rho


Marr = _np.arange(2980,3040)

def partitionInteger(Lbar_,z0bar_,M_):
	M0_ = z0bar_/Lbar_*M_
	resM0 = abs(M0_ - _np.round(M0_))
	N_ = M_/Lbar_
	resN = abs(N_ - _np.round(N_))
	res = resM0+resN
	index = _np.argmin(res)
	return M_[index]





M=partitionInteger(Lbar,z0bar,Marr)
h = Lbar/M
M0 = z0bar/h
M0 = int(_np.round(M0))
N = M0/z0bar
N = int(_np.round(N))


# Potential new definitions:
# M = partitionInteger(Lbar,z0bar,Marr)
# N = int(_np.round(M/Lbar))

def val(ind):
	return 1+1/(2*_np.floor(ind/M))

def val2(ind):
	return 1-1/(2*_np.floor(ind/M))

def realPotential(rho,z,rho0,z0,L,V0):
	ind = _np.arange(0,300)
	k = (ind+0.5)*_np.pi/L
	return 2*V0/L*_np.sum(1/k*_np.sin(k*z0)/_sp.iv(0,k*rho0)*_sp.iv(0,k*rho)*_np.cos(k*z))

def embed(T,Te=-100):
	NFull = T.shape[0] + 1
	MFull = T.shape[1] + 1
	Tfull = _np.zeros((NFull,MFull))
	Tfull[NFull-1][0:M0+1] = Te
	Tfull[0:-1, 0:-1] = T
	return Tfull


kMax = N*M

diag = -4*_np.ones(kMax)
diag[0:M]=-6

upperDiag = _np.ones(kMax)
upperDiag[(2*M-1):kMax:M]=0
upperDiag[M:M*(N-1)+1:M]=2
upperDiag[0]=2
upperDiag[M-1]=0
upperDiag=upperDiag[:-1]


lowerDiag=_np.ones(kMax)
lowerDiag[M:M*(N-1)+1:M]=0
lowerDiag[0]=0
lowerDiag=lowerDiag[1:]


upperUpperDiag=4*_np.ones(kMax)
upperUpperDiag[M:kMax] = val(_np.arange(M,kMax))
upperUpperDiag[M*(N-1)+1:kMax]=0
upperUpperDiag[M*(N-1)]=0
upperUpperDiag=upperUpperDiag[:-M]


lowerLowerDiag = _np.zeros(kMax)
lowerLowerDiag[M:kMax]=val2(_np.arange(M,kMax))
lowerLowerDiag=lowerLowerDiag[M:]


diagonals = [lowerLowerDiag,lowerDiag,diag,upperDiag,upperUpperDiag]
A = _sparse.diags(diagonals,[-M,-1,0,1,M])
b  = _np.zeros(kMax)
b[M*(N-1):M*(N-1)+M0+1]=100


u = _sparselinalg.spsolve(A,b)
T = u.reshape(N,M)
T = embed(T)

T1=_np.concatenate((T[-1:0:-1],T))
mirror=_np.transpose(_np.transpose(_np.fliplr(T1))[:-1])
T2=_np.transpose(_np.concatenate((_np.transpose(mirror),_np.transpose(T1))))

x = _np.linspace(-Lbar*rho,Lbar*rho,2*M+1)
y = _np.linspace(-rho,rho,2*N+1)

f = _plt.figure(figsize=(8,3))
ax = f.add_subplot(111)
im = ax.imshow(T2, extent=[_np.min(x),_np.max(x),_np.min(y),_np.max(y)],aspect='equal')
ax.set_xlabel(r'$z$ (mm)')
ax.set_ylabel(r'$\rho$ (mm)')
f.tight_layout()
_plt.colorbar(im,aspect=10,label='Voltage (V)')



# Legendre Coefficients
def coeff(j,V0,L,z0,rho0):
	ind = _np.arange(0,300)
	k = (ind+0.5)*_np.pi/L
	A_n = 2*V0/(L*k)*_np.sin(k*z0)/_sp.iv(0,k*rho0)
	d = _np.sqrt(1/2.0*(z0**2 + rho0**2/2))
	return 2/V0*(-1)**(j/2)/_sp.factorial(j)*_np.sum(A_n*(k*d)**j)


# get even coeff values
coeff_array = []
for i in range(0,600,2):
	coeff_array.append(coeff(i,-100.0,50.0,0.849*18,18.0))


def legendre(V0,z0,rho0,rho,z,L):
	# just get the 0th and 2nd order terms for quadrupole potential
	j = _np.arange(0,3,2)
	d = _np.sqrt(1/2.0*(z0**2 + rho0**2/2))
	r = _np.sqrt(rho**2 + z**2)
	frac = (r/d)**j
	legendre_polys = list(map(_sp.legendre,j))
	legendre_vals = []
	for i in _np.arange(0,2):
		legendre_vals.append(legendre_polys[i](z/r))
	return V0/2*_np.sum(coeff_array[:2]*frac*legendre_vals)


ideal_quadrupole = []
zarr = x
zarr[3000] = zarr[2999]
for i in zarr:
	ideal_quadrupole.append(legendre(-100.0,.849*18,18.0,0.0,i,50.0))




fig, ax = _plt.subplots()
ax = _plt.subplot(111)
_plt.plot(x,T2[1080,:],c='k',label='Penning potential')
_plt.plot(x,ideal_quadrupole,c='k',linestyle='--',label='Ideal quadrupole')
_plt.title('On-axis potential',size=16,pad=10)
_plt.legend(loc=(.57,.06),fontsize=13,fancybox=False,framealpha=1).get_frame().set_edgecolor('k')
# ax1.annotate("", xy=(.5, 10),
# 	arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
_plt.ylabel('Potential (Volts)',size=14)
_plt.xlabel(r'$z$' + ' (mm)',size=14)
ax.set_ylim([-100,0])
ax.set_xlim([_np.amin(x),_np.amax(x)])
_plt.xticks(size=13)
_plt.yticks(size=13)
_plt.minorticks_on()
ax.tick_params(which='both',direction='in',top=True,right=True)
_plt.setp(ax.spines.values(), linewidth=1)
#_plt.tight_layout()
_plt.show()


# #ideal quadrupole potential (Jerry's paper)
# def idealQuad(z0,rho0,V0,z,rho):
# 	d_squared = 1/2.0*(z0**2 + rho0**2/2)
# 	return V0/(2*d_squared)*(z**2 - rho**2/2)










