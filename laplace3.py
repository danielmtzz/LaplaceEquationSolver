import numpy as _np
import scipy.sparse as _sparse
import scipy.sparse.linalg as _sparselinalg
from scipy.linalg import solve_banded as _solve_banded
from matplotlib import pyplot as _plt
import scipy.special as _sp

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


