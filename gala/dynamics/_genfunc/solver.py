#################
#   AA Solvers  #
#################
import numpy as np
from itertools import product
from scipy.linalg import solve
from scipy.sparse.linalg import spsolve


def check_each_direction(n,angs,ifprint=True):
    """ returns a list of the index of elements of n which do not have adequate
    toy angle coverage. The criterion is that we must have at least one sample
    in each Nyquist box when we project the toy angles along the vector n """
    checks = np.array([])
    P = np.array([])
    if(ifprint):
        print("\nChecking modes:\n====")
    for k,i in enumerate(n):
        N_matrix = np.linalg.norm(i)
        X = np.dot(angs,i)
        if(np.abs(np.max(X)-np.min(X))<2.*np.pi):
            if(ifprint):
                print "Need a longer integration window for mode ", i
            checks=np.append(checks,i)
            P = np.append(P,(2.*np.pi-np.abs(np.max(X)-np.min(X))))
        elif(np.abs(np.max(X)-np.min(X))/len(X)>np.pi):
            if(ifprint):
                print "Need a finer sampling for mode ", i
            checks=np.append(checks,i)
            P = np.append(P,(2.*np.pi-np.abs(np.max(X)-np.min(X))))
    if(ifprint):
        print("====\n")
    return checks,P

def solver(AA, N_max, symNx = 2, throw_out_modes=False):
    """ Constructs the matrix A and the vector b from a timeseries of toy
    action-angles AA to solve for the vector x = (J_0,J_1,J_2,S...) where
    x contains all Fourier components of the generating function with |n|<N_max """

    # Find all integer component n_vectors which lie within sphere of radius N_max
    # Here we have assumed that the potential is symmetric x->-x, y->-y, z->-z
    # This can be relaxed by changing symN to 1
    # Additionally due to time reversal symmetry S_n = -S_-n so we only consider
    # "half" of the n-vector-space

    angs = unroll_angles(AA.T[3:].T,np.ones(3))

    symNz = 2
    NNx = range(-N_max, N_max+1, symNx)
    NNy = range(-N_max, N_max+1, symNz)
    NNz = range(-N_max, N_max+1, symNz)
    n_vectors = np.array([[i,j,k] for (i,j,k) in product(NNx,NNy,NNz)
                          if(not(i==0 and j==0 and k==0)            # exclude zero vector
                             and (k>0                               # northern hemisphere
                                  or (k==0 and j>0)                 # half of x-y plane
                                  or (k==0 and j==0 and i>0))       # half of x axis
                             and np.sqrt(i*i+j*j+k*k)<=N_max)])     # inside sphere

    xxx = check_each_direction(n_vectors,angs)

    if(throw_out_modes):
        n_vectors = np.delete(n_vectors,check_each_direction(n_vectors,angs),axis=0)

    n = len(n_vectors)+3
    b = np.zeros(shape=(n, ))
    a = np.zeros(shape=(n,n))

    a[:3,:3]=len(AA)*np.identity(3)

    for i in AA:
        a[:3,3:]+=2.*n_vectors.T[:3]*np.cos(np.dot(n_vectors,i[3:]))
        a[3:,3:]+=4.*np.dot(n_vectors,n_vectors.T)*np.outer(np.cos(np.dot(n_vectors,i[3:])),np.cos(np.dot(n_vectors,i[3:])))
        b[:3]+=i[:3]
        b[3:]+=2.*np.dot(n_vectors,i[:3])*np.cos(np.dot(n_vectors,i[3:]))

    a[3:,:3]=a[:3,3:].T

    return np.array(solve(a,b)), n_vectors

from itertools import izip


def unroll_angles(A,sign):
    """ Unrolls the angles, A, so they increase continuously """
    n = np.array([0,0,0])
    P = np.zeros(np.shape(A))
    P[0]=A[0]
    for i in xrange(1,len(A)):
        n = n+((A[i]-A[i-1]+0.5*sign*np.pi)*sign<0)*np.ones(3)*2.*np.pi
        P[i] = A[i]+sign*n
    return P

import matplotlib.pyplot as plt
from scipy.stats import linregress as lr

def angle_solver(AA, timeseries, N_max, sign, symNx = 2, throw_out_modes=False):
    """ Constructs the matrix A and the vector b from a timeseries of toy
    action-angles AA to solve for the vector x = (theta_0,theta_1,theta_2,omega_1,
    omega_2,omega_3, dSdx..., dSdy..., dSdz...) where x contains all derivatives
    of the Fourier components of the generating function with |n| < N_max """

    # First unroll angles
    angs = unroll_angles(AA.T[3:].T,sign)

    # Same considerations as above
    symNz = 2
    NNx = range(-N_max, N_max+1, symNx)
    NNy = range(-N_max, N_max+1, symNz)
    NNz = range(-N_max, N_max+1, symNz)
    n_vectors = np.array([[i,j,k] for (i,j,k) in product(NNx,NNy,NNz)
                          if(not(i==0 and j==0 and k==0)    # exclude zero vector
                             and (k>0                          # northern hemisphere
                                  or (k==0 and j>0)                 # half of x-y plane
                                  or (k==0 and j==0 and i>0))       # half of x axis
                             and np.sqrt(i*i+j*j+k*k)<=N_max     # inside sphere
                             )])

    if(throw_out_modes):
        n_vectors = np.delete(n_vectors,check_each_direction(n_vectors,angs),axis=0)

    nv = len(n_vectors)
    n = 3*nv+6

    b = np.zeros(shape=(n, ))
    a = np.zeros(shape=(n,n))

    a[:3,:3]=len(AA)*np.identity(3)
    a[:3,3:6]=np.sum(timeseries)*np.identity(3)
    a[3:6,:3]=a[:3,3:6]
    a[3:6,3:6]=np.sum(timeseries*timeseries)*np.identity(3)

    for i,j in izip(angs,timeseries):
        a[6:6+nv,0]+=-2.*np.sin(np.dot(n_vectors,i))
        a[6:6+nv,3]+=-2.*j*np.sin(np.dot(n_vectors,i))
        a[6:6+nv,6:6+nv]+=4.*np.outer(np.sin(np.dot(n_vectors,i)),np.sin(np.dot(n_vectors,i)))

        b[:3]+=i
        b[3:6]+=j*i

        b[6:6+nv]+=-2.*i[0]*np.sin(np.dot(n_vectors,i))
        b[6+nv:6+2*nv]+=-2.*i[1]*np.sin(np.dot(n_vectors,i))
        b[6+2*nv:6+3*nv]+=-2.*i[2]*np.sin(np.dot(n_vectors,i))

    a[6+nv:6+2*nv,1]=a[6:6+nv,0]
    a[6+2*nv:6+3*nv,2]=a[6:6+nv,0]
    a[6+nv:6+2*nv,4]=a[6:6+nv,3]
    a[6+2*nv:6+3*nv,5]=a[6:6+nv,3]
    a[6+nv:6+2*nv,6+nv:6+2*nv]=a[6:6+nv,6:6+nv]
    a[6+2*nv:6+3*nv,6+2*nv:6+3*nv]=a[6:6+nv,6:6+nv]

    a[:6,:]=a[:,:6].T

    return np.array(solve(a,b))
