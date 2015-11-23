##################
# Toy Potentials #
##################
import numpy as np
from scipy.optimize import fmin_bfgs, leastsq, fmin_powell,fmin

# in units kpc, km/s and 10^11 M_solar
# Grav = 430091.5694
Grav = 430211.34883729415 # This was a bug in Sanders' code!

# Triaxial harmonic

def H_ho(x,omega):
    """ Simple harmonic oscillator Hamiltonian = 0.5 * omega**2 * x**2"""
    return 0.5*np.sum(x[3:]**2+(omega*x[:3])**2)


def angact_ho(x,omega):
    """ Calculate angle and action variable in sho potential with
    parameter omega """
    action = (x[3:]**2+(omega*x[:3])**2)/(2.*omega)
    angle = np.array([np.arctan(-x[3+i]/omega[i]/x[i]) if x[i]!=0. else -np.sign(x[3+i])*np.pi/2. for i in range(3)])
    for i in range(3):
        if(x[i]<0):
            angle[i]+=np.pi
    return np.concatenate((action,angle % (2.*np.pi)))


def deltaH_ho(omega,xsamples):
    if(np.any(omega<1e-5)):
        return np.nan
    H = 0.5*np.sum(xsamples.T[3:]**2,axis=0)+0.5*np.sum((omega[:3]*xsamples.T[:3].T)**2,axis=1)
    return H-np.mean(H)

def Jac_deltaH_ho(omega,xsamples):
    dHdparams = omega[:3]*xsamples.T[:3].T**2
    return dHdparams-np.mean(dHdparams,axis=0)

def findbestparams_ho(xsamples):
    """ Minimize sum of square differences of H_sho-<H_sho> for timesamples """
    return np.abs(leastsq(deltaH_ho,np.array([10.,10.,10.]), Dfun = Jac_deltaH_ho, args=(xsamples,))[0])[:3]


# Isochrone

def cart2spol(X):
    """ Performs coordinate transformation from cartesian
    to spherical polar coordinates with (r,phi,theta) having
    usual meanings. """
    x,y,z,vx,vy,vz=X
    r=np.sqrt(x*x+y*y+z*z)
    p=np.arctan2(y,x)
    t=np.arccos(z/r)
    vr=(vx*np.cos(p)+vy*np.sin(p))*np.sin(t)+np.cos(t)*vz
    vp=-vx*np.sin(p)+vy*np.cos(p)
    vt=(vx*np.cos(p)+vy*np.sin(p))*np.cos(t)-np.sin(t)*vz
    return np.array([r,p,t,vr,vp,vt])


def H_iso(x,params):
    """ Isochrone Hamiltonian = -GM/(b+sqrt(b**2+(r-r0)**2))"""
    #r = (np.sqrt(np.sum(x[:3]**2))-params[2])**2
    r = np.sum(x[:3]**2)
    return 0.5*np.sum(x[3:]**2)-Grav*params[0]/(params[1]+np.sqrt(params[1]**2+r))


def angact_iso(x,params):
    """ Calculate angle and action variable in isochrone potential with
    parameters params = (M,b) """
    GM = Grav*params[0]
    E = H_iso(x,params)
    r,p,t,vr,vphi,vt=cart2spol(x)
    st=np.sin(t)
    Lz=r*vphi*st
    L=np.sqrt(r*r*vt*vt+Lz*Lz/st/st)
    if(E>0.):  # Unbound
        return (np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)
    Jr=GM/np.sqrt(-2*E)-0.5*(L+np.sqrt(L*L+4*GM*params[1]))
    action = np.array([Jr,Lz,L-abs(Lz)])

    c=GM/(-2*E)-params[1]
    e=np.sqrt(1-L*L*(1+params[1]/c)/GM/c)
    eta=np.arctan2(r*vr/np.sqrt(-2.*E),params[1]+c-np.sqrt(params[1]**2+r*r))
    OmR=np.power(-2*E,1.5)/GM
    Omp=0.5*OmR*(1+L/np.sqrt(L*L+4*GM*params[1]))
    thetar=eta-e*c*np.sin(eta)/(c+params[1])

    if(abs(vt)>1e-10):
        psi=np.arctan2(np.cos(t),-np.sin(t)*r*vt/L)
    else:
        psi=np.pi/2.
    a=np.sqrt((1+e)/(1-e))
    ap=np.sqrt((1+e+2*params[1]/c)/(1-e+2*params[1]/c))
    F = lambda x,y: np.pi/2.-np.arctan(np.tan(np.pi/2.-0.5*y)/x) if y>np.pi/2. \
        else -np.pi/2.+np.arctan(np.tan(np.pi/2.+0.5*y)/x) if y<-np.pi/2. \
        else np.arctan(x*np.tan(0.5*y))

    thetaz=psi+Omp*thetar/OmR-F(a,eta)-F(ap,eta)/np.sqrt(1+4*GM*params[1]/L/L)

    LR=Lz/L
    sinu = LR/np.sqrt(1.-LR**2)/np.tan(t)
    u = 0
    if(sinu>1.):
        u=np.pi/2.
    elif(sinu<-1.):
        u = -np.pi/2.
    else:
        u = np.arcsin(sinu)
    if(vt>0.):
        u=np.pi-u
    thetap=p-u+np.sign(Lz)*thetaz
    angle = np.array([thetar,thetap,thetaz])
    return np.concatenate((action,angle % (2.*np.pi)))


def deltaH_iso(params,p,r):
    deltaH = p-Grav*params[0]/(params[1]+np.sqrt(params[1]**2+r))
    if(params[0]<0. or params[1]<0. or np.any(deltaH>0.)):
        return np.nan
    return (deltaH-np.mean(deltaH))
    # return JR-np.mean(JR)


def Jac_deltaH_iso(params,p,r):
    H_o = -Grav/(params[1]+np.sqrt(params[1]**2+r))
    H_1 = Grav*params[0]*(1.+params[1]/np.sqrt(params[1]**2+r))/(params[1]+np.sqrt(params[1]**2+r))**2
    return np.array([(H_o-np.mean(H_o)),(H_1-np.mean(H_1))])


def findbestparams_iso(xsamples):
    """ Minimize sum of square differences of H_iso-<H_iso> for timesamples"""
    p = 0.5*np.sum(xsamples.T[3:]**2,axis=0)
    r = np.sum(xsamples.T[:3]**2,axis=0)
    return np.abs(leastsq(deltaH_iso,np.array([10.,10.]), Dfun = None , col_deriv=1,args=(p,r,))[0]) #Jac_deltaH_iso
