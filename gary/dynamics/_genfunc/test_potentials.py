#################
#   Potentials  #
#################
import numpy as np
from scipy.integrate import quad, ode

Grav = 430091.5694

class LMPot(object):
    """ Potential used in Law-Majewski 2010 """
    def __init__(self):
        """ Best-fit parameters - units = kpc, km/s and 10^11 M_sol """
        self.M_disk = 1.
        self.a_disk = 6.5
        self.b_disk = 0.26
        self.M_bulge = 0.34
        self.c_bulge = 0.7
        self.vhalo2 = 121.7**2
        phi = 97./180.*np.pi
        q1 = 1.38
        q2 = 1.
        self.C_1 = (np.cos(phi)/q1)**2+(np.sin(phi)/q2)**2
        self.C_2 = (np.cos(phi)/q2)**2+(np.sin(phi)/q1)**2
        self.C_3 = 2.*np.sin(phi)*np.cos(phi)*(1./q1/q1-1./q2/q2)
        self.q_z = 1.36
        self.rhalo2 = 144.
        rot90 = np.array([[0.,1.],[-1.,0.]])
        self.rotmatrix =  np.dot(rot90,np.linalg.svd(np.array([[self.C_1,self.C_3/2.],[self.C_3/2.,self.C_2]]))[0])
        self.invrotmatrix = np.linalg.inv(self.rotmatrix)

    def disk_pot(self,x,y,z):
        R = np.sqrt(x*x+y*y)
        return -Grav*self.M_disk/np.sqrt(R*R+(self.a_disk+np.sqrt(z*z+self.b_disk*self.b_disk))**2)

    def disk_force(self,x,y,z):
        R = np.sqrt(x*x+y*y)
        e = self.a_disk+np.sqrt(z*z+self.b_disk*self.b_disk)
        d = -Grav*self.M_disk/np.sqrt(R*R+e**2)**3
        return np.array([x*d, y*d, z*d*e/(e-self.a_disk)])

    def bulge_pot(self,x,y,z):
        r = np.sqrt(x*x+y*y+z*z)
        return -Grav*self.M_bulge/(r+self.c_bulge)

    def bulge_force(self,x,y,z):
        r = np.sqrt(x*x+y*y+z*z)
        if(r==0.):
            return -Grav*self.M_bulge/(r+self.c_bulge)**2
        else:
            return -Grav*self.M_bulge/(r+self.c_bulge)**2/r*np.array([x,y,z])

    def halo_pot(self,x,y,z):
        return self.vhalo2*np.log(self.C_1*x*x+self.C_2*y*y+self.C_3*x*y+(z/self.q_z)**2+self.rhalo2)

    def halo_force(self,x,y,z):
        p = -self.vhalo2/(self.C_1*x*x+self.C_2*y*y+self.C_3*x*y+(z/self.q_z)**2+self.rhalo2)
        return np.array([(2.*x*self.C_1+self.C_3*y)*p,(2.*y*self.C_2+self.C_3*x)*p,2.*z*p/self.q_z**2])

    def tot_pot(self,x,y,z):
        return self.disk_pot(x,y,z)+self.bulge_pot(x,y,z)+self.halo_pot(x,y,z)

    def H(self,X):
        return 0.5*np.sum(X[3:]**2)+self.tot_pot(*X[:3])

    def tot_force(self,x,y,z):
        return self.disk_force(x,y,z)+self.bulge_force(x,y,z)+self.halo_force(x,y,z)

    def coordrot(self,x,y):
        return np.dot(self.rotmatrix,np.array([x,y]))

    def invcoordrot(self,x,y):
        return np.dot(self.invrotmatrix,np.array([x,y]))


class log_triax(object):
    """ test triaxial logarithmic potential
        Phi(x,y,z) = 0.5 v_c^2 log(Rc^2+x^2+(y/qy)^2+(z/qz)^2) """
    def __init__(self,vc,Rc,qy,qz):
        self.vc2=vc*vc
        self.Rc2=Rc*Rc
        self.qy2=qy*qy
        self.qz2=qz*qz

    def pot(self,x,y,z):
        return self.vc2/2.*np.log(self.Rc2+x*x+y*y/self.qy2+z*z/self.qz2)

    def H(self, X):
        return 0.5*np.sum(X[3:]**2)+self.pot(*X[:3])

    def tot_force(self,x,y,z):
        p = self.Rc2+x*x+y*y/self.qy2+z*z/self.qz2
        return -self.vc2/p*np.array([x,y/self.qy2,z/self.qz2])


class quartic(object):
    """ Quartic potential
        Phi(x,y,z) = 0.25(lam[0] x^4+lam[1] y^4+lam[2] z^4.
    """

    def __init__(self,lam = np.array([1.,0.8,3.3])):
        self.lambd = lam

    def H(self,x):
        """ Quartic potential Hamiltonian """
        return 0.5*np.sum(x[3:]**2+0.5*self.lambd*x[:3]**4)

    def tot_force(self,x,y,z):
        """ Derivatives of quartic potential for orbit integration """
        return np.array([-self.lambd[0]*x**3,-self.lambd[1]*y**3,-self.lambd[2]*z**3])

    def action(self,x):
        """ Find true action for quartic potential \Phi = \sum_i 0.25*x_i**4 """
        acts = np.ones(3)
        for i in range(3):
            En = 0.5*x[i+3]**2+0.25*self.lambd[i]*x[i]**4
            xlim=(4.*En/self.lambd[i])**0.25
            acts[i]=2.*quad(lambda y:np.sqrt(2.*En-0.5*self.lambd[i]*y**4),0.,xlim)[0]/np.pi
        return acts

    def freq(self,x):
        """ Find true freq. for quartic potential \Phi = 0.25*x**4 """
        freq = np.ones(3)
        for i in range(3):
            En = 0.5*x[i+3]**2+0.25*self.lambd[i]*x[i]**4
            xlim=(4.*En/self.lambd[i])**0.25
            freq[i]=np.pi/quad(lambda y:2./np.sqrt(2.*En-0.5*self.lambd[i]*y**4),0.,xlim)[0]
        return freq

import sys
# sys.path.append("new_struct")
# import triax_py

class stackel_triax(object):
    """ For interface with C code to find actions in triaxial Stackel potential """
    def __init__(self):
        pass

    def H(self,x):
        """ triaxial stackel potential Hamiltonian """
        return triax_py.Stack_Triax_H(x)

    def tot_force(self,x,y,z):
        """ Derivatives of triaxial stackel potential for orbit integration """
        X = np.array([x,y,z])
        return triax_py.Stack_Triax_Forces(X)

    def action(self,x):
        """ Find true action for triaxial stackel potential """
        return triax_py.Stack_Triax_Actions(x)

    def freq(self,x):
        """ Find true action for triaxial stackel potential """
        return triax_py.Stack_Triax_Freqs(x)


class harmonic_oscillator(object):
    """
        Triaxial harmonic oscillator
        Phi(x,y,z) = 0.5*(omega[0]^2 x^2+omega[1]^2 y^2 + omega[2]^2 z^2
    """
    def __init__(self,omega=np.array([1.,1.,1.])):
        self.omega = omega

    def H(self,x):
        """ Hamiltonian """
        return 0.5*np.sum(x[3:]**2+(self.omega*x[:3])**2)

    def tot_force(self,x,y,z):
        """ Derivatives of ho potential for orbit integration """
        return -np.array([self.omega[0]**2*x,self.omega[1]**2*y,self.omega[2]**2*z])


class isochrone(object):
    """
        Isochrone potential
        Phi(r) = -GM/(b+sqrt(b^2+r^2))
    """
    def __init__(self,par = np.array([1./Grav,4.2,0.])):
        """ params = {M, b, r0} """
        self.params = par

    def H(self,x):
        """ Hamiltonian """
        r = (np.sqrt(np.sum(x[:3]**2))-self.params[2])**2
        return 0.5*np.sum(x[3:]**2)-Grav*self.params[0]/(self.params[1]+np.sqrt(self.params[1]**2+r))

    def pot(self,x):
        r = (np.sqrt(np.sum(x[:3]**2))-self.params[2])**2
        return -Grav*self.params[0]/(self.params[1]+np.sqrt(self.params[1]**2+r))

    def tot_force(self,x,y,z):
        """ Derivatives of isochrone potential for orbit integration """
        r = (np.sqrt(x*x+y*y+z*z)-self.params[2])**2
        fac = np.sqrt(r)/(np.sqrt(r)+self.params[2])
        return np.array([x,y,z])*fac*-Grav*self.params[0]/(self.params[1]+np.sqrt(self.params[1]**2+r))**2/np.sqrt(self.params[1]**2+r)


def orbit_derivs(t,x,Pot):
    """ Simple interface for derivatives for orbit integration
        t = time
        x = Cartesian coordinates
        Pot is an object which has a function tot_force(x,y,z) which
        calculates the total force at Cartesian x,y,z """
    X=x[0]
    Y=x[1]
    Z=x[2]
    return np.concatenate((x[3:],Pot.tot_force(X,Y,Z)))

def orbit_derivs2(x,t,Pot):
    return orbit_derivs(t,x,Pot)

import warnings

def orbit_integrate(x,tmax,Pot):
    """ Integrates an orbit with initial coordinates x for time tmax in
    potential Pot using Dormund Prince 8 adaptive step size """
    solver = ode(orbit_derivs).set_integrator('dopri5', n_steps=1, rtol=1e-10,atol=1e-10)
    solver.set_initial_value(x,0.).set_f_params(Pot)
    solver._integrator.iwork[2] = -1
    warnings.filterwarnings("ignore", category=UserWarning)
    t = np.array([0.])
    while solver.t < tmax:
        solver.integrate(tmax)
        x=np.vstack((x,solver.y))
        t=np.append(t,solver.t)
    warnings.resetwarnings()
    return x,t

def leapfrog_integrator(x,tmax,NT,Pot):
    deltat = tmax/NT
    h = deltat/100.
    t = 0.
    counter = 0
    X = np.copy(x)
    results = np.array([x])
    while(t<tmax):
        X[3:] += 0.5*h*Pot.tot_force(X[0],X[1],X[2])
        X[:3] += h*X[3:]
        X[3:] += 0.5*h*Pot.tot_force(X[0],X[1],X[2])
        # if(t==0.1):
        if(counter % 100 == 0):
            results=np.vstack((results,X))
        t+=h
        counter+=1
    return results
