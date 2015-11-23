# Solving the series of linear equations for true action
# and generating function Fourier components

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
import time

# in units kpc, km/s and 10^11 M_solar
Grav = 430091.5694
Conv = 0.97775

import toy_potentials as toy
import test_potentials as pot
import solver
import visualize_surfaces as vs
from solver import unroll_angles as ua


def choose_NT(N_max,iffreq=True):
    """ calculates number of time samples required to constrain N_max modes
        --- equation (21) from Sanders & Binney (2014) """
    if(iffreq):
        return max(200,9*N_max**3/4)
    else:
        return max(100,N_max**3/2)

def check_angle_solution(ang,n_vec,toy_aa,timeseries):
    """ Plots the toy angle solution against the toy angles ---
        Takes true angles and frequencies ang,
        the Fourier vectors n_vec,
        the toy action-angles toy_aa
        and the timeseries """
    f,a=plt.subplots(3,1)
    for i in range(3):
        a[i].plot(toy_aa.T[i+3],'.')
        size = len(ang[6:])/3
        AA = np.array([np.sum(ang[6+i*size:6+(i+1)*size]*np.sin(np.sum(n_vec*K,axis=1))) for K in toy_aa.T[3:].T])
        a[i].plot((ang[i]+ang[i+3]*timeseries-2.*AA) % (2.*np.pi),'.')
        a[i].set_ylabel(r'$\theta$'+str(i+1))
    a[2].set_xlabel(r'$t$')
    plt.show()

def check_target_angle_solution(ang,n_vec,toy_aa,timeseries):
    """ Plots the angle solution and the toy angles ---
        Takes true angles and frequencies ang,
        the Fourier vectors n_vec,
        the toy action-angles toy_aa
        and the timeseries """
    f,a=plt.subplots(3,1)
    for i in range(3):
        # a[i].plot(toy_aa.T[i+3],'.')
        size = len(ang[6:])/3
        AA = np.array([np.sum(ang[6+i*size:6+(i+1)*size]*np.sin(np.sum(n_vec*K,axis=1))) for K in toy_aa.T[3:].T])
        a[i].plot(((toy_aa.T[i+3]+2.*AA) % (2.*np.pi))-(ang[i]+timeseries*ang[i+3]) % (2.*np.pi),'.')
        a[i].plot(toy_aa.T[i+3],'.')
        a[i].set_ylabel(r'$\theta$'+str(i+1))
    a[2].set_xlabel(r'$t$')
    plt.show()

def eval_mean_error_functions(act,ang,n_vec,toy_aa,timeseries,withplot=False):
    """ Calculates sqrt(mean(E)) and sqrt(mean(F)) """

    Err = np.zeros(6)
    NT = len(timeseries)
    size = len(ang[6:])/3
    UA = ua(toy_aa.T[3:].T,np.ones(3))
    fig,axis=None,None
    if(withplot):
        fig,axis=plt.subplots(3,2)
        plt.subplots_adjust(wspace=0.3)
    for K in range(3):
        ErrJ = np.array([(i[K]-act[K]-2.*np.sum(n_vec.T[K]*act[3:]*np.cos(np.dot(n_vec,i[3:]))))**2 for i in toy_aa])
        Err[K] = np.sum(ErrJ)
        ErrT = np.array(((ang[K]+timeseries*ang[K+3]-UA.T[K]-2.*np.array([np.sum(ang[6+K*size:6+(K+1)*size]*np.sin(np.sum(n_vec*i,axis=1))) for i in toy_aa.T[3:].T])))**2)
        Err[K+3] = np.sum(ErrT)
        if(withplot):
            axis[K][0].plot(ErrJ,'.')
            axis[K][0].set_ylabel(r'$E$'+str(K+1))
            axis[K][1].plot(ErrT,'.')
            axis[K][1].set_ylabel(r'$F$'+str(K+1))

    if(withplot):
        for i in range(3):
            axis[i][0].set_xlabel(r'$t$')
            axis[i][1].set_xlabel(r'$t$')
        plt.show()

    EJ = np.sqrt(Err[:3]/NT)
    ET = np.sqrt(Err[3:]/NT)

    return np.array([EJ,ET])

def box_actions(results, times, N_matrix, ifprint):
    """
        Finds actions, angles and frequencies for box orbit.
        Takes a series of phase-space points from an orbit integration at times t and returns
        L = (act,ang,n_vec,toy_aa, pars) -- explained in find_actions() below.
    """
    if(ifprint):
        print("\n=====\nUsing triaxial harmonic toy potential")

    t = time.time()
    # Find best toy parameters
    omega = toy.findbestparams_ho(results)
    if(ifprint):
        print("Best omega "+str(omega)+" found in "+str(time.time()-t)+" seconds")

    # Now find toy actions and angles
    AA = np.array([toy.angact_ho(i,omega) for i in results])
    AA = AA[~np.isnan(AA).any(1)]
    if(len(AA)==0):
        return

    t = time.time()
    act = solver.solver(AA, N_matrix)
    if act==None:
        return

    if(ifprint):
        print("Action solution found for N_max = "+str(N_matrix)+", size "+str(len(act[0]))+" symmetric matrix in "+str(time.time()-t)+" seconds")

    np.savetxt("GF.Sn_box",np.vstack((act[1].T,act[0][3:])).T)

    ang = solver.angle_solver(AA,times,N_matrix,np.ones(3))
    if(ifprint):
        print("Angle solution found for N_max = "+str(N_matrix)+", size "+str(len(ang))+" symmetric matrix in "+str(time.time()-t)+" seconds")

    # Just some checks
    if(len(ang)>len(AA)):
        print("More unknowns than equations")

    return act[0], ang, act[1], AA, omega


def loop_actions(results, times, N_matrix, ifprint):
    """
        Finds actions, angles and frequencies for loop orbit.
        Takes a series of phase-space points from an orbit integration at times t and returns
        L = (act,ang,n_vec,toy_aa, pars) -- explained in find_actions() below.
        results must be oriented such that circulation is about the z-axis
    """
    if(ifprint):
        print("\n=====\nUsing isochrone toy potential")

    t = time.time()
    # First find the best set of toy parameters
    params = toy.findbestparams_iso(results)
    if(params[0]!=params[0]):
        params = np.array([10.,10.])
    if(ifprint):
        print("Best params "+str(params)+" found in "+str(time.time()-t)+" seconds")

    # Now find the toy angles and actions in this potential
    AA = np.array([toy.angact_iso(i,params) for i in results])
    AA = AA[~np.isnan(AA).any(1)]
    if(len(AA)==0):
        return

    t = time.time()
    act = solver.solver(AA, N_matrix,symNx = 1)
    if act==None:
        return

    if(ifprint):
        print("Action solution found for N_max = "+str(N_matrix)+", size "+str(len(act[0]))+" symmetric matrix in "+str(time.time()-t)+" seconds")

    # Store Sn
    np.savetxt("GF.Sn_loop",np.vstack((act[1].T,act[0][3:])).T)

    # Find angles
    sign = np.array([1.,np.sign(results[0][0]*results[0][4]-results[0][1]*results[0][3]),1.])
    ang = solver.angle_solver(AA,times,N_matrix,sign,symNx = 1)
    if(ifprint):
        print("Angle solution found for N_max = "+str(N_matrix)+", size "+str(len(ang))+" symmetric matrix in "+str(time.time()-t)+" seconds")

    # Just some checks
    if(len(ang)>len(AA)):
        print("More unknowns than equations")

    return act[0], ang, act[1], AA, params


def angmom(x):
    """ returns angular momentum vector of phase-space point x"""
    return np.array([x[1]*x[5]-x[2]*x[4],x[2]*x[3]-x[0]*x[5],x[0]*x[4]-x[1]*x[3]])


def assess_angmom(X):
    """
        Checks for change of sign in each component of the angular momentum.
        Returns an array with ith entry 1 if no sign change in i component
        and 0 if sign change.
        Box = (0,0,0)
        S.A loop = (0,0,1)
        L.A loop = (1,0,0)
    """
    L=angmom(X[0])
    loop = np.array([1,1,1])
    for i in X[1:]:
        L0 = angmom(i)
        if(L0[0]*L[0]<0.):
            loop[0] = 0
        if(L0[1]*L[1]<0.):
            loop[1] = 0
        if(L0[2]*L[2]<0.):
            loop[2] = 0
    return loop


def flip_coords(X,loop):
    """ Align circulation with z-axis """
    if(loop[0]==1):
        return np.array(map(lambda i: np.array([i[2],i[1],i[0],i[5],i[4],i[3]]),X))
    else:
        return X


def find_actions(results, t, N_matrix=8, use_box=False, ifloop=False, ifprint = True):
    """
        Main routine:
        Takes a series of phase-space points from an orbit integration at times t and returns
        L = (act,ang,n_vec,toy_aa, pars) where act is the actions, ang the initial angles and
        frequencies, n_vec the n vectors of the Fourier modes, toy_aa the toy action-angle
        coords, and pars are the toy potential parameters
        N_matrix sets the maximum |n| of the Fourier modes used,
        use_box forces the routine to use the triaxial harmonic oscillator as the toy potential,
        ifloop=True returns orbit classification,
        ifprint=True prints progress messages.
    """

    # Determine orbit class
    loop = assess_angmom(results)
    arethereloops = np.any(loop>0)
    if(arethereloops and not use_box):
        L = loop_actions(flip_coords(results,loop),t,N_matrix, ifprint)
        if(L==None):
            if(ifprint):
                print "Failed to find actions for this orbit"
            return
        # Used for switching J_2 and J_3 for long-axis loop orbits
        # This is so the orbit classes form a continuous plane in action space
        # if(loop[0]):
        #     L[0][1],L[0][2]=L[0][2],L[0][1]
        #     L[1][1],L[1][2]=L[1][2],L[1][1]
        #     L[1][4],L[1][5]=L[1][5],L[1][4]
        #     L[3].T[1],L[3].T[2]=L[3].T[2],L[3].T[1]
    else:
        L = box_actions(results,t,N_matrix, ifprint)
        if(L==None):
            if(ifprint):
                print "Failed to find actions for this orbit"
            return
    if(ifloop):
        return L,loop
    else:
        return L

###################
#  Plotting tests #
###################
from solver import check_each_direction as ced

def plot_Sn_timesamples(PSP):
    """ Plots Fig. 5 from Sanders & Binney (2014) """
    TT = pot.stackel_triax()
    f,a = plt.subplots(2,1,figsize=[3.32,3.6])
    plt.subplots_adjust(hspace=0.,top=0.8)

    LowestPeriod = 2.*np.pi/38.86564386
    Times = np.array([2.,4.,8.,12.])
    Sr = np.arange(2,14,2)

    # Loop over length of integration window
    for i,P,C in zip(Times,['.','s','D','^'],['k','r','b','g']):
        diffact = np.zeros((len(Sr),3))
        difffreq = np.zeros((len(Sr),3))
        MAXGAPS = np.array([])
        # Loop over N_max
        for k,j in enumerate(Sr):
            NT = choose_NT(j)
            timeseries=np.linspace(0.,i*LowestPeriod,NT)
            results = odeint(pot.orbit_derivs2,PSP,timeseries,args=(TT,),rtol=1e-13,atol=1e-13)
            act,ang,n_vec,toy_aa, pars = find_actions(results, timeseries,N_matrix=j,ifprint=False,use_box=True)
            # Check all modes
            checks,maxgap = ced(n_vec,ua(toy_aa.T[3:].T,np.ones(3)))
            if len(maxgap)>0:
                maxgap = np.max(maxgap)
            else:
                maxgap = 0
            diffact[k] = act[:3]/TT.action(results[0])
            print i,j,print_max_average(n_vec,toy_aa.T[3:].T,act[3:]),str(ang[3:6]-TT.freq(results[0])).replace('[','').replace(']',''),str(np.abs(act[:3]-TT.action(results[0]))).replace('[','').replace(']',''),len(checks),maxgap
            MAXGAPS = np.append(MAXGAPS,maxgap)
            difffreq[k] = ang[3:6]/TT.freq(results[0])
        size = 15
        if(P=='.'):
            size = 30
        LW = np.array(map(lambda i: 0.5+i*0.5, MAXGAPS))
        a[0].scatter(Sr,np.log10(np.abs(diffact.T[2]-1)),marker=P,s=size, color=C,facecolors="none",lw=LW,label=r'$T =\,$'+str(i)+r'$\,T_F$')
        a[1].scatter(Sr,np.log10(np.abs(difffreq.T[2]-1)),marker=P,s=size, color=C,facecolors="none", lw=LW)
    a[1].get_yticklabels()[-1].set_visible(False)
    a[0].set_xticklabels([])
    a[0].set_xlim(1,13)
    a[0].set_ylabel(r"$\log_{10}|J_3^\prime/J_{3, \rm true}-1|$")
    leg = a[0].legend(loc='upper center',bbox_to_anchor=(0.5,1.4),ncol=2, scatterpoints = 1)
    leg.draw_frame(False)
    a[1].set_xlim(1,13)
    a[1].set_xlabel(r'$N_{\rm max}$')
    a[1].set_ylabel(r"$\log_{10}|\Omega_3^\prime/\Omega_{3,\rm true}-1|$")
    plt.savefig('Sn_T_box.pdf',bbox_inches='tight')


def plot3D_stacktriax(initial,final_t,N_MAT,file_output):
    """ For producing plots from paper """

    # Setup Stackel potential
    TT = pot.stackel_triax()
    times = choose_NT(N_MAT)
    timeseries=np.linspace(0.,final_t,times)
    # Integrate orbit
    results = odeint(pot.orbit_derivs2,initial,timeseries,args=(TT,),rtol=1e-13,atol=1e-13)
    # Find actions, angles and frequencies
    (act,ang,n_vec,toy_aa, pars),loop = find_actions(results, timeseries,N_matrix=N_MAT,ifloop=True)

    toy_pot = 0
    if(loop[2]>0.5 or loop[0]>0.5):
        toy_pot = pot.isochrone(par=np.append(pars,0.))
    else:
        toy_pot = pot.harmonic_oscillator(omega=pars[:3])
    # Integrate initial condition in toy potential
    timeseries_2=np.linspace(0.,2.*final_t,3500)
    results_toy = odeint(pot.orbit_derivs2,initial,timeseries_2,args=(toy_pot,))

    print "True actions: ", TT.action(results[0])
    print "Found actions: ", act[:3]
    print "True frequencies: ",TT.freq(results[0])
    print "Found frequencies: ",ang[3:6]


    # and plot
    f,a = plt.subplots(2,3,figsize=[3.32,5.5])
    a[0,0] = plt.subplot2grid((3,2), (0, 0))
    a[1,0] = plt.subplot2grid((3,2), (0, 1))
    a[0,1] = plt.subplot2grid((3,2), (1, 0))
    a[1,1] = plt.subplot2grid((3,2), (1, 1))
    a[0,2] = plt.subplot2grid((3,2), (2, 0),colspan=2)
    plt.subplots_adjust(wspace=0.5,hspace=0.45)

    # xy orbit
    a[0,0].plot(results.T[0],results.T[1],'k')
    a[0,0].set_xlabel(r'$x/{\rm kpc}$')
    a[0,0].set_ylabel(r'$y/{\rm kpc}$')
    a[0,0].xaxis.set_major_locator(MaxNLocator(5))
    # xz orbit
    a[1,0].plot(results.T[0],results.T[2],'k')
    a[1,0].set_xlabel(r'$x/{\rm kpc}$')
    a[1,0].set_ylabel(r'$z/{\rm kpc}$')
    a[1,0].xaxis.set_major_locator(MaxNLocator(5))
    # toy orbits
    a[0,0].plot(results_toy.T[0],results_toy.T[1],'r',alpha=0.2,linewidth=0.3)
    a[1,0].plot(results_toy.T[0],results_toy.T[2],'r',alpha=0.2,linewidth=0.3)

    # Toy actions
    a[0,2].plot(Conv*timeseries,toy_aa.T[0],'k:',label='Toy action')
    a[0,2].plot(Conv*timeseries,toy_aa.T[1],'r:')
    a[0,2].plot(Conv*timeseries,toy_aa.T[2],'b:')
    # Arrows to show approx. actions
    arrow_end = a[0,2].get_xlim()[1]
    arrowd = 0.08*(arrow_end-a[0,2].get_xlim()[0])
    a[0,2].annotate('',(arrow_end+arrowd,act[0]),(arrow_end,act[0]),arrowprops=dict(arrowstyle='<-',color='k'),annotation_clip=False)
    a[0,2].annotate('',(arrow_end+arrowd,act[1]),(arrow_end,act[1]),arrowprops=dict(arrowstyle='<-',color='r'),annotation_clip=False)
    a[0,2].annotate('',(arrow_end+arrowd,act[2]),(arrow_end,act[2]),arrowprops=dict(arrowstyle='<-',color='b'),annotation_clip=False)
    # True actions
    a[0,2].plot(Conv*timeseries,TT.action(results[0])[0]*np.ones(len(timeseries)),'k',label='True action')
    a[0,2].plot(Conv*timeseries,TT.action(results[0])[1]*np.ones(len(timeseries)),'k')
    a[0,2].plot(Conv*timeseries,TT.action(results[0])[2]*np.ones(len(timeseries)),'k')
    a[0,2].set_xlabel(r'$t/{\rm Gyr}$')
    a[0,2].set_ylabel(r'$J/{\rm kpc\,km\,s}^{-1}$')
    leg = a[0,2].legend(loc='upper center',bbox_to_anchor=(0.5,1.2),ncol=3, numpoints = 1)
    leg.draw_frame(False)

    # Toy angle coverage
    a[0,1].plot(toy_aa.T[3]/(np.pi),toy_aa.T[4]/(np.pi),'k.',markersize=0.4)
    a[0,1].set_xlabel(r'$\theta_1/\pi$')
    a[0,1].set_ylabel(r'$\theta_2/\pi$')
    a[1,1].plot(toy_aa.T[3]/(np.pi),toy_aa.T[5]/(np.pi),'k.',markersize=0.4)
    a[1,1].set_xlabel(r'$\theta_1/\pi$')
    a[1,1].set_ylabel(r'$\theta_3/\pi$')

    plt.savefig(file_output,bbox_inches='tight')
    return act

if __name__=="__main__":
    BoxP = np.array([0.1,0.1,0.1,142.,140.,251.])
    LoopP = np.array([10.,1.,8.,40.,152.,63.])
    ResP = np.array([0.1,0.1,0.1,142.,150.,216.5])
    LongP = np.array([-0.5,18.,0.5,25.,20.,-133.1])

    # Short-axis Loop
    LowestPeriodLoop = 2*np.pi/15.30362865
    # Fig 1
    loop = plot3D_stacktriax(LoopP,8*LowestPeriodLoop,6,'genfunc_3d_example_LT_Stack_Loop.pdf')
    # Fig 3
    vs.Sn_plots('GF.Sn_loop','loop',loop,1)

    # Box
    LowestPeriodBox = 2.*np.pi/38.86564386
    # Fig 2
    box = plot3D_stacktriax(BoxP,8*LowestPeriodBox,6,'genfunc_3d_example_LT_Stack_Box.pdf')
    # Fig 4
    vs.Sn_plots('GF.Sn_box','box',box,0)

    # Res
    LowestPeriodRes = 2.*np.pi/42.182
    # Fig 5
    res = plot3D_stacktriax(ResP,8*LowestPeriodBox,6,'genfunc_3d_example_LT_Stack_Res.pdf')
    # vs.Sn_plots('GF.Sn_box','box',res,0)

    # Long-axis loop
    LowestPeriodLong = 2.*np.pi/12.3
