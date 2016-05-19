# For plotting the S_n

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

def meshgrid2(*arrs):
    arrs = tuple(reversed(arrs))  #edit
    lens = map(len, arrs)
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz*=s

    ans = []    
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j!=i:
                arr2 = arr2.repeat(sz, axis=j) 
        ans.append(arr2)

    return tuple(ans)

def conv(x,acts):
    return np.log((np.exp(x)-0.1)/acts+0.1)

def Sn_plots(inp,outp,actions,loop):
    Sn = np.genfromtxt(inp)
    loop_acts = actions
    acts = np.sum(loop_acts)
    acts = 1
    dx = 2
    if(loop):
        dx = 1
    x = np.arange(np.min(Sn.T[0])-2,np.max(Sn.T[0])+2,dx)
    y = np.arange(np.min(Sn.T[1])-4,np.max(Sn.T[1])+4,2)
    z = np.arange(-np.max(Sn.T[2])-2,np.max(Sn.T[2])+4,2)
    length_x = len(x)
    length_y = len(y)
    length_z = len(z)
    
    X,Y,Z = meshgrid2(x,y,z)
    S = np.zeros(np.shape(X.T))
    for i in Sn:
        xindex = np.where(np.abs(x-i[0])<1e-5)[0][0]
        yindex = np.where(np.abs(y-i[1])<1e-5)[0][0]
        zindex = np.where(np.abs(z-i[2])<1e-5)[0][0]
        S[xindex][yindex][zindex]=np.log(np.abs(i[3]/acts)+0.1)
        S[length_x-xindex-1][length_y-yindex-1][length_z-zindex-1]=np.log(np.abs(i[3]/acts)+0.1)
    
    R = np.zeros((length_x,length_z))
    R2 = np.zeros((length_x,length_y))
    for j,i in enumerate(S):
        R[j]=i[length_y/2]

    f,a = plt.subplots(2,1,figsize=[3.32,3.6])
    plt.subplots_adjust(right=0.75,wspace=0.4,hspace=0.3)
    conts = map(lambda x: conv(x,acts),[0.15,0.5,1.,1.5,2.,2.5])
    if(loop):
        a[0].set_xlim(-6,6)
        a[0].set_ylim(-6,6)
        a[1].set_xlim(-6,6)
        a[1].set_ylim(-6,6)
    else:
        a[0].set_xlim(-6,6)
        a[0].set_ylim(-6,6)
        a[1].set_xlim(-6,6)
        a[1].set_ylim(-6,6)

    a[0].contour(x,y,S.T[length_z/2],levels=conts)
    a[0].set_xlabel(r'$n_1$')
    a[0].set_ylabel(r'$n_2$')
    a[0].set_aspect('equal')
    a[0].text(2,4,r'$n_3=0$')
    a[0].text(3,7,outp)

    R = np.zeros((length_x,length_z))
    for j,i in enumerate(S):
        R[j]=i[length_y/2]
    
    cNorm = colors.Normalize(vmin=np.min(conts),vmax=np.max(conts))
    sM = cmx.ScalarMappable(norm=cNorm)
    sM._A = []
    C = a[1].contour(x,z,R.T,levels=conts)
    # print R.T
    a[1].set_aspect('equal')
    a[1].set_xlabel(r'$n_1$')
    a[1].set_ylabel(r'$n_3$')
    a[1].text(2,4.5,r'$n_2=0$')
    # a[1].set_xlim(np.min(y),np.max(y))
    cbar_ax = f.add_axes([0.75,0.15,0.05,0.7])
    ccc = f.colorbar(sM,cax=cbar_ax)
    ccc.set_label(r'$\log(|S_n / \rm{kpc}\,\rm{km}\,\rm{s}^{-1}|+0.1)$')
    plt.savefig(outp+"_planes.pdf")
