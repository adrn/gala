"""
Note: these functions are only used in tests to compare against Koposov's implementation.
"""

from numpy import sin, cos, deg2rad, rad2deg, arctan2, sqrt
import numpy
import numexpr

def cv_coord(a, b, c, fr=None, to=None, degr=False):
    if degr:
        degrad = deg2rad
        raddeg = rad2deg
    else:
        degrad = lambda x: x
        raddeg = lambda x: x
    if fr=='sph':
        x=c*cos(degrad(a))*cos(degrad(b))
        y=c*sin(degrad(a))*cos(degrad(b))
        z=c*sin(degrad(b))
    elif fr=='rect':
        x=a
        y=b
        z=c
    elif fr is None:
        raise Exception('You must specify the input coordinate system')
    else:
        raise Exception('Unknown input coordinate system')
    if to=='rect':
        return (x, y, z)
    elif to=='sph':
        ra = raddeg(arctan2(y, x))
        dec = raddeg(arctan2(z, sqrt(x**2+y**2)))
        rad = sqrt(x**2+y**2+z**2)
        return (ra, dec, rad)
    elif to is None:
        raise Exception('You must specify the output coordinate system')
    else:
        raise Exception('Unknown output coordinate system')

def torect(ra, dec):
    x=numexpr.evaluate('cos(ra/57.295779513082323)*cos(dec/57.295779513082323)')
    y=numexpr.evaluate('sin(ra/57.295779513082323)*cos(dec/57.295779513082323)')
    z=numexpr.evaluate('sin(dec/57.295779513082323)')
    return x, y, z

def fromrect(x, y, z):
    ra=numexpr.evaluate('arctan2(y, x)*57.295779513082323')
    dec=numexpr.evaluate('57.295779513082323*arctan2(z, sqrt(x**2+y**2))')
    return ra, dec

def sphere_rotate(ra, dec, rapol, decpol, ra0):
    """ rotate ra, dec to a new spherical coordinate system where the pole is
        at rapol, decpol and the zeropoint is at ra=ra0
        revert flag allows to reverse the transformation
    """

    x, y, z=torect(ra, dec)

    tmppol=cv_coord(rapol, decpol, 1, degr=True, fr='sph',to='rect') #pole axis
    tmpvec1=cv_coord(ra0, 0, 1, degr=True, fr='sph',to='rect') #x axis
    tmpvec1=numpy.array(tmpvec1)

    tmpvec1[2]=(-tmppol[0]*tmpvec1[0]-tmppol[1]*tmpvec1[1])/tmppol[2]
    tmpvec1/=numpy.sqrt((tmpvec1**2).sum())
    tmpvec2=numpy.cross(tmppol, tmpvec1) # y axis

    Axx, Axy, Axz=tmpvec1
    Ayx, Ayy, Ayz=tmpvec2
    Azx, Azy, Azz=tmppol

    xnew = numexpr.evaluate('x*Axx+y*Axy+z*Axz')
    ynew = numexpr.evaluate('x*Ayx+y*Ayy+z*Ayz')
    znew = numexpr.evaluate('x*Azx+y*Azy+z*Azz')

    del x, y, z
    tmp = fromrect(xnew, ynew, znew)
    return (tmp[0],tmp[1])
