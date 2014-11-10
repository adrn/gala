# coding: utf-8

""" Utility functions for handling astronomical time conversions """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import astropy.units as u
from astropy.time import Time
from datetime import datetime, timedelta, time

__all__ = ["sex_to_dec", "dec_to_sex", "gmst_to_utc", "utc_to_gmst", \
           "gmst_to_lmst", "lmst_to_gmst"]

def sex_to_dec(x, ms=False):
    """ Convert a sexagesimal representation to a decimal value.

        Parameters
        ----------
        x : tuple
            A length 3 tuple containing the components.
    """
    if ms:
        return x[0] + x[1]/60. + (x[2]+x[3]/1E6)/3600.
    else:
        return x[0] + x[1]/60. + x[2]/3600.

def dec_to_sex(x, ms=False):
    """ Convert a decimal value to a sexigesimal tuple.

        Parameters
        ----------
        x : numeric
    """
    a = int(x)
    _b = (x-a)*60.
    b = int(_b)

    if not ms:
        c = (_b - b)*60.
        return (a,b,c)
    else:
        _c = (_b - b)*60.
        c = int(_c)
        d = (_c-c)*1E6
        return (a,b,c,int(d))

def gmst_to_utc(t, utc_date):
    """ Convert a Greenwich Mean Sidereal Time into UTC time given a
        UTC date.

        Parameters
        ----------
        t : datetime.time
        utc_date : datetime.datetime
    """
    jd = int(Time(utc_date,scale='utc').jd) + 0.5

    S = jd - 2451545.0
    T = S / 36525.0
    T0 = 6.697374558 + (2400.051336*T) + (0.000025862*T**2)
    T0 = T0 % 24

    h = sex_to_dec((t.hour, t.minute, t.second))
    GST = (h - T0) % 24
    UT = GST * 0.9972695663

    tt = Time(jd, format='jd', scale='utc')
    dt = tt.datetime + timedelta(hours=UT)

    return Time(dt, scale='utc')

def utc_to_gmst(t):
    """ Convert a UTC time to Greenwich Mean Sidereal Time

        Parameters
        ----------
        t : astropy.time.Time
    """
    epoch = Time(datetime(2000,1,1,12,0,0), scale='utc')
    D = t - epoch
    D = (D.sec*u.second).to(u.day).value
    gmst = 18.697374558 + 24.06570982441908 * D
    return time(*dec_to_sex(gmst % 24, ms=True))

def gmst_to_lmst(t, longitude_w):
    """ Convert Greenwich to Local mean sidereal time, given a Longitude West

        Parameters
        ----------
        t : datetime.time
        longitude_w : astropy.units.Quantity
    """
    gmst_hours = sex_to_dec((t.hour,t.minute,t.second,t.microsecond), ms=True)
    long_hours = longitude_w.to(u.hourangle).value
    lmst_hours = (gmst_hours + (24. - long_hours)) % 24.
    return time(*dec_to_sex(lmst_hours, ms=True))

def lmst_to_gmst(t, longitude_w):
    """ Convert Local to Greenwich mean sidereal time, given a Longitude West

        Parameters
        ----------
        t : datetime.time
        longitude_w : astropy.units.Quantity
    """
    lmst_hours = sex_to_dec((t.hour,t.minute,t.second,t.microsecond), ms=True)
    long_hours = longitude_w.to(u.hourangle).value
    gmst_hours = (lmst_hours - (24. - long_hours)) % 24.
    return time(*dec_to_sex(gmst_hours, ms=True))