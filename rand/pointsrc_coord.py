# -*- coding: utf-8 -*-
#
# Aaron LI
# 2015/07/01
#

"""
Generate random coordinates for point sources with respect to the r^{1/4}
distribution.
"""

import numpy as np
import random


def cdf(r, N0=1.0):
    """
    Cumulative distribution function of the number of point sources.

    r^{1/4} distribution law: de Vaucouleurs 1948
    """
    return N0 * r**(1.0/4.0)


def pdf(r, N0=1.0):
    """
    Density function of the number of point sources.

    pdf = d(pdf) / d(r)
    """
    if isinstance(r, np.ndarray):
        p = np.zeros(r.shape)
        p[r<=0.0] = 0.0
        p[r>0.0] = 0.25 * N0 * r[r>0.0]**(-3.0/4.0)
    else:
        if r <= 0.0:
            p = 0.0
        else:
            p = 0.25 * N0 * r**(-3.0/4.0)
    return p


def sampler(min, max, number=1):
    """
    Generate a sample of coordinates (only r) within [min, max] from
    the above density distribution.

    min, max: the minimum and maximum r values (in degree)
    """
    # Get the maximum value of the density function
    M = pdf(min)
    results = []
    for i in range(number):
        while True:
            u = random.random() * M
            y = random.random() * (max-min) + min
            if u <= pdf(y):
                results.append(y)
                break
    if len(results) == 1:
        return results[0]
    else:
        return np.array(results)


def add_angle(r):
    """
    Add angle for each r value to make up a coordinate of a polar coordinate.
    """
    coords = []
    for ri in r:
        theta = random.random() * 360
        coords.append((ri, theta))
    if len(coords) == 1:
        return coords[0]
    else:
        return coords


def to_radec(coords, xc=0, yc=0):
    """
    Convert the generated coordinates to (ra, dec) (unit: degree).

    xc, yc: the center coordinate (ra, dec)
    """
    results = []
    for r, theta in coords:
        # FIXME: spherical algebra should be used!!!
        dx = r * np.cos(theta*np.pi/180)
        dy = r * np.sin(theta*np.pi/180)
        x = xc + dx
        y = yc + dy
        results.append((x, y))
    if len(results) == 1:
        return results[0]
    else:
        return results

