#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Aaron LI
# 2015/07/01
#

"""
Generate random numbers (i.e., fluxes) with respect to the
provided luminosity function.
"""

import numpy as np
import random

def luminosity_func(Lx, N0=1.0):
    """
    The *cumulative* luminosity function: N(>=L)
    The number of objects with luminosities >= L(x) for each L(x).
    """
    # broken power-law model (Xu et al. 2005)
    # Nx = (1) N0 * (Lx/L_b)^(-alpha_l);  for Lx <= L_b
    #      (2) N0 * (Lx/L_b)^(-alpha_h);  for Lx > L_b
    L_b = 4.4e38 # break point (erg/s) (+2.0/-1.4)
    alpha_h = 2.28 # (+1.72/-0.53)
    alpha_l = 1.08 # (+0.15/-0.33)
    if isinstance(Lx, np.ndarray):
        Nx = np.zeros(Lx.shape)
        Nx[Lx <= 0] = 0.0
        Nx[Lx <= L_b] = N0 * (Lx[Lx <= L_b] / L_b)**(-alpha_l)
        Nx[Lx > L_b] = N0 * (Lx[Lx > L_b] / L_b)**(-alpha_h)
    else:
        # Lx is a single number
        if Lx <= 0.0:
            Nx = 0.0
        elif Lx <= L_b:
            Nx = N0 * (Lx/L_b)**(-alpha_l)
        else:
            Nx = N0 * (Lx/L_b)**(-alpha_h)
    return Nx


def luminosity_density(Lx, N0=1.0):
    """
    Function of number density at luminosity at Lx. => PDF

    PDF(Lx) = - d(luminosity_func(Lx) / d(Lx)
    """
    L_b = 4.4e38 # break point (erg/s) (+2.0/-1.4)
    alpha_h = 2.28 # (+1.72/-0.53)
    alpha_l = 1.08 # (+0.15/-0.33)
    if isinstance(Lx, np.ndarray):
        Px = np.zeros(Lx.shape)
        Px[Lx<=0] = 0.0
        Px[Lx<=L_b] = N0 * (alpha_l/L_b) * (Lx[Lx<=L_b] / L_b)**(-alpha_l-1)
        Px[Lx>L_b] = N0 * (alpha_h/L_b) * (Lx[Lx>L_b] / L_b)**(-alpha_h-1)
    else:
        # Lx is a single number
        if Lx <= 0.0:
            Px = 0.0
        elif Lx <= L_b:
            Px = N0 * (alpha_l/L_b) * (Lx/L_b)**(-alpha_l-1)
        else:
            Px = N0 * (alpha_h/L_b) * (Lx/L_b)**(-alpha_h-1)
    return Px


def luminosity_pdf(Lx):
    """
    Probability density function
    """
    h = 1e-5 * Lx # step size for numerical deviation
    p = - (luminosity_func(Lx+0.5*h) - luminosity_func(Lx-0.5*h)) / h
    return p


def sampler(min, max, number=1):
    """
    Generate a sample of luminosity values within [min, max] from
    the above luminosity distribution.
    """
    # Get the maximum value of the density function
    M = luminosity_density(min)
    results = []
    for i in range(number):
        while True:
            u = random.random() * M
            y = random.random() * (max-min) + min
            if u <= luminosity_density(y):
                results.append(y)
                break
    if len(results) == 1:
        return results[0]
    else:
        return np.array(results)

