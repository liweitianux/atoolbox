#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Randomly pick point on the sphere surface.
#
# References:
# [1] Shpere Poin Picking -- from Wolfram MathWorld
#     http://mathworld.wolfram.com/SpherePointPicking.html
# [2] Random Points on a Sphere
#     https://www.jasondavies.com/maps/random-points/
#
# Aaron LI
# 2015/06/18

__version__ = "0.1.0"
__date__ = "2015/06/16"

import math
import random

def sphere_point(n=1, unit="rad"):
    """
    Randomly uniformly pick a point on the sphere surface.
    Using the method "Sphere Point Picking" from Wolfram MathWorld.

    Arguments:
        n: number of points to be generated
        unit: unit of output values: rad/deg

    Return:
        (theta, phi): spherical coordinate (unit: rad).
                      theta: [0, 2\pi); phi: [0 - \pi]
        If n > 1, then return a list of (theta, phi)
    """
    points = []
    for i in range(n):
        u = random.random()
        v = random.random()
        theta = 2.0 * math.pi * u
        phi = math.acos(2.0*v - 1.0)
        if unit == "deg":
            theta = rad2deg(theta)
            phi = rad2deg(phi)
        points.append((theta, phi))
    if n == 1:
        return points[0]
    else:
        return points


def rad2deg(x):
    return x * 180.0 / math.pi

def deg2rad(x):
    return x * math.pi / 180.0


