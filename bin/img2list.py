#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Aaron LI
# 2015/06/23
#


import numpy as np
from astropy.io import fits


def img2list(imgdata, mask=None):
    """
    Convert a image matrix to list of point coordinates.
    The input image matrix is taken as an integer matrix.
    If one pixel has value n (>1), then it is repeated n times.
    """
    img = imgdata.astype(int)
    points = []
    ii, jj = np.nonzero(img >= 1)
    while len(ii) > 0:
        for i, j in zip(ii, jj):
            points.append([i, j])
        img[ii, jj] -= 1
        ii, jj = np.nonzero(img >= 1)
    return np.array(points)

