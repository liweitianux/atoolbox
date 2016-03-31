# -*- coding: utf -*-
#
# Force field transform (Hurley et al., 2002, 2005)
#

"""
Force field transform
"""

import sys
import time
import numpy as np


def force(p1, p2):
    """
    The force between two points of the image.

    Arguments:
        p1, p2: (value, x, y)

    Return:
    #    (force, angle): value and direction of the force.
    #                    angle: (-pi, pi], with respect to p1.
        (f_x, f_y): x and y components of the force
    """
    v1, x1, y1 = p1
    v2, x2, y2 = p2
    #force = v1*v2 / ((x1-x2)**2 + (y1-y2)**2)
    #angle = np.atan2(y2-y1, x2-x1)
    #return (force, angle)
    f_x = v1 * v2 * (x2-x1) / ((x2-x1)**2 + (y2-y1)**2)**1.5
    f_y = v1 * v2 * (y2-y1) / ((x2-x1)**2 + (y2-y1)**2)**1.5
    return (f_x, f_y)


def force_array(p0, img):
    """
    The forces between the input point with respect to the image.

    Arguments:
        p0: (x, y), note (x, y) start with zero.
        img: input image, a numpy array

    Return:
        (f_x, f_y): x and y components of the forces of the same size
                    of the input image
    """
    x0, y0 = p0
    v0 = img[y0, x0]
    img[y0, x0] = 0.0
    x, y = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
    x[y0, x0] = -1
    y[y0, x0] = -1
    f_x = v0 * img * (x-x0) / ((x-x0)**2 + (y-y0)**2)**1.5
    f_y = v0 * img * (y-y0) / ((x-x0)**2 + (y-y0)**2)**1.5
    return (f_x, f_y)


def vector_add(v1, v2):
    """
    Add two vectors and return the results.

    Arguments:
        v1, v2: two input vectors of format (f_x, f_y)

    Return:
        (F_x, F_y)
    """
    f1_x, f1_y = v1
    f2_x, f2_y = v2
    return (f1_x+f2_x, f1_y+f2_y)


def force_summation(pixel, img):
    """
    Calculate the resulting force of the specified pixel with respect to
    the image.

    Argument:
        pixel: the position (x, y) of the pixel to be calculated
        img: the input image

    Return:
        (F_x, F_y): x and y components of the resulting force.
    """
    img = np.array(img)
    x0, y0 = pixel
    f_x, f_y = force_array((x0, y0), img)
    return (f_x.sum(), f_y.sum())


def force_field_transform(img):
    """
    Perform the "force field transform" on the input image.

    Arguments:
        img: input 2D image

    Return:
        (amplitudes, angles)
        amplitudes: the amplitudes of the resulting forces of each pixel
        angles: the directions of the resulting forces of each pixel,
                    in unit radian.
    """
    img = np.array(img)
    amplitudes = np.zeros(img.shape)
    angles = np.zeros(img.shape)
    rows, cols = img.shape
    t0 = time.time()
    t_p = t0 + 30  # in 30 seconds
    for y in range(rows):
        for x in range(cols):
            t1 = time.time()
            if t1 >= t_p:
                percent = 100 * (y*cols + x + 1) / (rows * cols)
                print("progress: %.3f%%; %.1f min" % (percent, (t1-t0)/60.0),
                        file=sys.stderr)
                t_p += 30  # in 30 seconds
            f_x, f_y = force_array((x, y), img)
            F_x, F_y = f_x.sum(), f_y.sum()
            amplitudes[y, x] = np.sqrt(F_x**2 + F_y**2)
            angles[y, x] = np.math.atan2(F_y, F_x)
    return (amplitudes, angles)


