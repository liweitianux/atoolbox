#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Credit:
# [1]  Radially averaged power spectrum of 2D real-valued matrix
#      Evan Ruzanski
#      'raPsd2d.m'
#      https://www.mathworks.com/matlabcentral/fileexchange/23636-radially-averaged-power-spectrum-of-2d-real-valued-matrix
#
# Aaron LI <aaronly.me@gmail.com>
# Created: 2015-04-22
# Updated: 2016-04-26
#
# Changelog:
# 2016-04-26:
#   * Adjust plot function
#   * Update normalize argument; Add pixel argument
# 2016-04-25:
#   * Update plot function
#   * Add command line scripting support
#   * Encapsulate the functions within class 'PSD'
#   * Update docs/comments
#

"""
Compute the radially averaged power spectral density (i.e., power spectrum).
"""

__version__ = "0.3.1"
__date__    = "2016-04-25"


import sys
import os
import argparse

import numpy as np
from scipy import fftpack
from astropy.io import fits

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

plt.style.use("ggplot")


class PSD:
    """
    Computes the 2D power spectral density and the radially averaged power
    spectral density (i.e., 1D power spectrum).
    """
    # 2D image data
    img = None
    # value and unit of 1 pixel for the input image
    pixel = (None, None)
    # whether to normalize the power spectral density by image size
    normalize = True
    # 2D power spectral density
    psd2d = None
    # 1D (radially averaged) power spectral density
    freqs     = None
    psd1d     = None
    psd1d_err = None

    def __init__(self, img, pixel=(1.0, "pixel"), normalize=True):
        self.img = img.astype(np.float)
        self.pixel = pixel
        self.normalize = normalize

    def calc_psd2d(self):
        """
        Computes the 2D power spectral density of the given image.
        Note that the low frequency components are shifted to the center
        of the FFT'ed image.

        Return:
            2D power spectral density, which is dimensionless if normalized,
            otherwise has unit ${pixel_unit}^2.
        """
        rows, cols = self.img.shape
        ## Compute the power spectral density (i.e., power spectrum)
        imgf = fftpack.fftshift(fftpack.fft2(self.img))
        if self.normalize:
            norm = rows * cols * self.pixel[0]**2
        else:
            norm = 1.0  # Do not normalize
        self.psd2d = (np.abs(imgf) / norm) ** 2
        return self.psd2d

    def calc_radial_psd1d(self, k_geometric=True, k_step=1.2):
        """
        Computes the radially averaged power spectral density from the
        provided 2D power spectral density.

        XXX/TODO:

        Arguments:
          * k_geometric: whether the k (i.e., frequency) varies as
                         geometric sequences (i.e., k, k*k_step, ...),
                         otherwise, k varies as (k, k+k_step, ...)
          * k_step: the step ratio or step length for k

        Return:
            (freqs, radial_psd, radial_psd_err)
            freqs: spatial freqencies (unit: ${pixel_unit}^(-1))
                   if k_geometric=True, frequencies are taken as the
                   geometric means.
            radial_psd: radially averaged power spectral density for each
                        frequency
            radial_psd_err: standard deviations of each radial_psd
        """
        psd2d = self.psd2d.copy()
        rows, cols = psd2d.shape
        ## Adjust the PSD array size
        dim_diff = np.abs(rows - cols)
        dim_max  = max(rows, cols)
        # Pad the 2D PSD array to be sqaure
        if rows > cols:
            # pad columns
            if np.mod(dim_diff, 2) == 0:
                cols_left     = np.zeros((rows, dim_diff/2))
                cols_left[:]  = np.nan
                cols_right    = np.zeros((rows, dim_diff/2))
                cols_right[:] = np.nan
                psd2d         = np.hstack((cols_left, psd2d, cols_right))
            else:
                cols_left     = np.zeros((rows, np.floor(dim_diff/2)))
                cols_left[:]  = np.nan
                cols_right    = np.zeros((rows, np.floor(dim_diff/2)+1))
                cols_right[:] = np.nan
                psd2d         = np.hstack((cols_left, psd2d, cols_right))
        elif rows < cols:
            # pad rows
            if np.mod(dim_diff, 2) == 0:
                rows_top       = np.zeros((dim_diff/2, cols))
                rows_top[:]    = np.nan
                rows_bottom    = np.zeros((dim_diff/2, cols))
                rows_bottom[:] = np.nan
                psd2d          = np.vstack((rows_top, psd2d, rows_bottom))
            else:
                rows_top       = np.zeros((np.floor(dim_diff/2), cols))
                rows_top[:]    = np.nan
                rows_bottom    = np.zeros((np.floor(dim_diff/2)+1, cols))
                rows_bottom[:] = np.nan
                psd2d          = np.vstack((rows_top, psd2d, rows_bottom))
        ## Compute radially average power spectrum
        px       = np.arange(-dim_max/2, dim_max/2)
        x, y     = np.meshgrid(px, px)
        rho, phi = self.cart2pol(x, y)
        rho      = np.around(rho).astype(np.int)
        dim_half = int(np.floor(dim_max/2) + 1)
        radial_psd     = np.zeros(dim_half)
        radial_psd_err = np.zeros(dim_half) # standard error
        for r in range(dim_half):
            # Get the indices of the elements satisfying rho[i,j]==r
            ii, jj = (rho == r).nonzero()
            # Calculate the mean value at a given radii
            data              = psd2d[ii, jj]
            radial_psd[r]     = np.nanmean(data)
            radial_psd_err[r] = np.nanstd(data)
        # Calculate frequencies
        f = fftpack.fftfreq(dim_max, d=1)  # sample spacing: set to 1 pixel
        freqs = np.abs(f[:dim_half])
        #
        self.freqs     = freqs
        self.psd1d     = radial_psd
        self.psd1d_err = radial_psd_err
        return (freqs, radial_psd, radial_psd_err)

    @staticmethod
    def cart2pol(x, y):
        """
        Convert Cartesian coordinates to polar coordinates.
        """
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return (rho, phi)

    @staticmethod
    def pol2cart(rho, phi):
        """
        Convert polar coordinates to Cartesian coordinates.
        """
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    def plot(self, ax=None, fig=None):
        """
        Make a plot of the radial (1D) PSD with matplotlib.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        #
        xmin = self.freqs[1] / 1.2  # ignore the first 0
        xmax = self.freqs[-1]
        ymin = np.nanmin(self.psd1d) / 10.0
        ymax = np.nanmax(self.psd1d + self.psd1d_err)
        #
        eb = ax.errorbar(self.freqs, self.psd1d, yerr=self.psd1d_err,
                         fmt="none")
        ax.plot(self.freqs, self.psd1d, "ko")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title("Radially Averaged Power Spectral Density")
        ax.set_xlabel(r"k (%s$^{-1}$)" % self.pixel[1])
        if self.normalize:
            ax.set_ylabel("Power")
        else:
            ax.set_ylabel(r"Power (%s$^2$)" % self.pixel[1])
        fig.tight_layout()
        return (fig, ax)


def main():
    parser = argparse.ArgumentParser(
            description="Compute the radially averaged power spectral density",
            epilog="Version: %s (%s)" % (__version__, __date__))
    parser.add_argument("-V", "--version", action="version",
            version="%(prog)s " + "%s (%s)" % (__version__, __date__))
    parser.add_argument("-i", "--infile", dest="infile",
            required=True, help="input image")
    parser.add_argument("-o", "--outfile", dest="outfile",
            required=True, help="output file to store the PSD data")
    parser.add_argument("-p", "--png", dest="png",
            help="plot the PSD and save to the given PNG file")
    parser.add_argument("-v", "--verbose", dest="verbose",
            action="store_true", help="show verbose information")
    parser.add_argument("-C", "--clobber", dest="clobber",
            action="store_true",
            help="overwrite the output files if already exist")
    args = parser.parse_args()

    # Check output files whether already exists
    if (not args.clobber) and os.path.exists(args.outfile):
        raise ValueError("outfile '%s' already exists" % args.outfile)
    if (not args.clobber) and os.path.exists(args.png):
        raise ValueError("output png '%s' already exists" % args.png)

    # Load image data
    if args.verbose:
        print("Loading input image ...", file=sys.stderr)
    with fits.open(args.infile) as ffile:
        img = ffile[0].data
        psd = PSD(img, normalize=True)

    # Calculate the power spectral density
    if args.verbose:
        print("Calculate 2D power spectral density ...", file=sys.stderr)
    psd.calc_psd2d()
    if args.verbose:
        print("Calculate radially averaged (1D) power spectral density ...",
                file=sys.stderr)
    freqs, psd1d, psd1d_err = psd.calc_radial_psd1d()

    # Write out PSD results
    psd_data = np.column_stack((freqs, psd1d, psd1d_err))
    np.savetxt(args.outfile, psd_data, header="freqs  psd1d  psd1d_err")

    # Make and save a plot
    fig = Figure(figsize=(10, 8))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    psd.plot(ax=ax, fig=fig)
    fig.savefig(args.png, format="png", dpi=150)


if __name__ == "__main__":
    main()

