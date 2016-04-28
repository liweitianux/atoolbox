#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Credit:
# [1]  Radially averaged power spectrum of 2D real-valued matrix
#      Evan Ruzanski
#      'raPsd2d.m'
#      https://www.mathworks.com/matlabcentral/fileexchange/23636-radially-averaged-power-spectrum-of-2d-real-valued-matrix
#
# XXX:
# * If the input image is NOT SQUARE; then are the horizontal frequencies
#   the same as the vertical frequencies ??
#
# Aaron LI <aaronly.me@gmail.com>
# Created: 2015-04-22
# Updated: 2016-04-28
#
# Changelog:
# 2016-04-28:
#   * Fix wrong meshgrid with respect to the shift zero-frequency component
#   * Use "numpy.fft" instead of "scipy.fftpack"
#   * Split method "pad_square()" from "calc_radial_psd()"
#   * Hide numpy warning when dividing by zero
#   * Add method "AstroImage.fix_shapes()"
#   * Add support for background subtraction and exposure correction
#   * Show verbose information during calculation
#   * Add class "AstroImage"
#   * Set default value for 'args.png'
#   * Rename from 'radialPSD2d.py' to 'calc_radial_psd.py'
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

__version__ = "0.5.0"
__date__    = "2016-04-28"


import sys
import os
import argparse

import numpy as np
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

    def calc_psd2d(self, verbose=False):
        """
        Computes the 2D power spectral density of the given image.
        Note that the low frequency components are shifted to the center
        of the FFT'ed image.

        NOTE:
        The zero-frequency component is shifted to position of index (0-based)
            (ceil((n-1) / 2), ceil((m-1) / 2)),
        where (n, m) are the number of rows and columns of the image/psd2d.

        Return:
            2D power spectral density, which is dimensionless if normalized,
            otherwise has unit ${pixel_unit}^2.
        """
        if verbose:
            print("Calculating 2D power spectral density ... ",
                    end="", flush=True)
        rows, cols = self.img.shape
        ## Compute the power spectral density (i.e., power spectrum)
        imgf = np.fft.fftshift(np.fft.fft2(self.img))
        if self.normalize:
            norm = rows * cols * self.pixel[0]**2
        else:
            norm = 1.0  # Do not normalize
        self.psd2d = (np.abs(imgf) / norm) ** 2
        if verbose:
            print("DONE", flush=True)
        return self.psd2d

    def calc_radial_psd1d(self, verbose=False):
        """
        Computes the radially averaged power spectral density from the
        provided 2D power spectral density.

        Return:
            (freqs, radial_psd, radial_psd_err)
            freqs: spatial freqencies (unit: ${pixel_unit}^(-1))
            radial_psd: radially averaged power spectral density for each
                        frequency
            radial_psd_err: standard deviations of each radial_psd
        """
        if verbose:
            print("Calculating radial (1D) power spectral density ... ",
                    end="", flush=True)
        if verbose:
            print("padding ... ", end="", flush=True)
        psd2d    = self.pad_square(self.psd2d, value=np.nan)
        dim      = psd2d.shape[0]
        dim_half = (dim+1) // 2
        # NOTE:
        # The zero-frequency component is shifted to position of index
        # (0-based): (ceil((n-1) / 2), ceil((m-1) / 2))
        px       = np.arange(dim_half-dim, dim_half)
        x, y     = np.meshgrid(px, px)
        rho, phi = self.cart2pol(x, y)
        rho      = np.around(rho).astype(np.int)
        radial_psd     = np.zeros(dim_half)
        radial_psd_err = np.zeros(dim_half)
        if verbose:
            print("radially averaging ... ",
                    end="", flush=True)
        for r in range(dim_half):
            # Get the indices of the elements satisfying rho[i,j]==r
            ii, jj = (rho == r).nonzero()
            # Calculate the mean value at a given radii
            data              = psd2d[ii, jj]
            radial_psd[r]     = np.nanmean(data)
            radial_psd_err[r] = np.nanstd(data)
        # Calculate frequencies
        f = np.fft.fftfreq(dim, d=self.pixel[0])
        freqs = np.abs(f[:dim_half])
        #
        self.freqs     = freqs
        self.psd1d     = radial_psd
        self.psd1d_err = radial_psd_err
        if verbose:
            print("DONE", end="", flush=True)
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

    @staticmethod
    def pad_square(data, value=np.nan):
        """
        Symmetrically pad the supplied data matrix to make it square.
        The padding rows are equally added to the top and bottom,
        as well as the columns to the left and right sides.
        The padded rows/columns are filled with the specified value.
        """
        mat = data.copy()
        rows, cols = mat.shape
        dim_diff = abs(rows - cols)
        dim_max  = max(rows, cols)
        if rows > cols:
            # pad columns
            if dim_diff // 2 == 0:
                cols_left     = np.zeros((rows, dim_diff/2))
                cols_left[:]  = value
                cols_right    = np.zeros((rows, dim_diff/2))
                cols_right[:] = value
                mat           = np.hstack((cols_left, mat, cols_right))
            else:
                cols_left     = np.zeros((rows, np.floor(dim_diff/2)))
                cols_left[:]  = value
                cols_right    = np.zeros((rows, np.floor(dim_diff/2)+1))
                cols_right[:] = value
                mat           = np.hstack((cols_left, mat, cols_right))
        elif rows < cols:
            # pad rows
            if dim_diff // 2 == 0:
                rows_top       = np.zeros((dim_diff/2, cols))
                rows_top[:]    = value
                rows_bottom    = np.zeros((dim_diff/2, cols))
                rows_bottom[:] = value
                mat            = np.vstack((rows_top, mat, rows_bottom))
            else:
                rows_top       = np.zeros((np.floor(dim_diff/2), cols))
                rows_top[:]    = value
                rows_bottom    = np.zeros((np.floor(dim_diff/2)+1, cols))
                rows_bottom[:] = value
                mat            = np.vstack((rows_top, mat, rows_bottom))
        return mat

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


class AstroImage:
    """
    Manipulate the astronimcal counts image, as well as the corresponding
    exposure map and background map.
    """
    # input counts image
    image = None
    # exposure map with respect to the input counts image
    expmap = None
    # background map (e.g., stowed background)
    bkgmap = None
    # exposure time of the input image
    exposure     = None
    # exposure time of the background map
    exposure_bkg = None

    def __init__(self, image, expmap=None, bkgmap=None, verbose=False):
        self.load_image(image,   verbose=verbose)
        self.load_expmap(expmap, verbose=verbose)
        self.load_bkgmap(bkgmap, verbose=verbose)

    def load_image(self, image, verbose=False):
        if verbose:
            print("Loading image ... ", end="", flush=True)
        with fits.open(image) as imgfits:
            self.image = imgfits[0].data.astype(np.float)
            self.exposure = imgfits[0].header["EXPOSURE"]
        if verbose:
            print("DONE", flush=True)

    def load_expmap(self, expmap, verbose=False):
        if expmap:
            if verbose:
                print("Loading exposure map ... ", end="", flush=True)
            with fits.open(expmap) as imgfits:
                self.expmap = imgfits[0].data.astype(np.float)
            if verbose:
                print("DONE", flush=True)

    def load_bkgmap(self, bkgmap, verbose=False):
        if bkgmap:
            if verbose:
                print("Loading background map ... ", end="", flush=True)
            with fits.open(bkgmap) as imgfits:
                self.bkgmap = imgfits[0].data.astype(np.float)
                self.exposure_bkg = imgfits[0].header["EXPOSURE"]
            if verbose:
                print("DONE", flush=True)

    def fix_shapes(self, tolerance=2, verbose=False):
        """
        Fix the shapes of self.expmap and self.bkgmap to make them have
        the same shape as the self.image.

        NOTE:
        * if the image is bigger than the reference image, then its
          columns on the right and rows on the botton are clipped;
        * if the image is smaller than the reference image, then padding
          columns on the right and rows on the botton are added.
        * Original images are REPLACED!

        Arguments:
          * tolerance: allow absolute difference between images
        """
        def _fix_shape(img, ref, tol=tolerance, verbose=verbose):
            if img.shape == ref.shape:
                if verbose:
                    print("SKIPPED", flush=True)
                return img
            elif np.allclose(img.shape, ref.shape, atol=tol):
                if verbose:
                    print(img.shape, "->", ref.shape, flush=True)
                rows, cols = img.shape
                rows_ref, cols_ref = ref.shape
                # rows
                if rows > rows_ref:
                    img_fixed = img[:rows_ref, :]
                else:
                    img_fixed = np.row_stack((img,
                        np.zeros((rows_ref-rows, cols), dtype=img.dtype)))
                # columns
                if cols > cols_ref:
                    img_fixed = img_fixed[:, :cols_ref]
                else:
                    img_fixed = np.column_stack((img_fixed,
                        np.zeros((rows_ref, cols_ref-cols), dtype=img.dtype)))
                return img_fixed
            else:
                raise ValueError("shape difference exceeds tolerance: " + \
                        "(%d, %d) vs. (%d, %d)" % (img.shape + ref.shape))
        #
        if self.bkgmap is not None:
            if verbose:
                print("Fixing shape for bkgmap ... ", end="", flush=True)
            self.bkgmap = _fix_shape(self.bkgmap, self.image)
        if self.expmap is not None:
            if verbose:
                print("Fixing shape for expmap ... ", end="", flush=True)
            self.expmap = _fix_shape(self.expmap, self.image)

    def subtract_bkg(self, verbose=False):
        if verbose:
            print("Subtracting background ... ", end="", flush=True)
        self.image -= (self.bkgmap / self.exposure_bkg * self.exposure)
        if verbose:
            print("DONE", flush=True)

    def correct_exposure(self, cut=0.015, verbose=False):
        """
        Correct the image for exposure by dividing by the expmap to
        create the exposure-corrected image.

        Arguments:
          * cut: the threshold percentage with respect to the maximum
                 exposure map value; and those pixels with lower values
                 than this threshold will be excluded/clipped (set to ZERO)
                 if set to None, then skip clipping image
        """
        if verbose:
            print("Correcting image for exposure ... ", end="", flush=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.image /= self.expmap
        # set invalid values to ZERO
        self.image[ ~ np.isfinite(self.image) ] = 0.0
        if verbose:
            print("DONE", flush=True)
        if cut is not None:
            # clip image according the exposure threshold
            if verbose:
                print("Clipping image (%s) ... " % cut, end="", flush=True)
            threshold = cut * np.max(self.expmap)
            self.image[ self.expmap < threshold ] = 0.0
            if verbose:
                print("DONE", flush=True)


def main():
    parser = argparse.ArgumentParser(
            description="Compute the radially averaged power spectral density",
            epilog="Version: %s (%s)" % (__version__, __date__))
    parser.add_argument("-V", "--version", action="version",
            version="%(prog)s " + "%s (%s)" % (__version__, __date__))
    parser.add_argument("-v", "--verbose", dest="verbose",
            action="store_true", help="show verbose information")
    parser.add_argument("-C", "--clobber", dest="clobber",
            action="store_true",
            help="overwrite the output files if already exist")
    parser.add_argument("-i", "--infile", dest="infile",
            required=True, help="input image")
    parser.add_argument("-b", "--bkgmap", dest="bkgmap", default=None,
            help="background map (for background subtraction)")
    parser.add_argument("-e", "--expmap", dest="expmap", default=None,
            help="exposure map (for exposure correction)")
    parser.add_argument("-o", "--outfile", dest="outfile",
            required=True, help="output file to store the PSD data")
    parser.add_argument("-p", "--png", dest="png", default=None,
            help="plot the PSD and save (default: same basename as outfile)")
    args = parser.parse_args()

    if args.png is None:
        args.png = os.path.splitext(args.outfile)[0] + ".png"

    # Check output files whether already exists
    if (not args.clobber) and os.path.exists(args.outfile):
        raise ValueError("outfile '%s' already exists" % args.outfile)
    if (not args.clobber) and os.path.exists(args.png):
        raise ValueError("output png '%s' already exists" % args.png)

    # Load image data
    image = AstroImage(image=args.infile, expmap=args.expmap,
                       bkgmap=args.bkgmap, verbose=args.verbose)
    image.fix_shapes(verbose=args.verbose)
    if args.bkgmap:
        image.subtract_bkg(verbose=args.verbose)
    if args.expmap:
        image.correct_exposure(verbose=args.verbose)

    # Calculate the power spectral density
    psd = PSD(img=image.image, normalize=True)
    psd.calc_psd2d(verbose=args.verbose)
    freqs, psd1d, psd1d_err = psd.calc_radial_psd1d(verbose=args.verbose)

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

