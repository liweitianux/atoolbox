#!/usr/bin/env python3
#
# Copyright (c) 2015-2017 Aaron LI
# MIT License
#

"""
Compute the radially averaged power spectral density (i.e., power spectrum)
of a 2D image (in FITS format).  The input image must be square.

Credit
------
* Radially averaged power spectrum of 2D real-valued matrix
  Evan Ruzanski
  'raPsd2d.m'
  https://www.mathworks.com/matlabcentral/fileexchange/23636-radially-averaged-power-spectrum-of-2d-real-valued-matrix
"""

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

    def __init__(self, image, pixel=(1.0, "pixel"), normalize=True, step=None):
        self.image = np.array(image, dtype=np.float)
        self.shape = self.image.shape
        if self.shape[0] != self.shape[1]:
            raise ValueError("input image is not square!")

        self.pixel = pixel
        self.normalize = normalize
        self.step = step

    @property
    def radii(self):
        pass

    def calc_psd2d(self):
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
        print("Calculating 2D power spectral density ... ", end="", flush=True)
        rows, cols = self.img.shape
        # Compute the power spectral density (i.e., power spectrum)
        imgf = np.fft.fftshift(np.fft.fft2(self.img))
        if self.normalize:
            norm = rows * cols * self.pixel[0]**2
        else:
            norm = 1.0  # Do not normalize
        self.psd2d = (np.abs(imgf) / norm) ** 2
        print("DONE", flush=True)
        return self.psd2d

    def calc_psd(self):
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
        if not hasattr(self, "ps2d") or self.psd2d is None:
            self.calc_psd2d()

        print("Radially averaging 2D power spectral density ... ")
        psd2d = self.psd2d
        dim = psd2d.shape[0]
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
        print("    -> radially averaging ... ", end="", flush=True)
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
        print("DONE", flush=True)
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


def open_image(infile):
    """
    Open the slice image and return its header and 2D image data.

    NOTE
    ----
    The input slice image may have following dimensions:
    * NAXIS=2: [Y, X]
    * NAXIS=3: [FREQ=1, Y, X]
    * NAXIS=4: [STOKES=1, FREQ=1, Y, X]

    NOTE
    ----
    Only open slice image that has only ONE frequency and ONE Stokes
    parameter.

    Returns
    -------
    header : `~astropy.io.fits.Header`
    image : 2D `~numpy.ndarray`
        The 2D [Y, X] image part of the slice image.
    """
    with fits.open(infile) as f:
        header = f[0].header
        data = f[0].data
    if data.ndim == 2:
        # NAXIS=2: [Y, X]
        image = data
    elif data.ndim == 3 and data.shape[0] == 1:
        # NAXIS=3: [FREQ=1, Y, X]
        image = data[0, :, :]
    elif data.ndim == 4 and data.shape[0] == 1 and data.shape[1] == 1:
        # NAXIS=4: [STOKES=1, FREQ=1, Y, X]
        image = data[0, 0, :, :]
    else:
        raise ValueError("Slice '{0}' has invalid dimensions: {1}".format(
            infile, data.shape))
    return (header, image)


def main():
    parser = argparse.ArgumentParser(
            description="Calculate radially averaged power spectral density")
    parser.add_argument("-C", "--clobber", dest="clobber", action="store_true",
                        help="overwrite the output files if already exist")
    parser.add_argument("-i", "--infile", dest="infile", required=True,
                        help="input FITS image")
    parser.add_argument("-o", "--outfile", dest="outfile", required=True,
                        help="output TXT file to save the PSD data")
    parser.add_argument("-p", "--plot", dest="plot", action="store_true",
                        help="plot the PSD and save as PNG image")
    args = parser.parse_args()

    if args.plot:
        plotfile = os.path.splitext(args.outfile)[0] + ".png"

    # Check output files whether already exists
    if (not args.clobber) and os.path.exists(args.outfile):
        raise OSError("outfile '%s' already exists" % args.outfile)
    if args.plot:
        if (not args.clobber) and os.path.exists(plotfile):
            raise OSError("output plot file '%s' already exists" % plotfile)

    header, image = open_image(args.infile)
    psd = PSD(image=image, normalize=True)
    psd.calc_psd2d()
    freqs, psd, psd_err = psd.calc_psd()

    # Write out PSD results
    psd_data = np.column_stack((freqs, psd, psd_err))
    np.savetxt(args.outfile, psd_data, header="freqs  psd  psd_err")
    print("Saved PSD data to: %s" % args.outfile)

    if args.plot:
        # Make and save a plot
        fig = Figure(figsize=(10, 8))
        FigureCanvas(fig)
        ax = fig.add_subplot(111)
        psd.plot(ax=ax, fig=fig)
        fig.savefig(plotfile, format="png", dpi=150)
        print("Plotted PSD and saved as: %s" % plotfile)


if __name__ == "__main__":
    main()
