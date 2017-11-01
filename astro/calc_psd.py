#!/usr/bin/env python3
#
# Copyright (c) 2015-2017 Aaron LI
# MIT License
#

"""
Compute the radial (i.e., azimuthally averaged) power spectral density
(a.k.a. power spectrum) of a FITS image.

NOTE: The input image must be square.

Credit
------
* Radially averaged power spectrum of 2D real-valued matrix
  Evan Ruzanski
  'raPsd2d.m'
  https://www.mathworks.com/matlabcentral/fileexchange/23636-radially-averaged-power-spectrum-of-2d-real-valued-matrix
"""

import os
import argparse
from functools import lru_cache

import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

plt.style.use("ggplot")


class PSD:
    """
    Computes the 2D power spectral density and the azimuthally averaged power
    spectral density (i.e., 1D radial power spectrum).
    """
    def __init__(self, image, pixel=(1.0, "pixel"), step=None):
        self.image = np.array(image, dtype=np.float)
        self.shape = self.image.shape
        if self.shape[0] != self.shape[1]:
            raise ValueError("input image is not square!")

        self.pixel = pixel
        self.step = step
        if step is not None and step <= 1:
            raise ValueError("step must be greater than 1")

    @property
    @lru_cache()
    def radii(self):
        """
        The radial (frequency) points where to calculate the powers.
        If ``self.step`` is ``None``, then the powers at every frequency
        point are calculated.  If ``self.step`` is specified, then a
        log-even grid is adopted, which can greatly save computation time
        for large images.
        """
        dim_half = (self.shape[0] + 1) // 2
        x = np.arange(dim_half)
        if self.step is None:
            return x
        else:
            xmax = x.max()
            x2 = list(x[x*(self.step-1) <= 1])
            v1 = x[len(x2)]
            while v1 < xmax:
                x2.append(v1)
                v1 *= self.step
            x2.append(xmax)
            return np.array(x2)

    @property
    @lru_cache()
    def frequencies(self):
        """
        The (spatial) frequencies w.r.t. the above radii.
        """
        radii = self.radii
        freqs = (1 / (self.shape[0] * self.pixel[0])) * radii
        return freqs

    def calc_psd2d(self):
        """
        Computes the 2D power spectral density of the given image.
        Note that the low frequency components are shifted to the center
        of the FFT'ed image.

        NOTE
        ----
        The zero-frequency component is shifted to position of index (0-based)
            (ceil((n-1) / 2), ceil((m-1) / 2)),
        where (n, m) are the number of rows and columns of the image/psd2d.

        Returns
        -------
        2D power spectral density, which has dimension of ${input_unit}^2.
        """
        print("Calculating 2D power spectral density ... ", end="", flush=True)
        rows, cols = self.shape
        # Compute the power spectral density (i.e., power spectrum)
        imgf = np.fft.fftshift(np.fft.fft2(self.image))
        # Normalization w.r.t. image size
        norm = rows * cols * self.pixel[0]**2
        self.psd2d = (np.abs(imgf) ** 2) / norm
        print("DONE", flush=True)
        return self.psd2d

    def calc_psd(self):
        """
        Azimuthally average the above 2D power spectral density to generate
        the 1D radial power spectral density.

        Returns
        -------
        frequencies: 1D float `~numpy.ndarray`
            Spatial frequencies, [{pixel_unit}^(-1)]
        psd1d, psd1d_err: 1D float `~numpy.ndarray`
            Azimuthally averaged powers and their standard deviations at
            each (radial) spatial frequency bin.
        """
        if not hasattr(self, "ps2d") or self.psd2d is None:
            self.calc_psd2d()

        print("Azimuthally averaging 2D power spectral density ... ",
              end="", flush=True)
        dim = self.shape[0]
        dim_half = (dim+1) // 2
        # NOTE:
        # The zero-frequency component is shifted to position of index
        # (0-based): (ceil((n-1) / 2), ceil((m-1) / 2))
        px = np.arange(dim_half-dim, dim_half)
        x, y = np.meshgrid(px, px)
        rho, phi = self.cart2pol(x, y)

        radii = self.radii
        nr = len(radii)
        if nr > 100:
            print("\n    ... many points to calculate, may take a while ... ",
                  end="", flush=True)
        else:
            print(" %d data points ... " % nr, end="", flush=True)
        psd1d = np.zeros(nr)
        psd1d_err = np.zeros(nr)
        for i, r in enumerate(radii):
            if (i+1) % 100 == 0:
                percent = 100 * (i+1) / nr
                print("%.1f%% ... " % percent, end="", flush=True)
            ii, jj = (rho <= r).nonzero()
            rho[ii, jj] = np.inf
            data = self.psd2d[ii, jj]
            psd1d[i] = np.mean(data)
            psd1d_err[i] = np.std(data)
        print("DONE", flush=True)

        self.psd1d = psd1d
        self.psd1d_err = psd1d_err
        return (self.frequencies, psd1d, psd1d_err)

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
        Make a plot of the 1D radial PSD with matplotlib.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        freqs = self.frequencies
        xmin = freqs[1] / 1.2  # ignore the first 0
        xmax = freqs[-1] * 1.1
        ymin = np.min(self.psd1d) / 10.0
        ymax = np.max(self.psd1d[1:] + self.psd1d_err[1:]) * 2

        ax.errorbar(freqs, self.psd1d, yerr=self.psd1d_err, fmt="none")
        ax.plot(freqs, self.psd1d, "ko")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title("Radially Averaged Power Spectral Density")
        ax.set_xlabel(r"k [%s$^{-1}$]" % self.pixel[1])
        ax.set_ylabel("Power")
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
            description="Calculate radial power spectral density")
    parser.add_argument("-C", "--clobber", dest="clobber", action="store_true",
                        help="overwrite the output files if already exist")
    parser.add_argument("-s", "--step", dest="step", type=float, default=None,
                        help="step ratio between 2 consecutive radial " +
                        "frequency points, must be > 1, thus a log-even " +
                        "grid is adopted; if not specified, then the power " +
                        "at every frequency point will be calculated, " +
                        "i.e., using a even grid, which may be very slow " +
                        "for very large images!")
    parser.add_argument("-p", "--plot", dest="plot", action="store_true",
                        help="plot the PSD and save as a PNG image")
    parser.add_argument("-i", "--infile", dest="infile", required=True,
                        help="input FITS image")
    parser.add_argument("-o", "--outfile", dest="outfile", required=True,
                        help="output TXT file to save the PSD data")
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
    psdobj = PSD(image=image, step=args.step)
    freqs, psd, psd_err = psdobj.calc_psd()

    # Write out PSD results
    psd_data = np.column_stack((freqs, psd, psd_err))
    np.savetxt(args.outfile, psd_data, header="freqs  psd  psd_err")
    print("Saved PSD data to: %s" % args.outfile)

    if args.plot:
        # Make and save a plot
        fig = Figure(figsize=(10, 8))
        FigureCanvas(fig)
        ax = fig.add_subplot(111)
        psdobj.plot(ax=ax, fig=fig)
        fig.savefig(plotfile, format="png", dpi=150)
        print("Plotted PSD and saved to image: %s" % plotfile)


if __name__ == "__main__":
    main()
