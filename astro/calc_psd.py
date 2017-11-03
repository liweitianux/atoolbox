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

    Parameters
    ----------
    image : 2D `~numpy.ndarray`
        Input image array
    pixel : (float, str), optional
        Specify the pixel size and its unit of the image.
        e.g., (0.33, "arcmin")
    step : float, optional
        If specified, then a log-even grid with the given step ratio will
        be used to do the azimuthal averages.  Otherwise, a evenly
        pixel-by-pixel (along radial direction) is adopted.
    meanstd : bool, optional
        By default, the median and 16% and 84% percentiles (i.e., 68% IQR)
        will be calculated for each averaged annulus.  If this option is
        ``True`` then calculate the mean and standard deviation instead.
    """
    def __init__(self, image, pixel=(1.0, "pixel"), step=None, meanstd=False):
        self.image = np.array(image, dtype=float)
        self.shape = self.image.shape
        if self.shape[0] != self.shape[1]:
            raise ValueError("input image is not square!")

        self.pixel = pixel
        self.step = step
        if step is not None and step <= 1:
            raise ValueError("step must be greater than 1")

        self.meanstd = meanstd

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
        frequencies : 1D float `~numpy.ndarray`
            Spatial frequencies, [{pixel_unit}^(-1)]
        psd1d : 1D float `~numpy.ndarray`
            The median or mean (``self.meanstd=True``) of the powers within
            each (radial) spatial frequency bin.
        psd1d_errl, psd1d_erru : 1D float `~numpy.ndarray`
            The lower and upper errors of the powers.  By default, they are
            determined from the 16% and 84% percentiles w.r.t. the median.
            If ``self.meanstd=True`` then they are the standard deviation.

        Attributes
        ----------
        psd1d
        psd1d_errl
        psd1d_erru
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
            print("\n    ... %d data points, may take a while ... " % nr,
                  end="", flush=True)
        else:
            print(" %d data points ... " % nr, end="", flush=True)
        psd1d = np.zeros(nr)
        psd1d_errl = np.zeros(nr)  # lower error
        psd1d_erru = np.zeros(nr)  # upper error
        for i, r in enumerate(radii):
            if (i+1) % 100 == 0:
                percent = 100 * (i+1) / nr
                print("%.1f%% ... " % percent, end="", flush=True)
            ii, jj = (rho <= r).nonzero()
            rho[ii, jj] = np.inf
            data = self.psd2d[ii, jj]
            if self.meanstd:
                psd1d[i] = np.mean(data)
                std = np.std(data)
                psd1d_errl[i] = std
                psd1d_erru[i] = std
            else:
                median, q16, q84 = np.percentile(data, q=(50, 16, 84))
                psd1d[i] = median
                psd1d_errl[i] = median - q16
                psd1d_erru[i] = q84 - median
        print("DONE", flush=True)

        self.psd1d = psd1d
        self.psd1d_errl = psd1d_errl
        self.psd1d_erru = psd1d_erru
        return (self.frequencies, psd1d, psd1d_errl, psd1d_erru)

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

    def save(self, outfile):
        data = np.column_stack((self.frequencies, self.psd1d,
                                self.psd1d_errl, self.psd1d_erru))
        header = [
            "pixel: %s [%s]" % self.pixel,
            "frequency: [%s^-1]" % self.pixel[1],
        ]
        if self.meanstd:
            header += [
                "psd1d: *mean* powers of radial spectral annuli",
                "psd1d_errl, psd1d_erru: *standard deviation* (lower, upper)",
            ]
        else:
            header += [
                "psd1d: *median* powers of radial spectral annuli",
                "psd1d_errl, psd1d_erru: 16% and 84% *percentiles*",
            ]
        header += ["", "frequency   psd1d   psd1d_errl   psd1d_erru"]
        np.savetxt(outfile, data, header="\n".join(header))
        print("Saved PSD data to: %s" % outfile)

    def plot(self, ax):
        """
        Make a plot of the 1D radial power spectrum.
        """
        freqs = self.frequencies
        xmin = freqs[1] / 1.2  # ignore the first 0
        xmax = freqs[-1] * 1.1
        ymin = np.min(self.psd1d) / 10.0
        ymax = np.max(self.psd1d[1:] + self.psd1d_erru[1:]) * 1.5

        if self.meanstd:
            label = "mean"
            labelerr = "standard deviation"
        else:
            label = "median"
            labelerr = "68% IQR"
        yerr = np.row_stack((self.psd1d_errl, self.psd1d_erru))
        ax.errorbar(freqs, self.psd1d, yerr=yerr,
                    fmt="none", label=labelerr)
        ax.plot(freqs, self.psd1d, marker="o", label=label)
        ax.set(xscale="log", yscale="log",
               xlim=(xmin, xmax), ylim=(ymin, ymax),
               title="Radial (Azimuthally Averaged) Power Spectral Density",
               xlabel=r"k [%s$^{-1}$]" % self.pixel[1],
               ylabel="Power")
        ax.legend()

        if self.pixel[1] != "pixel":
            # Add an additional X axis for pixel-based frequencies
            ax2 = ax.twiny()
            ax2.set_xscale(ax.get_xscale())
            pix_ticks = np.logspace(-4, 0, num=5)  # [pixel^-1]
            ax2.set_xticks(pix_ticks)
            ax2.set_xticklabels([r"10$^{%d}$" % ep
                                 for ep in np.log10(pix_ticks)])
            x1_min, x1_max = ax.get_xlim()
            x2_min, x2_max = x1_min*self.pixel[0], x1_max*self.pixel[0]
            ax2.set_xlim(x2_min, x2_max)
            ax2.set_xlabel(r"k [pixel$^{-1}$] (1 pixel = %.2f %s)" %
                           self.pixel)
            ax2.grid(False)
            # Raise title position to avoid overlapping
            ax.title.set_position([0.5, 1.1])
            return (ax, ax2)
        else:
            return ax


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
    parser.add_argument("-p", "--pixelsize", dest="pixelsize", type=float,
                        help="image spatial pixel size [arcsec] " +
                        "(will try to obtain from FITS header)")
    parser.add_argument("-m", "--mean-std", dest="meanstd",
                        action="store_true",
                        help="calculate the mean and standard deviation " +
                        "for each averaged annulus instead of the median " +
                        "16%% and 84%% percentiles (i.e., 68%% IQR)")
    parser.add_argument("-P", "--plot", dest="plot", action="store_true",
                        help="plot the PSD and save as a PNG image")
    parser.add_argument("-i", "--infile", dest="infile", nargs="+",
                        help="input FITS image(s); if multiple images " +
                        "are provided, they are added first.")
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

    header, image = open_image(args.infile[0])
    bunit = header.get("BUNIT", "???")
    print("Read image from: %s" % args.infile[0])
    print("Image size: %dx%d" % tuple(reversed(image.shape)))
    print("Data unit: %s" % bunit)

    if args.pixelsize:
        pixel = (args.pixelsize/60, "arcmin")  # [arcsec]->[arcmin]
    else:
        try:
            pixel = (header["PixSize"]/60, "arcmin")  # [arcsec]->[arcmin]
        except KeyError:
            try:
                pixel = (abs(header["CDELT1"])*60, "arcmin")  # [deg]->[arcmin]
            except KeyError:
                pixel = (1.0, "pixel")
    print("Image pixel size: %.2f [%s]" % pixel)

    for fn in args.infile[1:]:
        print("Adding additional image: %s" % fn)
        header2, image2 = open_image(fn)
        bunit2 = header2.get("BUNIT", "???")
        if bunit2 == bunit:
            image += image2
        else:
            raise ValueError("image has different unit: %s" % bunit2)

    psd = PSD(image=image, pixel=pixel, step=args.step, meanstd=args.meanstd)
    psd.calc_psd()
    psd.save(args.outfile)

    if args.plot:
        # Make and save a plot
        fig = Figure(figsize=(8, 8))
        FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)
        psd.plot(ax=ax)
        fig.tight_layout()
        fig.savefig(plotfile, format="png", dpi=150)
        print("Plotted PSD and saved to image: %s" % plotfile)


if __name__ == "__main__":
    main()
