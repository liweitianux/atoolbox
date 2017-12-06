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

import matplotlib
import matplotlib.style
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# Matplotlib settings
matplotlib.style.use("ggplot")
for k, v in [("font.family",       "monospace"),
             ("xtick.major.size",  7.0),
             ("xtick.major.width", 2.0),
             ("xtick.minor.size",  4.0),
             ("xtick.minor.width", 1.5),
             ("ytick.major.size",  7.0),
             ("ytick.major.width", 2.0),
             ("ytick.minor.size",  4.0),
             ("ytick.minor.width", 1.5)]:
    matplotlib.rcParams[k] = v


class PSD:
    """
    Calculate the 2D power spectral density and then apply azimuthal
    averaging to obtain the 1D radial power spectrum.

    Parameters
    ----------
    image : 2D `~numpy.ndarray`
        Input image array
    pixel : (float, str), optional
        Specify the pixel size and its unit of the image.
        e.g., (0.33, "arcmin")
    step : float, optional
        By default, a logarithmic grid with the specified step ratio
        (default: 1.1) will be used to do the azimuthal averages.
        If specified a value <=1 or None, then an evenly pixel-by-pixel
        (along radial direction) averaging scheme is adopted.
    meanstd : bool, optional
        By default, the median and 1.4826*MAD will be calculated for
        each averaging annulus.  If ``meanstd=True``, then calculate
        the mean and standard deviation instead.
    """
    def __init__(self, image, pixel=(1.0, "pixel"), step=1.1,
                 meanstd=False, bunit=None):
        self.image = np.array(image, dtype=float)
        if image.ndim != 2:
            raise ValueError("input image is not 2D!")
        self.shape = self.image.shape
        if self.shape[0] != self.shape[1]:
            raise ValueError("input image is not square!")

        self.pixel = pixel
        if step is None or step <= 1:
            self.step = None
        else:
            self.step = step

        self.meanstd = meanstd
        self.bunit = bunit

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
        imgf = np.fft.fftshift(np.fft.fft2(self.image))
        # NOTE: normalize w.r.t. image size
        norm = rows * cols * self.pixel[0]**2
        self.psd2d = (np.abs(imgf) ** 2) / norm
        print("DONE", flush=True)
        return self.psd2d

    def calc_psd(self):
        """
        Azimuthally average the above 2D power spectral density to obtain
        the 1D radial power spectral density.

        Returns
        -------
        psd1d : 2D `~numpy.ndarray`
            2D array of shape (nbins, 4) including such 4 columns:
            + spatial frequencies, [{pixel_unit}^(-1)]
            + average (median / mean) powers within each averaging bin
            + power errors (1.4826*MAD / standard deviation)
            + number of averaging cells

        Attributes
        ----------
        psd1d
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
        rho = np.sqrt(x**2 + y**2)

        radii = self.radii
        nr = len(radii)
        if nr > 100:
            print("\n    ... %d data points, may take a while ... " % nr,
                  end="", flush=True)
        else:
            print(" %d data points ... " % nr, end="", flush=True)
        psd1d = np.zeros(shape=(nr, 4))
        psd1d[:, 0] = self.frequencies

        for i, r in enumerate(radii):
            if (i+1) % 100 == 0:
                percent = 100 * (i+1) / nr
                print("%.1f%% ... " % percent, end="", flush=True)
            ii, jj = (rho <= r).nonzero()
            rho[ii, jj] = np.inf
            cells = self.psd2d[ii, jj]
            psd1d[i, 3] = len(cells)
            if self.meanstd:
                psd1d[i, 1] = np.mean(cells)
                psd1d[i, 2] = np.std(cells)
            else:
                median = np.median(cells)
                mad = np.median(np.abs(cells - median))
                psd1d[i, 1] = median
                psd1d[i, 2] = mad * 1.4826
        print("DONE", flush=True)

        self.psd1d = psd1d
        return psd1d

    def save(self, outfile):
        data = self.psd1d
        header = [
            "pixel: %s [%s]" % self.pixel,
            "frequency: [%s^-1]" % self.pixel[1],
        ]
        if self.meanstd:
            header += [
                "psd1d: *mean* powers of radial averaging annuli",
                "psd1d_err: *standard deviation*",
            ]
        else:
            header += [
                "psd1d: *median* powers of radial averaging annuli",
                "psd1d_err: 1.4826*MAD (median absolute deviation)",
            ]
        header += [
            "n_cells: number of averaging cells",
            "",
            "frequency   psd1d   psd1d_err   n_cells"
        ]
        np.savetxt(outfile, data, header="\n".join(header))
        print("Saved PSD data to: %s" % outfile)

    def plot(self, ax):
        """
        Make a plot of the 1D radial power spectrum.
        """
        data = self.psd1d
        freqs = data[:, 0]
        psd1d = data[:, 1]
        psd1d_err = data[:, 2]

        xmin = freqs[1] / 1.2  # ignore the first 0
        xmax = freqs[-1] * 1.1
        ymin = np.min(psd1d) / 10.0
        ymax = np.max(psd1d[1:] + psd1d_err[1:]) * 1.5

        if self.meanstd:
            label = "mean"
            labelerr = "standard deviation"
        else:
            label = "median"
            labelerr = "1.4826*MAD"
        if self.bunit:
            ylabel = r"Power [(%s)$^2$]" % self.bunit
        else:
            ylabel = "Power"

        ax.errorbar(freqs, psd1d, yerr=psd1d_err,
                    fmt="none", label=labelerr)
        ax.plot(freqs, psd1d, marker="o", label=label)
        ax.set(xscale="log", yscale="log",
               xlim=(xmin, xmax), ylim=(ymin, ymax),
               title="Radial (Azimuthally Averaged) Power Spectral Density",
               xlabel=r"$k$ [%s$^{-1}$]" % self.pixel[1],
               ylabel=ylabel)
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
            ax2.set_xlabel(r"$k$ [pixel$^{-1}$] (1 pixel = %.2f %s)" %
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
    parser.add_argument("-c", "--center", dest="center", type=int,
                        help="crop the central box region of specified " +
                        "size before calculating the power spectrum")
    parser.add_argument("-s", "--step", dest="step", type=float, default=1.1,
                        help="step ratio (>1; default: 1.1) between 2 " +
                        "consecutive radial frequency points, " +
                        "i.e., a logarithmic grid used. " +
                        "if specified a value <=1, then the power at every " +
                        "radial frequency point will be calculated, " +
                        "i.e., using a even grid, which may be very slow " +
                        "for very large images!")
    parser.add_argument("-p", "--pixelsize", dest="pixelsize", type=float,
                        help="image spatial pixel size [arcsec] " +
                        "(will try to obtain from FITS header)")
    parser.add_argument("-m", "--mean-std", dest="meanstd",
                        action="store_true",
                        help="calculate the mean and standard deviation " +
                        "for each averaging annulus instead of the median " +
                        "and 1.4826*MAD")
    parser.add_argument("-P", "--no-plot", dest="noplot", action="store_true",
                        help="do NOT plot the PSD and save")
    parser.add_argument("-i", "--infile", dest="infile", nargs="+",
                        help="input FITS image(s); if multiple images " +
                        "are provided, they are added first.")
    parser.add_argument("-o", "--outfile", dest="outfile", required=True,
                        help="output TXT file to save the PSD data")
    args = parser.parse_args()

    if (not args.clobber) and os.path.exists(args.outfile):
        raise OSError("outfile '%s' already exists" % args.outfile)

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

    if args.center:
        csize = args.center
        if csize >= min(image.shape):
            raise ValueError("--center %d exceeds image size" % csize)
        rows, cols = image.shape
        rc, cc = rows//2, cols//2
        cs1, cs2 = csize//2, (csize+1)//2
        image = image[(rc-cs1):(rc+cs2), (cc-cs1):(cc+cs2)]
        print("Cropped image with central box of size: %dx%d" % image.shape)

    psd = PSD(image=image, pixel=pixel, step=args.step,
              meanstd=args.meanstd, bunit=bunit)
    psd.calc_psd()
    psd.save(args.outfile)

    if not args.noplot:
        fig = Figure(figsize=(8, 8), dpi=150)
        FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)
        psd.plot(ax=ax)
        fig.tight_layout()
        plotfile = os.path.splitext(args.outfile)[0] + ".png"
        fig.savefig(plotfile)
        print("Plotted PSD and saved to image: %s" % plotfile)


if __name__ == "__main__":
    main()
