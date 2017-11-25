#!/usr/bin/env python3
#
# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license
#

"""
Calculate the 2D cylindrical-averaged power spectrum from the
3D image spectral cube.

References
----------
.. [liu2014]
   Liu, Parsons & Trott 2014, PhRvD, 90, 023018
   http://adsabs.harvard.edu/abs/2014PhRvD..90b3018L
   Appendix.A

.. [morales2004]
   Morales & Hewitt 2004, ApJ, 615, 7
   http://adsabs.harvard.edu/abs/2004ApJ...615....7M
   Sec.3

.. [matlab-psd-fft]
   MATLAB - Power Spectral Density Estimates Using FFT
   https://cn.mathworks.com/help/signal/ug/power-spectral-density-estimates-using-fft.html

.. [matlab-answer-psd]
   MATLAB Answers - How to create power spectral density from FFT
   https://cn.mathworks.com/matlabcentral/answers/43548-how-to-create-power-spectral-density-from-fft-fourier-transform
"""

import os
import sys
import argparse
import logging
from functools import lru_cache

import numpy as np
from scipy import fftpack
from scipy import signal
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import astropy.constants as ac

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


plt.style.use("ggplot")

logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s:%(lineno)d] %(message)s")
logger = logging.getLogger()


# HI line frequency
freq21cm = 1420.405751  # [MHz]
# Adopted cosmology
H0 = 71.0  # [km/s/Mpc]
OmegaM0 = 0.27
cosmo = FlatLambdaCDM(H0=H0, Om0=OmegaM0)


def freq2z(freq):
    z = freq21cm / freq - 1.0
    return z


def get_frequencies(header):
    """
    Get the frequencies for each cube slice.
    Unit: [MHz]
    """
    nfreq = header["NAXIS3"]
    freq0 = header["CRVAL3"]  # [Hz]
    freqstep = header["CDELT3"]  # [Hz]
    frequencies = freq0 + freqstep*np.arange(nfreq)
    return frequencies / 1e6  # [MHz]


def get_pixelsize(header):
    """
    Get the pixel size of cube image.
    Unit: [arcsec]
    """
    try:
        pixelsize = header["PixSize"]  # [arcsec]
    except KeyError:
        try:
            pixelsize = header["CDELT1"]  # [deg]
            if abs(pixelsize-1.0) < 1e-8:
                # Place-holder value set by ``fitscube.py``
                pixelsize = None
            else:
                pixelsize = abs(pixelsize)*3600  # [arcsec]
        except KeyError:
            pixelsize = None
    return pixelsize


class PS2D:
    """
    2D cylindrically averaged power spectrum

    NOTE
    ----
    * Cube dimensions: [nfreq, height, width] <-> [Z, Y, X]
    * Cube data unit: [K] (brightness temperature)

    Parameters
    ----------
    cube : 3D `~numpy.ndarray`
        3D spectral cube of shape (nfreq, height, width)
    pixelsize : float
        cube image pixel size [arcsec]
    frequencies : 1D `~numpy.ndarray`
        frequencies at each image slice [MHz]
    meanstd : bool, optional
        if ``True``, calculate the mean and standard deviation for each
        power bin instead of the median and 68% percentile range.
    unit : str, optional
        unit of the cube data; will be used to determine the power spectrum
        unit as well as the plot labels.
    window_name : str, optional
        if specified, taper the cube along the frequency axis using the
        specified window.
    """
    def __init__(self, cube, pixelsize, frequencies, meanstd=False,
                 unit="???", window_name=None):
        logger.info("Initializing PS2D instance ...")
        self.cube = np.array(cube, dtype=float)
        self.pixelsize = pixelsize  # [arcsec]
        self.unit = unit
        logger.info("Loaded data cube: %dx%d (cells) * %d (channels)" %
                    (self.Nx, self.Ny, self.Nz))
        logger.info("Image pixel size: %.2f [arcsec]" % pixelsize)
        logger.info("Data unit: %s" % unit)

        self.frequencies = np.asarray(frequencies)  # [MHz]
        self.nfreq = len(self.frequencies)
        self.dfreq = self.frequencies[1] - self.frequencies[0]  # [MHz]
        if self.nfreq != self.Nz:
            raise RuntimeError("data cube and frequencies do not match!")
        logger.info("Frequency band: %.2f-%.2f [MHz]" %
                    (self.frequencies.min(), self.frequencies.max()))
        logger.info("Frequency channel width: %.2f [MHz], %d channels" %
                    (self.dfreq, self.nfreq))

        # Central frequency and redshift
        self.freqc = self.frequencies.mean()
        self.zc = freq2z(self.freqc)
        logger.info("Central frequency %.2f [MHz] <-> redshift %.4f" %
                    (self.freqc, self.zc))

        # Transverse comoving distance at zc; unit: [Mpc]
        self.DMz = cosmo.comoving_transverse_distance(self.zc).value
        self.meanstd = meanstd
        self.window_name = window_name
        self.window = self.gen_window(name=window_name)

    def gen_window(self, name=None):
        if name is None:
            return None

        window_func = getattr(signal.windows, name)
        nfreq = self.nfreq
        window = window_func(nfreq, sym=False)
        width_pix = self.nfreq
        logger.info("Generated window: %s (%d pixels)" % (name, width_pix))
        return window

    def calc_ps3d(self):
        """
        Calculate the 3D power spectrum of the image cube.

        The power spectrum is properly normalized to have dimension
        of [K^2 Mpc^3].
        """
        if self.window is not None:
            logger.info("Applying window along frequency axis ...")
            self.cube *= self.window[:, np.newaxis, np.newaxis]

        logger.info("3D FFTing data cube ...")
        cubefft = fftpack.fftshift(fftpack.fftn(self.cube))

        logger.info("Calculating 3D power spectrum ...")
        ps3d = np.abs(cubefft) ** 2  # [K^2]
        # Normalization
        norm1 = 1 / (self.Nx * self.Ny * self.Nz)
        norm2 = 1 / (self.fs_xy**2 * self.fs_z)  # [Mpc^3]
        norm3 = 1 / (2*np.pi)**3
        self.ps3d = ps3d * norm1 * norm2 * norm3  # [K^2 Mpc^3]
        return self.ps3d

    def calc_ps2d(self):
        """
        Calculate the 2D power spectrum by cylindrically binning
        the above 3D power spectrum.

        Returns
        -------
        ps2d : 3D `~numpy.ndarray`
            3D array of shape (3, n_k_los, n_k_perp) including the median
            and lower and upper errors (68% percentile range).
            If ``self.meanstd=True`` then the mean and standard deviation
            are calculated instead.

        Attributes
        ----------
        ps2d
        """
        logger.info("Calculating 2D power spectrum ...")
        n_k_perp = len(self.k_perp)
        n_k_los = len(self.k_los)
        ps2d = np.zeros(shape=(3, n_k_los, n_k_perp))  # value, errl, erru

        eps = 1e-8
        ic_xy = (np.abs(self.k_xy) < eps).nonzero()[0][0]
        ic_z = (np.abs(self.k_z) < eps).nonzero()[0][0]
        p_xy = np.arange(self.Nx) - ic_xy
        p_z = np.abs(np.arange(self.Nz) - ic_z)
        mx, my = np.meshgrid(p_xy, p_xy)
        rho, phi = self.cart2pol(mx, my)
        rho = np.around(rho).astype(int)

        logger.info("Cylindrically averaging 3D power spectrum ...")
        for r in range(n_k_perp):
            ix, iy = (rho == r).nonzero()
            for s in range(n_k_los):
                iz = (p_z == s).nonzero()[0]
                cells = np.concatenate([self.ps3d[z, iy, ix] for z in iz])
                if self.meanstd:
                    ps2d[0, s, r] = cells.mean()
                    std = cells.std()
                    ps2d[1, s, r] = std
                    ps2d[2, s, r] = std
                else:
                    median, q16, q84 = np.percentile(cells, q=(50, 16, 84))
                    ps2d[0, s, r] = median
                    ps2d[1, s, r] = median - q16
                    ps2d[2, s, r] = q84 - median

        self.ps2d = ps2d
        return ps2d

    def save(self, outfile, clobber=False):
        """
        Save the calculated 2D power spectrum as a FITS image.
        """
        hdu = fits.PrimaryHDU(data=self.ps2d, header=self.header)
        try:
            hdu.writeto(outfile, overwrite=clobber)
        except TypeError:
            hdu.writeto(outfile, clobber=clobber)
        logger.info("Wrote 2D power spectrum to file: %s" % outfile)

    def plot(self, ax, ax_err, colormap="jet"):
        """
        Plot the calculated 2D power spectrum.
        """
        x = self.k_perp
        y = self.k_los

        if self.meanstd:
            title = "2D Power Spectrum (mean)"
            title_err = "Error (standard deviation)"
        else:
            title = "2D Power Spectrum (median)"
            title_err = "Error (68% percentile range)"

        # median/mean
        mappable = ax.pcolormesh(x[1:], y[1:],
                                 np.log10(self.ps2d[0, 1:, 1:]),
                                 cmap=colormap)
        vmin, vmax = mappable.get_clim()
        ax.set(xscale="log", yscale="log",
               xlim=(x[1], x[-1]), ylim=(y[1], y[-1]),
               xlabel=r"k$_{\perp}$ [Mpc$^{-1}$]",
               ylabel=r"k$_{||}$ [Mpc$^{-1}$]",
               title=title)
        cb = ax.figure.colorbar(mappable, ax=ax, pad=0.01, aspect=30)
        cb.ax.set_xlabel(r"[%s$^2$ Mpc$^3$]" % self.unit)

        # error (68% percentile range / standard deviation)
        error = 0.5 * (self.ps2d[1, :, :] + self.ps2d[2, :, :])
        mappable = ax_err.pcolormesh(x[1:], y[1:],
                                     np.log10(error[1:, 1:]),
                                     cmap=colormap)
        mappable.set_clim(vmin, vmax)
        ax_err.set(xscale="log", yscale="log",
                   xlim=(x[1], x[-1]), ylim=(y[1], y[-1]),
                   xlabel=r"k$_{\perp}$ [Mpc$^{-1}$]",
                   ylabel=r"k$_{||}$ [Mpc$^{-1}$]",
                   title=title_err)
        cb = ax_err.figure.colorbar(mappable, ax=ax_err, pad=0.01, aspect=30)
        cb.ax.set_xlabel(r"[%s$^2$ Mpc$^3$]" % self.unit)

        return (ax, ax_err)

    @property
    def Nx(self):
        """
        Number of cells/pixels along the X axis.
        Cube shape/dimensions: [Z, Y, X]
        """
        return self.cube.shape[2]

    @property
    def Ny(self):
        return self.cube.shape[1]

    @property
    def Nz(self):
        return self.cube.shape[0]

    @property
    @lru_cache()
    def d_xy(self):
        """
        The sampling interval along the (X, Y) spatial dimensions,
        translated from the pixel size.
        Unit: [Mpc]

        Reference: Ref.[liu2014].Eq.(A7)
        """
        pixelsize = self.pixelsize / 3600  # [arcsec] -> [deg]
        d_xy = self.DMz * np.deg2rad(pixelsize)
        return d_xy

    @property
    @lru_cache()
    def d_z(self):
        """
        The sampling interval along the Z line-of-sight dimension,
        translated from the frequency channel width.
        Unit: [Mpc]

        Reference: Ref.[liu2014].Eq.(A9)
        """
        dfreq = self.dfreq  # [MHz]
        c = ac.c.to("km/s").value  # [km/s]
        Hz = cosmo.H(self.zc).value  # [km/s/Mpc]
        d_z = c * (1+self.zc)**2 * dfreq / Hz / freq21cm
        return d_z

    @property
    @lru_cache()
    def fs_xy(self):
        """
        The sampling frequency along the (X, Y) spatial dimensions:
            Fs = 1/T (inverse of interval)
        Unit: [Mpc^-1]
        """
        return 1/self.d_xy

    @property
    @lru_cache()
    def fs_z(self):
        """
        The sampling frequency along the Z line-of-sight dimension.
        Unit: [Mpc^-1]
        """
        return 1/self.d_z

    @property
    @lru_cache()
    def df_xy(self):
        """
        The spatial frequency bin size (i.e., resolution) along the
        (X, Y) dimensions.
        Unit: [Mpc^-1]
        """
        return self.fs_xy / self.Nx

    @property
    @lru_cache()
    def df_z(self):
        """
        The spatial frequency bin size (i.e., resolution) along the
        line-of-sight (Z) direction.
        Unit: [Mpc^-1]
        """
        return self.fs_z / self.Nz

    @property
    def dk_xy(self):
        """
        The k-space (spatial) frequency bin size (i.e., resolution).
        """
        return 2*np.pi * self.df_xy

    @property
    @lru_cache()
    def dk_z(self):
        return 2*np.pi * self.df_z

    @property
    @lru_cache()
    def k_xy(self):
        """
        The k-space coordinates along the (X, Y) spatial dimensions,
        which describe the spatial frequencies.

        NOTE:
        k = 2*pi * f, where "f" is the spatial frequencies, and the
        Fourier dual to spatial transverse distances x/y.

        Unit: [Mpc^-1]
        """
        f_xy = fftpack.fftshift(fftpack.fftfreq(self.Nx, d=self.d_xy))
        k_xy = 2*np.pi * f_xy
        return k_xy

    @property
    @lru_cache()
    def k_z(self):
        f_z = fftpack.fftshift(fftpack.fftfreq(self.Nz, d=self.d_z))
        k_z = 2*np.pi * f_z
        return k_z

    @property
    @lru_cache()
    def k_perp(self):
        """
        Comoving wavenumbers perpendicular to the LoS

        NOTE: The Nyquist frequency just located at the first element
              after fftshift when the length is even, and it is negative.
        """
        k_x = self.k_xy
        return k_x[k_x >= 0]

    @property
    @lru_cache()
    def k_los(self):
        """
        Comoving wavenumbers along the LoS
        """
        k_z = self.k_z
        return k_z[k_z >= 0]

    @staticmethod
    def cart2pol(x, y):
        """
        Convert Cartesian coordinates to polar coordinates.
        """
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return (rho, phi)

    @property
    def header(self):
        dk_xy = self.dk_xy
        dk_z = self.dk_z
        hdr = fits.Header()
        hdr["HDUNAME"] = ("PS2D", "block name")
        hdr["CONTENT"] = ("2D cylindrically averaged power spectrum",
                          "data product")
        hdr["BUNIT"] = ("%s^2 Mpc^3" % self.unit, "data unit")
        if self.meanstd:
            hdr["AvgType"] = ("mean + standard deviation", "average type")
        else:
            hdr["AvgType"] = ("median + 68% percentile range", "average type")

        hdr["WINDOW"] = (self.window_name, "window applied along LoS")

        # Physical coordinates: IRAF LTM/LTV
        # Li{Image} = LTMi_i * Pi{Physical} + LTVi
        # Reference: ftp://iraf.noao.edu/iraf/web/projects/fitswcs/specwcs.html
        hdr["LTV1"] = 0.0
        hdr["LTM1_1"] = 1.0 / dk_xy
        hdr["LTV2"] = 0.0
        hdr["LTM2_2"] = 1.0 / dk_z

        # WCS physical coordinates
        hdr["WCSTY1P"] = "PHYSICAL"
        hdr["CTYPE1P"] = ("k_perp", "wavenumbers perpendicular to LoS")
        hdr["CRPIX1P"] = (0.5, "reference pixel")
        hdr["CRVAL1P"] = (0.0, "coordinate of the reference pixel")
        hdr["CDELT1P"] = (dk_xy, "coordinate delta/step")
        hdr["CUNIT1P"] = ("Mpc^-1", "coordinate unit")
        hdr["WCSTY2P"] = "PHYSICAL"
        hdr["CTYPE2P"] = ("k_los", "wavenumbers along LoS")
        hdr["CRPIX2P"] = (0.5, "reference pixel")
        hdr["CRVAL2P"] = (0.0, "coordinate of the reference pixel")
        hdr["CDELT2P"] = (dk_z, "coordinate delta/step")
        hdr["CUNIT2P"] = ("Mpc^-1", "coordinate unit")

        # Data information
        hdr["PixSize"] = (self.pixelsize, "[arcsec] data cube pixel size")
        hdr["Z_C"] = (self.zc, "data cube central redshift")
        hdr["Freq_C"] = (self.freqc, "[MHz] data cube central frequency")
        hdr["Freq_Min"] = (self.frequencies.min(),
                           "[MHz] data cube minimum frequency")
        hdr["Freq_Max"] = (self.frequencies.max(),
                           "[MHz] data cube maximum frequency")
        # Command history
        hdr.add_history(" ".join(sys.argv))
        return hdr


def main():
    parser = argparse.ArgumentParser(
        description="Calculate 2D power spectrum from 3D image cube")
    parser.add_argument("-C", "--clobber", dest="clobber",
                        action="store_true",
                        help="overwrite existing file")
    parser.add_argument("-c", "--center", dest="center", type=int,
                        help="crop the central box region of specified " +
                        "size before calculating the power spectrum; " +
                        "only crop the spatial dimensions, while " +
                        "the frequency axis is unmodified.")
    parser.add_argument("-m", "--mean-std", dest="meanstd",
                        action="store_true",
                        help="calculate the mean and standard deviation " +
                        "for each averaged annulus instead of the median " +
                        "16%% and 84%% percentiles (i.e., 68%% error)")
    parser.add_argument("-P", "--no-plot", dest="noplot", action="store_true",
                        help="do NOT plot the 2D power spectrum and save")
    parser.add_argument("-p", "--pixelsize", dest="pixelsize", type=float,
                        help="spatial pixel size [arcsec] (required " +
                        "if cannot obtain from FITS header)")
    parser.add_argument("-w", "--window", dest="window",
                        choices=["nuttall"],
                        help="apply window along frequency axis " +
                        "(default: None)")
    parser.add_argument("-i", "--infile", dest="infile", nargs="+",
                        help="input FITS image cube(s); if multiple cubes " +
                        "are provided, they are added first.")
    parser.add_argument("-o", "--outfile", dest="outfile", required=True,
                        help="output 2D power spectrum FITS file")
    args = parser.parse_args()

    if (not args.clobber) and os.path.exists(args.outfile):
        raise OSError("outfile '%s' already exists" % args.outfile)

    with fits.open(args.infile[0]) as f:
        cube = f[0].data
        header = f[0].header
    logger.info("Cube shape: %dx%dx%d" % cube.shape)
    bunit = header.get("BUNIT", "???")
    logger.info("Cube data unit: %s" % bunit)
    if bunit.upper() not in ["K", "KELVIN", "MK"]:
        logger.warning("input cube in unknown unit: %s" % bunit)

    for fn in args.infile[1:]:
        logger.info("Adding additional FITS cube: %s" % fn)
        with fits.open(fn) as f:
            cube2 = f[0].data
            header2 = f[0].header
            bunit2 = header2.get("BUNIT", "???")
        if bunit2.upper() == bunit.upper():
            cube += cube2
        else:
            raise ValueError("cube has different unit: %s" % bunit2)

    if args.center:
        csize = args.center
        if csize >= min(cube.shape[1:]):
            raise ValueError("--center %d exceeds image size" % csize)
        logger.info("Central spatial crop box size: %dx%d" % (csize, csize))
        rows, cols = cube.shape[1:]
        rc, cc = rows//2, cols//2
        cs1, cs2 = csize//2, (csize+1)//2
        cube = cube[:, (rc-cs1):(rc+cs2), (cc-cs1):(cc+cs2)]
        logger.info("Cropped cube shape: %dx%dx%d" % cube.shape)

    frequencies = get_frequencies(header)
    if args.pixelsize:
        pixelsize = args.pixelsize  # [arcsec]
    else:
        pixelsize = get_pixelsize(header)
    if pixelsize is None:
        raise RuntimeError("--pixelsize required")

    ps2d = PS2D(cube=cube, pixelsize=pixelsize, frequencies=frequencies,
                meanstd=args.meanstd, unit=bunit, window_name=args.window)
    ps2d.calc_ps3d()
    ps2d.calc_ps2d()
    ps2d.save(outfile=args.outfile, clobber=args.clobber)

    if not args.noplot:
        fig = Figure(figsize=(16, 8), dpi=150)
        FigureCanvas(fig)
        ax = fig.add_subplot(1, 2, 1)
        ax_err = fig.add_subplot(1, 2, 2)
        ps2d.plot(ax=ax, ax_err=ax_err)
        fig.tight_layout()
        plotfile = os.path.splitext(args.outfile)[0] + ".png"
        fig.savefig(plotfile)
        logger.info("Plotted 2D PSD and saved to image: %s" % plotfile)


if __name__ == "__main__":
    main()
