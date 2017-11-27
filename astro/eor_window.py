#!/usr/bin/env python3
#
# Copyright (c) Weitian LI <weitian@aaronly.me>
# MIT license
#

"""
Calculate the total power within the EoR window on the 2D power spectrum.

The adopted EoR window definition is from [thyagarajan2013],Eq.(26),Fig.(11).

.. [thyagarajan2013]
   Thyagarajan et al. 2013, ApJ, 776, 6
"""

import argparse
from functools import lru_cache

import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import astropy.constants as ac

import matplotlib
import matplotlib.style
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


# Matplotlib settings
matplotlib.style.use("ggplot")
for k, v in [("font.family",       "monospace"),
             ("image.cmap",        "jet"),
             ("xtick.major.size",  7.0),
             ("xtick.major.width", 2.0),
             ("xtick.minor.size",  4.0),
             ("xtick.minor.width", 1.5),
             ("ytick.major.size",  7.0),
             ("ytick.major.width", 2.0),
             ("ytick.minor.size",  4.0),
             ("ytick.minor.width", 1.5)]:
    matplotlib.rcParams[k] = v

# HI line frequency
freq21cm = 1420.405751  # [MHz]
# Adopted cosmology
H0 = 71.0  # [km/s/Mpc]
OmegaM0 = 0.27
cosmo = FlatLambdaCDM(H0=H0, Om0=OmegaM0)


@lru_cache()
def freq2z(freq):
    z = freq21cm / freq - 1.0
    return z


class PS2D:
    """
    2D cylindrically averaged power spectrum; calculated by ``ps2d.py``.

    Parameters
    ----------
    fov : float
        instrumental field of view (FoV)
        Unit: [deg]
    e : float, optional
        Thyagarajan proposed characteristic convolution width factor,
        generally 0-2; default: 2.0

    Attributes
    ----------
    ps2d : 2D `~numpy.ndarray`
        Shape: [n_k_los, n_k_perp]
    """
    def __init__(self, infile, fov, e=2.0,
                 k_perp_min=None, k_perp_max=None,
                 k_los_min=None, k_los_max=None):
        self.infile = infile
        with fits.open(infile) as f:
            self.header = f[0].header
            self.ps2d = f[0].data[0, :, :]  # errors ignored
        self.freqc = self.header["Freq_C"]
        self.freqmin = self.header["Freq_Min"]
        self.freqmax = self.header["Freq_Max"]
        self.bandwidth = self.freqmax - self.freqmin  # [MHz]
        self.zc = self.header["Z_C"]
        self.pixelsize = self.header["PixSize"]
        self.unit = self.header["BUNIT"]
        self.set(fov=fov, e=e, k_perp_min=k_perp_min, k_perp_max=k_perp_max,
                 k_los_min=k_los_min, k_los_max=k_los_max)

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if key in ["fov", "e", "k_perp_min", "k_perp_max",
                       "k_los_min", "k_los_max"]:
                if value is not None:
                    setattr(self, key, value)
            else:
                raise ValueError("invalid item: %s" % key)

    @property
    def power_unit(self):
        return self.unit.split(" ")[0]  # [K^2] or [mK^2]

    @property
    def k_perp(self):
        dk = self.header["CDELT1P"]
        nk = self.header["NAXIS1"]
        return np.arange(nk) * dk

    @property
    def k_los(self):
        dk = self.header["CDELT2P"]
        nk = self.header["NAXIS2"]
        return np.arange(nk) * dk

    @property
    def k_perp_min(self):
        try:
            return self._k_perp_min
        except AttributeError:
            return self.k_perp[1]  # ignore the first 0

    @k_perp_min.setter
    def k_perp_min(self, value):
        self._k_perp_min = value

    @property
    def k_perp_max(self):
        try:
            return self._k_perp_max
        except AttributeError:
            return self.k_perp[-1]

    @k_perp_max.setter
    def k_perp_max(self, value):
        self._k_perp_max = value

    @property
    def k_los_min(self):
        try:
            return self._k_los_min
        except AttributeError:
            return self.k_los[1]  # ignore the first 0

    @k_los_min.setter
    def k_los_min(self, value):
        self._k_los_min = value

    @property
    def k_los_max(self):
        try:
            return self._k_los_max
        except AttributeError:
            return self.k_los[-1]

    @k_los_max.setter
    def k_los_max(self, value):
        self._k_los_max = value

    def sum_power(self, window):
        """
        Sum the power within the defined window.

        NOTE: The cylindrical average should be accounted for.
        """
        k_perp = self.k_perp
        k_los = self.k_los
        dk_perp = k_perp[1] - k_perp[0]
        dk_los = k_los[1] - k_los[0]
        volume = np.zeros_like(self.ps2d)
        volume[0, :] = 2*np.pi * k_perp * dk_perp * dk_los
        for i in range(1, len(k_los)):
            # The extra "2" to account for the average on +k_los and -k_los
            volume[i, :] = 2*np.pi * k_perp * dk_perp * dk_los * 2

        power = np.sum(self.ps2d * window * volume)
        return power

    def eor_window(self):
        """
        Determine the EoR window region.

        Attributes
        ----------
        window : 2D bool `~numpy.ndarray`
            2D array mask of the same size of the power spectrum indicating
            the defined EoR window region.

        Returns
        -------
        window
        """
        print("k_perp: [%g, %g] [Mpc^-1]" % (self.k_perp_min, self.k_perp_max))
        print("k_los: [%g, %g] [Mpc^-1]" % (self.k_los_min, self.k_los_max))
        print("FoV: %.1f [deg]" % self.fov)
        print("e_ConvWidth: %.1f" % self.e)
        window = np.ones_like(self.ps2d, dtype=bool)
        k_perp = self.k_perp
        k_los = self.k_los
        k_wedge = self.wedge_edge()
        window[k_los < self.k_los_min, :] = False
        window[k_los > self.k_los_max, :] = False
        window[:, k_perp < self.k_perp_min] = False
        window[:, k_perp > self.k_perp_max] = False
        for i, k in enumerate(k_wedge):
            window[k_los < k, i] = False
        self.window = window
        return window

    def header_eor_windowr(self):
        header = self.header.copy(strip=True)
        header["FoV"] = (self.fov,
                         "[deg] Field of view to determine EoR window")
        header["e_ConvW"] = (self.e, "characteristic convolution width")
        header["kper_min"] = (self.k_perp_min, "[Mpc^-1] minimum k_perp")
        header["kper_max"] = (self.k_perp_max, "[Mpc^-1] maximum k_perp")
        header["klos_min"] = (self.k_los_min, "[Mpc^-1] minimum k_los")
        header["klos_max"] = (self.k_los_max, "[Mpc^-1] maximum k_los")
        return header

    def wedge_edge(self):
        """
        The boundary/edge between the EoR window (top-left) and the
        foreground wedge (bottom-right).
        """
        Hz = cosmo.H(self.zc).value  # [km/s/Mpc]
        Dc = cosmo.comoving_distance(self.zc).value  # [Mpc]
        c = ac.c.to("km/s").value  # [km/s]
        coef = Hz * Dc / (c * (1+self.zc))
        term1 = np.sin(np.deg2rad(self.fov)) * self.k_perp  # [Mpc^-1]
        term2 = ((2*np.pi * self.e * freq21cm / self.bandwidth) /
                 ((1 + self.zc) * Dc))  # [Mpc^-1]
        k_los = coef * (term1 + term2)
        return k_los

    def save_eor_window(self, outfile, clobber=False):
        header = self.header_eor_windowr()
        hdu = fits.PrimaryHDU(data=self.window.astype(np.int16), header=header)
        try:
            hdu.writeto(outfile, overwrite=clobber)
        except TypeError:
            hdu.writeto(outfile, clobber=clobber)

    def plot(self, ax, power=None, colormap="jet"):
        """
        Plot the 2D power spectrum with EoR window marked on.
        """
        x = self.k_perp
        y = self.k_los
        y_wedge = self.wedge_edge()
        if power is None:
            title = "EoR Window (fov=%.1f[deg], e=%.1f)" % (self.fov, self.e)
        else:
            title = ("EoR Window (fov=%.1f[deg], e=%.1f, power=%g[%s])" %
                     (self.fov, self.e, power, self.power_unit))

        # data
        mappable = ax.pcolormesh(x[1:], y[1:],
                                 np.log10(self.ps2d[1:, 1:]),
                                 cmap=colormap)
        # EoR window
        ax.axvline(x=self.k_perp_min, color="black",
                   linewidth=2, linestyle="--")
        ax.axvline(x=self.k_perp_max, color="black",
                   linewidth=2, linestyle="--")
        ax.axhline(y=self.k_los_min, color="black",
                   linewidth=2, linestyle="--")
        ax.axhline(y=self.k_los_max, color="black",
                   linewidth=2, linestyle="--")
        ax.plot(x, y_wedge, color="black", linewidth=2, linestyle="--")
        #
        ax.set(xscale="log", yscale="log",
               xlim=(x[1], x[-1]), ylim=(y[1], y[-1]),
               xlabel=r"k$_{\perp}$ [Mpc$^{-1}$]",
               ylabel=r"k$_{||}$ [Mpc$^{-1}$]",
               title=title)
        cb = ax.figure.colorbar(mappable, ax=ax, pad=0.01, aspect=30)
        cb.ax.set_xlabel("[%s]" % self.unit)
        return ax


def main():
    parser = argparse.ArgumentParser(
        description="Determine EoR window region and calculate total power")
    parser.add_argument("-F", "--fov", dest="fov",
                        type=float, required=True,
                        help="instrumental FoV to determine the EoR window; " +
                        "SKA1-Low has FoV ~ 3.12 / (nu/200MHz) [deg], i.e., " +
                        "~5.03 @ 124, ~3.95 @ 158, ~3.18 @ 196")
    parser.add_argument("-e", "--conv-width", dest="conv_width",
                        type=float, default=3.0,
                        help="characteristic convolution width (default: 3.0)")
    parser.add_argument("-p", "--k-perp-min", dest="k_perp_min", type=float,
                        help="minimum k wavenumber perpendicular to LoS; " +
                        "unit: [Mpc^-1]")
    parser.add_argument("-P", "--k-perp-max", dest="k_perp_max", type=float,
                        help="maximum k wavenumber perpendicular to LoS")
    parser.add_argument("-l", "--k-los-min", dest="k_los_min", type=float,
                        help="minimum k wavenumber along LoS")
    parser.add_argument("-L", "--k-los-max", dest="k_los_max", type=float,
                        help="maximum k wavenumber along LoS")
    parser.add_argument("--save-window", dest="save_window",
                        help="save the determined EoR window into FITS " +
                        "file with the provided filename")
    parser.add_argument("--plot", dest="plot",
                        help="plot the 2D power spectrum with the " +
                        "determined EoR window marked, and save into " +
                        "the specified file")
    parser.add_argument("infile", help="2D power spectrum FITS file")
    args = parser.parse_args()

    ps2d = PS2D(args.infile, fov=args.fov, e=args.conv_width,
                k_perp_min=args.k_perp_min, k_perp_max=args.k_perp_max,
                k_los_min=args.k_los_min, k_los_max=args.k_los_max)
    window = ps2d.eor_window()
    power = ps2d.sum_power(window)
    print("Total power within EoR window: %g [%s]" % (power, ps2d.power_unit))

    if args.save_window:
        ps2d.save_eor_window(outfile=args.save_window)
        print("Saved EoR window into file: %s" % args.save_window)

    if args.plot:
        fig = Figure(figsize=(8, 8), dpi=150)
        FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)
        ps2d.plot(ax=ax, power=power)
        fig.tight_layout()
        fig.savefig(args.plot)
        print("Plotted 2D PSD with EoR window and saved to: %s" % args.plot)


if __name__ == "__main__":
    main()
