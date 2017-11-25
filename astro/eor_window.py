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

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


plt.style.use("ggplot")

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

    Attributes
    ----------
    ps2d : 2D `~numpy.ndarray`
        Shape: [n_k_los, n_k_perp]
    """
    def __init__(self, infile):
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
        return self.k_perp[1]  # ignore the first 0

    @property
    def k_perp_max(self):
        return self.k_perp[-1]

    @property
    def k_los_min(self):
        return self.k_los[1]  # ignore the first 0

    @property
    def k_los_max(self):
        return self.k_los[-1]

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

    def eor_window(self, fov, e,
                   k_perp_min=None, k_perp_max=None,
                   k_los_min=None, k_los_max=None):
        """
        Determine the EoR window region.

        Parameters
        ----------
        fov : float
            instrumental field of view (FoV)
            Unit: [deg]
        e : float
            Thyagarajan proposed characteristic convolution width factor,
            generally 0-3

        Returns
        -------
        window : 2D bool `~numpy.ndarray`
            2D array mask of the same size of the power spectrum indicating
            the defined EoR window region.
        header : fits.Header
            FITS header with the keywords recording the EoR window variables
        """
        if k_perp_min is None:
            k_perp_min = self.k_perp_min
        if k_perp_max is None:
            k_perp_max = self.k_perp_max
        if k_los_min is None:
            k_los_min = self.k_los_min
        if k_los_max is None:
            k_los_max = self.k_los_max

        window = np.ones_like(self.ps2d, dtype=bool)
        k_perp = self.k_perp
        k_los = self.k_los
        k_wedge = self.wedge_edge(k_perp, fov=fov, e=e)
        window[k_los < k_los_min, :] = False
        window[k_los > k_los_max, :] = False
        window[:, k_perp < k_perp_min] = False
        window[:, k_perp > k_perp_max] = False
        for i, k in enumerate(k_wedge):
            window[k_los < k, i] = False

        header = self.eor_window_header(fov=fov, e=e,
                                        k_perp_min=k_perp_min,
                                        k_perp_max=k_perp_max,
                                        k_los_min=k_los_min,
                                        k_los_max=k_los_max)
        return (window, header)

    def eor_window_header(self, fov, e, k_perp_min, k_perp_max,
                          k_los_min, k_los_max):
        header = self.header.copy(strip=True)
        header["FoV"] = (fov, "[deg] Field of view to determine EoR window")
        header["e_ConvW"] = (e, "characteristic convolution width")
        header["kper_min"] = (k_perp_min, "[Mpc^-1] minimum k_perp")
        header["kper_max"] = (k_perp_max, "[Mpc^-1] maximum k_perp")
        header["klos_min"] = (k_los_min, "[Mpc^-1] minimum k_los")
        header["klos_max"] = (k_los_max, "[Mpc^-1] maximum k_los")
        return header

    def wedge_edge(self, k_perp, fov, e):
        """
        The boundary/edge between the EoR window (top-left) and the
        foreground wedge (bottom-right).
        """
        Hz = cosmo.H(self.zc).value  # [km/s/Mpc]
        Dc = cosmo.comoving_distance(self.zc).value  # [Mpc]
        c = ac.c.to("km/s").value  # [km/s]
        coef = Hz * Dc / (c * (1+self.zc))
        term1 = np.sin(np.deg2rad(fov)) * k_perp  # [Mpc^-1]
        term2 = ((2*np.pi * e * freq21cm / self.bandwidth) /
                 ((1 + self.zc) * Dc))  # [Mpc^-1]
        k_los = coef * (term1 + term2)
        return k_los

    def plot(self, ax, fov, e,
             k_perp_min=None, k_perp_max=None,
             k_los_min=None, k_los_max=None,
             colormap="jet"):
        """
        Plot the 2D power spectrum with EoR window marked on.
        """
        if k_perp_min is None:
            k_perp_min = self.k_perp_min
        if k_perp_max is None:
            k_perp_max = self.k_perp_max
        if k_los_min is None:
            k_los_min = self.k_los_min
        if k_los_max is None:
            k_los_max = self.k_los_max

        x = self.k_perp
        y = self.k_los
        y_wedge = self.wedge_edge(x, fov=fov, e=e)
        title = "EoR Window (fov=%.1f[deg], e=%.1f)" % (fov, e)

        # data
        mappable = ax.pcolormesh(x[1:], y[1:],
                                 np.log10(self.ps2d[1:, 1:]),
                                 cmap=colormap)
        # EoR window
        ax.axvline(x=k_perp_min, color="black", linewidth=2, linestyle="--")
        ax.axvline(x=k_perp_max, color="black", linewidth=2, linestyle="--")
        ax.axhline(y=k_los_min, color="black", linewidth=2, linestyle="--")
        ax.axhline(y=k_los_max, color="black", linewidth=2, linestyle="--")
        ax.plot(x, y_wedge, color="black", linewidth=2, linestyle="--")
        #
        ax.set(xscale="log", yscale="log",
               xlim=(x[1], x[-1]), ylim=(y[1], y[-1]),
               xlabel=r"k$_{\perp}$ [Mpc$^{-1}$]",
               ylabel=r"k$_{||}$ [Mpc$^{-1}$]",
               title=title)
        cb = ax.figure.colorbar(mappable, ax=ax, pad=0.01, aspect=30)
        cb.ax.set_xlabel(r"[%s$^2$ Mpc$^3$]" % self.unit)
        return ax


def main():
    parser = argparse.ArgumentParser(
        description="Determine EoR window region and calculate total power")
    parser.add_argument("-F", "--fov", dest="fov",
                        type=float, required=True,
                        help="instrumental FoV to determine the EoR window")
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

    ps2d = PS2D(args.infile)
    window, header = ps2d.eor_window(fov=args.fov, e=args.conv_width,
                                     k_perp_min=args.k_perp_min,
                                     k_perp_max=args.k_perp_max,
                                     k_los_min=args.k_los_min,
                                     k_los_max=args.k_los_max)
    power = ps2d.sum_power(window)
    print("Total power within EoR window: %g [%s]" % (power, ps2d.unit))

    if args.save_window:
        hdu = fits.PrimaryHDU(data=window.astype(np.int16), header=header)
        hdu.writeto(args.save_window)
        print("Saved EoR window into file: %s" % args.save_window)

    if args.plot:
        fig = Figure(figsize=(8, 8), dpi=150)
        FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)
        ps2d.plot(ax=ax, fov=args.fov, e=args.conv_width,
                  k_perp_min=args.k_perp_min, k_perp_max=args.k_perp_max,
                  k_los_min=args.k_los_min, k_los_max=args.k_los_max)
        fig.tight_layout()
        fig.savefig(args.plot)
        print("Plotted 2D PSD with EoR window and saved to: %s" % args.plot)


if __name__ == "__main__":
    main()
