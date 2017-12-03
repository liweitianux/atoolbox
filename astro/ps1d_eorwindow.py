#!/usr/bin/env python3
#
# Copyright (c) 2017 Weitna LI <weitian@aaronly.me>
# MIT License
#

"""
Average the 2D power spectrum within the EoR window (i.e., excluding the
foreground contaminated wedge) to derive the 1D spherically averaged
power spectrum.
"""

import os
import argparse

import numpy as np

import matplotlib
import matplotlib.style
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from eor_window import PS2D


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


class PS1D:
    """
    Calculate the 1D spherically averaged power spectrum from 2D PS.
    """
    def __init__(self, ps2d):
        self.ps2d = ps2d
        self.data = ps2d.ps2d  # shape: [n_k_los, n_k_perp]
        self.data_err = ps2d.ps2d_err

    @property
    def k_perp(self):
        return self.ps2d.k_perp

    @property
    def k_los(self):
        return self.ps2d.k_los

    @property
    def eor_window(self):
        return self.ps2d.eor_window()

    def calc_ps1d(self, normalize=True):
        """
        Calculate the 1D spherically averaged power spectrum by averaging
        the 2D cylindrical power spectrum.

        Parameters
        ----------
        normalize : bool
            Whether to normalize the 1D power spectrum to obtain the
            dimensionless power spectrum, i.e.,
                Δ^2(k) = (k^3 / (2*π^2)) P(k)
        """
        eor_window = self.eor_window
        data = self.data.copy()
        data_err = self.data_err.copy()
        data[~eor_window] = np.nan
        data_err[~eor_window] = np.nan

        k_perp = self.k_perp
        k_los = self.k_los
        dk_perp = k_perp[1] - k_perp[0]
        dk_los = k_los[1] - k_los[0]
        dk = np.sqrt(dk_perp * dk_los)
        print("dk = %.6f [Mpc^-1]" % dk)
        k_max = np.sqrt(k_perp[-1]**2 + k_los[-1]**2)
        nk = int(k_max / dk) + 1
        print("number of k points: %d" % nk)
        ps1d_k = np.arange(nk) * dk

        # PS1D's 3 columns: [k, ps1d, ps1d_err]
        ps1d = np.zeros(shape=(nk, 3))
        ps1d[:, 0] = ps1d_k

        print("averaging 2D power spectrum ...")
        mx, my = np.meshgrid(k_perp, k_los)
        mk = np.sqrt(mx**2 + my**2)
        for i, k in enumerate(ps1d_k):
            ii, jj = (mk <= k).nonzero()
            mk[ii, jj] = np.inf
            cells = data[ii, jj]
            cells = cells[np.isfinite(cells)]
            if len(cells) > 0:
                ps1d[i, 1] = np.mean(cells)
                cells = data_err[ii, jj]
                cells = cells[np.isfinite(cells)]
                ps1d[i, 2] = np.sqrt(np.sum((cells/len(cells))**2))

        if normalize:
            coef = ps1d_k**3 / (2*np.pi**2)
            ps1d[:, 1] *= coef
            ps1d[:, 2] *= coef
            self.ps1d_normalized = True
        else:
            self.ps1d_normalized = False

        self.ps1d = ps1d
        return ps1d

    def save(self, outfile):
        ps1d = self.ps1d
        header = [
            "EoR window:",
            "  FoV: %f [deg]" % self.ps2d.fov,
            "  e_ConvWidth: %f" % self.ps2d.e,
            "  k_perp_min: %f [Mpc^-1]" % self.ps2d.k_perp_min,
            "  k_perp_max: %f [Mpc^-1]" % self.ps2d.k_perp_max,
            "  k_los_min: %f [Mpc^-1]" % self.ps2d.k_los_min,
            "  k_los_max: %f [Mpc^-1]" % self.ps2d.k_los_max,
            "",
            "k: wavenumber [Mpc^-1]",
        ]
        if self.ps1d_normalized:
            header += ["ps1d: normalized power [K^2]"]
        else:
            header += ["ps1d: power [K^2 Mpc^3]"]
        header += [
            "ps1d_err: power errors",
            "",
            "k   ps1d   ps1d_err"
        ]
        np.savetxt(outfile, ps1d, header="\n".join(header))
        print("saved 1D power spectrum to file: %s" % outfile)

    def plot(self, ax):
        ps1d = self.ps1d
        if self.ps1d_normalized:
            ylabel = r"$\Delta^2(k)$ [K$^2$]"
        else:
            ylabel = r"$P(k)$ [K$^2$ Mpc$^3$]"

        x = ps1d[:, 0]
        y = ps1d[:, 1]
        yerr = ps1d[:, 2]
        ax.errorbar(x[1:], y[1:], yerr=yerr[1:], fmt="none")
        ax.plot(x[1:], y[1:], marker="o")
        ax.set(xscale="log", yscale="log",
               xlabel=r"[Mpc$^{-1}$]", ylabel=ylabel,
               title="1D Spherically Average Power Spectrum")
        return ax


def main():
    parser = argparse.ArgumentParser(
        description="Calculate 1D power spectrum within the EoR window")
    parser.add_argument("-C", "--clobber", dest="clobber", action="store_true",
                        help="overwrite the output files if already exist")
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
    parser.add_argument("--no-plot", dest="noplot", action="store_true",
                        help="do not plot and save the calculated 1D power " +
                        "power within the EoR window")
    parser.add_argument("-i", "--infile", dest="infile", required=True,
                        help="2D power spectrum FITS file")
    parser.add_argument("-o", "--outfile", dest="outfile", required=True,
                        help="output TXT file to save the PSD data")
    args = parser.parse_args()

    if (not args.clobber) and os.path.exists(args.outfile):
        raise OSError("outfile '%s' already exists" % args.outfile)

    ps2d = PS2D(args.infile, fov=args.fov, e=args.conv_width,
                k_perp_min=args.k_perp_min, k_perp_max=args.k_perp_max,
                k_los_min=args.k_los_min, k_los_max=args.k_los_max)
    ps1d = PS1D(ps2d)
    ps1d.calc_ps1d()
    ps1d.save(args.outfile)

    if not args.noplot:
        fig = Figure(figsize=(8, 8), dpi=150)
        FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)
        ps1d.plot(ax=ax)
        fig.tight_layout()
        plotfile = os.path.splitext(args.outfile)[0] + ".png"
        fig.savefig(plotfile)
        print("Plotted 1D power spectrum within EoR window: %s" % plotfile)


if __name__ == "__main__":
    main()
