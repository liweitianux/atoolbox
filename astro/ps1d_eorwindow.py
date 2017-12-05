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

    Parameters
    ----------
    ps2d : `~PS2D`
        A `~PS2D` instance
    step : float, optional
        By default, a logarithmic grid with the specified step ratio
        (default: 1.1) will be used to do the azimuthal averages.
        If specified a value <=1 or None, then an equal-width pixel-by-pixel
        (along radial direction) grid is adopted.
    """
    def __init__(self, ps2d, step=1.1):
        self.ps2d = ps2d
        self.data = ps2d.ps2d  # shape: [n_k_los, n_k_perp]
        self.data_err = ps2d.ps2d_err
        self.eor_window = ps2d.eor_window()

        if step is None or step <= 1:
            self.step = None
        else:
            self.step = step

    @property
    def k_perp(self):
        return self.ps2d.k_perp

    @property
    def k_los(self):
        return self.ps2d.k_los

    @property
    def dk(self):
        """
        The wavenumber k bin size that will be used to determine the
        averaging grid.  Considering that the angular and line-of-sight
        wavenumber bin sizes are very different, their geometric mean
        is used instead.
        """
        k_perp = self.k_perp
        k_los = self.k_los
        dk_perp = k_perp[1] - k_perp[0]
        dk_los = k_los[1] - k_los[0]
        return np.sqrt(dk_perp * dk_los)

    @property
    def k(self):
        """
        The radial k positions to determine the averaging bins to derive
        the 1D power spectrum.
        """
        k_max = np.sqrt(self.k_perp[-1]**2 + self.k_los[-1]**2)
        dk = self.dk
        nk = int(k_max / dk) + 1
        x = np.arange(nk)
        if self.step is None:
            return x * dk
        else:
            xmax = x.max()
            x2 = list(x[x*(self.step-1) <= 1])
            v1 = x[len(x2)]
            while v1 < xmax:
                x2.append(v1)
                v1 *= self.step
            x2.append(xmax)
            return np.array(x2) * dk

    def calc_ps1d(self, normalize=True):
        """
        Calculate the 1D spherically averaged power spectrum by averaging
        the 2D cylindrical power spectrum.

        Parameters
        ----------
        normalize : bool
            Whether to normalize the 1D power spectrum to obtain the
            dimensionless one, i.e.,
                Δ^2(k) = (k^3 / (2*π^2)) P(k)

        Attributes
        ----------
        ps1d : 2D `~numpy.ndarray`
            3-column array storing the calculated 1D power spectrum,
            ``[k, ps1d, ps1d_err]``
        ps1d_normalized : bool
            Whether the calculated 1D power spectrum is normalized?

        Returns
        -------
        ps1d
        """
        ps2d = self.data
        ps2d_err = self.data_err
        k_perp = self.k_perp
        k_los = self.k_los
        ps1d_k = self.k
        nk = len(ps1d_k)
        print("Number of k points: %d" % nk)

        # PS1D's 3 columns: [k, ps1d, ps1d_err]
        ps1d = np.zeros(shape=(nk, 3))
        ps1d[:, 0] = ps1d_k

        print("Averaging 2D power spectrum ...")
        mx, my = np.meshgrid(k_perp, k_los)
        mk = np.sqrt(mx**2 + my**2)
        mk[~self.eor_window] = np.inf  # constrain within EoR window

        for i, k in enumerate(ps1d_k):
            ii, jj = (mk <= k).nonzero()
            mk[ii, jj] = np.inf
            data = ps2d[ii, jj]
            errors = ps2d_err[ii, jj]
            ncell = len(data)
            if ncell > 0:
                ps1d[i, 1] = np.mean(data)
                ps1d[i, 2] = np.sqrt(np.sum(errors ** 2)) / ncell

        if normalize:
            # XXX is this normalization correct???
            coef = ps1d_k**3 / (2*np.pi**2)
            ps1d[:, 1] *= coef
            ps1d[:, 2] *= coef
            self.ps1d_normalized = True
        else:
            self.ps1d_normalized = False

        self.ps1d = ps1d
        return ps1d

    def save(self, outfile):
        if self.ps1d_normalized:
            ps1d_desc = "normalized power [K^2]"
        else:
            ps1d_desc = "power [K^2 Mpc^3]"
        header = [
            "EoR window definition:",
            "+ FoV: %f [deg]" % self.ps2d.fov,
            "+ e_ConvWidth: %f" % self.ps2d.e,
            "+ k_perp_min: %f [Mpc^-1]" % self.ps2d.k_perp_min,
            "+ k_perp_max: %f [Mpc^-1]" % self.ps2d.k_perp_max,
            "+ k_los_min: %f [Mpc^-1]" % self.ps2d.k_los_min,
            "+ k_los_max: %f [Mpc^-1]" % self.ps2d.k_los_max,
            "",
            "Columns:",
            "1. k: wavenumber [Mpc^-1]",
            "2. ps1d: %s" % ps1d_desc,
            "ps1d_err: power errors",
            "",
            "k   ps1d   ps1d_err",
        ]
        np.savetxt(outfile, self.ps1d, header="\n".join(header))
        print("Saved 1D power spectrum to file: %s" % outfile)

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
               xlabel=r"k [Mpc$^{-1}$]", ylabel=ylabel,
               title="1D Spherically Averaged Power Spectrum")
        return ax


def main():
    parser = argparse.ArgumentParser(
        description="Calculate 1D power spectrum within the EoR window")
    parser.add_argument("-C", "--clobber", dest="clobber", action="store_true",
                        help="overwrite the output files if already exist")
    parser.add_argument("-s", "--step", dest="step", type=float, default=1.1,
                        help="step ratio (>1; default: 1.1) between 2 " +
                        "consecutive radial k bins, i.e., logarithmic grid. " +
                        "if specified a value <=1, then an equal-width grid " +
                        "of current k bin size will be used.")
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
    ps1d = PS1D(ps2d, step=args.step)
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
