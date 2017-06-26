#!/usr/bin/env python3
#
# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license
#

"""
Calculate the 2D cylindrical-averaged power spectrum from the
3D image spectral cube.
"""

import os
import sys
import argparse
import logging

import numpy as np
from scipy import fftpack
from scipy.signal import windows
from astropy.io import fits
from astropy.wcs import WCS
from astropy.cosmology import FlatLambdaCDM
import astropy.constants as ac


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(os.path.basename(sys.argv[0]))


# present Hubble parameter
H0 = 71.0  # [km/s/Mpc]
# present density parameter of matter
OmegaM0 = 0.27
# HI line frequency
freq21cm = 1420.405751  # [MHz]


def freq2z(freq):
    z = freq21cm / freq - 1.0
    return z


def get_frequencies(wcs, nfreq):
    pix = np.zeros(shape=(nfreq, 3), dtype=np.int)
    pix[:, -1] = np.arange(nfreq)
    world = wcs.wcs_pix2world(pix, 0)
    freqMHz = world[:, -1] / 1e6
    return freqMHz


class PS2D:
    """
    2D cylindrical-averaged power spectrum

    cube dimensions: [nfreq, height, width] / [Z, Y, X]
    """
    def __init__(self, cube, pixelsize, frequencies,
                 window="nuttall", width="extended"):
        logger.info("Initializing PS2D instance ...")
        self.cube = cube
        self.pixelsize = pixelsize  # [arcmin]
        self.frequencies = np.array(frequencies)  # [MHz]
        self.nfreq = len(self.frequencies)
        # Central frequency and redshift
        self.freqc = self.frequencies.mean()
        self.zc = freq2z(self.freqc)
        logger.info("Central frequency %.2f [MHz] <-> redshift %.4f" %
                    (self.freqc, self.zc))
        self.cosmo = FlatLambdaCDM(H0=H0, Om0=OmegaM0)
        # Transverse comoving distance at zc; unit: [Mpc]
        self.DMz = self.cosmo.comoving_transverse_distance(self.zc).value
        self.set_window(name=window, width=width)

    def set_window(self, name, width="extended"):
        self.window = {
            "name": name,
            "func": getattr(windows, name),
            "width": width
        }
        filter = self.window["func"](self.window_width, sym=False)
        if len(filter) > self.nfreq:
            # cut the filter
            midx = int(len(filter) / 2)  # index of the peak element
            nleft = int(self.nfreq / 2)  # number of element on the left
            nright = int((self.nfreq-1) / 2)  # number of element on the right
            filter = filter[(midx-nleft):(midx+nright+1)]
        self.window["filter"] = filter
        logger.info("Set window: %s (%s)" % (name, width))

    @property
    def window_width(self):
        if self.window["width"] == "extended":
            w = self.window["func"](self.nfreq, sym=False)
            ex = 1.0 / (w.sum() / self.nfreq)
            return int(ex * self.nfreq)
        else:
            return self.nfreq

    def pad_cube(self):
        # Pad the image cube to be square in spatial dimensions.
        # TODO
        __, ny, nz = self.cube.shape
        if ny != nz:
            logger.info("Padding image to be square ...")
            raise RuntimeError("image must be square!")

    def calc_ps3d(self):
        """
        Calculate the 3D power spectrum of the image cube.
        """
        logger.info("Applying window to frequency axis ...")
        w = self.window["filter"]
        cube2 = self.cube * w[:, np.newaxis, np.newaxis]
        logger.info("Calculating 3D FFT and PS ...")
        cubefft = fftpack.fftshift(fftpack.fftn(cube2))
        self.ps3d = np.abs(cubefft) ** 2
        return self.ps3d

    def calc_ps2d(self):
        """
        Calculate the 2D power spectrum by cylindrically binning
        the above 3D power spectrum.
        """
        nz, ny, nx = self.cube.shape
        k_x, k_y = self.k_xy
        k_z = self.k_z
        dkx = np.abs(k_x[0] - k_x[1])
        dkz = np.abs(k_z[0] - k_z[1])
        vcell = dkx**2 * dkz  # volume of each cell [Mpc^-3]
        eps = 1e-8
        ic_x = (np.abs(k_x) < eps).nonzero()[0][0]
        ic_z = (np.abs(k_z) < eps).nonzero()[0][0]
        p_x = np.arange(nx) - ic_x
        p_z = np.abs(np.arange(ny) - ic_z)
        mx, my = np.meshgrid(p_x, p_x)
        rho, phi = self.cart2pol(mx, my)
        rho = np.around(rho).astype(np.int)
        n_k_prep = (nx+1) // 2
        n_k_los = (nz+1) // 2
        ps2d = np.zeros(shape=(n_k_los, n_k_prep))  # (k_los, k_prep)
        logger.info("Calculating 2D PS by binning 3D PS ...")
        for r in range(n_k_prep):
            ix, iy = (rho == r).nonzero()
            for s in range(n_k_los):
                iz = (p_z == s).nonzero()[0]
                cells = np.concatenate([self.ps3d[z, iy, ix] for z in iz])
                volume = cells.size * vcell
                ps2d[s, r] = cells.sum() / volume
        self.ps2d = ps2d
        return ps2d

    def save(self, outfile, clobber=False):
        """
        Save the calculated 2D power spectrum as a FITS image.
        """
        hdu = fits.PrimaryHDU(data=self.ps2d, header=self.header)
        hdu.writeto(outfile, clobber=clobber)
        logger.info("PS2D results saved to file: %s" % outfile)

    @property
    def k_xy(self):
        __, ny, nx = self.cube.shape
        dxy = self.DMz * np.deg2rad(self.pixelsize / 60.0)  # [Mpc]
        kx = 2*np.pi * fftpack.fftshift(fftpack.fftfreq(nx, dxy))
        ky = 2*np.pi * fftpack.fftshift(fftpack.fftfreq(ny, dxy))
        return (kx, ky)  # [Mpc^-1]

    @property
    def k_z(self):
        freq_step = 1e6 * (self.frequencies[1] - self.frequencies[0])  # [Hz]
        eta = fftpack.fftshift(fftpack.fftfreq(self.nfreq, freq_step))  # [s]
        c = ac.c.si.value  # [m/s]
        h = H0 * 1000.0  # [m/s/Mpc]
        f21cm = freq21cm * 1e6  # [Hz]
        denom = c * (1+self.zc)**2 / h / f21cm / self.cosmo.efunc(self.zc)
        kz = 2*np.pi * eta / denom
        return kz  # [Mpc^-1]

    @property
    def k_prep(self):
        """
        Comoving wavenumbers perpendicular to the LoS

        NOTE: The Nyquist frequency just located at the first element
              after fftshift when the length is even, and it is negative.
        """
        k_x, k_y = self.k_xy
        return k_x[k_x >= 0]

    @property
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
        kx, __ = self.k_xy
        kz = self.k_z
        dkx = np.abs(kx[0] - kx[1])
        dkz = np.abs(kz[0] - kz[1])
        hdr = fits.Header()
        hdr["HDUNAME"] = ("PS2D", "block name")
        hdr["CONTENT"] = ("2D cylindrical-averaged power spectrum",
                          "data product")
        hdr["BUNIT"] = ("K^2 Mpc^3", "data unit")
        # Physical coordinates: IRAF LTM/LTV
        # Li{Image} = LTMi_i * Pi{Physical} + LTVi
        # Reference: ftp://iraf.noao.edu/iraf/web/projects/fitswcs/specwcs.html
        hdr["LTV1"] = 0.0
        hdr["LTM1_1"] = 1.0 / dkx
        hdr["LTV2"] = 0.0
        hdr["LTM2_2"] = 1.0 / dkz
        # WCS physical coordinates
        hdr["WCSTY1P"] = "PHYSICAL"
        hdr["CTYPE1P"] = ("k_prep", "wavenumbers perpendicular to LoS")
        hdr["CRPIX1P"] = (0.5, "reference pixel")
        hdr["CRVAL1P"] = (0.0, "coordinate of the reference pixel")
        hdr["CDELT1P"] = (dkx, "coordinate delta/step")
        hdr["CUNIT1P"] = ("Mpc^-1", "coordinate unit")
        hdr["WCSTY2P"] = "PHYSICAL"
        hdr["CTYPE2P"] = ("k_los", "wavenumbers along LoS")
        hdr["CRPIX2P"] = (0.5, "reference pixel")
        hdr["CRVAL2P"] = (0.0, "coordinate of the reference pixel")
        hdr["CDELT2P"] = (dkz, "coordinate delta/step")
        hdr["CUNIT2P"] = ("Mpc^-1", "coordinate unit")
        # Command history
        hdr.add_history(" ".join(sys.argv))
        return hdr


def main():
    parser = argparse.ArgumentParser(
        description="Calculate 2D PS from 3D image cube")
    parser.add_argument("-C", "--clobber", dest="clobber",
                        action="store_true",
                        help="overwrite existing file")
    parser.add_argument("-p", "--pixelsize", dest="pixelsize", required=True,
                        help="image cube pixel size; unit: [arcmin]")
    parser.add_argument("-i", "--infile", dest="infile", required=True,
                        help="input FITS image cube")
    parser.add_argument("-o", "--outfile", dest="outfile", required=True,
                        help="output 2D power spectrum FITS file")
    args = parser.parse_args()

    with fits.open(args.infile) as f:
        cube = f[0].data
        wcs = WCS(f[0].header)
    nfreq = cube.shape[0]
    frequencies = get_frequencies(wcs, nfreq)
    logger.info("%d frequencies [MHz]:" % nfreq)
    for f in frequencies:
        logger.info("* %.2f" % f)
    ps2d = PS2D(cube=cube, pixelsize=args.pixelsize, frequencies=frequencies)
    ps2d.calc_ps3d()
    ps2d.calc_ps2d()
    ps2d.save(outfile=args.outfile, clobber=args.clobber)


if __name__ == "__main__":
    main()
