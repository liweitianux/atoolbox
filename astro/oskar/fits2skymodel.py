#!/usr/bin/env python3
#
# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT License
#

"""
Convert a FITS image to OSKAR sky model for simulation usage.

NOTE
----
The OSKAR sky model consists of all the valid (>threshold) pixels
from the given image (slice), and fluxes are given in unit [Jy],
therefore, the input image should be converted from brightness
temperature [K] to unit [Jy/pixel].

References
----------
[1] GitHub: OxfordSKA/OSKAR
    https://github.com/OxfordSKA/OSKAR
[2] OSKAR - Sky Model
    http://www.oerc.ox.ac.uk/~ska/oskar2/OSKAR-Sky-Model.pdf
[3] OSKAR - Settings
    http://www.oerc.ox.ac.uk/~ska/oskar2/OSKAR-Settings.pdf
"""

import os
import sys
import argparse
import logging
from datetime import datetime

import numpy as np
import astropy.io.fits as fits
import astropy.constants as ac
from astropy.wcs import WCS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(os.path.basename(sys.argv[0]))


class SkyModel:
    """
    OSKAR sky model.

    Parameters
    ----------
    image : 2D float `~numpy.ndarray`
        Input image array; unit [K] (brightness temperature)
    freq : float
        Frequency of the input image slice; unit [MHz]
    pixsize : float
        Pixel size of the input image; unit [arcmin]
    ra0, dec0 : float
        The coordinate of the image center; unit [deg]
    minvalue : float, optional
        The minimum threshold for the image values
    projection : str, optional
        The WCS projection for the image; default "TAN"
        TODO: support "SIN" etc.
    """
    def __init__(self, image, freq, pixsize, ra0, dec0,
                 minvalue=1e-4, projection="TAN"):
        self.image = image  # K (brightness temperature)
        self.freq = freq  # MHz
        self.pixsize = pixsize  # arcmin
        self.ra0 = ra0  # deg
        self.dec0 = dec0  # deg
        self.minvalue = minvalue
        self.projection = projection
        logger.info("SkyModel: Loaded image @ %.2f [MHz]" % freq)

    @property
    def wcs(self):
        """
        WCS for the given image slice.
        """
        shape = self.image.shape
        delta = self.pixsize / 60.0  # deg
        wcs_ = WCS(naxis=2)
        wcs_.wcs.ctype = ["RA---"+self.projection, "DEC--"+self.projection]
        wcs_.wcs.crval = np.array([self.ra0, self.dec0])
        wcs_.wcs.crpix = np.array([shape[1], shape[0]]) / 2.0 + 1
        wcs_.wcs.cdelt = np.array([delta, delta])
        return wcs_

    @property
    def fits_header(self):
        header = self.wcs.to_header()
        header["BUNIT"] = ("Jy/pixel", "Brightness unit")
        header["FREQ"] = (self.freq, "Frequency [MHz]")
        header["RA0"] = (self.ra0, "Center R.A. [deg]")
        header["DEC0"] = (self.dec0, "Center Dec. [deg]")
        header["PIXSIZE"] = (self.pixsize, "Pixel size [arcmin]")
        return header

    @property
    def factor_K2JyPixel(self):
        """
        Conversion factor to convert brightness unit from 'K' to 'Jy/pixel'

        http://www.iram.fr/IRAMFR/IS/IS2002/html_1/node187.html
        """
        pixarea = np.deg2rad(self.pixsize/60.0) ** 2  # [sr]
        kB = ac.k_B.si.value  # Boltzmann constant [J/K]
        c0 = ac.c.si.value  # speed of light in vacuum [m/s]
        freqHz = self.freq * 1e6  # [Hz]
        factor = 2*kB * 1.0e26 * pixarea * (freqHz/c0)**2
        return factor

    @property
    def ra_dec(self):
        """
        Calculate the (ra, dec) of each image pixel using the above WCS.

        NOTE: axis ordering difference between numpy array and FITS
        """
        shape = self.image.shape
        wcs = self.wcs
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        pix = np.column_stack([x.flatten(), y.flatten()])
        world = wcs.wcs_pix2world(pix, 0)
        ra = world[:, 0].reshape(shape)
        dec = world[:, 1].reshape(shape)
        return (ra, dec)

    @property
    def sky(self):
        """
        OSKAR sky model array converted from the input image.

        Columns
        -------
        ra : (J2000) right ascension (deg)
        dec : (J2000) declination (deg)
        flux : source (Stokes I) flux density (Jy)
        """
        idx = self.image.flatten() >= self.minvalue
        ra, dec = self.ra_dec
        ra = ra.flatten()[idx]
        dec = dec.flatten()[idx]
        flux = self.image.flatten()[idx] * self.factor_K2JyPixel
        sky_ = np.column_stack([ra, dec, flux])
        return sky_

    def write_sky_model(self, outfile, clobber=False):
        """
        Write the converted sky model for simulation.
        """
        if os.path.exists(outfile) and (not clobber):
            raise OSError("OSKAR sky model file already exists: " % outfile)
        sky = self.sky
        nsources = sky.shape[0]
        logger.info("Number of sources: %d" % nsources)
        header = ("Frequency = %.3f [MHz]\n" % self.freq +
                  "Pixel size = %.2f [arcmin]\n" % self.pixsize +
                  "RA0 = %.4f [deg]\n" % self.ra0 +
                  "Dec0 = %.4f [deg]\n" % self.dec0 +
                  "Number of sources = %d\n\n" % len(sky) +
                  "R.A.[deg]    Dec.[deg]    flux[Jy]")
        np.savetxt(outfile, sky, fmt='%.10e, %.10e, %.10e', header=header)
        logger.info("Wrote OSKAR sky model to file: %s" % outfile)

    def write_fits(self, outfile, oldheader=None, clobber=False):
        if os.path.exists(outfile) and (not clobber):
            raise OSError("Sky FITS already exists: " % outfile)
        if oldheader is not None:
            header = oldheader
            header.extend(self.fits_header, update=True)
        else:
            header = self.fits_header
        header.add_history(datetime.now().isoformat())
        header.add_history(" ".join(sys.argv))
        image = self.image
        image[image < self.minvalue] = np.nan
        image *= self.factor_K2JyPixel
        hdu = fits.PrimaryHDU(data=image, header=header)
        try:
            hdu.writeto(outfile, overwrite=True)
        except TypeError:
            hdu.writeto(outfile, clobber=True)  # old astropy versions
        logger.info("Wrote FITS image of sky model to file: %s" % outfile)


def main():
    parser = argparse.ArgumentParser(
        description="Convert FITS image to OSKAR sky model")
    parser.add_argument("-C", "--clobber", dest="clobber",
                        action="store_true",
                        help="overwrite existing file")
    parser.add_argument("-r", "--ra0", dest="ra0", type=float, required=True,
                        help="R.A. of the image center")
    parser.add_argument("-d", "--dec0", dest="dec0", type=float, required=True,
                        help="Dec. of the image center")
    parser.add_argument("-p", "--pix-size", dest="pixsize", type=float,
                        help="image pixel size [arcmin]; " +
                        "(default: obtain from the FITS header 'PIXSIZE')")
    parser.add_argument("-f", "--freq", dest="freq", type=float,
                        help="frequency [MHz] the image measured; " +
                        "(default: obtain from the FITS header 'FREQ')")
    exgrp = parser.add_mutually_exclusive_group()
    exgrp.add_argument("-m", "--min-value", dest="minvalue", type=float,
                       help="minimum threshold to the output sky model " +
                       "(default: 1e-4, i.e., 0.1 mK)")
    exgrp.add_argument("-M", "--min-peak-fraction", dest="minpfrac",
                       type=float,
                       help="minimum threshold determined as the fraction " +
                       "the peak value to the output sky model")
    #
    parser.add_argument("-F", "--osm-fits", dest="osmfits",
                        action="store_true",
                        help="save a FITS version of the converted sky model")
    parser.add_argument("infile", help="input FITS image")
    parser.add_argument("outfile", nargs="?",
                        help="output OSKAR sky model (default: " +
                        "save basename as the input FITS image)")
    args = parser.parse_args()

    if args.outfile:
        outfile = args.outfile
    else:
        outfile = os.path.splitext(args.infile)[0] + ".osm"

    with fits.open(args.infile) as f:
        image = f[0].data
        header = f[0].header
    logger.info("Read image slice: %s" % args.infile)
    freq = args.freq if args.freq else header["FREQ"]  # [MHz]
    pixsize = args.pixsize if args.pixsize else header["PIXSIZE"]  # [arcmin]
    logger.info("Frequency: %.2f [MHz]" % freq)
    logger.info("Pixel size: %.2f [arcmin]" % pixsize)
    minvalue = 1e-4  # i.e., 0.1 [mK]
    if args.minvalue:
        minvalue = args.minvalue
    if args.minpfrac:
        minvalue = args.minpfrac * image.max()
    logger.info("Minimum threshold: %g [K]" % minvalue)
    skymodel = SkyModel(image=image, freq=freq, ra0=args.ra0, dec0=args.dec0,
                        pixsize=pixsize, minvalue=minvalue)
    skymodel.write_sky_model(outfile, clobber=args.clobber)
    if args.osmfits:
        outfits = outfile + ".fits"
        skymodel.write_fits(outfits, oldheader=header, clobber=args.clobber)


if __name__ == "__main__":
    main()
