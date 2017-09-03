#!/usr/bin/env python3
#
# Copyright (c) 2017 Aaron LI <aly@aaronly.me>
# MIT license
#

"""
Rescale the FITS image to increase/decrease the pixel resolution
by interpolation, meanwhile keep the image FoV and update FITS WCS.

For example, we may simulate the galaxy clusters (and point sources)
foreground component with a (much) higher resolution (smaller pixels)
compared to other foreground components (e.g., Galactic synchrotron)
and EoR signal.  After simulating the observed images using OSKAR and
WSClean, the output images may in different sizes/resolution, which
may cause some troubles in subsequent usage, e.g., power spectrum
calculation.
"""

import sys
import argparse
from datetime import datetime, timezone

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage


class FITSImage:
    def __init__(self, filename, pixelsize=None):
        self.filename = filename
        self.header, self.image = self.open_image(filename)
        if pixelsize is not None:
            self.pixelsize = pixelsize
        print("Loaded FITS image from file: %s" % filename)
        print("FITS image size: %dx%d" % (self.Nx, self.Ny))
        print("Pixel size: %.1f [arcsec]" % self.pixelsize)
        print("FoV: %.2fx%.2f [deg^2]" % self.fov)

    def rescale(self, shape, order=1):
        print("Rescaling FITS image ...")
        try:
            Ny2, Nx2 = shape
        except TypeError:
            Ny2 = Nx2 = shape
        print("Scale output size: %dx%d" % (Nx2, Ny2))
        print("Scale interpolation order: %d" % order)
        zoom = ((Ny2+0.1)/self.Ny, (Nx2+0.1)/self.Nx)
        pixelsize_old = self.pixelsize
        self.image = ndimage.zoom(self.image, zoom=zoom, order=order)
        self.pixelsize = pixelsize_old / zoom[0]
        print("Scaled pixel size: %.1f [arcsec]" % self.pixelsize)

    def crop(self, size):
        print("Cropping FITS image ...")
        try:
            xsize, ysize = size  # [deg]
        except TypeError:
            xsize = ysize = size
        Nx2 = round(xsize * 3600 / self.pixelsize)
        Ny2 = round(ysize * 3600 / self.pixelsize)
        if Nx2 > self.Nx or Ny2 > self.Ny:
            raise ValueError("Crop region too large!")

        print("Central crop box size: %.2fx%.2f [deg^2]" % (xsize, ysize))
        print("Cropped image size: %dx%d" % (Nx2, Ny2))
        xi0 = int((self.Nx-Nx2) / 2)
        yi0 = int((self.Ny-Ny2) / 2)
        self.image = self.image[yi0:(yi0+Ny2), xi0:(xi0+Nx2)]

    def write(self, outfile, clobber=False):
        header = self.header.copy(strip=True)
        header.extend(self.wcs.to_header(), update=True)
        header["DATE"] = (datetime.now(timezone.utc).astimezone().isoformat(),
                          "File creation date")
        header.add_history(" ".join(sys.argv))
        hdu = fits.PrimaryHDU(data=self.image, header=header)
        try:
            hdu.writeto(outfile, overwrite=clobber)
        except TypeError:
            hdu.writeto(outfile, clobber=clobber)
        print("Wrote FITS image to file: %s" % outfile)

    @property
    def fov(self):
        # Unit: [deg]
        return (self.Nx*self.pixelsize/3600, self.Ny*self.pixelsize/3600)

    @property
    def pixelsize(self):
        # Unit: [arcsec]
        if hasattr(self, "_pixelsize"):
            return self._pixelsize

        try:
            return self.header["PixSize"]  # [arcsec]
        except KeyError:
            try:
                return abs(self.header["CDELT1"]) * 3600  # [deg] -> [arcsec]
            except KeyError:
                raise ValueError("Cannot obtain pixel size from header; " +
                                 "please manually provide --pixelsize")

    @pixelsize.setter
    def pixelsize(self, value):
        # Unit: [arcsec]
        self._pixelsize = value

    @property
    def Nx(self):
        return self.image.shape[1]

    @property
    def Ny(self):
        return self.image.shape[0]

    @property
    def wcs(self):
        hdr = self.header
        w = WCS(naxis=2)
        w.wcs.equinox = hdr.get("EQUINOX", 2000.0)
        w.wcs.ctype = [hdr.get("CTYPE1", "RA---SIN"),
                       hdr.get("CTYPE2", "DEC--SIN")]
        w.wcs.crval = np.array([hdr.get("CRVAL1", 0.0),
                                hdr.get("CRVAL2", 0.0)])
        w.wcs.crpix = np.array([self.Ny/2+1, self.Nx/2+1])
        w.wcs.cdelt = np.array([-self.pixelsize/3600, self.pixelsize/3600])
        w.wcs.cunit = [hdr.get("CUNIT1", "deg"), hdr.get("CUNIT2", "deg")]
        return w

    @staticmethod
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
        description="Rescale a FITS image to the desired size/resolution")
    parser.add_argument("-C", "--clobber", action="store_true",
                        help="overwrite existing output file")
    parser.add_argument("--order", type=int, default=1,
                        help="scale interpolation order (default: 1)")
    parser.add_argument("-s", "--size", type=int, required=True,
                        help="output image size (number of pixels)")
    parser.add_argument("-p", "--pixelsize", type=float,
                        help="input FITS image pixel size [arcsec] " +
                        "(default: try to obtain from FITS header)")
    parser.add_argument("-i", "--infile", required=True,
                        help="input FITS image")
    parser.add_argument("-o", "--outfile", required=True,
                        help="output FITS image")
    args = parser.parse_args()

    fitsimage = FITSImage(args.infile, pixelsize=args.pixelsize)
    fitsimage.rescale(shape=args.size, order=args.order)
    fitsimage.write(args.outfile, clobber=args.clobber)


if __name__ == "__main__":
    main()
