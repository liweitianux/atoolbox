#!/usr/bin/env python3
#
# Copyright (c) 2017-2018 Weitian LI <weitian@aaronly.me>
# MIT license
#

"""
FITS image manipulate tool.
"""

import sys
import argparse

import numpy as np
from astropy.io import fits
from scipy import ndimage


class FITSImage:
    """
    FITS image class that deals with plain 2D image (NAXIS=2), but also
    handles single-frequency single-polarized image cube (NAXIS=3, 4),
    e.g., created by WSClean.
    """
    def __init__(self, infile, pixelsize=None):
        self.infile = infile
        with fits.open(infile) as f:
            self.header = f[0].header.copy(strip=True)
            self.data = f[0].data
        self.ndim = self.data.ndim
        self.shape = self.data.shape
        if pixelsize is not None:
            self.pixelsize = pixelsize  # [arcsec]

    @property
    def bunit(self):
        return self.header.get("BUNIT")

    @property
    def Nx(self):
        """
        Number of pixels along the X axis (i.e., image width)
        """
        return self.shape[-1]

    @property
    def Ny(self):
        """
        Number of pixels along the Y axis (i.e., image height)
        """
        return self.shape[-2]

    @property
    def image(self):
        """
        Deal with single-frequency and single-polarized image cube.
        """
        if self.ndim == 2:
            # NAXIS=2: [Y, X]
            image = self.data[:, :].copy()
        elif self.ndim == 3 and self.shape[0] == 1:
            # NAXIS=3: [FREQ=1, Y, X]
            image = self.data[0, :, :].copy()
        elif self.ndim == 4 and self.shape[0] == 1 and self.shape[1] == 1:
            # NAXIS=4: [STOKES=1, FREQ=1, Y, X]
            image = self.data[0, 0, :, :].copy()
        else:
            raise ValueError("invalid data shape: {1}".format(self.shape))
        return image

    @image.setter
    def image(self, value):
        if self.ndim == 2:
            # NAXIS=2: [Y, X]
            self.data = np.array(value)
        elif self.ndim == 3:
            # NAXIS=3: [FREQ=1, Y, X]
            self.data = np.array(value)[np.newaxis, :, :]
        else:
            # NAXIS=4: [STOKES=1, FREQ=1, Y, X]
            self.data = np.array(value)[np.newaxis, np.newaxis, :, :]

    @property
    def pixelsize(self):
        """
        Image pixel size, in units of [arcsec]
        """
        if hasattr(self, "_pixelsize"):
            return self._pixelsize

        try:
            return self.header["PixSize"]  # [arcsec]
        except KeyError:
            try:
                return abs(self.header["CDELT1"]) * 3600  # [deg] -> [arcsec]
            except KeyError:
                return None

    @pixelsize.setter
    def pixelsize(self, value):
        # Unit: [arcsec]
        oldvalue = self.pixelsize
        self._pixelsize = value
        # Update header
        self.header["PixSize"] = value  # [arcsec]
        try:
            self.header["CDELT1"] *= value / oldvalue
            self.header["CDELT2"] *= value / oldvalue
        except KeyError:
            pass

    @property
    def fov(self):
        """
        Image FoV coverage: (fov_x, fov_y)
        Unit: [deg]
        """
        pixelsize = self.pixelsize
        if pixelsize:
            return (self.Nx*pixelsize/3600, self.Ny*pixelsize/3600)
        else:
            return None

    def zoom(self, newsize, order=1):
        """
        Zoom the image to the specified ``newsize``, meanwhile the header
        information will be updated accordingly to preserve the FoV coverage.

        NOTE
        ----
        The image aspect ratio cannot be changed.

        Parameters
        ----------
        newsize : (Nx, Ny) or N
            The size of the zoomed image.
        order : int, optional
            The interpolation order, default: 1
        """
        try:
            Nx2, Ny2 = newsize
        except TypeError:
            Nx2 = Ny2 = newsize
        zoom = ((Ny2+0.1)/self.Ny, (Nx2+0.1)/self.Nx)
        if abs(zoom[0] - zoom[1]) > 1e-3:
            raise RuntimeError("image aspect ratio cannot be changed")

        pixelsize_old = self.pixelsize
        self.image = ndimage.zoom(self.image, zoom=zoom, order=order)
        self.pixelsize = pixelsize_old * (self.Nx / Nx2)
        return self.image

    def write(self, outfile, clobber=False):
        self.header.add_history(" ".join(sys.argv))
        hdu = fits.PrimaryHDU(data=self.data, header=self.header)
        try:
            hdu.writeto(outfile, overwrite=clobber)
        except TypeError:
            hdu.writeto(outfile, clobber=clobber)


def show_info(filename, abs_=None, center=None):
    """
    Show FITS image information.
    """
    fimage = FITSImage(filename)
    print("Image data shape: {0}".format(fimage.shape))
    print("Image size: %dx%d" % (fimage.Nx, fimage.Ny))
    print("Data unit: [%s]" % fimage.bunit)
    pixelsize = fimage.pixelsize
    if pixelsize:
        print("Pixel size: %.1f [arcsec]" % pixelsize)
        print("Field of view: (%.2f, %.2f) [deg]" % fimage.fov)
    data = fimage.image
    if abs_:
        data = np.abs(data)
    if center:
        print("Central box size: %d" % center)
        rows, cols = data.shape
        rc, cc = rows//2, cols//2
        cs1, cs2 = center//2, (center+1)//2
        data = data[(rc-cs1):(rc+cs2), (cc-cs1):(cc+cs2)]
    min_ = np.nanmin(data)
    max_ = np.nanmax(data)
    mean = np.nanmean(data)
    median = np.nanmedian(data)
    std = np.nanstd(data)
    iqr = np.diff(np.nanpercentile(data, q=(25, 75)))
    mad = np.nanmedian(np.abs(data - median))
    rms = np.sqrt(np.nanmean(data**2))
    print("min:    %13.6e" % min_)
    print("max:    %13.6e" % max_)
    print("range:  %13.6e (max - min)" % (max_ - min_))
    print("mean:   %13.6e" % mean)
    print("median: %13.6e" % median)
    print("std:    %13.6e  (standard deviation)" % std)
    print("iqr:    %13.6e  (interquartile range)" % iqr)
    print("mad:    %13.6e  (median absolute deviation)" % mad)
    print("rms:    %13.6e  (root-mean-squared)" % rms)


def cmd_info(args):
    """
    Sub-command: "info", show FITS image information
    """
    for fn in args.files:
        print(">>> %s <<<" % fn)
        show_info(fn, abs_=args.abs, center=args.center)
        print("")


def cmd_add(args):
    """
    Sub-command: "add", add the image by a number or other image(s)
    """
    fimage = FITSImage(args.infile)
    image = fimage.image
    if args.number:
        print("Add by number: %g" % args.number)
        image += args.number
    else:
        for fn in args.files:
            print("Add by another image from: %s" % fn)
            fimage2 = FITSImage(fn)
            image += fimage2.image
    fimage.image = image
    fimage.write(args.outfile, clobber=args.clobber)
    print("Saved FITS image to: %s" % args.outfile)


def cmd_sub(args):
    """
    Sub-command: "sub", subtract the image by a number or other image(s)
    """
    fimage = FITSImage(args.infile)
    image = fimage.image
    if args.number:
        print("Subtract by number: %g" % args.number)
        image -= args.number
    else:
        for fn in args.files:
            print("Subtract by another image from: %s" % fn)
            fimage2 = FITSImage(fn)
            image -= fimage2.image
    fimage.image = image
    fimage.write(args.outfile, clobber=args.clobber)
    print("Saved FITS image to: %s" % args.outfile)


def cmd_mul(args):
    """
    Sub-command: "mul", multiply the image by a number or other image(s)
    """
    fimage = FITSImage(args.infile)
    image = fimage.image
    if args.number:
        print("Multiply by number: %g" % args.number)
        image *= args.number
    else:
        for fn in args.files:
            print("Multiply by another image from: %s" % fn)
            fimage2 = FITSImage(fn)
            image *= fimage2.image
    fimage.image = image
    fimage.write(args.outfile, clobber=args.clobber)
    print("Saved FITS image to: %s" % args.outfile)


def cmd_div(args):
    """
    Sub-command: "div", divide the image by a number or other image(s)
    """
    fimage = FITSImage(args.infile)
    image = fimage.image
    if args.number:
        print("Divide by number: %g" % args.number)
        image /= args.number
    else:
        for fn in args.files:
            print("Divide by another image from: %s" % fn)
            fimage2 = FITSImage(fn)
            with np.errstate(divide="warn"):
                image /= fimage2.image

    if args.fill_value:
        print("Filling invalid data with: %s" % args.fill_value)
        image[~np.isfinite(image)] = float(args.fill_value)
    fimage.image = image
    fimage.write(args.outfile, clobber=args.clobber)
    print("Saved FITS image to: %s" % args.outfile)


def cmd_zoom(args):
    """
    Sub-command: "zoom", zoom the image to a new size with FoV coverage
    preserved.
    """
    fimage = FITSImage(args.infile)
    print("Image size: %dx%d" % (fimage.Nx, fimage.Ny))
    pixelsize = fimage.pixelsize
    if pixelsize is None:
        raise RuntimeError("--pixelsize required")
    else:
        print("Pixel size: %.1f [arcsec]" % pixelsize)
        print("Field of view: (%.2f, %.2f) [deg]" % fimage.fov)

    print("Zooming image ...")
    print("Interpolation order: %d" % args.order)
    print("Zoomed image size: %dx%d" % (args.size, args.size))
    fimage.zoom(newsize=args.size, order=args.order)
    print("Zoomed image pixel size: %.1f [arcsec]" % fimage.pixelsize)
    fimage.write(args.outfile, clobber=args.clobber)
    print("Saved zoomed FITS image to: %s" % args.outfile)


def main():
    parser = argparse.ArgumentParser(
        description="FITS image manipulation tool")
    subparsers = parser.add_subparsers(dest="subparser_name",
                                       title="sub-commands",
                                       help="additional help")

    # sub-command: "info"
    parser_info = subparsers.add_parser(
        "info", aliases=["show"],
        help="show FITS image info")
    parser_info.add_argument("-c", "--center", dest="center", type=int,
                             help="choose central region of specified size")
    parser_info.add_argument("-a", "--abs", dest="abs", action="store_true",
                             help="take absolute values of image pixels")
    parser_info.add_argument("files", nargs="+", help="FITS image filename")
    parser_info.set_defaults(func=cmd_info)

    # sub-command: "add"
    parser_add = subparsers.add_parser(
        "add",
        help="add the image by a number or other image(s)")
    parser_add.add_argument("-C", "--clobber", dest="clobber",
                            action="store_true",
                            help="overwrite existing output file")
    parser_add.add_argument("-i", "--infile", dest="infile", required=True,
                            help="input FITS image")
    parser_add.add_argument("-o", "--outfile", dest="outfile", required=True,
                            help="output FITS image")
    exgrp_add = parser_add.add_mutually_exclusive_group(required=True)
    exgrp_add.add_argument("-n", "--number", dest="number", type=float,
                           help="number to be added by")
    exgrp_add.add_argument("-f", "--files", dest="files", nargs="+",
                           help="FITS image(s) to be added by")
    parser_add.set_defaults(func=cmd_add)

    # sub-command: "sub"
    parser_sub = subparsers.add_parser(
        "sub", aliases=["subtract"],
        help="subtract the image by a number or other image(s)")
    parser_sub.add_argument("-C", "--clobber", dest="clobber",
                            action="store_true",
                            help="overwrite existing output file")
    parser_sub.add_argument("-i", "--infile", dest="infile", required=True,
                            help="input FITS image")
    parser_sub.add_argument("-o", "--outfile", dest="outfile", required=True,
                            help="output FITS image")
    exgrp_sub = parser_sub.add_mutually_exclusive_group(required=True)
    exgrp_sub.add_argument("-n", "--number", dest="number", type=float,
                           help="number to be subtracted by")
    exgrp_sub.add_argument("-f", "--files", dest="files", nargs="+",
                           help="FITS image(s) to be subtracted by")
    parser_sub.set_defaults(func=cmd_sub)

    # sub-command: "mul"
    parser_mul = subparsers.add_parser(
        "mul", aliases=["multiply"],
        help="multiply the image by a number or other image(s)")
    parser_mul.add_argument("-C", "--clobber", dest="clobber",
                            action="store_true",
                            help="overwrite existing output file")
    parser_mul.add_argument("-i", "--infile", dest="infile", required=True,
                            help="input FITS image")
    parser_mul.add_argument("-o", "--outfile", dest="outfile", required=True,
                            help="output FITS image")
    exgrp_mul = parser_mul.add_mutually_exclusive_group(required=True)
    exgrp_mul.add_argument("-n", "--number", dest="number", type=float,
                           help="number to be multiplied by")
    exgrp_mul.add_argument("-f", "--files", dest="files", nargs="+",
                           help="FITS image(s) to be multiplied by")
    parser_mul.set_defaults(func=cmd_mul)

    # sub-command: "div"
    parser_div = subparsers.add_parser(
        "div", aliases=["divide"],
        help="divide the image by a number or other image(s)")
    parser_div.add_argument("-C", "--clobber", dest="clobber",
                            action="store_true",
                            help="overwrite existing output file")
    parser_div.add_argument("-F", "--fill-value", dest="fill_value",
                            help="value to fill the invalid elements")
    parser_div.add_argument("-i", "--infile", dest="infile", required=True,
                            help="input FITS image")
    parser_div.add_argument("-o", "--outfile", dest="outfile", required=True,
                            help="output FITS image")
    exgrp_div = parser_div.add_mutually_exclusive_group(required=True)
    exgrp_div.add_argument("-n", "--number", dest="number", type=float,
                           help="number to be divided by")
    exgrp_div.add_argument("-f", "--files", dest="files", nargs="+",
                           help="FITS image(s) to be divided by")
    parser_div.set_defaults(func=cmd_div)

    # sub-command: "zoom"
    parser_zoom = subparsers.add_parser(
        "zoom", aliases=["rescale"],
        help="zoom the image to a new size with FoV coverage preserved")
    parser_zoom.add_argument("-C", "--clobber", dest="clobber",
                             action="store_true",
                             help="overwrite existing output file")
    parser_zoom.add_argument("--order", type=int, default=1,
                             help="zoom interpolation order (default: 1)")
    parser_zoom.add_argument("-s", "--size", type=int, required=True,
                             help="zoomed image size (number of pixels)")
    parser_zoom.add_argument("-p", "--pixelsize", type=float,
                             help="input FITS image pixel size [arcsec] " +
                             "(default: try to obtain from FITS header)")
    parser_zoom.add_argument("-i", "--infile", dest="infile", required=True,
                             help="input FITS image")
    parser_zoom.add_argument("-o", "--outfile", dest="outfile", required=True,
                             help="output zoomed FITS image")
    parser_zoom.set_defaults(func=cmd_zoom)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
