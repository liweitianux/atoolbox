#!/usr/bin/env python3
#
# Copyright (c) Weitian LI <weitian@aaronly.me>
# MIT license
#

"""
FITS image manipulate tool.
"""

import sys
import argparse

import numpy as np
from astropy.io import fits


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

    def write(self, outfile, clobber=False):
        self.header.add_history(" ".join(sys.argv))
        hdu = fits.PrimaryHDU(data=self.data, header=self.header)
        try:
            hdu.writeto(outfile, overwrite=clobber)
        except TypeError:
            hdu.writeto(outfile, clobber=clobber)


def cmd_info(args):
    """
    Sub-command: "info", show FITS image information
    """
    fimage = FITSImage(args.infile)
    print("Image size: %dx%d" % (fimage.Nx, fimage.Ny))
    print("Data unit: [%s]" % fimage.bunit)
    pixelsize = fimage.pixelsize
    if pixelsize:
        print("Pixel size: %.1f [arcsec]" % pixelsize)
        print("Field of view: (%.2f, %.2f) [deg]" % fimage.fov)
    data = fimage.image
    if args.abs:
        data = np.abs(data)
    if args.center:
        print("Central box size: %d" % args.center)
        rows, cols = data.shape
        rc, cc = rows//2, cols//2
        cs1, cs2 = args.center//2, (args.center+1)//2
        data = data[(rc-cs1):(rc+cs2), (cc-cs1):(cc+cs2)]
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    iqr = np.diff(np.percentile(data, q=(25, 75)))
    mad = np.median(np.abs(data - median))
    rms = np.sqrt(np.mean(data**2))
    print("mean:   %13.6e" % mean)
    print("median: %13.6e" % median)
    print("std:    %13.6e  (standard deviation)" % std)
    print("iqr:    %13.6e  (interquartile range)" % iqr)
    print("mad:    %13.6e  (median absolute deviation)" % mad)
    print("rms:    %13.6e  (root-mean-squared)" % rms)


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
    parser_info.add_argument("infile", help="FITS image filename")
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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
