#!/usr/bin/env python3
#
# Copyright (c) Weitian LI <weitian@aaronly.me>
# MIT license
#

"""
Create FITS image cube from a series of image slices.
"""

import os
import argparse

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


class FITSCube:
    """
    FITS image cube.
    """
    def __init__(self, infile=None):
        if infile is not None:
            self.load(infile)

    def load(self, infile):
        with fits.open(infile) as f:
            self.data = f[0].data
            self.header = f[0].header
        print("Loaded FITS cube from file: %s" % infile)
        print("Cube dimensions: %dx%dx%d" %
              (self.width, self.height, self.nslice))

    def add_slices(self, slices, zbegin=0.0, zstep=1.0):
        """
        Create a FITS cube from input image slices.
        """
        nslice = len(slices)
        with fits.open(slices[0]) as f:
            image = f[0].data
            header = f[0].header
        shape = (nslice, ) + image.shape
        data = np.zeros(shape, dtype=image.dtype)
        for i, s in enumerate(slices):
            print("Adding image slice: %s ..." % s)
            data[i, :, :] = fits.open(s)[0].data
        self.data = data
        wcs = self.make_wcs(header, zbegin=zbegin, zstep=zstep)
        self.header = header.copy(strip=True)
        self.header.extend(wcs.to_header(), update=True)
        print("Created FITS cube of dimensions: %dx%dx%d" %
              (self.width, self.height, self.nslice))

    def make_wcs(self, header, zbegin, zstep):
        w = WCS(naxis=3)
        w.wcs.ctype = ["pixel", "pixel", "pixel"]
        w.wcs.crpix = np.array([header.get("CRPIX1", 1.0),
                                header.get("CRPIX2", 1.0),
                                1.0])
        w.wcs.crval = np.array([header.get("CRVAL1", 0.0),
                                header.get("CRVAL2", 0.0),
                                zbegin])
        w.wcs.cdelt = np.array([header.get("CDELT1", 1.0),
                                header.get("CDELT2", 1.0),
                                zstep])
        return w

    def write(self, outfile, clobber):
        hdu = fits.PrimaryHDU(data=self.data, header=self.header)
        hdu.writeto(outfile, clobber=clobber)

    @property
    def width(self):
        __, __, w = self.data.shape
        return w

    @property
    def height(self):
        __, h, __ = self.data.shape
        return h

    @property
    def nslice(self):
        ns, __, __ = self.data.shape
        return ns

    @property
    def zbegin(self):
        """
        The Z-axis position of the first slice.
        """
        return self.header["CRVAL3"]

    @property
    def zstep(self):
        """
        The Z-axis step/spacing between slices.
        """
        return self.header["CDELT3"]

    @property
    def zvalues(self):
        """
        Calculate the Z-axis positions for all slices
        """
        nslice = self.nslice
        wcs = WCS(self.header)
        pix = np.zeros(shape=(nslice, 3), dtype=np.int)
        pix[:, 2] = np.arange(nslice)
        world = wcs.wcs_pix2world(pix, 0)
        return world[:, 2]


def cmd_info(args):
    """
    Sub-command: "info", show FITS cube information
    """
    cube = FITSCube(args.infile)
    print("Image/slice size: %dx%d" % (cube.width, cube.height))
    print("Number of slices: %d" % cube.nslice)
    print("Slice step/spacing: %.3f" % cube.zstep)
    print("Slice positions: {0}".format(cube.zvalues))


def cmd_create(args):
    """
    Sub-command: "create", create a FITS cube
    """
    if not args.clobber and os.path.exists(args.outfile):
        raise FileExistsError("output file already exists: %s" % args.outfile)
    cube = FITSCube()
    cube.add_slices(args.infiles, zbegin=args.zbegin, zstep=args.zstep)
    cube.write(args.outfile, clobber=args.clobber)
    print("Created FITS cube: %s" % args.outfile)


def main():
    parser = argparse.ArgumentParser(
        description="Create FITS cube from a series of image slices.")
    subparsers = parser.add_subparsers(dest="subparser_name",
                                       title="sub-commands",
                                       help="additional help")
    # sub-command: "info"
    parser_info = subparsers.add_parser("info", help="show FITS cube info")
    parser_info.add_argument("infile", help="FITS cube filename")
    parser_info.set_defaults(func=cmd_info)
    # sub-command: "create"
    parser_create = subparsers.add_parser("create", help="create a FITS cube")
    parser_create.add_argument("-C", "--clobber", dest="clobber",
                               action="store_true",
                               help="overwrite existing output file")
    parser_create.add_argument("-z", "--z-begin", dest="zbegin",
                               type=float, default=0.0,
                               help="Z-axis position of the first slice")
    parser_create.add_argument("-s", "--z-step", dest="zstep",
                               type=float, default=1.0,
                               help="Z-axis step/spacing between slices")
    parser_create.add_argument("-o", "--outfile", dest="outfile",
                               required=True,
                               help="output FITS cube filename")
    parser_create.add_argument("-i", "--infiles", dest="infiles",
                               nargs="+", required=True,
                               help="input image slices (in order)")
    parser_create.set_defaults(func=cmd_create)
    #
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
