#!/usr/bin/env python3
#
# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT License
#

"""
Get the slice of the specified redshift/frequency (21cm signal) from
the redshift cube (created by `extract_slice.py` and `fitscube.py`)
using linear interpolation in redshift.

NOTE:
The cube slices are ordered in increasing redshifts.
"""

import os
import sys
import argparse

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from z2freq import z2freq, freq2z


class FITSCube:
    def __init__(self, infile):
        with fits.open(infile) as f:
            self.data = f[0].data
            self.header = f[0].header
        self.wcs = WCS(self.header)

    @property
    def nslice(self):
        ns, __, __ = self.data.shape
        return ns

    @property
    def redshifts(self):
        # Z-axis
        nslice = self.nslice
        pix = np.zeros(shape=(nslice, 3), dtype=np.int)
        pix[:, 2] = np.arange(nslice)
        world = self.wcs.wcs_pix2world(pix, 0)
        return world[:, 2]

    def get_slice(self, z):
        redshifts = self.redshifts
        if z < redshifts.min() or z > redshifts.max():
            raise ValueError("requested redshift out of range: %.2f" % z)
        idx2 = (redshifts <= z).sum()
        idx1 = idx2 - 1
        z1, slice1 = redshifts[idx1], self.data[idx1, :, :]
        z2, slice2 = redshifts[idx2], self.data[idx2, :, :]
        if os.environ.get("DEBUG"):
            print("DEBUG: redshifts: {0}".format(redshifts), file=sys.stderr)
            print("DEBUG: z={0}".format(z), file=sys.stderr)
            print("DEBUG: z1={0}, idx1={1}".format(z1, idx1), file=sys.stderr)
            print("DEBUG: z2={0}, idx2={1}".format(z2, idx2), file=sys.stderr)
        return slice1 + (slice2-slice1) * (z-z1) / (z2-z1)


def main():
    outfile_default = "{prefix}_z{z:05.2f}_f{freq:06.2f}.fits"

    parser = argparse.ArgumentParser(
        description="Get slices at requested redshifts/frequencies")
    parser.add_argument("-C", "--clobber", dest="clobber",
                        action="store_true",
                        help="overwrite existing files")
    parser.add_argument("-i", "--infile", dest="infile", required=True,
                        help="input redshift data cube")
    parser.add_argument("-o", "--outfile", dest="outfile",
                        default=outfile_default,
                        help="output FITS image slice (default: %s)" %
                        outfile_default)
    parser.add_argument("-p", "--prefix", dest="prefix", required=True,
                        help="prefix for the output FITS image")
    exgrp = parser.add_mutually_exclusive_group(required=True)
    exgrp.add_argument("-z", "--redshifts", dest="redshifts", nargs="+",
                       help="redshifts where to interpolate slices")
    exgrp.add_argument("-f", "--freqs", dest="freqs", nargs="+",
                       help="21cm frequencies [MHz] to interpolate slices")
    args = parser.parse_args()

    if args.redshifts:
        redshifts = [float(z) for z in args.redshifts]
        freqs = z2freq(redshifts, print_=False)
    else:
        freqs = [float(f) for f in args.freqs]
        redshifts = freq2z(freqs, print_=False)

    cube = FITSCube(args.infile)
    for z, f in zip(redshifts, freqs):
        outfile = args.outfile.format(prefix=args.prefix, z=z, freq=f)
        print("z=%05.2f, freq=%06.2f MHz : %s" % (z, f, outfile))
        zslice = cube.get_slice(z)
        header = fits.Header()
        try:
            header["BUNIT"] = (cube.header["BUNIT"],
                               cube.header.comments["BUNIT"])
        except KeyError:
            pass
        try:
            header["LSIDE"] = (cube.header["LSIDE"],
                               cube.header.comments["LSIDE"])
        except KeyError:
            pass
        header["REDSHIFT"] = (z, "Slice where interpolated")
        header["FREQ"] = (f, "21cm signal frequency [MHz]")
        header.add_history(" ".join(sys.argv))
        hdu = fits.PrimaryHDU(data=zslice, header=header)
        try:
            hdu.writeto(outfile, overwrite=args.clobber)
        except TypeError:
            hdu.writeto(outfile, clobber=args.clobber)


if __name__ == "__main__":
    main()
