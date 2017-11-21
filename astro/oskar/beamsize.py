#!/usr/bin/env python3
#
# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license
#

"""
Read the beam parameters from one or more FITS images (e.g., imaged by
WSClean), and briefly summarize the results.
"""

import argparse

import numpy as np

from astropy.io import fits


def main():
    parser = argparse.ArgumentParser(
        description="Read and summarize the beam parameters")
    parser.add_argument("infile", nargs="+", help="input FITS images")
    args = parser.parse_args()

    nfile = len(args.infile)
    beam = np.zeros((nfile, 4))
    for i, fn in enumerate(args.infile):
        header = fits.open(fn)[0].header
        bmajor = header["BMAJ"] * 3600  # [arcsec]
        bminor = header["BMIN"] * 3600  # [arcsec]
        bpa = header["BPA"]  # [deg]
        bsize = np.pi * bmajor*bminor / (4*np.log(2))  # [arcsec^2]
        beam[i, :] = (bsize, bmajor, bminor, bpa)

    if nfile == 1:
        print("size: %.1f [arcsec^2]" % beam[0, 0])
        print("major: %.1f [arcsec]" % beam[0, 1])
        print("minor: %.1f [arcsec]" % beam[0, 2])
        print("position angle: %.1f [deg]" % beam[0, 3])
        return 0

    # Two or more input images ...
    fnlen = max([len(fn) for fn in args.infile])
    fmt_header = "%-{fnlen}s: %8s  %6s  %6s  %6s".format(fnlen=fnlen)
    fmt_beam = "%-{fnlen}s: %8.0f  %6.1f  %6.1f  %6.1f".format(fnlen=fnlen)
    print(fmt_header % ("filename", "size", "major", "minor", "pangle"))
    print(fmt_header % (" ", "arcsec^2", "arcsec", "arcsec", "deg"))
    print("------------------------------------------------------------"
          "--------------------")
    for i, fn in enumerate(args.infile):
        print(fmt_beam % ((fn,) + tuple(beam[i, :])))
    print("------------------------------------------------------------"
          "--------------------")
    print(fmt_beam % (("mean",) + tuple(beam.mean(axis=0))))
    print(fmt_beam % (("std",) + tuple(beam.std(axis=0))))


if __name__ == "__main__":
    main()
