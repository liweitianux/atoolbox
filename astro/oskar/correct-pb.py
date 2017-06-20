#!/usr/bin/env python3
#
# Copyright (c) Weitna LI <weitian@aaronly.me>
# MIT License
#
# Correct for primary beam by dividing the image by the (simulated)
# primary beam pattern.
#
# 2017-06-20
#

import os
import sys
import argparse

from astropy.io import fits


def main():
    parser = argparse.ArgumentParser(
        description="Correct for the primary beam pattern")
    parser.add_argument("-C", "--clobber", dest="clobber",
                        action="store_true",
                        help="overwrite existing files")
    parser.add_argument("-p", "--primary-beam", dest="pb",
                        required=True,
                        help="file of the primary beam pattern")
    parser.add_argument("infile", help="input image to be corrected for")
    parser.add_argument("outfile", nargs="?",
                        help="output pb-corrected image (default: add " +
                        "'pbcorr.fits' suffix)")
    args = parser.parse_args()

    if args.outfile:
        outfile = args.outfile
    else:
        outfile = os.path.splitext(args.infile)[0] + ".pbcorr.fits"

    with fits.open(args.infile) as f:
        imgin = f[0].data
        header = f[0].header
    pb = fits.open(args.pb)[0].data
    imgout = imgin / pb
    header.add_history(" ".join(sys.argv))
    hdu = fits.PrimaryHDU(data=imgout, header=header)
    try:
        hdu.writeto(outfile, overwrite=args.clobber)
    except TypeError:
        hdu.writeto(outfile, clobber=args.clobber)
    print("Wrote pb-corrected image: %s" % outfile)


if __name__ == "__main__":
    main()
