#!/usr/bin/env python3
#
# Copyright (c) Weitian LI <weitian@aaronly.me>
# MIT license
#

"""
Add multiple FITS images of same shape.
"""

import os
import sys
import argparse

from astropy.io import fits


def main():
    parser = argparse.ArgumentParser(
        description="Add two or more FITS images of same shape")
    parser.add_argument("-C", "--clobber", action="store_true",
                        help="overwrite existing files")
    parser.add_argument("-i", "--infile", nargs="+",
                        help=">=2 FITS images to be added")
    parser.add_argument("-o", "--outfile", required=True,
                        help="filename of added FITS image")
    args = parser.parse_args()

    nimg = len(args.infile)
    if nimg < 2:
        raise RuntimeError("more than 2 input FITS images required")

    if os.path.exists(args.outfile):
        if args.clobber:
            os.remove(args.outfile)
            print("WARNING: removed existing output file: %s" % args.outfile)
        else:
            raise OSError("output file already exists: %s" % args.outfile)

    with fits.open(args.infile[0]) as f:
        image = f[0].data
        header = f[0].header
        print("Opened 1st image: %s" % args.infile[0])
        print("Image shape: %s" % str(list(reversed(image.shape))))

    for fn in args.infile[1:]:
        print("Adding FITS image: %s ..." % fn)
        image += fits.open(fn)[0].data

    header.add_history(" ".join(sys.argv))
    hdu = fits.PrimaryHDU(data=image, header=header)
    hdu.writeto(args.outfile)
    print("Wrote added FITS image to: %s" % args.outfile)


if __name__ == "__main__":
    main()
