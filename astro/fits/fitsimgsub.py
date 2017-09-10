#!/usr/bin/env python3
#
# Copyright (c) Weitian LI <weitian@aaronly.me>
# MIT license
#

"""
Subtract a FITS image by one or more FITS images of same shape.
"""

import os
import sys
import argparse

from astropy.io import fits


def main():
    parser = argparse.ArgumentParser(
        description="Subtract a FITS image by >=1 images of same shape")
    parser.add_argument("-C", "--clobber", action="store_true",
                        help="overwrite existing files")
    parser.add_argument("-1", "--infile1", required=True,
                        help="the FITS image from which to be subtracted")
    parser.add_argument("-2", "--infile2", nargs="+",
                        help="one or more FITS images to be subtracted by")
    parser.add_argument("-o", "--outfile", required=True,
                        help="filename of subtracted FITS image")
    args = parser.parse_args()

    if os.path.exists(args.outfile):
        if args.clobber:
            os.remove(args.outfile)
            print("WARNING: removed existing output file: %s" % args.outfile)
        else:
            raise OSError("output file already exists: %s" % args.outfile)

    with fits.open(args.infile1) as f:
        image = f[0].data
        header = f[0].header
        print("Opened FITS image: %s" % args.infile1)
        print("Image shape: %s" % str(list(reversed(image.shape))))

    for fn in args.infile2:
        print("Subtracting FITS image: %s ..." % fn)
        image -= fits.open(fn)[0].data

    header.add_history(" ".join(sys.argv))
    hdu = fits.PrimaryHDU(data=image, header=header)
    hdu.writeto(args.outfile)
    print("Wrote subtracted FITS image to: %s" % args.outfile)


if __name__ == "__main__":
    main()
