#!/usr/bin/env python3
#
# Copyright (c) 2017 Aaron LI <aly@aaronly.me>
# MIT license
#

"""
Crop out the central region of specified size from the FITS image.
"""

import argparse

from rescale_image import FITSImage


def main():
    parser = argparse.ArgumentParser(
        description="Rescale a FITS image to the desired size/resolution")
    parser.add_argument("-C", "--clobber", action="store_true",
                        help="overwrite existing output file")
    parser.add_argument("-s", "--size", type=float, required=True,
                        help="central crop box size [deg]")
    parser.add_argument("-i", "--infile", required=True,
                        help="input FITS image")
    parser.add_argument("-o", "--outfile", required=True,
                        help="output cropped FITS image")
    args = parser.parse_args()

    fitsimage = FITSImage(args.infile)
    fitsimage.crop(size=args.size)
    fitsimage.write(args.outfile, clobber=args.clobber)


if __name__ == "__main__":
    main()
