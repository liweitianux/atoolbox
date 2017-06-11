#!/usr/bin/env python3
#
# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT License
#

"""
Extract a slice from the 21cm cube and save as FITS image.
"""

import os
import sys
import argparse

import numpy as np
import astropy.io.fits as fits


def main():
    outfile_default = "{prefix}_z{z:05.2f}_N{Nside}_L{Lside}_s{sidx}.fits"

    parser = argparse.ArgumentParser(
        description="Extract a slice from cube and save as FITS image")
    parser.add_argument("-C", "--clobber", dest="clobber",
                        action="store_true",
                        help="overwrite existing files")
    parser.add_argument("-d", "--dtype", dest="dtype", default="float32",
                        help="NumPy dtype of data cubes (default: float32)")
    parser.add_argument("-z", "--redshift", dest="redshift",
                        type=float, required=True,
                        help="redshift of the input data cube")
    parser.add_argument("-L", "--len-side", dest="Lside",
                        type=float, required=True,
                        help="Side length of the cube [comoving Mpc]")
    parser.add_argument("-s", "--slice-idx", dest="sidx",
                        type=int, default=None,
                        help="slice index to be extracted (default: "
                        "the central slice)")
    parser.add_argument("-u", "--unit", dest="unit",
                        help="data unit (e.g., K, mK)")
    parser.add_argument("-i", "--infile", dest="infile", required=True,
                        help="input data cube")
    parser.add_argument("-o", "--outfile", dest="outfile",
                        default=outfile_default,
                        help="output FITS image slice (default: %s)" %
                        outfile_default)
    parser.add_argument("-p", "--prefix", dest="prefix", required=True,
                        help="prefix for the output FITS image")
    args = parser.parse_args()

    cube = np.fromfile(open(args.infile, "rb"), dtype=args.dtype)
    Nside = round(cube.shape[0] ** (1.0/3))
    print("Read cube: %s (Nside=%d)" % (args.infile, Nside))
    if args.sidx is None:
        sidx = int(Nside / 2.0)
    elif args.idx >= 0 and args.idx < Nside:
        sidx = args.idx
    else:
        raise ValueError("invalid slice index: %s" % args.sidx)
    outfile = args.outfile.format(prefix=args.prefix, z=args.redshift,
                                  Nside=Nside, Lside=args.Lside, sidx=sidx)
    if os.path.exists(outfile) and not args.clobber:
        raise OSError("output file already exists: %s" % outfile)

    cube = cube.reshape((Nside, Nside, Nside))
    simg = cube[:, :, sidx]
    header = fits.Header()
    header["REDSHIFT"] = args.redshift
    header["Lside"] = (args.Lside, "Cube side length [comoving Mpc]")
    header["Nside"] = (Nside, "Number of pixels on each cube side")
    header["SliceIdx"] = (sidx, "Index of this extracted slice")
    if args.unit:
        header["BUNIT"] = (args.unit, "Data unit")
    header.add_history(" ".join(sys.argv))
    hdu = fits.PrimaryHDU(data=simg, header=header)
    try:
        hdu.writeto(outfile, overwrite=args.clobber)
    except TypeError:
        hdu.writeto(outfile, clobber=args.clobber)
    print("Extracted #%d slice: %s" % (sidx, outfile))


if __name__ == "__main__":
    main()
