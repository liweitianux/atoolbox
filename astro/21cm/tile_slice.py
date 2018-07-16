#!/usr/bin/env python3
#
# Copyright (c) 2017-2018 Weitian LI <wt@liwt.net>
# MIT License
#

"""
Tile the given slices to the required FoV size, scale down to the
wanted size (for faster simulation later).
"""

import os
import sys
import argparse

import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import astropy.units as au
import scipy.ndimage

from z2freq import z2freq, freq2z


# Adopted cosmology
cosmo = FlatLambdaCDM(H0=71.0, Om0=0.27, Ob0=0.046)


def main():
    outfile_default = "{prefix}_f{freq:06.2f}_N{Nside}_fov{fov:.1f}.fits"

    parser = argparse.ArgumentParser(
        description="Tile slice to match FoV size and scale to required size")
    parser.add_argument("-C", "--clobber", dest="clobber",
                        action="store_true",
                        help="overwrite existing files")
    parser.add_argument("-F", "--fov", dest="fov", default=10.0, type=float,
                        help="required FoV [deg] of the output slice " +
                        "(default: 10.0 [deg])")
    parser.add_argument("-N", "--n-side", dest="Nside", default=1800, type=int,
                        help="required image size of output slice " +
                        "(default: 1800)")
    parser.add_argument("-i", "--infile", dest="infile", required=True,
                        help="input slice")
    parser.add_argument("-o", "--outfile", dest="outfile",
                        default=outfile_default,
                        help="output tiled slice (default: %s)" %
                        outfile_default)
    parser.add_argument("-p", "--prefix", dest="prefix", default="deltaTb",
                        help="prefix for the output tiled slice " +
                        "(default: 'deltaTb')")
    exgrp = parser.add_mutually_exclusive_group(required=True)
    exgrp.add_argument("-z", "--redshift-c", dest="zc", type=float,
                       help="central redshift of the selected cube")
    exgrp.add_argument("-f", "--freq-c", dest="fc", type=float,
                       help="central frequency [MHz] of the selected cube")
    args = parser.parse_args()

    if os.path.exists(args.outfile):
        if args.clobber:
            os.remove(args.outfile)
        else:
            raise FileExistsError('output file already exists: %s' %
                                  args.outfile)

    if args.zc:
        zc = args.zc
        fc = z2freq(zc, print_=False)
    else:
        fc = args.fc
        zc = freq2z(fc, print_=False)

    with fits.open(args.infile) as f:
        img_in = f[0].data
        header = f[0].header
    freq = header["FREQ"]  # [MHz]
    Lside = header["LSIDE"]  # [Mpc]
    print("frequency = %.2f [MHz], Lside = %.1f [cMpc]" % (freq, Lside))
    DM = cosmo.comoving_distance(zc).value  # [Mpc]
    print("Comoving distance (@z=%.3f/f=%.2fMHz) = %.2f [Mpc]" % (zc, fc, DM))
    Nside = img_in.shape[0]
    fov_in = (Lside / DM) * au.rad.to(au.deg)  # [deg]
    Nup = int(np.ceil(args.fov / fov_in))
    print("Input FoV: %s [deg], Tiling repeats: %d" % (fov_in, Nup))
    img_tiled = np.tile(img_in, reps=(Nup, Nup))
    Nside2 = round(Nside * args.fov / fov_in)
    img2 = img_tiled[:Nside2, :Nside2]
    # Rescale to the output size
    zoom = (args.Nside + 0.1) / Nside2  # +0.1 to workaround the scipy warning
    img_out = scipy.ndimage.zoom(img2, zoom=zoom, order=1)
    # Record information to header
    header["Z_C"] = (zc, "Central redshift")
    header["FREQ_C"] = (fc, "[MHz] Frequency w.r.t. to central redshift")
    header["FoV"] = (args.fov, "[deg] FoV of this slice")
    header["PixSize"] = (3600.0*args.fov/args.Nside, "[arcsec] Pixel size")
    header.add_history(" ".join(sys.argv))
    hdu = fits.PrimaryHDU(data=img_out, header=header)
    outfile = args.outfile.format(prefix=args.prefix, freq=freq,
                                  Nside=args.Nside, fov=args.fov)
    try:
        hdu.writeto(outfile, overwrite=args.clobber)
    except TypeError:
        hdu.writeto(outfile, clobber=args.clobber)
    print("Tiled and scaled slice: %s" % outfile)


if __name__ == "__main__":
    main()
