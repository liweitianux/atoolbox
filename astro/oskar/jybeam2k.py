#!/usr/bin/env python3
#
# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license
#

"""
Convert the image (e.g., by WSClean) from units [Jy/beam] to [K]
by taking into account the telescope's beam size.
"""

import os
import sys
import argparse

import numpy as np

from astropy.io import fits
import astropy.units as au


def open_image(infile):
    """
    Open the FITS image and return its header and data, but requiring
    the input image has only ONE frequency.

    The input FITS image may have following dimensions:
    * NAXIS=2: [Y, X]
    * NAXIS=3: [FREQ=1, Y, X]
    * NAXIS=4: [STOKES, FREQ=1, Y, X]
    """
    with fits.open(infile) as f:
        header = f[0].header
        data = f[0].data
    if ((data.ndim == 3 and data.shape[0] != 1) or
            (data.ndim == 4 and data.shape[1] != 1)):
        # NAXIS=3: [FREQ!=1, Y, X]
        # NAXIS=4: [STOKES, FREQ!=1, Y, X]
        raise ValueError("input file '{0}' has invalid dimensions: {1}".format(
            infile, data.shape))
    print("Read in FITS image from: %s" % infile)
    return (header, data)


def calc_beam_size(header):
    """
    Calculate the beam/PSF size using the 'WSCNORMF' keyword recorded by
    the WSClean imager, which applies to the natural-weighted image, and
    is more accurate than the fitted beam (stored as 'BMAJ' and 'BMIN'.)
    """
    try:
        weight = header["WSCWEIGH"]
        normf = header["WSCNORMF"]
        print("WSCWEIGH: %s" % weight)
        print("WSCNORMF: %.1f" % normf)
    except KeyError:
        raise RuntimeError("NO necessary WSC* keyword; switch to " +
                           "--use-fitted-beam instead")
    if weight.upper() != "NATURAL":
        print("WARNING: weighting scheme is '%s' != natural!" % weight)
    width = header["NAXIS1"]
    height = header["NAXIS2"]
    pixelsize = np.abs(header["CDELT1"]) * 3600  # [arcsec]
    print("Image: %dx%d, %.1f [arcsec/pixel]" % (width, height, pixelsize))
    beam_size = (width*height * pixelsize**2) / normf
    return beam_size


def main():
    parser = argparse.ArgumentParser(
        description="Convert WSClean created image from [Jy/beam] to [K]",
        epilog=("By default the 'WSCNORMF' keyword is taken to derive " +
                "the beam/PSF size for natural-weighted image created " +
                "by WSClean, which is more accurate than the fitted " +
                "beam stored as 'BMAJ' and 'BMIN'."))
    parser.add_argument("-C", "--clobber", dest="clobber",
                        action="store_true",
                        help="overwrite existing output file")
    parser.add_argument("-B", "--use-fitted-beam", dest="use_fitted_beam",
                        action="store_true",
                        help="use the fitted beam (i.e., BMAJ and BMIN) " +
                        "or the beam size specified by --beam-size.")
    parser.add_argument("-b", "--beam-size", dest="beam_size", type=float,
                        help="instrumental beam size [arcsec^2] " +
                        "(=pi*bmajor*bminor/4/ln(2)) (default: obtain " +
                        "from the header BMAJ and BMIN keywords); this " +
                        "argument also implies --use-fitted-beam")
    parser.add_argument("-f", "--frequency", dest="frequency",
                        help="frequency [MHz] of the input image (NOTE: " +
                        "required if failed to obtain from the header)")
    parser.add_argument("infile",
                        help="input FITS image file (NOTE: only single " +
                        "frequency supported)")
    parser.add_argument("outfile",
                        help="output filename of the converted image")
    args = parser.parse_args()

    header, data = open_image(args.infile)
    bunit = header["BUNIT"]
    if bunit.upper() == "JY/BEAM":
        unit = "Jy"
    elif bunit.upper() == "MJY/BEAM":
        unit = "mJy"
    else:
        raise ValueError("input image has wrong unit: %s" % bunit)
    print("Data unit: %s/beam" % unit)

    if args.frequency:
        freq = args.frequency  # [MHz]
    else:
        try:
            freq = header["FREQ"]  # [MHz]
        except KeyError:
            if header.get("CTYPE3", "").upper() == "FREQ":
                freq = header["CRVAL3"] / 1e6  # [MHz]
            else:
                raise ValueError("--frequency required")
    print("Frequency: %.2f [MHz]" % freq)

    # Elliptical Gaussian beam (full width at half power; FWHP)
    if args.beam_size:
        beam_size = args.beam_size
    elif args.use_fitted_beam:
        bmajor = header["BMAJ"] * 3600  # [arcsec]
        bminor = header["BMIN"] * 3600  # [arcsec]
        beam_size = np.pi * bmajor*bminor / (4*np.log(2))  # [arcsec^2]
        print("Beam: (%.2f, %.2f) [arcsec]" % (bmajor, bminor))
    else:
        # Derive beam size using 'WSCNORMF' keyword
        beam_size = calc_beam_size(header)
    print("Beam size: %.2f [arcsec^2]" % beam_size)

    equiv = au.brightness_temperature(beam_size*au.arcsec**2, freq*au.MHz)
    jybeam2k = au.Unit(unit).to(au.K, equivalencies=equiv)
    print("Conversion factor [%s/beam] -> [K]: %g" % (unit, jybeam2k))

    header["BUNIT"] = ("K", "Kelvin; converted from [%s/beam]" % unit)
    header["JyBeam2K"] = (jybeam2k,
                          "[%s/beam] -> [K] conversion factor" % unit)
    header["FREQ"] = (freq, "[MHz] frequency")
    header.add_history(" ".join(sys.argv))
    data = data * jybeam2k

    if os.path.exists(args.outfile):
        if args.clobber:
            os.remove(args.outfile)
        else:
            raise OSError("output file already existed: %s" % args.outfile)
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(args.outfile)
    print("Converted image wrote to: %s" % args.outfile)


if __name__ == "__main__":
    main()
