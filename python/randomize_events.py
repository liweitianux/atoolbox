#!/usr/bin/env python3
#
# Randomize the (X,Y) position of each X-ray photon events according
# to a Gaussian distribution of given sigma.
#
# References:
# [1] G. Scheellenberger, T.H. Reiprich, L. Lovisari, J. Nevalainen & L. David
#     2015, A&A, 575, A30
#
#
# Aaron LI
# Created: 2016-03-24
# Updated: 2016-03-24
#

from astropy.io import fits
import numpy as np

import os
import sys
import datetime
import argparse


CHANDRA_ARCSEC_PER_PIXEL = 0.492

def randomize_events(infile, outfile, sigma, clobber=False):
    """
    Randomize the position (X,Y) of each X-ray event according to a
    specified size/sigma Gaussian distribution.
    """
    sigma_pix = sigma / CHANDRA_ARCSEC_PER_PIXEL
    evt_fits = fits.open(infile)
    evt_table = evt_fits[1].data
    # (X,Y) physical coordinate
    evt_x = evt_table["x"]
    evt_y = evt_table["y"]
    rand_x = np.random.normal(scale=sigma_pix, size=evt_x.shape)\
            .astype(evt_x.dtype)
    rand_y = np.random.normal(scale=sigma_pix, size=evt_y.shape)\
            .astype(evt_y.dtype)
    evt_x += rand_x
    evt_y += rand_y
    # Add history to FITS header
    evt_hdr = evt_fits[1].header
    evt_hdr.add_history("TOOL: %s @ %s" % (
            os.path.basename(sys.argv[0]),
            datetime.datetime.utcnow().isoformat()))
    evt_hdr.add_history("COMMAND: %s" % " ".join(sys.argv))
    evt_fits.writeto(outfile, clobber=clobber, checksum=True)


def main():
    parser = argparse.ArgumentParser(
            description="Randomize the (X,Y) of each X-ray event")
    parser.add_argument("infile", help="input event file")
    parser.add_argument("outfile", help="output randomized event file")
    parser.add_argument("-s", "--sigma", dest="sigma",
            required=True, type=float,
            help="sigma/size of the Gaussian distribution used" + \
                 "to randomize the position of events (unit: arcsec)")
    parser.add_argument("-C", "--clobber", dest="clobber",
            action="store_true", help="overwrite output file if exists")
    args = parser.parse_args()

    randomize_events(args.infile, args.outfile,
            sigma=args.sigma, clobber=args.clobber)


if __name__ == "__main__":
    main()

