#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Aaron LI
# 2015/06/16

"""
Add a time column for the chandra blanksky event file.
The time data are generated with a uniform distribution
between TSTART and TSTOP.
"""

__version__ = "0.1.0"
__date__ = "2015/06/16"

import sys
import argparse

import numpy as np
try:
    from astropy.io import fits
except ImportError:
    try:
        import pyfits as fits
    except ImportError:
        raise ImportError("cannot import 'astropy.io.fits' or 'pyfits'")


def add_time_column(fitsfile, blockname="EVENTS"):
    """
    Add a time column to the specified block of the input fits file.
    The time data are generated with a uniform distribution
    between TSTART and TSTOP.

    Return:
        A fits object with the new time column.
    """
    if isinstance(fitsfile, str):
        fitsfile = fits.open(fitsfile)
    table = fitsfile[blockname]
    tstart = table.header["TSTART"]
    tstop = table.header["TSTOP"]
    counts = len(table.data)
    time_data = np.random.uniform(tstart, tstop, counts)
    time_col = fits.Column(name="time", format="1D", unit="s", array=time_data)
    newtable = fits.BinTableHDU.from_columns(
            fits.ColDefs([time_col]) + table.columns)
    fitsfile[blockname].data = newtable.data
    return fitsfile


def main():
    parser = argparse.ArgumentParser(
            description="Add a time column for Chandra blanksky event file.")
    parser.add_argument("-V", "--version", action="version",
            version="%(prog)s " + "%s (%s)" % (__version__, __date__))
    parser.add_argument("infile", help="input chandra blanksky file")
    parser.add_argument("outfile", nargs="?", default=None,
            help="modified blanksky file. IN-PLACE modification if omitted.")
    parser.add_argument("-C", "--clobber", dest="clobber",
            action="store_true", help="overwrite output file if exists")
    args = parser.parse_args()

    newfits = add_time_column(args.infile)
    if args.outfile:
        newfits.writeto(args.outfile, clobber=args.clobber)
    else:
        newfits.writeto(args.infile, clobber=True)


if __name__ == "__main__":
    main()

