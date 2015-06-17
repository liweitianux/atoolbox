#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Aaron LI
# 2015/06/16
#
# Changelogs:
# 0.2.0, 2015/06/16, Aaron LI
#   * append the new time column to the *last*, rather than inserting
#     to the beginning
#   * explicitly update header from the new generated table
#
# BUGS:
#   * comments of columns will lost after modified by astropy.io.fits,
#     which is a bug with this package
#

"""
Add a time column for the chandra blanksky event file.
The time data are generated with a uniform distribution
between TSTART and TSTOP.
"""

__version__ = "0.2.0"
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
    # NOTE: append the new time column to the *last*!
    # Otherwise the TLMIN??/TLMAX?? keyword pairs, which record the
    # minimum/maximum values of corresponding columns, will become
    # *out of order*. Therefore the output FITS file causes weird problems
    # with DS9 and DM tools.
    newtable = fits.BinTableHDU.from_columns(
            table.columns + fits.ColDefs([time_col]))
    fitsfile[blockname].data = newtable.data
    # update header
    fitsfile[blockname].header.update(newtable.header)
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

