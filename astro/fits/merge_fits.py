#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Aaron LI
# 2015/06/16
#
# Changelogs:
# 0.3.0, 2015/06/17, Aaron LI
#   * added argument '-c/--columns' to specify columns to be merged
#   * added argument 'columns' to function 'merge2fits()'
# 0.2.0, 2015/06/17, Aaron LI
#   * added function 'del_key_startswith()' to delete header keywords,
#     and the deletion must be repeated until the header length does not
#     decrease any more
#   * ignore the header of the second FITS file to avoid keyword conflictions
#
# BUGS:
#   * astropy.io.fits may have bug with header keywords deletion
#
# TODO:
#   * to support image FITS merge
#   * to allow specify column list to be merged
#

"""
Merge several (>=2) of FITS file.

By default the *first* extend tables are merged and write out to a new
FITS file containing the *common* columns.  If the data types of the
columns of each FITS table do not match, then the data type of the column
of the *first* FITS table is used, and other columns are coerced.

If the FITS files have only *1* HDU (i.e., the Primary HDU), then data of
these HDU's are summed up to make up the output FITS file (an image),
on conditional that the shapes of all these HDU's are the same.
"""

__version__ = "0.3.0"
__date__ = "2015/06/17"

# default blockname to be merged
BLOCKNAME_DFT = "EVENTS"

DEBUG = True

import sys
import argparse
import re

import numpy as np
try:
    from astropy.io import fits
except ImportError:
    try:
        import pyfits as fits
    except ImportError:
        raise ImportError("cannot import 'astropy.io.fits' or 'pyfits'")


def merge2fits(file1, file2, block1=1, block2=1, columns=None):
    """
    Merge *two* FITS files of the given blocks (extension table),
    and return the merged FITS object.

    TODO:
        * log history to header

    Arguments:
        file1, file2: input two FITS files
        block1, block2: table number or table name to be merged
        columns: the columns to be merged; by default to merge the
            common columns

    Return:
        the merged FITS object
    """
    # open file if provide filename
    if isinstance(file1, str):
        file1 = fits.open(file1)
    if isinstance(file2, str):
        file2 = fits.open(file2)
    # if has only *1* HDU => image
    if len(file1) == 1:
        block1 = 0
    if len(file2) == 1:
        block2 = 0
    if block1 == 0 or block2 == 0:
        # TODO
        raise NotImplementedError("image FITS merge currently not supported!")
    # get table to be merged
    table1 = file1[block1]
    table2 = file2[block2]
    # create column names to be merged
    # get names of all columns (convert to upper case)
    colnames1 = [col.name.upper() for col in table1.columns]
    colnames2 = [col.name.upper() for col in table2.columns]
    colnames_common = list(set(colnames1).intersection(set(colnames2)))
    # sort the common column names acoording original column orders
    colnames_common.sort(key = lambda x: colnames1.index(x))
    if columns is not None:
        if isinstance(columns, list):
            columnlist = list(map(str.upper, columns))
        else:
            columnlist = list(columns.upper())
        # check the specified columns whether in the above colnames_common
        for name in columnlist:
            if name not in colnames_common:
                raise ValueError("column '%s' not found in both files" % name)
        # use the specified columns
        colnames_common = columnlist
    # "STATUS" columns don't have equal-length format, so remove it
    if "STATUS" in colnames_common:
        colnames_common.remove("STATUS")
    if DEBUG:
        print("DEBUG: columns to merge: ", colnames_common, file=sys.stderr)
    # filter out the common columns
    nrow1 = table1.data.shape[0]
    nrow2 = table2.data.shape[0]
    hdu_merged = fits.BinTableHDU.from_columns(
            fits.ColDefs([table1.columns[name] for name in colnames_common]),
            nrows=nrow1+nrow2)
    for name in colnames_common:
        if DEBUG:
            print("DEBUG: merging column: ", name, file=sys.stderr)
        dtype = hdu_merged.columns[name].array.dtype
        hdu_merged.columns[name].array[nrow1:] = \
                table2.columns[name].array.astype(dtype)
    # process headers, based on the header of the first FITS file
    # DO NOT strip the base header, in order to keep the position of
    # XTENSION/BITPIX/NAXIS/NAXIS1/NAXIS2/PCOUNT/GCOUNT/TFIELDS keywords.
    header = table1.header.copy()  # do not strip
    # IGNORE the header of the second FITS file to avoid keyword conflictions.
    #header2 = table2.header.copy(strip=True)
    ## merge two headers; COMMENT and HISTORY needs special handle
    #for comment in header2["COMMENT"]:
    #    header.add_comment(comment)
    #for history in header2["HISTORY"]:
    #    header.add_history(history)
    #if "COMMENT" in header2:
    #    del header2["COMMENT"]
    #if "HISTORY" in header2:
    #    del header2["HISTORY"]
    #if "" in header2:
    #    del header2[""]
    #header.update(header2)
    # remove the original TLMIN??/TLMAX??/TTYPE??/TFORM??/TUNIT?? keywords
    del_key_startswith(header,
            startswith=["TLMIN", "TLMAX", "TTYPE", "TFORM", "TUNIT"],
            lastlength=len(header))
    # update with new TLMIN??/TLMAX??/TTYPE??/TFORM??/TUNIT?? keywords
    header.update(hdu_merged.header)
    hdu_merged.header = header
    # copy PrimaryHDU from first FITS
    primary_hdu = file1[0].copy()
    # make HDUList and return
    return fits.HDUList([primary_hdu, hdu_merged])


def del_key_startswith(header, startswith, lastlength=0):
    """
    Delete the keys which start with the specified strings.

    Arguments:
        header: FITS table header
        startswith: a list of strings; If a key starts with any
            of these strings, then the key-value pair is removed.

    XXX: the deletion must be repeated several times until the
         length of the header does not decrease any more.
         (This may be a bug related to the astropy.io.fits???)
    """
    if not isinstance(startswith, list):
        startswith = list(startswith)
    re_key = re.compile(r"^(%s)" % "|".join(startswith), re.I)
    for k in header.keys():
        if re_key.match(k):
            del header[k]
    curlength = len(header)
    if lastlength == curlength:
        return
    else:
        # recursively continue deletion
        if DEBUG:
            print("DEBUG: recursively continue header keywords deleteion",
                    file=sys.stderr)
        del_key_startswith(header, startswith, curlength)


def get_filename_blockname(pstr):
    """
    Separate privided 'pstr' (parameter string) into filename and
    blockname.  If does not have a blockname, then the default
    blockname returned.
    """
    try:
        filename, blockname = re.sub(r"[\[\]]", " ", pstr).split()
    except ValueError:
        filename = pstr
        blockname = BLOCKNAME_DFT
    return (filename, blockname)


def main():
    parser = argparse.ArgumentParser(
            description="Merge several FITS files with the common columns.")
    parser.add_argument("-V", "--version", action="version",
            version="%(prog)s " + "%s (%s)" % (__version__, __date__))
    parser.add_argument("infile1", help="input FITS file 1; " + \
            "The blockname can be appended, e.g., infile1.fits[EVENTS]")
    parser.add_argument("infile2", nargs="+",
            help="input FITS file 2 and more")
    parser.add_argument("outfile", help="merged output file")
    parser.add_argument("-c", "--columns", dest="columns",
            help="list of columns to be merged (comma separated)")
    parser.add_argument("-C", "--clobber", dest="clobber",
            action="store_true", help="overwrite output file if exists")
    args = parser.parse_args()
    if DEBUG:
        print("DEBUG: infile2: ", args.infile2, file=sys.stderr)

    if args.columns:
        columns = args.columns.upper().replace(",", " ").split()
    file1, block1 = get_filename_blockname(args.infile1)
    merged_fits = fits.open(file1)
    for fitsfile in args.infile2:
        # split filename and block name
        file2, block2 = get_filename_blockname(fitsfile)
        merged_fits = merge2fits(merged_fits, file2, block1, block2, columns)
    merged_fits.writeto(args.outfile, checksum=True, clobber=args.clobber)


if __name__ == "__main__":
    main()

