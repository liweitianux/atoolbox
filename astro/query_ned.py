#!/usr/bin/env python3
#
# Copyright (c) 2016-2017 Weitian LI <liweitianux@live.com>
# MIT license
#
# References:
# [1] astroquery: NedClass
#     https://astroquery.readthedocs.org/en/latest/api/astroquery.ned.NedClass.html
#
# Change log:
# 2017-02-11:
#   * Add argument "--brief" to not print header
# 2016-05-25:
#   * Also output RA, DEC results
#   * Update argument process
#   * Simplify queried results process
#   * Improve comments a bit
#   * Some PEP8 fixes
#
# TODO:
#   * allow to query by coordinates & radius range
#   * filter queried results according to the type/other...
#   * if not queried by name, then try query by coordinates
#

"""
Query NED with the provided name or coordinate.
NASA/IPAC Extragalactic Database: http://ned.ipac.caltech.edu/
"""

import sys
import argparse
import csv
from collections import OrderedDict

from astroquery.ned import Ned
from astroquery.exceptions import RemoteServiceError
# from astropy import coordinates
# import astropy.units as u


# Ned configurations
Ned.TIMEOUT = 20


def query_name(name, verbose=False):
    """
    Query NED by source name.
    """
    try:
        q = Ned.query_object(name)
        objname  = q["Object Name"][0].decode("utf-8")
        objtype  = q["Type"][0].decode("utf-8")
        ra       = q["RA(deg)"][0]
        dec      = q["DEC(deg)"][0]
        velocity = q["Velocity"][0]
        z        = q["Redshift"][0]
        z_flag   = q["Redshift Flag"][0].decode("utf-8")
        refs     = q["References"][0]
        notes    = q["Notes"][0]
        if verbose:
            print("%s: %s,%s,%s,%s,%s,%s,%s,%s,%s" %
                  (name, objname, objtype, ra, dec, velocity, z, z_flag,
                   refs, notes),
                  file=sys.stderr)
    except RemoteServiceError as e:
        objname  = None
        objtype  = None
        ra       = None
        dec      = None
        velocity = None
        z        = None
        z_flag   = None
        refs     = None
        notes    = None
        if verbose:
            print("*** %s: not found ***" % name, file=sys.stderr)
    #
    results = OrderedDict([
        ("Name",       name),
        ("NED_Name",   objname),
        ("Type",       objtype),
        ("RA",         ra),
        ("DEC",        dec),
        ("Velocity",   velocity),
        ("z",          z),
        ("z_Flag",     z_flag),
        ("References", refs),
        ("Notes",      notes),
    ])
    return results


def main():
    parser = argparse.ArgumentParser(
            description="Query NED database by source name")
    parser.add_argument("-v", "--verbose", dest="verbose",
                        action="store_true",
                        help="show verbose information")
    parser.add_argument("-b", "--brief", dest="brief",
                        action="store_true",
                        help="be brief and do not print header")
    parser.add_argument("-i", "--input", dest="input", required=True,
                        help="source names to be queried (sep by comma); " +
                             "or a file contains the names (one per line)")
    parser.add_argument("-o", "--output", dest="output", default=sys.stdout,
                        help="output CSV file with queried data")
    args = parser.parse_args()

    try:
        names = list(map(str.strip, open(args.input).readlines()))
    except FileNotFoundError:
        names = list(map(str.strip, args.input.split(",")))

    results_list = []

    for name in names:
        qr = query_name(name, verbose=args.verbose)
        results_list.append(qr)

    try:
        of = open(args.output, "w")
    except TypeError:
        of = args.output
    writer = csv.writer(of)
    if not args.brief:
        writer.writerow(results_list[0].keys())
    for res in results_list:
        writer.writerow(res.values())
    if of is not sys.stdout:
        of.close()


if __name__ == "__main__":
    main()
