#!/usr/bin/env python3
#
# Copyright (c) 2016-2018 Weitian LI <weitian@aaronly.me>
# MIT License
#
# TODO:
#   * allow to query by coordinates & radius range
#   * filter queried results according to the type/other...
#   * if not queried by name, then try query by coordinates
#

"""
Query NED with the provided name or coordinate.
NASA/IPAC Extragalactic Database: http://ned.ipac.caltech.edu/

References
----------
* astroquery: NedClass
  https://astroquery.readthedocs.org/en/latest/api/astroquery.ned.NedClass.html
"""

import sys
import argparse
import csv
from collections import OrderedDict

from astroquery.ned import Ned
from astroquery.exceptions import RemoteServiceError


# Ned configurations
Ned.TIMEOUT = 20


def query_name(name, verbose=False, print_header=False):
    """
    Query NED by source name.
    """
    try:
        q = Ned.query_object(name)
        objname  = q["Object Name"][0]
        objtype  = q["Type"][0].decode("utf-8")
        ra       = q["RA(deg)"][0]
        dec      = q["DEC(deg)"][0]
        velocity = q["Velocity"][0]
        z        = q["Redshift"][0]
        z_flag   = q["Redshift Flag"][0].decode("utf-8")
        refs     = q["References"][0]
        notes    = q["Notes"][0]
    except RemoteServiceError:
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
    if verbose:
        if print_header:
            print(",".join(results.keys()))
        print(",".join([str(v) for v in results.values()]))
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
    parser.add_argument("-o", "--output", dest="output",
                        default=sys.stdout,
                        help="output CSV file with queried data")
    args = parser.parse_args()

    try:
        names = list(map(str.strip, open(args.input).readlines()))
    except FileNotFoundError:
        names = list(map(str.strip, args.input.split(",")))

    results_list = []

    print_header = True
    for name in names:
        qr = query_name(name, verbose=args.verbose,
                        print_header=print_header)
        print_header = False
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
