#!/usr/bin/env python3
#
# Copyright (c) 2016-2018 Weitian LI <weitian@aaronly.me>
# MIT License
#
# TODO:
# * allow to query by coordinates & radius range
# * filter queryed results according to the type/other...
# * if not queryed by name, then try query by coordinates
#

"""
Query SIMBAD with the provided name or coordinate.
http://simbad.u-strasbg.fr/simbad/

References
----------
* astroquery: SimbadClass
  https://astroquery.readthedocs.org/en/latest/api/astroquery.simbad.SimbadClass.html
* All available VOTable fields:
  http://simbad.u-strasbg.fr/simbad/sim-help?Page=sim-fscript#VotableFields
"""

import sys
import argparse
import csv
from collections import OrderedDict

from astroquery.simbad import Simbad
from astroquery.exceptions import RemoteServiceError


# Simbad configurations
Simbad.ROW_LIMIT = 30
Simbad.TIMEOUT = 20

# Add query items/fields:
# otype:    standard name of the object type
# rv_value: Radial velocity value. Eventually translated from a redshift
# z_value:  Redshift value. Eventually translated from a radial velocity
# rvz_qual: Quality code (A: best, .., E: worst)
# rvz_type: stored type of velocity: 'v'=radial velocity, 'z'=redshift
Simbad.reset_votable_fields()
Simbad.add_votable_fields('otype', 'rv_value', 'z_value',
                          'rvz_qual', 'rvz_type')


def query_name(name, verbose=False, print_header=False):
    """
    Query SIMBAD by name.
    """
    try:
        q = Simbad.query_object(name)
        main_id  = str(q['MAIN_ID'][0], encoding='utf-8')
        otype    = str(q['OTYPE'][0], encoding='utf-8')
        rv       = q['RV_VALUE'][0]
        z        = q['Z_VALUE'][0]
        rvz_qual = q['RVZ_QUAL'][0]
        rvz_type = q['RVZ_TYPE'][0]
    except RemoteServiceError:
        main_id  = None
        otype    = None
        rv       = None
        z        = None
        rvz_qual = None
        rvz_type = None
        if verbose:
            print('*** %s: not found ***' % name, file=sys.stderr)
    #
    results = OrderedDict([
        ("Name",         name),
        ("SIMBAD_ID",    main_id),
        ("Type",         otype),
        ("RV",           rv),
        ("z",            z),
        ("RV/z_Quality", rvz_qual),
        ("RV/z_Type",    rvz_type),
    ])
    if verbose:
        if print_header:
            print(",".join(results.keys()))
        print(",".join([str(v) for v in results.values()]))
    return results


def main():
    parser = argparse.ArgumentParser(
            description="Query SIMBAD database by source name")
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
