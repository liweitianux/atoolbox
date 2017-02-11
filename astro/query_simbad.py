#!/usr/bin/env python3
#
# Copyright (c) 2016-2017 Weitian LI <liweitianux@live.com>
# MIT license
#
# NOTE:
# * SimbadClass
#   https://astroquery.readthedocs.org/en/latest/api/astroquery.simbad.SimbadClass.html
# * All available VOTable fields:
#   http://simbad.u-strasbg.fr/simbad/sim-help?Page=sim-fscript#VotableFields
#
# Change logs:
# 2017-02-11:
#   * Add argument "--brief" to not print header
#   * Sync with 'query_ned.py'
# 2016-01-14:
#   * Add 'z_value'
#
# TODO:
# * allow to query by coordinates & radius range
# * filter queryed results according to the type/other...
# * if not queryed by name, then try query by coordinates
#

"""
Query SIMBAD with the provided name or coordinate.
http://simbad.u-strasbg.fr/simbad/
"""

import sys
import argparse
import csv

from collections import OrderedDict

from astroquery.simbad import Simbad
# from astropy import coordinates
# import astropy.units as u


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


def query_name(name, verbose=False):
    """
    Query SIMBAD by name.
    """
    q = Simbad.query_object(name)
    try:
        main_id  = str(q['MAIN_ID'][0], encoding='utf-8')
        otype    = str(q['OTYPE'][0], encoding='utf-8')
        rv       = q['RV_VALUE'][0]
        z        = q['Z_VALUE'][0]
        rvz_qual = q['RVZ_QUAL'][0]
        rvz_type = q['RVZ_TYPE'][0]
        if verbose:
            print('%s: %s,%s,%s,%s,%s,%s' % (name, main_id, otype, rv, z,
                                             rvz_qual, rvz_type))
    except (TypeError, KeyError) as e:
        main_id  = ''
        otype    = ''
        rv       = ''
        z        = ''
        rvz_qual = ''
        rvz_type = ''
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
        names = map(str.strip, open(args.input).readlines())
    except FileNotFoundError:
        names = map(str.strip, args.input.split(","))

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


#  vim: set ts=4 sw=4 tw=0 fenc=utf-8 ft=python: #
