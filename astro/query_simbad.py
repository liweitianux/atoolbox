#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# NOTE:
# * SimbadClass
#   https://astroquery.readthedocs.org/en/latest/api/astroquery.simbad.SimbadClass.html
# * All available VOTable fields:
#   http://simbad.u-strasbg.fr/simbad/sim-help?Page=sim-fscript#VotableFields
#
# ChangeLog:
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

__version__ = "0.1.1"
__date__ = "2016-01-14"


import sys
import argparse
import csv

from astroquery.simbad import Simbad
from astropy import coordinates
import astropy.units as u


## Simbad configurations
Simbad.ROW_LIMIT = 30
Simbad.TIMEOUT = 20

## Add query items/fields
# otype:    standard name of the object type
# rv_value: Radial velocity value. Eventually translated from a redshift
# z_value:  Redshift value. Eventually translated from a radial velocity
# rvz_qual: Quality code (A: best, .., E: worst)
# rvz_type: stored type of velocity: 'v'=radial velocity, 'z'=redshift
Simbad.reset_votable_fields()
Simbad.add_votable_fields('otype', 'rv_value', 'z_value', 'rvz_qual', 'rvz_type')


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
    results = {
        'main_id':  main_id,
        'otype':    otype,
        'rv':       rv,
        'z':        z,
        'rvz_qual': rvz_qual,
        'rvz_type': rvz_type,
    }
    return results


def main():
    parser = argparse.ArgumentParser(
            description="Query SIMBAD ...")
    parser.add_argument("-V", "--version", action="version",
            version="%(prog)s " + "%s (%s)" % (__version__, __date__))
    parser.add_argument("infile",
            help="file contains list of names; one per line")
    parser.add_argument("outfile",
            help="output with queryed data, empty if not found; CSV format")
    parser.add_argument("-v", "--verbose", dest="verbose",
            action="store_true", help="show verbose information")
    args = parser.parse_args()

    name_list     = []
    main_id_list  = []
    otype_list    = []
    rv_list       = []
    z_list        = []
    rvz_qual_list = []
    rvz_type_list = []

    with open(args.infile) as f:
        for name in f:
            name = str.strip(name)
            name_list.append(name)
            qr = query_name(name, verbose=args.verbose)
            main_id_list.append(qr['main_id'])
            otype_list.append(qr['otype'])
            rv_list.append(qr['rv'])
            z_list.append(qr['z'])
            rvz_qual_list.append(qr['rvz_qual'])
            rvz_type_list.append(qr['rvz_type'])

    with open(args.outfile, 'w') as of:
        writer = csv.writer(of)
        writer.writerow([ "Name", "SIMBAD_ID", "Type",
                          "RV", "z", "RV/z_Quality", "RV/z_Type" ])
        for i in range(len(name_list)):
            writer.writerow([ name_list[i],
                              main_id_list[i],
                              otype_list[i],
                              rv_list[i],
                              z_list[i],
                              rvz_qual_list[i],
                              rvz_type_list[i] ])


if __name__ == "__main__":
    main()


#  vim: set ts=4 sw=4 tw=0 fenc=utf-8 ft=python: #
