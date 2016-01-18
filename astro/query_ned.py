#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# NOTE:
# * NedClass
#   https://astroquery.readthedocs.org/en/latest/api/astroquery.ned.NedClass.html
#
# ChangeLog:
#
# TODO:
# * allow to query by coordinates & radius range
# * filter queryed results according to the type/other...
# * if not queryed by name, then try query by coordinates
#

"""
Query NED with the provided name or coordinate.
NASA/IPAC Extragalactic Database: http://ned.ipac.caltech.edu/
"""

__version__ = "0.1.0"
__date__ = "2016-01-14"


import sys
import argparse
import csv

from astroquery.ned import Ned
from astroquery.exceptions import RemoteServiceError
from astropy import coordinates
import astropy.units as u


## Ned configurations
Ned.TIMEOUT = 20


def query_name(name, verbose=False):
    """
    Query NED by name.
    """
    try:
        q = Ned.query_object(name)
        objname  = str(q['Object Name'][0], encoding='utf-8')
        objtype  = str(q['Type'][0], encoding='utf-8')
        velocity = q['Velocity'][0]
        z        = q['Redshift'][0]
        z_flag   = str(q['Redshift Flag'][0], encoding='utf-8')
        refs     = q['References'][0]
        notes    = q['Notes'][0]
        if verbose:
            print('%s: %s,%s,%s,%s,%s,%s,%s' % (name, objname, objtype,
                                                velocity, z, z_flag,
                                                refs, notes))
    except RemoteServiceError as e:
        objname  = ''
        objtype  = ''
        velocity = ''
        z        = ''
        z_flag   = ''
        refs     = ''
        notes    = ''
        if verbose:
            print('*** %s: not found ***' % name, file=sys.stderr)
    #
    results = {
        'objname':  objname,
        'objtype':  objtype,
        'velocity': velocity,
        'z':        z,
        'z_flag':   z_flag,
        'refs':     refs,
        'notes':    notes,
    }
    return results


def main():
    parser = argparse.ArgumentParser(
            description="Query NED ...")
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
    objname_list  = []
    objtype_list  = []
    velocity_list = []
    z_list        = []
    z_flag_list   = []
    refs_list     = []
    notes_list    = []

    with open(args.infile) as f:
        for name in f:
            name = str.strip(name)
            name_list.append(name)
            qr = query_name(name, verbose=args.verbose)
            objname_list.append(qr['objname'])
            objtype_list.append(qr['objtype'])
            velocity_list.append(qr['velocity'])
            z_list.append(qr['z'])
            z_flag_list.append(qr['z_flag'])
            refs_list.append(qr['refs'])
            notes_list.append(qr['notes'])

    with open(args.outfile, 'w') as of:
        writer = csv.writer(of)
        writer.writerow([ "Name", "NED_Name", "Type", "Velocity",
                          "z", "z_Flag", "References", "Notes" ])
        for i in range(len(name_list)):
            writer.writerow([ name_list[i],
                              objname_list[i],
                              objtype_list[i],
                              velocity_list[i],
                              z_list[i],
                              z_flag_list[i],
                              refs_list[i],
                              notes_list[i] ])


if __name__ == "__main__":
    main()


#  vim: set ts=4 sw=4 tw=0 fenc=utf-8 ft=python: #
