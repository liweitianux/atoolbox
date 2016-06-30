#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Aaron LI
# Created: 2015-04-17
# Updated: 2016-06-30
#

"""
Convert the coordinates data in format (??h??m??s, ??d??m??s)
to format (degree, degree).
"""

import os
import sys
import re
import getopt
import math


USAGE = """Usage:
    %(prog)s [ -h ] -i coords_file

Required arguments:
    -i, --infile
        infile containing the coordinates

Optional arguments:
    -h, --help
""" % {'prog': os.path.basename(sys.argv[0])}


def usage():
    print(USAGE)


def ra2deg(h, m, s):
    return h * 15.0 + m * 15.0/60.0 + s * 15.0/3600.0


def dec2deg(d, m, s):
    if (d >= 0):
        sign = 1.0
    else:
        sign = -1.0
    return sign * (math.fabs(d) + m/60.0 + s/3600.0)


def s_ra2deg(hms):
    h, m, s = map(float, re.sub('[hms]', ' ', hms).split())
    return h * 15.0 + m * 15.0/60.0 + s * 15.0/3600.0


def s_dec2deg(dms):
    d, m, s = map(float, re.sub('[dms]', ' ', dms).split())
    if (d >= 0):
        sign = 1.0
    else:
        sign = -1.0
    return sign * (math.fabs(d) + m/60.0 + s/3600.0)


def calc_offset(coord1, coord2):
    ra1, dec1 = coord1
    ra2, dec2 = coord2
    return math.sqrt((ra1-ra2)**2 + (dec1-dec2)**2)


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:",
                                   ["help", "infile="])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit(1)
        elif opt in ("-i", "--infile"):
            infile = arg
        else:
            assert False, "unhandled option"

    for line in open(infile):
        if re.match(r"^\s*#", line) or re.match(r"^\s*$", line):
            continue
        ra, dec = line.split()
        ra_deg = s_ra2deg(ra)
        dec_deg = s_dec2deg(dec)
        print("%.8f %.8f" % (ra_deg, dec_deg))


if __name__ == "__main__":
    main()
