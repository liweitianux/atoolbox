#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Aaron LI
# 2015/06/19

"""
Generate the required number of random points within the required region.
"""

__version__ = "0.1.0"
__date__ = "2015/06/19"
DEBUG = True

import sys
import argparse
import random
import time
import re

from rand.sphere import sphere_point
from region.region import Region

random.seed(time.time())


def parse_region(regstring):
    reg_par = re.sub(r"[(),]", " ", regstring).split()
    regtype = reg_par[0].lower()
    if regtype == "box":
        xc = float(reg_par[1])
        yc = float(reg_par[2])
        width = parse_reg_value(reg_par[3])
        height = parse_reg_value(reg_par[4])
        rotation = float(reg_par[5])
        reg = Region(regtype, xc=xc, yc=yc,
                width=width, height=height, rotation=rotation)
    else:
        raise ValueError("region type '%s' currently not implemented" % regtype)
    return reg


def parse_reg_value(valstring):
    if valstring[-1] == '"':
        # arcsec -> deg
        value = float(valstring.split('"')[0]) / 60.0 / 60.0
    elif valstring[-1] == "'":
        # arcmin -> deg
        value = float(valstring.split("'")[0]) / 60.0
    else:
        value = float(valstring)
    return value


def main():
    parser = argparse.ArgumentParser(
            description="Generate random point within the given region.")
    parser.add_argument("-V", "--version", action="version",
            version="%(prog)s " + "%s (%s)" % (__version__, __date__))
    parser.add_argument("-n", "--number", dest="number",
            type=int, default=1,
            help="number of points to be generated")
    parser.add_argument("-r", "--region", dest="region", required=True,
            help="DS9 region")
    args = parser.parse_args()

    reg = parse_region(args.region)
    if DEBUG:
        print("DEBUG: region: ", reg.dump(), file=sys.stderr)

    points = []
    while len(points) < args.number:
        p = sphere_point(unit="deg")
        if reg.is_inside(p):
            points.append(p)
            print("%s  %s" % p)


if __name__ == "__main__":
    main()

