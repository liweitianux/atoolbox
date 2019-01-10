#!/usr/bin/env python3
#
# Copyright (c) 2017,2019 Aaron LI <aly@aaronly.me>
# MIT License
#

"""
Convert among various (R.A., Dec.) coordinate formats.
"""

import re
import argparse

from astropy import units as au
from astropy.coordinates import Angle


def parse_coord(c):
    ra_is_deg = True
    if len(c) == 2:
        ra, dec = c
        # ?h[our]  ?[deg]
        if ra.endswith('h') or ra.endswith('hour'):
            ra = ra.rstrip('hour')
            ra_is_deg = False

        # ?d[eg] ?[deg]
        # ?[deg] ?[deg]
        ra = ra.rstrip('deg')
        dec = dec.rstrip('deg')

        # ?h?m?s   ?d?m?s
        # ?:?:?    ?:?:?
        ra = re.split(r'[hms:]', ra.rstrip('s'))
        dec = re.split(r'[dms:]', dec.rstrip('s'))
    elif len(c) == 6:
        # h m s  d m s
        ra, dec = c[:3], c[-3:]
    else:
        raise ValueError('invalid coordinate: {0}'.format(c))

    if len(ra) == 1:
        ra = float(ra[0])
    else:
        ra = (float(ra[0]), float(ra[1]), float(ra[2]))
    if len(dec) == 1:
        dec = float(dec[0])
    else:
        dec = (float(dec[0]), float(dec[1]), float(dec[2]))

    if ra_is_deg:
        ra = Angle(ra, unit=au.deg)
    else:
        ra = Angle(ra, unit=au.hourangle)
    dec = Angle(dec, unit=au.deg)
    return (ra, dec)


def main():
    parser = argparse.ArgumentParser(
        description="Convert among multiple coordinate formats")
    parser.add_argument(
        "coord", nargs="+",
        help="syntax: (1) deg deg; (2) hour deg; (3) h m s d m s; " +
        "(4) ?h?m?s ?d?m?s; (5) ?:?:? ?:?:?")
    args = parser.parse_args()

    ra, dec = parse_coord(args.coord)
    info = (
        "%-14s  %-14s\n" % ("R.A.", "Dec.") +
        "%s--%s\n" % ("-"*14, "-"*14) +
        "%-9.4f[deg]  %-+9.4f[deg]\n" % (ra.deg, dec.deg) +
        "%-8.4f[hour]  %-+9.4f[deg]\n" % (ra.hour, dec.deg) +
        "%-14s  %-14s\n" % (
            ra.to_string(unit=au.hourangle, precision=4),
            dec.to_string(unit=au.deg, alwayssign=True, precision=3)) +
        "%-14s  %-14s\n" % (
            ra.to_string(unit=au.hourangle, sep=":", precision=4),
            dec.to_string(unit=au.deg, alwayssign=True,
                          sep=":", precision=3)) +
        "%-14s  %-14s\n" % (
            ra.to_string(unit=au.hourangle, sep=" ", precision=4),
            dec.to_string(unit=au.deg, alwayssign=True,
                          sep=" ", precision=3))
    )
    print(info)


if __name__ == "__main__":
    main()
