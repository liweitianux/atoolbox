#!/usr/bin/env python3
#
# Copyright (c) 2017 Aaron LI <aly@aaronly.me>
# MIT License
#

"""
Convert among various (R.A., Dec.) coordinate formats.
"""

import argparse

from astropy import units as au
from astropy.coordinates import Angle


def parse_coord(c):
    if len(c) == 6:
        # h m s d m s
        ra = Angle((float(c[0]), float(c[1]), float(c[2])), unit=au.hourangle)
        dec = Angle((float(c[3]), float(c[4]), float(c[5])), unit=au.deg)
    elif len(c) == 2:
        ra = Angle(float(c[0]), unit=au.deg)
        dec = Angle(float(c[1]), unit=au.deg)
    else:
        raise ValueError("invalid coordinate: {0}".format(c))
    return (ra, dec)


def main():
    parser = argparse.ArgumentParser(
        description="Convert among multiple coordinate formats")
    parser.add_argument("coord", nargs="+")
    args = parser.parse_args()

    ra, dec = parse_coord(args.coord)
    info = (
        "%-14s  %-14s\n" % ("R.A.", "Dec.") +
        "%s--%s\n" % ("-"*14, "-"*14) +
        "%-14.3f  %-+14.3f\n" % (ra.deg, dec.deg) +
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
