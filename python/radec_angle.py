#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Aaron LI
# Created: 2015-04-17
# Updated: 2016-06-30
#

"""
Calculate the angles between the given center point to other points
on the sphere.

The following input formats are supported:
    longitude  latitude     => FMT
    ------------------------------------
    ??h??m??s  ??d??m??s    => "radec"
    ?? ?? ??   ?? ?? ??     => "ra3dec3"
    deg        deg          => "degdeg"
"""

import os
import sys
import re
import getopt
import math


USAGE = """Usage:
    %(prog)s [ -b -h ] -r RA -d DEC -i infile -f FMT -u UNIT

Required arguments:
    -r, --ra
        RA (??h??m??s) of the center
    -d, --dec
        DEC (??d??m??s) of the center
    -i, --infile
        input file containing the coordinates data
    -f, --format
        value: radec | ra3dec3 | degdeg
        coordinates format of the input data file
    -u, --unit
        value: deg | arcmin | arcsec
        unit of the output data

Optional arguments:
    -b, --brief
        brief mode: only output results
    -h, --help
""" % {'prog': os.path.basename(sys.argv[0])}


def usage():
    print(USAGE, file=sys.stderr)


def ra2deg(h, m, s):
    """
    Convert RA (hour, minute, second) to degree.
    """
    return h * 15.0 + m * 15.0/60.0 + s * 15.0/3600.0


def dec2deg(d, m, s):
    """
    Convert DEC (deg, arcmin, arcsec) to degree.
    """
    if (d >= 0):
        sign = 1.0
    else:
        sign = -1.0
    return sign * (math.fabs(d) + m/60.0 + s/3600.0)


def s_ra2deg(hms):
    """
    Convert RA string ("??h??m??s") to degree.
    """
    h, m, s = map(float, re.sub('[hms]', ' ', hms).split())
    return ra2deg(h, m, s)


def s_dec2deg(dms):
    """
    Convert DEC string ("??d??m??s") to degree.
    """
    d, m, s = map(float, re.sub('[dms]', ' ', dms).split())
    return dec2deg(d, m, s)


def deg2rad(deg):
    """
    Convert unit from deg to rad.
    """
    return deg * math.pi / 180.0


def rad2deg(rad):
    """
    Convert unit from rad to deg.
    """
    return rad * 180.0 / math.pi


def central_angle(p1, p2, unit="deg"):
    """
    Calculate the central angle between the two points on the sphere.

    Input parameters:
        p1, p2: (longitude, latitude), coorindates of the two points

    Algorithm:
        (radial, azimuthal, polar): (r, theta, phi)
        central_angle: alpha
        longitude: lambda = theta
        latitude: delta = 90 - phi
        colatitude: phi

        Unit vector:
        \hat{r}_1 = (cos(theta1) sin(phi1), sin(theta1) sin(phi1), cos(phi1))
            = (cos(lambda1) cos(delta1), sin(lambda1) cos(delta1), sin(delta1))
        \hat{r}_2 = (cos(theta2) sin(phi2), sin(theta2) sin(phi2), cos(phi2))
            = (cos(lambda2) cos(delta2), sin(lambda2) cos(delta2), sin(delta2))

        Therefore the angle (alpha) between \hat{r}_1 and \hat{r}_2:
        cos(alpha) = \hat{r}_1 \cdot \hat{r}_2
            = cos(delta1) cos(delta2) cos(lambda1-lambda2)
              + sin(delta1) sin(delta2)

    References:
    [1] Spherical Coordinates - Wolfram MathWorld
        http://mathworld.wolfram.com/SphericalCoordinates.html
        Equation (19)
    [2] Great Circle - Wolfram MathWorld
        http://mathworld.wolfram.com/GreatCircle.html
        Equation (1), (2), (4)
    """
    lbd1, delta1 = map(deg2rad, p1)
    lbd2, delta2 = map(deg2rad, p2)
    dotvalue = (math.cos(delta1) * math.cos(delta2) * math.cos(lbd1-lbd2) +
                math.sin(delta1) * math.sin(delta2))
    alpha = math.acos(dotvalue)
    if unit == "rad":
        return alpha
    elif unit == "arcmin":
        return rad2deg(alpha) * 60.0
    elif unit == "arcsec":
        return rad2deg(alpha) * 60.0*60.0
    else:
        # default: degree
        return rad2deg(alpha)


def main():
    # Mandatory arguments
    center_ra = None
    center_dec = None
    infile = None

    # Default parameters
    unit = "arcmin"
    fmt = "radec"  # default format: "??h??m??s  ??d??m??s"

    # Process command line arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:], "bd:f:hi:r:u:",
                                   ["brief", "dec=", "format=", "help",
                                    "infile=", "ra=", "unit="])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    brief = False  # brief mode
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit(1)
        elif opt in ("-b", "--brief"):
            brief = True
        elif opt in ("-d", "--dec"):
            center_dec = arg
        elif opt in ("-r", "--ra"):
            center_ra = arg
        elif opt in ("-i", "--infile"):
            infile = arg
        elif opt in ("-f", "--format"):
            fmt = arg
        elif opt in ("-u", "--unit"):
            unit = arg
        else:
            assert False, "unhandled option"

    # Check mandatory arguments
    if not center_ra:
        print("Error: --ra argument required!", file=sys.stderr)
    if not center_dec:
        print("Error: --dec argument required!", file=sys.stderr)
    if not infile:
        print("Error: --infile argument required!", file=sys.stderr)

    if fmt == "radec":
        center_ra_deg = s_ra2deg(center_ra)
        center_dec_deg = s_dec2deg(center_dec)
    elif fmt == "ra3dec3":
        ra_h, ra_m, ra_s = map(float, center_ra.split())
        dec_d, dec_m, dec_s = map(float, center_dec.split())
        center_ra_deg = ra2deg(ra_h, ra_m, ra_s)
        center_dec_deg = dec2deg(dec_d, dec_m, dec_s)
    elif fmt == "degdeg":
        center_ra_deg = float(center_ra)
        center_dec_deg = float(center_dec)
    else:
        print("Error: unknown format type: %s" % fmt, file=sys.stderr)
        sys.exit(2)

    if not brief:
        print("# Central_angle (unit: %s)" % unit)

    datafile = open(infile, "r")
    for line in datafile:
        if re.match(r"^\s*#", line):
            # skip comments
            continue
        elif re.match(r"^\s*$", line):
            # skip blank line
            continue
        # coordinate format
        if fmt == "radec":
            ra, dec = line.split()
            ra_deg = s_ra2deg(ra)
            dec_deg = s_dec2deg(dec)
        elif fmt == "ra3dec3":
            ra_h, ra_m, ra_s, dec_d, dec_m, dec_s = map(float, line.split())
            ra_deg = ra2deg(ra_h, ra_m, ra_s)
            dec_deg = dec2deg(dec_d, dec_m, dec_s)
        elif fmt == "degdeg":
            ra_deg, dec_deg = map(float, line.split())
        else:
            print("Error: unknown format type: %s" % fmt, file=sys.stderr)
            sys.exit(2)
        # calculate angle
        angle = central_angle((center_ra_deg, center_dec_deg),
                              (ra_deg, dec_deg), unit=unit)
        print("%.10f" % angle)
    datafile.close()


if __name__ == "__main__":
    main()
