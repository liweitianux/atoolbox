#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Aaron LI
# Created: 2015-12-03
# Updated: 2015-12-05
#
# ChangeLog:
# 2015-12-05:
#   * Add class "RegionDS9" to parse region
# 2015-12-10:
#   * Support both "erg/cm^2/s" and "photon/cm^2/s" flux units
#   * Add argument "--outregion" to also save a region file
#

"""
Generate random point source information for MARX simulation.
And make a point source data list for "marx_pntsrc.py" usage.
"""

__version__ = "0.3.1"
__date__    = "2015-12-10"


import sys
import argparse
import re
import numpy as np


class RegionDS9:
    """
    Process DS9 regions.
    """
    def __init__(self, shape=None, xc=None, yc=None,
            width=None, height=None, rotation=None):
        self.shape    = shape
        self.xc       = xc
        self.yc       = yc
        self.width    = width
        self.height   = height
        self.rotation = rotation

    def parse(self, region):
        """
        Parse a DS9 region string and update the instance.

        Region syntax:
            box(xc,yc,width,height,rotation)

        Note:
        "width", "height" may have a '"' suffix which means "arcsec" instead
        of "degree".
        """
        re_box = re.compile(r'^\s*(?P<shape>box)\(\s*(?P<xc>[\d.-]+)\s*,\s*(?P<yc>[\d.-]+)\s*,\s*(?P<width>[\d.-]+"?)\s*,\s*(?P<height>[\d.-]+"?)\s*,\s*(?P<rotation>[\d.-]+)\s*\).*$', re.I)
        m_box  = re_box.match(region)
        if m_box is not None:
            self.shape    = "box"
            self.xc       = float(m_box.group("xc"))
            self.yc       = float(m_box.group("yc"))
            self.width    = self.parse_dms(m_box.group("width"))
            self.height   = self.parse_dms(m_box.group("height"))
            self.rotation = float(m_box.group("rotation"))
        else:
            raise NotImplementedError("Only 'box' region supported")

    @staticmethod
    def parse_dms(ms):
        """
        Parse a value in format ?'?" into degree.
        """
        re_arcmin = re.compile(r'^\s*(?P<arcmin>[\d.]+)\'.*')
        re_arcsec = re.compile(r'^([^\']*\'|)\s*(?P<arcsec>[\d.]+)".*')
        m_arcmin  = re_arcmin.match(ms)
        m_arcsec  = re_arcsec.match(ms)
        degree = 0.0
        if m_arcmin is not None:
            degree += float(m_arcmin.group("arcmin")) / 60.0
        if m_arcsec is not None:
            degree += float(m_arcsec.group("arcsec")) / 3600.0
        return degree


class RandCoord:
    """
    Randomly generate the coordinates of point sources within a given box
    region for MARX simulation.

    Arguments:
      xc      - central X position of the box (degree)
      yc      - central Y position of the box (degree)
      width   - width of the box (degree)
      height  - height of the box (degree)
      mindist - minimum distance between each generated coordinate (degree)
    """

    def __init__(self, xc, yc, width, height, mindist=0):
        self.xc      = xc
        self.yc      = yc
        self.width   = width
        self.height  = height
        self.mindist = mindist
        # Record the generated coordinates: [(x1,y1), (x2,y2), ...]
        self.xy = []

    def clear(self):
        """
        Clear previously generated coordinates.
        """
        self.xy = []

    def generate(self, n=1):
        """
        Generate random coordinates.
        """
        coord = []
        xmin = self.xc - 0.5 * self.width
        xmax = self.xc + 0.5 * self.width
        ymin = self.yc - 0.5 * self.height
        ymax = self.yc + 0.5 * self.height
        i = 0
        while i < n:
            x = np.random.uniform(low=xmin, high=xmax)
            y = np.random.uniform(low=ymin, high=ymax)
            if self.checkDistance((x, y)):
                i += 1
                coord.append((x, y))
                self.xy.append((x, y))
        return coord

    def checkDistance(self, coord):
        """
        Check whether the given coordinate has a distance larger than
        the specified "mindist"
        """
        if len(self.xy) == 0:
            return True
        else:
            xy = np.array(self.xy)  # each row represents one coordinate
            dist2 = (xy[:, 0] - coord[0])**2 + (xy[:, 1] - coord[1])**2
            if all(dist2 >= self.mindist**2):
                return True
            else:
                return False


class RandFlux:
    """
    Randomly generate the flux of point sources for MARX simulation.

    Arguments:
      fmin - minimum flux
      fmax - maximum flux
    """

    def __init__(self, fmin, fmax):
        self.fmin = fmin
        self.fmax = fmax

    @staticmethod
    def fluxDensity(S):
        """
        The *differential* number count - flux function: dN(>S)/dS
        i.e., density function

        Broken power law:
            dN/dS = (1) K (S/S_ref)^(-gamma_1);  (S < S_b)
                    (2) K (S_b/S_ref)^(gamma_2-gamma_1) (S/S_ref)^(-gamma_2);  (S >= S_b)
            K: normalization constant
            S_ref: normalization flux; [10^-15 erg/cm^2/s]
            gamma_1: faint power-law index
            gamma_2: bright power-law index
            S_b: break flux; [10^-15 erg/cm^2/s]

        Reference:
        [1] Kim et al. 2007, ApJ, 659, 29
            http://adsabs.harvard.edu/abs/2007ApJ...659...29K
            http://hea-www.cfa.harvard.edu/CHAMP/PUBLICATIONS/ChaMP_ncounts.pdf
            Table 4: ChaMP: 9.6 [deg^2]: 0.5-8 [keV]: 1.4 (photon index)
                Differential number count; broken power law
                K (normalization constant): 1557 (+28 / -50)
                S_ref (normalization flux): 1.0 [10^-15 erg/cm^2/s]
                gamma_1 (faint power-law index): 1.64 (+/- 0.01)
                gamma_2 (bright power-law index): 2.48 (+/- 0.05)
                S_b (break flux): 22.9 (+/- 1.6) [10^-15 erg/cm^2/s]
                f_min (faint flux limit): 0.69 [10^-15 erg/cm^2/s]
                f_max (bright flux limit): 6767.74 [10^-15 erg/cm^2/s]
        """
        K       = 1557  # normalization constant: 1557 (+28 / -50)
        S_ref   = 1.0   # normalization flux: 1.0 [10^-15 erg/cm^2/s]
        gamma_1 = 1.64  # faint power-law index: 1.64 (+/- 0.01)
        gamma_2 = 2.48  # bright power-law index: 2.48 (+/- 0.05)
        S_b     = 22.9  # break flux: 22.9 (+/- 1.6) [10^-15 erg/cm^2/s]
        # Adjust unit/magnitude
        S  = S / 1e-15  # => unit: 10^-15 erg/cm^2/s
        if isinstance(S, np.ndarray):
            Np = np.zeros(S.shape)
            Np[S<=0]   = 0.0
            Np[S<=S_b] = K * (S[S<=S_b] / S_ref)**(-gamma_1)
            Np[S>S_b]  = K * (S_b/S_ref)**(gamma_2-gamma_1) * (S[S>S_b] / S_ref)**(-gamma_2)
        else:
            # "S" is a single number
            if S <= 0.0:
                Np = 0.0
            elif S <= S_b:
                Np = K * (S/S_ref)**(-gamma_1)
            else:
                Np = K * (S_b/S_ref)**(gamma_2-gamma_1) * (S/S_ref)**(-gamma_2)
        #
        return Np

    def generate(self, n=1):
        """
        Generate a sample of luminosity values within [min, max] from
        the above luminosity distribution.
        """
        results = []
        # Get the maximum value of the flux number density function,
        # which is a monotonically decreasing.
        M = self.fluxDensity(self.fmin)
        for i in range(n):
            while True:
                u = np.random.uniform() * M
                y = 10 ** np.random.uniform(low=np.log10(self.fmin),
                        high=np.log10(self.fmax))
                if u <= self.fluxDensity(y):
                    results.append(y)
                    break
        return results


def main():
    parser = argparse.ArgumentParser(
            description="Randomly generate point sources information for MARX")
    parser.add_argument("-V", "--version", action="version",
            version="%(prog)s " + "v%s (%s)" % (__version__, __date__))
    parser.add_argument("-n", "--number", dest="number", type=int, default=1,
            help="number of point sources (default: 1)")
    parser.add_argument("-m", "--fmin", dest="fmin",
            type=float, default=1e-15,
            help="minimum flux (default: 1e-15 erg/cm^2/s)")
    parser.add_argument("-M", "--fmax", dest="fmax",
            type=float, default=6000e-15,
            help="maximum flux (default: 6000e-15 erg/cm^2/s)")
    parser.add_argument("-r", "--region", dest="region", required=True,
            help="region within which to generate coordinates ('box' only)")
    parser.add_argument("-d", "--distance", dest="distance", default="0",
            help="minimum distance between coordinates (default: 0) [unit: deg/arcmin]")
    parser.add_argument("-u", "--unit", dest="unit", default="erg",
            help="unit for input and output flux; 'erg' (default) / 'photon'")
    parser.add_argument("-f", "--factor", dest="factor", type=float,
            help="conversion factor from 'photon/cm^s/s' to 'erg/cm^2/s' (required if unit='photon')")
    parser.add_argument("-o", "--outfile", dest="outfile",
            help="output file to save the generate information list")
    parser.add_argument("-O", "--outregion", dest="outregion",
            help="write the generate information list as a DS9 region file")

    args = parser.parse_args()

    # Check flux unit
    if args.unit == "erg":
        unit   = "erg/cm^2/s"
        fmin   = args.fmin
        fmax   = args.fmax
        factor = 1.0
    elif args.unit == "photon":
        unit   = "photon/cm^2/s"
        factor = args.factor
        try:
            fmin = args.fmin / factor
            fmax = args.fmax / factor
        except NameError:
            raise ValueError("argument '--factor' required")
    else:
        raise ValueError("unsupported flux unit")

    region = RegionDS9()
    region.parse(args.region)
    # Check the box rotation
    if not (abs(region.rotation) <= 1.0 or abs(region.rotation-360) <= 1.0):
        raise NotImplementedError("rotated 'box' region not supported")

    # Minimum distance between generated coordinates
    try:
        mindist = float(args.distance)
    except ValueError:
        mindist = region.parse_dms(args.distance)

    randcoord = RandCoord(region.xc, region.yc, region.width, region.height,
            mindist=mindist)
    randflux  = RandFlux(fmin, fmax)
    coord     = randcoord.generate(n=args.number)
    flux      = randflux.generate(n=args.number)

    if args.outfile:
        outfile = open(args.outfile, "w")
    else:
        outfile = sys.stdout

    print("# region: %s" % args.region, file=outfile)
    print("# mindist: %.9f [deg]" % mindist, file=outfile)
    print("# f_min: %.9g; f_max: %.9g [%s]" % (fmin, fmax, unit), file=outfile)
    print("# factor: %g [photon/cm^2/s] / [erg/cm^2/s]" % factor, file=outfile)
    print("# R.A.[deg]   Dec.[deg]   Flux[%s]" % unit, file=outfile)
    for ((ra, dec), f) in zip(coord, flux):
        print("%.9f   %.9f   %.9g" % (ra, dec, f*factor), file=outfile)

    if args.outfile:
        outfile.close()

    # Save the generated information as a DS9 region file if specified
    if args.outregion:
        reg_r = '3"'
        reg_header = ["# Region file format: DS9 version 4.1", "fk5"]
        regions = ["circle(%.9f,%.9f,%s) # text={%.9g}" % (ra, dec, reg_r, f)
                for ((ra, dec), f) in zip(coord, flux)]
        regfile = open(args.outregion, "w")
        regfile.write("\n".join(reg_header + regions))
        regfile.close()


if __name__ == "__main__":
    main()

