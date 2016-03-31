#!/usr/bin/env python3
# -*- coding: utf-8-
#
# Split the strip-shaped CCD gaps regions into a series of small
# square regions, which are used as the input regions of 'roi' to
# determine the corresponding background regions, and finally providied
# to 'dmfilth' in order to fill in the CCD gaps.
#
# Aaron LI
# 2015/08/12
#
# Changelogs:
# v0.1.0, 2015/08/12
#   * initial version
#


__version__ = "0.1.0"
__date__    = "2015/08/12"


import os
import sys
import re
import math
import argparse
from io import TextIOWrapper


## BoxRegion {{{
class BoxRegion(object):
    """
    CIAO/DS9 "rotbox"/"box" region class.

    rotbox/box format:
        rotbox(xc, yc, width, height, rotation)
        box(xc, yc, width, height, rotation)
    Notes:
        rotation: [0, 360) (degree)
    """
    def __init__(self, xc=None, yc=None,
                 width=None, height=None, rotation=None):
        self.regtype  = "rotbox"
        self.xc       = xc
        self.yc       = yc
        self.width    = width
        self.height   = height
        self.rotation = rotation

    def __str__(self):
        return "%s(%s,%s,%s,%s,%s)" % (self.regtype, self.xc, self.yc,
                self.width, self.height, self.rotation)

    @classmethod
    def parse(cls, regstr):
        """
        Parse region string.
        """
        regex_box = re.compile(r'^\s*(box|rotbox)\(([0-9. ]+),([0-9. ]+),([0-9. ]+),([0-9. ]+),([0-9. ]+)\)\s*$', re.I)
        m = regex_box.match(regstr)
        if m:
            regtype  = m.group(1)
            xc       = float(m.group(2))
            yc       = float(m.group(3))
            width    = float(m.group(4))
            height   = float(m.group(5))
            rotation = float(m.group(6))
            return cls(xc, yc, width, height, rotation)
        else:
            return None

    def split(self, filename=None):
        """
        Split strip-shaped box region into a series small square regions.
        """
        angle = self.rotation * math.pi / 180.0
        # to record the center coordinates of each split region
        centers = []
        if self.width > self.height:
            # number of regions after split
            nreg = math.ceil(self.width / self.height)
            # width & height of the split region
            width = self.width / nreg
            height = self.height
            # position of the left-most region
            x_l = self.xc - 0.5*self.width * math.cos(angle)
            y_l = self.yc - 0.5*self.width * math.sin(angle)
            for i in range(nreg):
                x = x_l + (0.5 + i) * width * math.cos(angle)
                y = y_l + (0.5 + i) * width * math.sin(angle)
                centers.append((x, y))
        else:
            # number of regions after split
            nreg = math.ceil(self.height / self.width)
            # width & height of the split region
            width = self.width
            height = self.height / nreg
            # position of the left-most region
            x_l = self.xc + 0.5*self.height * math.cos(angle + math.pi/2)
            y_l = self.yc + 0.5*self.height * math.sin(angle + math.pi/2)
            for i in range(nreg):
                x = x_l - (0.5 + i) * height * math.cos(angle + math.pi/2)
                y = y_l - (0.5 + i) * height * math.sin(angle + math.pi/2)
                centers.append((x, y))
        # create split regions
        regions = []
        for (x, y) in centers:
            regions.append(self.__class__(x, y, width+2, height+2,
                                          self.rotation))
        # write split regions into file if specified
        if isinstance(filename, str):
            regout = open(filename, "w")
            regout.write("\n".join(map(str, regions)))
            regout.close()
        else:
            return regions

## BoxRegion }}}


def main():
    # command line arguments
    parser = argparse.ArgumentParser(
            description="Split strip-shaped rotbox region into " + \
                    "a series of small square regions.",
            epilog="Version: %s (%s)" % (__version__, __date__))
    parser.add_argument("-V", "--version", action="version",
            version="%(prog)s " + "%s (%s)" % (__version__, __date__))
    parser.add_argument("infile", help="input rotbox region file")
    parser.add_argument("outfile", help="output file of the split regions")
    args = parser.parse_args()

    outfile = open(args.outfile, "w")
    regex_box = re.compile(r'^\s*(box|rotbox)\([0-9., ]+\)\s*$', re.I)
    for line in open(args.infile, "r"):
        if regex_box.match(line):
            reg = BoxRegion.parse(line)
            split_regs = reg.split()
            outfile.write("\n".join(map(str, split_regs)) + "\n")
        else:
            outfile.write(line)

    outfile.close()


if __name__ == "__main__":
    main()

