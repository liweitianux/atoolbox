# -*- coding: utf-8 -*-
#
# Aaron Li
# 2015/06/19

"""
Class Region for regions on the spherical surface.
Used in astronomy to select/define a certian region, e.g, DS9.
"""

import sys


class Region(object):
    """
    Basic region class for regions on the spherical surface,
    similar definition as to DS9 regions.

    Coordinate style: (ra, dec)
    Unit: degree
    ra: [0, 2\pi)
    dec: [-\pi/2, \pi/2]
    """

    # currently supported region types (similar to DS9)
    REGION_TYPES = ["circle", "ellipse", "box", "annulus", "pie", "panda"]

    def __init__(self, regtype, xc, yc,
            radius=None, radius2=None,
            width=None, height=None, rotation=None,
            start=None, end=None):
        if regtype.lower() not in self.REGION_TYPES:
            raise ValueError("only following region types supported: %s" %\
                    " ".join(self.REGION_TYPES))
        self.regtype = regtype.lower()
        self.xc = xc
        self.yc = yc
        self.radius = radius
        self.radius2 = radius2
        self.width = width
        self.height = height
        self.rotation = rotation

    def __repr__(self):
        return "Region: %s" % self.regtype

    def dump(self):
        return {"regtype": self.regtype,
                "xc": self.xc,
                "yc": self.yc,
                "radius": self.radius,
                "radius2": self.radius2,
                "width": self.width,
                "height": self.height,
                "rotation": self.rotation
               }

    def is_inside(self, point):
        """
        Determine whether the given point is inside the region.
        """
        x = point[0]
        y = point[1]
        if self.regtype == "box":
            #print("WARNING: rotation box currently not supported!",
            #        file=sys.stderr)
            xmin = self.xc - self.width/2.0
            xmax = self.xc + self.width/2.0
            ymin = self.yc - self.height/2.0
            ymax = self.yc + self.height/2.0
            if all([x >= xmin, x <= xmax, y >= ymin, y <= ymax]):
                return True
            else:
                return False
        else:
            raise ValueError("region type '%s' currently not implemented" %\
                    self.regtype)

