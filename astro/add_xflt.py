#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Aaron LI
# 2015/06/16

"""
Add XFLT#### keywords to the spectrum header according to the provided
region, in order to employ the "PROJCT" model in XSPEC.
"""

__version__ = "0.1.0"
__date__ = "2015-11-14"

import sys
import argparse
import subprocess
import re
import os
from collections import OrderedDict


def parse_region(regstr):
    """
    Parse the given region string into one of the following 4 cases:
    1. annulus
    2. pie (cxc)
    3. pie + annulus (ftools/xmm)
    4. other

    For the first 3 cases, the return is a dictionary like:
        { 'shape': 'pie', 'xc': 55, 'yc': 89,
          'radius_in': 50, 'radius_out': 100,
          'angle_begin': 30, 'angle_end': 120 }
    Otherwise, return None.
    """
    re_annulus = re.compile(r'^.*(?P<shape>annulus)\(\s*(?P<xc>[\d.-]+)\s*,\s*(?P<yc>[\d.-]+)\s*,\s*(?P<radius_in>[\d.-]+)\s*,\s*(?P<radius_out>[\d.-]+)\s*\).*$', re.I)
    re_pie_cxc = re.compile(r'^.*(?P<shape>pie)\(\s*(?P<xc>[\d.-]+)\s*,\s*(?P<yc>[\d.-]+)\s*,\s*(?P<radius_in>[\d.-]+)\s*,\s*(?P<radius_out>[\d.-]+)\s*,\s*(?P<angle_begin>[\d.-]+)\s*,\s*(?P<angle_end>[\d.-]+)\s*\).*$', re.I)
    re_pie_ft  = re.compile(r'^.*(?P<shape>pie)\(\s*(?P<xc>[\d.-]+)\s*,\s*(?P<yc>[\d.-]+)\s*,\s*(?P<angle_begin>[\d.-]+)\s*,\s*(?P<angle_end>[\d.-]+)\s*\).*$', re.I)
    m_annulus = re_annulus.match(regstr)
    m_pie_cxc = re_pie_cxc.match(regstr)
    m_pie_ft  = re_pie_ft.match(regstr)
    if m_pie_cxc is not None:
        # case 2: pie (cxc)
        region = OrderedDict([
            ('shape',       m_pie_cxc.group('shape').lower()),
            ('xc',          float(m_pie_cxc.group('xc'))),
            ('yc',          float(m_pie_cxc.group('yc'))),
            ('radius_in',   float(m_pie_cxc.group('radius_in'))),
            ('radius_out',  float(m_pie_cxc.group('radius_out'))),
            ('angle_begin', float(m_pie_cxc.group('angle_begin'))),
            ('angle_end',   float(m_pie_cxc.group('angle_end')))
        ])
    elif m_pie_ft is not None:
        # pie (ftools/xmm)
        if m_annulus is not None:
            # case 3: pie + annulus (ftools/xmm)
            region = OrderedDict([
                ('shape',       m_pie_ft.group('shape').lower()),
                ('xc',          float(m_pie_ft.group('xc'))),
                ('yc',          float(m_pie_ft.group('yc'))),
                ('radius_in',   float(m_annulus.group('radius_in'))),
                ('radius_out',  float(m_annulus.group('radius_out'))),
                ('angle_begin', float(m_pie_ft.group('angle_begin'))),
                ('angle_end',   float(m_pie_ft.group('angle_end')))
            ])
        else:
            region = None
    elif m_annulus is not None:
        # case 1: annulus
        region = OrderedDict([
            ('shape',      m_annulus.group('shape').lower()),
            ('xc',         float(m_annulus.group('xc'))),
            ('yc',         float(m_annulus.group('yc'))),
            ('radius_in',  float(m_annulus.group('radius_in'))),
            ('radius_out', float(m_annulus.group('radius_out')))
        ])
    else:
        region = None
    return region


def make_xflt(region):
    """
    Make a dictionary for the XFLT#### keywords and values according
    to the provided region.

    Return:
    a dictionary containing the XFLT#### keywords and values, e.g.,
    { 'XFLT0001': radius_out,  'XFLT0002': radius_out, 'XFLT0003': 0,
      'XFLT0004': angle_begin, 'XFLT0005': angle_end }
    """
    if region.get('shape') == 'annulus':
        xflt = OrderedDict([
            ('XFLT0001', region.get('radius_out')),
            ('XFLT0002', region.get('radius_out')),
            ('XFLT0003', 0)
        ])
    elif region.get('shape') == 'pie':
        xflt = OrderedDict([
            ('XFLT0001', region.get('radius_out')),
            ('XFLT0002', region.get('radius_out')),
            ('XFLT0003', 0),
            ('XFLT0004', region.get('angle_begin')),
            ('XFLT0005', region.get('angle_end'))
        ])
    else:
        xflt = None
    return xflt


def add_xflt(fitsfile, xflt):
    """
    Add XFLT#### keywords to the given FITS file.
    """
    if xflt is not None:
        for key, val in xflt.items():
            cmd = 'fthedit "%(file)s+1" keyword="%(key)s" operation=add value="%(val)s"' % \
                    {'file': fitsfile, 'key': key, 'val': val}
            print("CMD: %s" % cmd, file=sys.stderr)
            subprocess.call(cmd, shell=True)


def main():
    parser = argparse.ArgumentParser(
            description="Add XFLT???? keywords to spectrum header")
    parser.add_argument("-V", "--version", action="version",
            version="%(prog)s " + "%s (%s)" % (__version__, __date__))
    parser.add_argument("spectrum", help="input spectrum; @stack")
    parser.add_argument("region", help="extraction region of this spectrum; @stack")
    parser.add_argument("arcmin2pix", nargs='?', help="1 arcmin = ? pixel",
            default=1.0, type=float)
    args = parser.parse_args()

    if args.spectrum[0] == '@' and args.region[0] == '@':
        spectrum  = map(str.strip, open(args.spectrum[1:]).readlines())
        regionstr = map(str.strip, open(args.region[1:]).readlines())
    else:
        spectrum  = [ args.spectrum ]
        regionstr = [ args.region ]

    for spec, reg in zip(spectrum, regionstr):
        print("SPECTRUM: '%s'" % spec)
        print("REGION: '%s'" % reg)
        region = parse_region(reg)
        if region is None:
            print("ERROR: invalid region %s" % reg, file=sys.stderr)
            sys.exit(11)
        else:
            # Convert pixel to arcmin
            region['radius_in']  = region['radius_in']  / args.arcmin2pix
            region['radius_out'] = region['radius_out'] / args.arcmin2pix
            xflt = make_xflt(region)
            add_xflt(spec, xflt)


if __name__ == "__main__":
    main()

