#!/usr/bin/env python3
#
# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT License
#

"""
Taper the sky image (input of OSKAR simulation) to mitigate the
side lobes effects, which causes trouble in creating good images.

The circular Tukey window is adopted, which is also built in by
e.g., WSClean.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timezone

import numpy as np
from scipy import interpolate
import astropy.io.fits as fits


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(os.path.basename(sys.argv[0]))


def make2d(w1d, x=None):
    """
    Create 2D filter from the 1D one.

    Parameters
    ----------
    w1d : 1D `~numpy.ndarray`
        The input 1D filter/window
    x : 1D `~numpy.ndarray`, optional
        The X-axis values of the input ``w1d`` filter

    Returns
    -------
    w2d : 2D `~numpy.ndarray`
        Created 2D filter/window from the input 1D one.

    Credit
    ------
    [1] MATLAB - creating 2D convolution filters
        https://cn.mathworks.com/matlabcentral/newsreader/view_thread/23588
    """
    if x is None:
        L = len(w1d)
        M = (L-1) / 2
        x = np.linspace(-M, M, num=L)

    xmax = np.max(x)
    xsize = int(2 * xmax + 1)
    xx = np.linspace(-xmax, xmax, num=xsize)
    xg, yg = np.meshgrid(xx, xx)
    r = np.sqrt(xg**2 + yg**2)
    ridx = (r <= xmax)
    w2d = np.zeros(shape=(xsize, xsize))
    finterp = interpolate.interp1d(x=x, y=w1d, kind="linear")
    w2d[ridx] = finterp(r[ridx])

    return w2d


def tukey(L, Minner=0.5, Mouter=0.5, Lmin=0):
    """
    The 1D Tukey filter/window

    ^
    | Lmin | Minner*L | constant ones | Mouter*L |
    +--------------------------------------------. (L)

    Parameters
    ----------
    L : int
        The length (number of points) of the window/filter
    Minner : float
        The fraction of ``L`` for the inner (left), increasing part
    Mouter : float
        The fraction of ``L`` for the outer (left), decreasing part
    Lmin : int
        The beginning point for the inner (left) part

    References
    ----------
    * MATLAB - Tukey (tapered cosine) window
      https://cn.mathworks.com/help/signal/ref/tukeywin.html
    * WSClean - Weight tapering
      https://sourceforge.net/p/wsclean/wiki/Tapering/
    """
    w = np.zeros(L)
    Linner = int(Minner * L)
    Louter = int(Mouter * L)

    Lones = L - Lmin - Linner - Louter
    if Lones < 0:
        raise RuntimeError("invalid parameters combination")
    w[(Lmin+Linner):(Lmin+Linner+Lones)] = 1.0

    if Linner > 0:
        xinner = np.arange(0, Linner)
        x2 = (np.pi/Linner) * (xinner - Linner)
        w[Lmin:(Lmin+Linner)] = 0.5 + 0.5 * np.cos(x2)

    if Louter > 0:
        xouter = np.arange(0, Louter)
        x2 = (np.pi/Louter) * xouter
        w[(Lmin+Linner+Lones):] = 0.5 + 0.5 * np.cos(x2)

    return w


def main():
    parser = argparse.ArgumentParser(
        description="Taper sky image with Tukey window")
    parser.add_argument("-C", "--clobber", dest="clobber",
                        action="store_true",
                        help="overwrite existing file")
    parser.add_argument("-r", "--window-inner", dest="Rinner",
                        type=int, required=True,
                        help="inner radius [pixel] of the tukey window")
    parser.add_argument("-R", "--window-outer", dest="Router",
                        type=int, required=True,
                        help="outer radius [pixel] of the tukey window")
    parser.add_argument("-T", "--outfile-taper", dest="outfile_taper",
                        help="save the applied taper into a FITS image")
    parser.add_argument("infile", help="input FITS sky image")
    parser.add_argument("outfile", help="output tapered sky image")
    args = parser.parse_args()

    if os.path.exists(args.outfile):
        if args.clobber:
            logger.warning("Removed existing file: %s" % args.outfile)
            os.remove(args.outfile)
        else:
            raise OSError("Output file already exists: %s" % args.outfile)
    if os.path.exists(args.outfile_taper):
        if args.clobber:
            logger.warning("Removed existing file: %s" % args.outfile_taper)
            os.remove(args.outfile_taper)
        else:
            raise OSError("Output file already exists: %s" %
                          args.outfile_taper)

    with fits.open(args.infile) as f:
        image = f[0].data
        header = f[0].header
    logger.info("Read sky image from file: %s" % args.infile)

    L = args.Router
    Louter = L - args.Rinner
    Mouter = Louter / L
    logger.info("Create Tukey window: Rin=%d, Rout=%d" %
                (args.Rinner, args.Router))
    w1d = tukey(L, Minner=0, Mouter=Mouter)
    x1d = np.arange(len(w1d))
    logger.info("Make 2D Tukey window")
    w2d = make2d(w1d, x1d)

    xw = w2d.shape[0]
    ximg = image.shape[0]
    if xw == ximg:
        taper = w2d
    elif xw > ximg:
        i1 = int((xw-ximg) / 2)
        i2 = int((xw+ximg) / 2)
        taper = w2d[i1:i2, i1:i2]
    else:
        taper = np.zeros(shape=image.shape)
        i1 = int((ximg-xw) / 2)
        i2 = int((ximg+xw) / 2)
        taper[i1:i2, i1:i2] = w2d

    logger.info("Taper the sky image ...")
    image *= taper

    header["DATE"] = (datetime.now(timezone.utc).astimezone().isoformat(),
                      "File creation date")
    header.add_history(" ".join(sys.argv))
    hdu = fits.PrimaryHDU(data=image, header=header)
    hdu.writeto(args.outfile)
    logger.info("Wrote tapered sky image to file: %s" % args.outfile)

    if args.outfile_taper:
        header["OBJECT"] = ("Tukey Window", "Taper window")
        header["Rinner"] = (args.Rinner, "[pixel] inner radius")
        header["Router"] = (args.Router, "[pixel] outer radius")
        hdu = fits.PrimaryHDU(data=taper, header=header)
        hdu.writeto(args.outfile_taper)
        logger.info("Wrote tapered to file: %s" % args.outfile_taper)


if __name__ == "__main__":
    main()
