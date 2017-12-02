#!/usr/bin/env python3
#
# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT License
#

"""
Fit the spectral-smooth foreground along the frequency axis (i.e.,
line of sight for the 21 cm signal) using a polynomial in the linear
scale for a series of simulated images (e.g., made by WSClean), then
subtract the fitted smooth component to remove/reduce the foreground
contamination.

References
----------
* Liu, Tegmark & Zaldarriaga 2009, MNRAS, 394, 1575
"""

import os
import sys
import argparse
import time
import logging

import numpy as np
from astropy.io import fits


logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s:%(lineno)d] %(message)s")
logger = logging.getLogger()


def open_image(infile):
    """
    Open the slice image and return its header and 2D image data.

    NOTE
    ----
    The input slice image may have following dimensions:
    * NAXIS=2: [Y, X]
    * NAXIS=3: [FREQ=1, Y, X]
    * NAXIS=4: [STOKES=1, FREQ=1, Y, X]

    NOTE
    ----
    Only open slice image that has only ONE frequency and ONE Stokes
    parameter.

    Returns
    -------
    header : `~astropy.io.fits.Header`
    image : 2D `~numpy.ndarray`
        The 2D [Y, X] image part of the slice image.
    """
    with fits.open(infile) as f:
        header = f[0].header
        data = f[0].data
    if data.ndim == 2:
        # NAXIS=2: [Y, X]
        image = data
    elif data.ndim == 3 and data.shape[0] == 1:
        # NAXIS=3: [FREQ=1, Y, X]
        image = data[0, :, :]
    elif data.ndim == 4 and data.shape[0] == 1 and data.shape[1] == 1:
        # NAXIS=4: [STOKES=1, FREQ=1, Y, X]
        image = data[0, 0, :, :]
    else:
        raise ValueError("Slice '{0}' has invalid dimensions: {1}".format(
            infile, data.shape))
    return (header, image)


def get_frequency(header):
    freq = None
    try:
        freq = header["FREQ"]  # [MHz]
    except KeyError:
        try:
            ctype3 = header["CTYPE3"]
            if ctype3 == "FREQ":
                freq = header["CRVAL3"] / 1e6  # [MHz]
        except KeyError:
            pass
    return freq


def main():
    parser = argparse.ArgumentParser(
        description="Subtract foreground through polynomial fitting")
    parser.add_argument("-C", "--clobber", dest="clobber", action="store_true",
                        help="overwrite existing output file")
    parser.add_argument("-p", "--poly-order", dest="poly_order",
                        type=int, default=2,
                        help="order of polynomial used for fitting " +
                        "(default: 2, i.e., quadratic)")
    parser.add_argument("-o", "--outdir", dest="outdir", required=True,
                        help="output directory to store the subtracted images")
    parser.add_argument("-i", "--infiles", dest="infiles", nargs="+",
                        help="input images slices (in order)")
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
        logger.info("Created output directory: %s" % args.outdir)
    for infile in args.infiles:
        outfile = os.path.join(args.outdir, os.path.basename(infile))
        if os.path.exists(outfile):
            if args.clobber:
                os.remove(outfile)
                logger.warning("Removed existing output file: %s" % outfile)
            else:
                raise OSError("Output file already exists: %s" % outfile)

    nfiles = len(args.infiles)
    logger.info("Number of images: %d" % nfiles)
    headers = []
    images = []
    freqs = np.zeros(nfiles)
    for i, infile in enumerate(args.infiles):
        header, image = open_image(infile)
        headers.append(header)
        images.append(image)
        freq = get_frequency(header)
        if freq is None:
            raise ValueError("no frequency for image: %s" % infile)
        freqs[i] = freq
        logger.info("Loaded slice #%d: %s @ %.2f [MHz]" % (i+1, infile, freq))

    cube = np.stack(images)
    nz, ny, nx = cube.shape
    logger.info("Image cube dimensions: %dx%d * %d (slices)" % (nx, ny, nz))
    cubeout = cube.copy()
    npix = nx * ny
    logger.info("Polynomial fitting order: %d" % args.poly_order)
    t1 = time.perf_counter()
    for i in range(npix):
        if (i+1) % 10000 == 0:
            t2 = time.perf_counter()
            telapsed = (t2 - t1) / 60.0
            teta = telapsed * (npix - i) / i
            logger.info("%d/%d [%.2f%%] || elapsed %.1f / ETA %.1f [min] ..." %
                        (i+1, npix, 100*(i+1)/npix, telapsed, teta))
        iy = i // ny
        ix = i % ny
        vlos = cube[:, iy, ix]
        pfit = np.polyfit(freqs, vlos, deg=args.poly_order)
        vfit = np.polyval(pfit, freqs)
        cubeout[:, iy, ix] -= vfit

    logger.info("Done fitting and subtracting foreground!")

    for i, infile in enumerate(args.infiles):
        outfile = os.path.join(args.outdir, os.path.basename(infile))
        image = cubeout[i, :, :]
        header = headers[i]
        header.add_history(" ".join(sys.argv))
        fits.PrimaryHDU(data=image, header=header).writeto(outfile)
        logger.info("Wrote subtracted image slice: %s" % outfile)

    t2 = time.perf_counter()
    telapsed = (t2 - t1) / 60.0
    logger.info("Running time: %.1f [min]" % telapsed)


if __name__ == "__main__":
    main()
