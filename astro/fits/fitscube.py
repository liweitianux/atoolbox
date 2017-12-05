#!/usr/bin/env python3
#
# Copyright (c) Weitian LI <weitian@aaronly.me>
# MIT license
#

"""
Create FITS image cube from a series of image slices.
"""

import os
import sys
import argparse
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


class FITSCube:
    """
    FITS image cube.
    """
    def __init__(self, infile=None):
        if infile is not None:
            self.load(infile)

    def load(self, infile):
        with fits.open(infile) as f:
            self.data = f[0].data
            self.header = f[0].header
        print("Loaded FITS cube from file: %s" % infile)
        print("Cube dimensions: %dx%dx%d" %
              (self.width, self.height, self.nslice))
        # The Z-axis position of the first slice.
        self.zbegin = self.header["CRVAL3"]
        # The Z-axis step/spacing between slices.
        self.zstep = self.header["CDELT3"]

    def add_slices(self, infiles, zbegin=0.0, zstep=1.0):
        """
        Create a FITS cube from input image slices.
        """
        self.infiles = infiles
        self.zbegin = zbegin
        self.zstep = zstep
        nslice = len(infiles)
        header, image = self.open_image(infiles[0])
        shape = (nslice, ) + image.shape
        data = np.zeros(shape, dtype=image.dtype)
        for i, fn in enumerate(infiles):
            print("[%d/%d] Adding image slice: %s ..." % (i+1, nslice, fn))
            hdr, img = self.open_image(fn)
            data[i, :, :] = img
        self.data = data
        self.header = header.copy(strip=True)
        print("Created FITS cube of dimensions: %dx%dx%d" %
              (self.width, self.height, self.nslice))

    @staticmethod
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

    @property
    def header(self):
        if not hasattr(self, "header_"):
            self.header_ = fits.Header()
        return self.header_

    @header.setter
    def header(self, value):
        self.header_ = value
        for key in ["CTYPE4", "CRPIX4", "CRVAL4", "CDELT4", "CUNIT4"]:
            try:
                del self.header_[key]
            except KeyError:
                pass

    @property
    @lru_cache()
    def wcs(self):
        w = WCS(naxis=3)
        w.wcs.ctype = ["pixel", "pixel", "pixel"]
        w.wcs.crpix = np.array([self.header.get("CRPIX1", 1.0),
                                self.header.get("CRPIX2", 1.0),
                                1.0])
        w.wcs.crval = np.array([self.header.get("CRVAL1", 0.0),
                                self.header.get("CRVAL2", 0.0),
                                self.zbegin])
        w.wcs.cdelt = np.array([self.header.get("CDELT1", 1.0),
                                self.header.get("CDELT2", 1.0),
                                self.zstep])
        return w

    def write(self, outfile, clobber=False):
        header = self.header
        header.extend(self.wcs.to_header(), update=True)
        header["DATE"] = (datetime.now(timezone.utc).astimezone().isoformat(),
                          "File creation date")
        header.add_history(" ".join(sys.argv))
        hdu = fits.PrimaryHDU(data=self.data, header=header)
        try:
            hdu.writeto(outfile, overwrite=clobber)
        except TypeError:
            hdu.writeto(outfile, clobber=clobber)

    @property
    def width(self):
        __, __, w = self.data.shape
        return w

    @property
    def height(self):
        __, h, __ = self.data.shape
        return h

    @property
    def nslice(self):
        ns, __, __ = self.data.shape
        return ns

    @property
    @lru_cache()
    def zvalues(self):
        """
        Calculate the Z-axis positions for all slices
        """
        nslice = self.nslice
        wcs = self.wcs
        pix = np.zeros(shape=(nslice, 3), dtype=int)
        pix[:, 2] = np.arange(nslice)
        world = wcs.wcs_pix2world(pix, 0)
        return world[:, 2]

    @property
    def slices(self):
        """
        A list of slices in the cube w.r.t. ``zvalues``.
        """
        return (self.data[i, :, :] for i in range(self.nslice))

    def get_slice(self, i, csize=None):
        """
        Get the i-th (0-based) slice image, and crop out the central box
        of size ``csize`` if specified.
        """
        if csize is None:
            return self.data[i, :, :]
        else:
            rows, cols = self.height, self.width
            rc, cc = rows//2, cols//2
            cs1, cs2 = csize//2, (csize+1)//2
            return self.data[i, (rc-cs1):(rc+cs2), (cc-cs1):(cc+cs2)]

    def apply_gain(self, gain):
        """
        Multiply the supplied ``gain`` to each slice, to achieve slice
        or channel response calibration or corruption.
        """
        gain = np.asarray(gain)
        self.data *= gain[:, np.newaxis, np.newaxis]

    @property
    def unit(self):
        """
        Cube data unit.
        """
        return self.header.get("BUNIT")

    @unit.setter
    def unit(self, value):
        self.header["BUNIT"] = value

    @property
    def zunit(self):
        """
        Unit of the slice z-axis positions.
        """
        return self.header.get("CUNIT3")

    @zunit.setter
    def zunit(self, value):
        self.header["CUNIT3"] = value


def cmd_info(args):
    """
    Sub-command: "info", show FITS cube information
    """
    cube = FITSCube(args.infile)
    if cube.zunit:
        pzunit = " [%s]" % cube.zunit
    else:
        pzunit = ""
    zvalues = cube.zvalues
    print("Data cube unit: %s" % cube.unit)
    print("Image/slice size: %dx%d" % (cube.width, cube.height))
    print("Number of slices: %d" % cube.nslice)
    print("Slice step/spacing: %s%s" % (cube.zstep, pzunit))
    print("Slice positions: %s <-> %s%s" %
          (zvalues.min(), zvalues.max(), pzunit))
    if args.meanstd:
        mean = np.zeros(cube.nslice)
        std = np.zeros(cube.nslice)
        for i in range(cube.nslice):
            image = cube.get_slice(i, csize=args.center)
            if args.abs:
                image = np.abs(image)
            mean[i] = np.mean(image)
            std[i] = np.std(image)
        print("Slice <z>         <mean> +/- <std>:")
        for i, z in enumerate(zvalues):
            print("* %12.4e:  %-12.4e  %-12.4e" % (z, mean[i], std[i]))
        if args.outfile:
            data = np.column_stack([zvalues, mean, std])
            np.savetxt(args.outfile, data, header="z   mean   std")
            print("Saved mean/std data to file: %s" % args.outfile)


def cmd_create(args):
    """
    Sub-command: "create", create a FITS cube
    """
    if not args.clobber and os.path.exists(args.outfile):
        raise FileExistsError("output file already exists: %s" % args.outfile)
    cube = FITSCube()
    cube.add_slices(args.infiles, zbegin=args.zbegin, zstep=args.zstep)
    cube.zunit = args.zunit
    if args.unit:
        cube.unit = args.unit
    cube.write(args.outfile, clobber=args.clobber)
    print("Created FITS cube: %s" % args.outfile)


def cmd_calibrate(args):
    """
    Sub-command: "calibrate", calibrate the z-axis slice/channel responses
    by fitting a polynomial.
    """
    if not args.dryrun:
        if args.outfile is None:
            raise ValueError("--outfile required")
        elif not args.clobber and os.path.exists(args.outfile):
            raise OSError("output file already exists: %s" % args.outfile)

    cube = FITSCube(args.infile)
    zvalues = cube.zvalues
    print("Data cube unit: %s" % cube.unit)
    print("Image/slice size: %dx%d" % (cube.width, cube.height))
    print("Number of slices: %d" % cube.nslice)
    mean = np.zeros(cube.nslice)
    std = np.zeros(cube.nslice)
    for i in range(cube.nslice):
        image = cube.get_slice(i, csize=args.center)
        if args.abs:
            image = np.abs(image)
        threshold = np.percentile(image, q=100*args.threshold)
        data = image[image >= threshold]
        mean[i] = np.mean(data)
        std[i] = np.std(data)
    print("Fitting polynomial order: %d" % args.poly_order)
    weights = 1.0 / std
    pfit = np.polyfit(zvalues, mean, w=weights, deg=args.poly_order)
    mean_new = np.polyval(pfit, zvalues)
    coef = mean_new / mean

    if args.dryrun:
        print("*** DRY RUN MODE ***")
    else:
        print("Applying slice/channel calibration gains ...")
        cube.apply_gain(coef)
        print("Saving calibrated FITS cube ...")
        cube.write(args.outfile, clobber=args.clobber)
        print("Calibrated FITS cube wrote to: %s" % args.outfile)

    print("Slice <z>        <mean.old> +/- <std.old>     " +
          "<mean.new>     <gain.coef>")
    for i, z in enumerate(zvalues):
        print("* %12.4e:  %-12.4e  %-12.4e   %-12.4e   %.6f" %
              (z, mean[i], std[i], mean_new[i], coef[i]))

    if args.cal_out:
        data = np.column_stack([zvalues, mean, std, mean_new, coef])
        header = [
            "Parameters:",
            "+ center: %s" % args.center,
            "+ abs: %s" % args.abs,
            "+ threshold (percentile): %.2f" % args.threshold,
            "+ polynomial_order: %d" % args.poly_order,
            "",
            "Columns:",
            "+ z/frequency: z-axis position / frequency [%s]" % cube.zunit,
            "+ mean.old: mean before calibration [%s]" % cube.unit,
            "+ std.old: standard deviation before calibration",
            "+ mean.new: mean after calibration",
            "+ gain_coef: calibration coefficient",
            "",
        ]
        np.savetxt(args.cal_out, data, header="\n".join(header))
        print("Saved calibration data to file: %s" % args.cal_out)


def main():
    parser = argparse.ArgumentParser(
        description="Create FITS cube from a series of image slices.")
    subparsers = parser.add_subparsers(dest="subparser_name",
                                       title="sub-commands",
                                       help="additional help")
    # sub-command: "info"
    parser_info = subparsers.add_parser("info", help="show FITS cube info")
    parser_info.add_argument("-c", "--center", dest="center", type=int,
                             help="crop the central box region of specified " +
                             "size to calculate the mean/std.")
    parser_info.add_argument("-m", "--mean-std", dest="meanstd",
                             action="store_true",
                             help="calculate mean+/-std for each slice")
    parser_info.add_argument("-a", "--abs", dest="abs", action="store_true",
                             help="take absolute values for image pixels")
    parser_info.add_argument("-o", "--outfile", dest="outfile",
                             help="outfile to save mean/std values")
    parser_info.add_argument("infile", help="FITS cube filename")
    parser_info.set_defaults(func=cmd_info)
    # sub-command: "create"
    parser_create = subparsers.add_parser("create", help="create a FITS cube")
    parser_create.add_argument("-C", "--clobber", dest="clobber",
                               action="store_true",
                               help="overwrite existing output file")
    parser_create.add_argument("-U", "--data-unit", dest="unit",
                               help="cube data unit (will overwrite the " +
                               "slice data unit)")
    parser_create.add_argument("-z", "--z-begin", dest="zbegin",
                               type=float, default=0.0,
                               help="Z-axis position of the first slice")
    parser_create.add_argument("-s", "--z-step", dest="zstep",
                               type=float, default=1.0,
                               help="Z-axis step/spacing between slices")
    parser_create.add_argument("-u", "--z-unit", dest="zunit",
                               help="Z-axis unit (e.g., cm, Hz)")
    parser_create.add_argument("-o", "--outfile", dest="outfile",
                               required=True,
                               help="output FITS cube filename")
    parser_create.add_argument("-i", "--infiles", dest="infiles",
                               nargs="+", required=True,
                               help="input image slices (in order)")
    parser_create.set_defaults(func=cmd_create)
    # sub-command: "calibrate"
    parser_cal = subparsers.add_parser(
        "calibrate",
        help="calibrate z-axis slice/channel responses by fitting " +
        "a polynomial")
    parser_cal.add_argument("-n", "--dry-run", dest="dryrun",
                            action="store_true",
                            help="dry run mode")
    parser_cal.add_argument("-C", "--clobber", dest="clobber",
                            action="store_true",
                            help="overwrite existing output file")
    parser_cal.add_argument("-c", "--center", dest="center", type=int,
                            help="crop the central box region of specified " +
                            "size to calculate the mean/std.")
    parser_cal.add_argument("-t", "--threshold", dest="threshold",
                            type=float, default=0.0,
                            help="percentile threshold (0 -> 1) and only " +
                            "considers image pixels with values > threshold " +
                            "to determine the channel/slice responses; " +
                            "(default: 0, i.e., all pixels are accounted for)")
    parser_cal.add_argument("-a", "--abs", dest="abs", action="store_true",
                            help="take absolute values for image pixels")
    parser_cal.add_argument("-p", "--poly-order", dest="poly_order",
                            type=int, default=2,
                            help="order of polynomial used for fitting " +
                            "(default: 2, i.e., quadratic)")
    parser_cal.add_argument("-i", "--infile", dest="infile", required=True,
                            help="input FITS cube filename")
    parser_cal.add_argument("-o", "--outfile", dest="outfile",
                            help="output calibrated FITS cube (optional " +
                            "for dry-run model)")
    parser_cal.add_argument("-O", "--cal-out", dest="cal_out",
                            help="output TXT file to save the calibration " +
                            "coefficients for each channel/slice")
    parser_cal.set_defaults(func=cmd_calibrate)
    #
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
