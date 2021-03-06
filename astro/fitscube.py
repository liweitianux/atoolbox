#!/usr/bin/env python3
#
# Copyright (c) 2017-2019 Weitian LI <wt@liwt.net>
# MIT License
#

"""
FITS image cube manipulation tool.

This tool was originally developed to create a FITS image cube from a
series of CT scan slices to help better visualize/examine them in the
sophisticated SAOImage DS9 software.  Each slice in the cube is a CT
image at a position from the CT scan, with the z-axis tracking the slice
positions (equal-distant) in units of, e.g., [cm].

Then this tool was significantly improved to deal with the spectral cube
in radio astronomy, with each slice representing the radio sky at a
certain frequency (channel), so the z-axis records the frequency in
units of [Hz].

For example, we simulate the observed image using OSKAR and WSClean one
frequency channel at a time, then use this tool to combine them into
a spectral cube, from which the 2D and 1D power spectra is derived.

The ``calibrate`` sub-command is used to calibrate the frequency channel
responses to make them spectrally smooth by fitting a low-order polynomial.

The ``corrupt`` sub-command is used to corrupt the frequency channel
responses to simulate that real instrument suffers from calibration
imperfections.
"""

import os
import sys
import re
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

        NOTE: The infiles should be sorted appropriately.
        """
        self.infiles = infiles
        self.zbegin = zbegin
        self.zstep = zstep
        N = len(infiles)
        zvalues = zbegin + zstep * np.arange(N)
        header, image = self.open_image(infiles[0])
        height, width = image.shape
        data = np.zeros((N, height, width), dtype=image.dtype)
        for i, (z, fn) in enumerate(zip(zvalues, infiles)):
            print("[%2d/%2d] slice @ %s: %s ..." % (i+1, N, z, fn))
            hdr, img = self.open_image(fn)
            data[i, :, :] = img
        self.data = data
        self.header = header.copy(strip=True)
        print("Cube dimension: %dx%dx%d" % (width, height, N))

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

    def keyword(self, key, value=None, comment=None):
        header = self.header
        if value is None:
            return header[key]
        else:
            header[key] = (value, comment)

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

    def pool(self, blocksize, func=np.mean):
        """
        Down-sampling the images by pooling
        """
        try:
            from skimage.measure import block_reduce
        except ImportError:
            print("scikit-image not installed")
            raise

        self.data = block_reduce(self.data,
                                 block_size=(1, blocksize, blocksize),
                                 func=func)
        self.keyword(key="POOL_BS", value=blocksize,
                     comment="down-sampling block size")
        self.keyword(key="POOL_FUN", value=func.__name__,
                     comment="down-sampling function/method")

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
    nslice = cube.nslice
    print("Data cube unit: %s" % cube.unit)
    print("Image/slice size: %dx%d" % (cube.width, cube.height))
    print("Number of slices: %d" % nslice)
    print("Slice step/spacing: %s%s" % (cube.zstep, pzunit))
    print("Slice positions: %s <-> %s%s" %
          (zvalues.min(), zvalues.max(), pzunit))
    if args.stats:
        mean = np.zeros(nslice)
        std = np.zeros(nslice)
        rms = np.zeros(nslice)
        for i in range(nslice):
            image = cube.get_slice(i, csize=args.center)
            if args.abs:
                image = np.abs(image)
            mean[i] = np.mean(image)
            std[i] = np.std(image)
            rms[i] = np.sqrt(np.mean(image**2))
        print("Slice <z>          <mean>        <std>         <rms>")
        for i, z in enumerate(zvalues):
            print("* %12.4e:  %12.4e  %12.4e  %12.4e" %
                  (z, mean[i], std[i], rms[i]))
        if args.outfile:
            data = np.column_stack([zvalues, mean, std, rms])
            np.savetxt(args.outfile, data, header="z   mean   std   rms")
            print("Saved statistics data to file: %s" % args.outfile)


def cmd_create(args):
    """
    Sub-command: "create", create a FITS cube
    """
    def sorted_natural(l):
        # Credit: https://stackoverflow.com/a/4623518

        def _tryint(s):
            try:
                return int(s)
            except ValueError:
                return s

        def _alphanum_key(s):
            return [_tryint(c) for c in re.split(r'([0-9]+)', s)]

        return sorted(l, key=_alphanum_key)

    if not args.clobber and os.path.exists(args.outfile):
        raise FileExistsError("output file already exists: %s" % args.outfile)

    if args.sort:
        infiles = sorted_natural(args.infiles)
    else:
        infiles = args.infiles

    N = len(infiles)
    print("Input slices:")
    for i, fn in enumerate(infiles):
        print("  + [%2d/%2d] %s" % (i+1, N, fn))

    cube = FITSCube()
    print("Adding slices to cube ...")
    cube.add_slices(infiles, zbegin=args.zbegin, zstep=args.zstep)
    cube.zunit = args.zunit
    if args.unit:
        cube.unit = args.unit

    cube.write(args.outfile, clobber=args.clobber)
    print("Created FITS cube: %s" % args.outfile)


def cmd_crop(args):
    """
    Sub-command: "crop", crop the central region of a FITS cube
    """
    if not args.clobber and os.path.exists(args.outfile):
        raise FileExistsError("output file already exists: %s" % args.outfile)

    cube = FITSCube(args.infile)
    print("Image/slice size: %dx%d" % (cube.width, cube.height))
    print("Cropping region size: %dx%d" % (args.size, args.size))
    s_width = slice((cube.width - args.size) // 2,
                    (cube.width + args.size) // 2)
    s_height = slice((cube.height - args.size) // 2,
                     (cube.height + args.size) // 2)
    cube.data = cube.data[:, s_height, s_width]
    print("Saving FITS cube ...")
    cube.write(args.outfile, clobber=args.clobber)
    print("Created FITS cube: %s" % args.outfile)


def cmd_add(args):
    """
    Sub-command: "add", add two or more FITS cubes
    """
    if not args.clobber and os.path.exists(args.outfile):
        raise FileExistsError("output file already exists: %s" % args.outfile)
    if len(args.infiles) < 2:
        raise RuntimeError("Two or more input FITS cubes required")

    cube = FITSCube(args.infiles[0])
    print("Data cube unit: %s" % cube.unit)
    print("Image/slice size: %dx%d" % (cube.width, cube.height))
    print("Number of slices: %d" % cube.nslice)

    for f in args.infiles[1:]:
        cube2 = FITSCube(f)
        if cube.unit != cube2.unit:
            print("WARNING: data unit mismatch: %s <-> %s" %
                  (cube.unit, cube2.unit))
        if cube.zunit != cube2.zunit:
            print("WARNING: Z-axis unit mismatch: %s <-> %s" %
                  (cube.zunit, cube2.zunit))
        print("Adding cube %s ..." % f)
        cube.data = cube.data + cube2.data

    print("Saving FITS cube ...")
    cube.write(args.outfile, clobber=args.clobber)
    print("Saved FITS cube: %s" % args.outfile)


def cmd_mul(args):
    """
    Sub-command: "mul", multiply two or more FITS cubes
    """
    if not args.clobber and os.path.exists(args.outfile):
        raise FileExistsError("output file already exists: %s" % args.outfile)
    if len(args.infiles) < 2:
        raise RuntimeError("Two or more input FITS cubes required")

    cube = FITSCube(args.infiles[0])
    print("Data cube unit: %s" % cube.unit)
    print("Image/slice size: %dx%d" % (cube.width, cube.height))
    print("Number of slices: %d" % cube.nslice)

    for f in args.infiles[1:]:
        cube2 = FITSCube(f)
        assert (cube.unit, cube.zunit) == (cube2.unit, cube2.zunit)
        print("Multiplying cube %s ..." % f)
        cube.data = cube.data * cube2.data

    print("Saving FITS cube ...")
    cube.write(args.outfile, clobber=args.clobber)
    print("Saved FITS cube: %s" % args.outfile)


def cmd_sub(args):
    """
    Sub-command: "sub", subtract one FITS cube by another one
    """
    if not args.clobber and os.path.exists(args.outfile):
        raise FileExistsError("output file already exists: %s" % args.outfile)

    cube = FITSCube(args.infile)
    print("Data cube unit: %s" % cube.unit)
    print("Image/slice size: %dx%d" % (cube.width, cube.height))
    print("Number of slices: %d" % cube.nslice)

    cube2 = FITSCube(args.infile2)
    assert (cube.unit, cube.zunit) == (cube2.unit, cube2.zunit)
    print("Subtracting cube %s ..." % args.infile2)
    cube.data = cube.data - cube2.data

    print("Saving FITS cube ...")
    cube.write(args.outfile, clobber=args.clobber)
    print("Saved FITS cube: %s" % args.outfile)


def cmd_div(args):
    """
    Sub-command: "div", divide one FITS cube by another one
    """
    if not args.clobber and os.path.exists(args.outfile):
        raise FileExistsError("output file already exists: %s" % args.outfile)

    cube = FITSCube(args.infile)
    print("Data cube unit: %s" % cube.unit)
    print("Image/slice size: %dx%d" % (cube.width, cube.height))
    print("Number of slices: %d" % cube.nslice)

    cube2 = FITSCube(args.infile2)
    assert (cube.unit, cube.zunit) == (cube2.unit, cube2.zunit)
    print("Dividing cube %s ..." % args.infile2)
    with np.errstate(divide='warn'):
        cube.data = cube.data / cube2.data

    if args.fill_value:
        print("Filling invalid data with: %s" % args.fill_value)
        cube.data[~np.isfinite(cube.data)] = float(args.fill_value)

    print("Saving FITS cube ...")
    cube.write(args.outfile, clobber=args.clobber)
    print("Saved FITS cube: %s" % args.outfile)


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

    print("Slice <z>          <mean.old> +/- <std.old>     " +
          "<mean.new>   <gain.coef>")
    for i, z in enumerate(zvalues):
        print("* %12.4e:  %12.4e  %12.4e   %12.4e   %.6f" %
              (z, mean[i], std[i], mean_new[i], coef[i]))

    if args.save_info:
        data = np.column_stack([zvalues, mean, std, mean_new, coef])
        header = [
            "Arguments:",
            "+ center: %s" % args.center,
            "+ abs: %s" % args.abs,
            "+ threshold (percentile): %.2f" % args.threshold,
            "+ polynomial_order: %d" % args.poly_order,
            "",
            "Columns:",
            "1. z/frequency: z-axis position / frequency [%s]" % cube.zunit,
            "2. mean.old: mean before calibration [%s]" % cube.unit,
            "3. std.old: standard deviation before calibration",
            "4. mean.new: mean after calibration",
            "5. gain_coef: calibration coefficient",
            "",
        ]
        infofile = os.path.splitext(args.outfile)[0] + ".txt"
        np.savetxt(infofile, data, header="\n".join(header))
        print("Saved calibration information to file: %s" % infofile)


def cmd_corrupt(args):
    """
    Sub-command: "corrupt", corrupt z-axis slice/channel responses by
    applying random gain coefficients.
    """
    if not args.clobber and os.path.exists(args.outfile):
        raise OSError("output file already exists: %s" % args.outfile)

    cube = FITSCube(args.infile)
    zvalues = cube.zvalues
    print("Data cube unit: %s" % cube.unit)
    print("Image/slice size: %dx%d" % (cube.width, cube.height))
    print("Number of slices: %d" % cube.nslice)

    if args.gaus_sigma is not None:
        print("Gaussian sigma: %.1f%%" % args.gaus_sigma)
        sigma = args.gaus_sigma * 0.01
        gains = np.random.normal(loc=0.0, scale=sigma, size=cube.nslice)
        idx_outliers = np.abs(gains) > 3*sigma
        gains[idx_outliers] = np.sign(gains[idx_outliers]) * (3*sigma)
        gains += 1.0
    else:
        print("Use corruption information from file: %s" % args.infofile)
        args.save_info = False  # ``--info-file`` discards ``--save-info``
        crpdata = np.loadtxt(args.infofile)
        gains = crpdata[:, 1]

    print("Applying slice/channel corruptions ...")
    cube.apply_gain(gains)
    print("Saving corrupted FITS cube ...")
    cube.write(args.outfile, clobber=args.clobber)
    print("Corrupted FITS cube wrote to: %s" % args.outfile)

    print("Slice <z>    <gain.corruption>")
    for z, g in zip(zvalues, gains):
        print("* %12.4e:  %.6f" % (z, g))
    if args.save_info:
        data = np.column_stack([zvalues, gains])
        header = [
            "Arguments:",
            "+ gaus_sigma: %.1f%%" % args.gaus_sigma,
            "",
            "Columns:",
            "1. z/frequency: z-axis position / frequency [%s]" % cube.zunit,
            "2. gain_corruption: corruption coefficient",
            "",
        ]
        infofile = os.path.splitext(args.outfile)[0] + ".txt"
        np.savetxt(infofile, data, header="\n".join(header))
        print("Saved corruption information to file: %s" % infofile)


def cmd_pool(args):
    """
    Sub-command: "pool", down-sample image cube along the spatial dimension.
    """
    if not args.clobber and os.path.exists(args.outfile):
        raise OSError("output file already exists: %s" % args.outfile)

    cube = FITSCube(args.infile)
    print("Data cube unit: %s" % cube.unit)
    print("Image/slice size: %dx%d" % (cube.width, cube.height))
    print("Number of slices: %d" % cube.nslice)

    print("Pooling image cube ...")
    print("block size: %d, method: %s" % (args.blocksize, args.method))
    cube.pool(blocksize=args.blocksize, func=getattr(np, args.method))
    print("Pooled image/slice size: %dx%d" % (cube.width, cube.height))
    print("Saving pooled FITS cube ...")
    cube.write(args.outfile, clobber=args.clobber)
    print("Pooled FITS cube wrote to: %s" % args.outfile)


def cmd_d2f(args):
    """
    Sub-command: "d2f", convert data type from double to float(32).
    """
    if not args.clobber and os.path.exists(args.outfile):
        raise OSError("output file already exists: %s" % args.outfile)

    cube = FITSCube(args.infile)
    print("Data type: %s" % cube.data.dtype)
    print("Converting to float(32) ...")
    cube.data = cube.data.astype(np.float32)
    cube.write(args.outfile, clobber=args.clobber)
    print("FITS cube wrote to: %s" % args.outfile)


def main():
    parser = argparse.ArgumentParser(
        description="FITS image cube manipulation tool")
    subparsers = parser.add_subparsers(dest="subparser_name",
                                       title="sub-commands",
                                       help="additional help")

    # sub-command: "info"
    parser_info = subparsers.add_parser("info", help="show FITS cube info")
    parser_info.add_argument("-a", "--abs", action="store_true",
                             help="take absolute values for image pixels")
    parser_info.add_argument("-c", "--center", type=int,
                             help="crop the central box region of specified " +
                             "size to calculate the statistics")
    parser_info.add_argument("-s", "--stats", action="store_true",
                             help="calculate statistics (mean, std, rms) "
                             "for each slice")
    parser_info.add_argument("-o", "--outfile",
                             help="outfile to save the statistics")
    parser_info.add_argument("infile", help="FITS cube filename")
    parser_info.set_defaults(func=cmd_info)

    # sub-command: "create"
    parser_create = subparsers.add_parser(
        "create", aliases=["new"],
        help="create a FITS cube from a series of images/slices")
    parser_create.add_argument("-C", "--clobber", action="store_true",
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
    parser_create.add_argument("-S", "--sort", action="store_true",
                               help="Sort input files naturally")
    parser_create.add_argument("-o", "--outfile", dest="outfile",
                               required=True,
                               help="output FITS cube filename")
    parser_create.add_argument("-i", "--infiles", dest="infiles",
                               nargs="+", required=True,
                               help="input image slices (in order)")
    parser_create.set_defaults(func=cmd_create)

    # sub-command: "crop"
    parser_crop = subparsers.add_parser(
        "crop",
        help="crop the central spatial region of the FITS cube")
    parser_crop.add_argument("-C", "--clobber", action="store_true",
                             help="overwrite existing output file")
    parser_crop.add_argument("-n", "--size", type=int, required=True,
                             help="crop region size (number of pixels)")
    parser_crop.add_argument("-o", "--outfile", required=True,
                             help="output FITS cube filename")
    parser_crop.add_argument("-i", "--infile", required=True,
                             help="input FITS cube")
    parser_crop.set_defaults(func=cmd_crop)

    # sub-command: "add"
    parser_add = subparsers.add_parser(
        "add",
        help="add two or more FITS cubes")
    parser_add.add_argument("-C", "--clobber", action="store_true",
                            help="overwrite existing output file")
    parser_add.add_argument("-o", "--outfile", required=True,
                            help="output FITS cube filename")
    parser_add.add_argument("-i", "--infiles", nargs="+", required=True,
                            help="two or more input FITS cubes")
    parser_add.set_defaults(func=cmd_add)

    # sub-command: "multiply"
    parser_mul = subparsers.add_parser(
        "mul", aliases=["multiply"],
        help="multiply one FITS cube by another one")
    parser_mul.add_argument("-C", "--clobber", action="store_true",
                            help="overwrite existing output file")
    parser_mul.add_argument("-o", "--outfile", required=True,
                            help="output FITS cube filename")
    parser_mul.add_argument("-i", "--infiles", nargs="+", required=True,
                            help="two or more input FITS cubes")
    parser_mul.set_defaults(func=cmd_mul)

    # sub-command: "subtract"
    parser_sub = subparsers.add_parser(
        "sub", aliases=["subtract"],
        help="subtract one FITS cube by another one")
    parser_sub.add_argument("-C", "--clobber", action="store_true",
                            help="overwrite existing output file")
    parser_sub.add_argument("-o", "--outfile", required=True,
                            help="output FITS cube filename")
    parser_sub.add_argument("-i", "--infile", required=True,
                            help="input FITS cube as the minuend")
    parser_sub.add_argument("-I", "--infile2", required=True,
                            help="another input FITS cube as the subtrahend")
    parser_sub.set_defaults(func=cmd_sub)

    # sub-command: "divide"
    parser_div = subparsers.add_parser(
        "div", aliases=["divide"],
        help="divide one FITS cube by another one")
    parser_div.add_argument("-C", "--clobber", action="store_true",
                            help="overwrite existing output file")
    parser_div.add_argument("-F", "--fill-value", dest="fill_value",
                            help="value to fill the invalid elements")
    parser_div.add_argument("-o", "--outfile", required=True,
                            help="output FITS cube filename")
    parser_div.add_argument("-i", "--infile", required=True,
                            help="input FITS cube as the dividend")
    parser_div.add_argument("-I", "--infile2", required=True,
                            help="another input FITS cube as the divisor")
    parser_div.set_defaults(func=cmd_div)

    # sub-command: "calibrate"
    parser_cal = subparsers.add_parser(
        "cal", aliases=["calibrate"],
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
    parser_cal.add_argument("--save-info", dest="save_info",
                            action="store_true",
                            help="save the calibration information of each " +
                            "channel/slice to a text file")
    parser_cal.set_defaults(func=cmd_calibrate)

    # sub-command: "corrupt"
    parser_crp = subparsers.add_parser(
        "crp", aliases=["corrupt"],
        help="corrupt z-axis slice/channel responses by applying " +
        "random gain coefficients")
    exgrp_crp = parser_crp.add_mutually_exclusive_group(required=True)
    exgrp_crp.add_argument("-G", "--gaus-sigma", dest="gaus_sigma", type=float,
                           help="Gaussian sigma in percent from which " +
                           "random gain coefficients are sampled; " +
                           "specified in percent (e.g., 1 for 1%%)")
    exgrp_crp.add_argument("-I", "--info-file", dest="infofile",
                           help="use the gain coefficients from a " +
                           "(previously saved) corruption information " +
                           "file; will also discard argument --save-info")
    parser_crp.add_argument("-C", "--clobber", dest="clobber",
                            action="store_true",
                            help="overwrite existing output file")
    parser_crp.add_argument("-i", "--infile", dest="infile", required=True,
                            help="input FITS cube filename")
    parser_crp.add_argument("-o", "--outfile", dest="outfile", required=True,
                            help="output corrupted FITS cube")
    parser_crp.add_argument("--save-info", dest="save_info",
                            action="store_true",
                            help="save the corruption information of each " +
                            "channel/slice to a text file")
    parser_crp.set_defaults(func=cmd_corrupt)

    # sub-command: "pool"
    parser_pool = subparsers.add_parser(
        "pool",
        help="down-sample image cube along the spatial dimensions")
    parser_pool.add_argument("-C", "--clobber", dest="clobber",
                             action="store_true",
                             help="overwrite existing output file")
    parser_pool.add_argument("-i", "--infile", required=True,
                             help="input FITS cube filename")
    parser_pool.add_argument("-o", "--outfile", required=True,
                             help="output pooled FITS cube")
    parser_pool.add_argument("-n", "--block-size", dest="blocksize",
                             type=int, required=True,
                             help="down-sampling block size (i.e., factor)")
    parser_pool.add_argument("-m", "--method", default="mean",
                             choices=["mean", "min", "max"],
                             help="down-sampling method (default: mean)")
    parser_pool.set_defaults(func=cmd_pool)

    # sub-command: "d2f"
    parser_d2f = subparsers.add_parser(
        "d2f",
        help="convert data type from double to float(32)")
    parser_d2f.add_argument("-C", "--clobber", action="store_true",
                            help="overwrite existing output file")
    parser_d2f.add_argument("-i", "--infile", required=True,
                            help="input FITS cube filename")
    parser_d2f.add_argument("-o", "--outfile", required=True,
                            help="output converted FITS cube")
    parser_d2f.set_defaults(func=cmd_d2f)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
