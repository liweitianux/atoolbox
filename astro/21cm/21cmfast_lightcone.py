#!/usr/bin/env python3
#
# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT License
#

"""
Convert the 21cmFAST simulated and generated lighttravel cubes into the
FITS format with proper headers that are consistent with the lightcone
created by ``make_lightcone.py``, and therefore ``get_slice_zfreq.py``
can be used to extract needed slices.

The 21cmFAST lighttravel cube is created by the tool
``redshift_interpolate_boxes.c`` shipped with 21cmFAST.

For example, Mesinger et al. (2016) released their simulations at:
http://homepage.sns.it/mesinger/EOS.html
where the *faint galaxies* model is recommended by the authors.

The lighttravel cubes have filenames:
``delta_T_v3_no_halos__zstart*_zend*_FLIPBOXES0_1024_1600Mpc_lighttravel``
data type: float32, little endidian, C ordering
cube dimension: 1024x1024 (XY-spatial), 1024 (LoS/z)
length: 1600 comoving Mpc
unit: mK

.. 21cmFAST: https://github.com/andreimesinger/21cmFAST

.. Mesinger et al. 2016, MNRAS, 459, 2342
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timezone

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.cosmology import FlatLambdaCDM
import astropy.units as au


logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s:%(lineno)d] %(message)s")
logger = logging.getLogger()


class LightCone:
    """
    Light-cone cube mimic the observation of HI signal.

    cube shape:
      * [X, Y, LoS] (axes_swapped = False)
      * [LoS, Y, Y] (axes_swapped = True)
    """
    def __init__(self, cube, zmin, zmax, unit, Lside, cosmo,
                 axes_swapped=False):
        self.cube = cube
        self.zmin = zmin
        self.zmax = zmax
        self.unit = unit
        self.Lside = Lside  # [cMpc] simulation side length
        self.cosmo = cosmo
        self.Dc_min = cosmo.comoving_distance(zmin).value  # [cMpc]
        self.Dc_max = cosmo.comoving_distance(zmax).value  # [cMpc]
        self.axes_swapped = axes_swapped

    def swap_axes(self):
        self.cube = np.swapaxes(self.cube, 0, 2)
        self.axes_swapped = not self.axes_swapped
        logger.info("Axes swapped: %s" % self.axes_swapped)
        if self.axes_swapped:
            logger.info("Cube axes: [LoS, Y, X]")
        else:
            logger.info("Cube axes: [X, Y, LoS]")

    @property
    def Nside(self):
        return self.cube.shape[1]

    @property
    def Nslice(self):
        if self.axes_swapped:
            return self.cube.shape[0]
        else:
            return self.cube.shape[2]

    @property
    def Dc_cell(self):
        return self.Lside / self.Nslice

    @property
    def wcs(self):
        w = WCS(naxis=3)
        w.wcs.ctype = ["pixel", "pixel", "pixel"]
        w.wcs.cunit = ["Mpc", "Mpc", "Mpc"]  # comoving
        w.wcs.crpix = np.array([1.0, 1.0, 1.0])
        w.wcs.crval = np.array([0.0, 0.0, self.Dc_min])
        w.wcs.cdelt = np.array([self.Dc_cell, self.Dc_cell, self.Dc_cell])
        return w

    @property
    def header(self):
        dDc = self.Dc_cell
        header = fits.Header()
        header["BUNIT"] = (str(self.unit), "Data unit")
        header["zmin"] = (self.zmin, "HI simulation minimum redshift")
        header["zmax"] = (self.zmax, "HI simulation maximum redshift")
        header["Dc_min"] = (self.Dc_min, "[cMpc] comoving distance at zmin")
        header["Dc_max"] = (self.Dc_max, "[cMpc] comoving distance at zmax")
        header["Dc_step"] = (dDc, "[cMpc] comoving distance between slices")
        header["Lside"] = (self.Lside, "[cMpc] Simulation side length")
        header["Nside"] = (self.Nside, "Number of cells at each side")
        header["DATE"] = (datetime.now(timezone.utc).astimezone().isoformat(),
                          "File creation date")
        header.add_history(" ".join(sys.argv))
        header.extend(self.wcs.to_header(), update=True)
        return header

    def write(self, outfile, clobber=False):
        hdu = fits.PrimaryHDU(data=self.cube, header=self.header)
        logger.info("Created FITS object, writing to disk ...")
        try:
            hdu.writeto(outfile, overwrite=clobber)
        except TypeError:
            hdu.writeto(outfile, clobber=clobber)
        logger.info("Wrote light-cone cube to: %s" % outfile)


def main():
    parser = argparse.ArgumentParser(
        description="convert 21cmFAST lighttravel cube to FITS lightcone cube")
    parser.add_argument("-C", "--clobber", dest="clobber",
                        action="store_true", help="overwrite existing file")
    parser.add_argument("--data-type", dest="data_type", default="float32",
                        help="input cube data type (default: float32)")
    parser.add_argument("--unit", dest="unit", default="mK",
                        help="input cube data unit (default: mK)")
    parser.add_argument("--unit-out", dest="unit_out", default="K",
                        help="output data unit (default: K)")
    parser.add_argument("--side-length", dest="side_length",
                        type=float, default=1600.0,
                        help="input cube simulation side length [cMpc]")
    parser.add_argument("--dimension", dest="dimension",
                        nargs=3, type=int, default=[1024, 1024, 1024],
                        help="input cube dimension (C ordering assumed)")
    parser.add_argument("--H0", dest="H0", type=float, default=67.8,
                        help="simulation adopted H0 (default: 67.8)")
    parser.add_argument("--omega-m0", dest="Om0", type=float, default=0.308,
                        help="simulation adopted OmegaM0 (default: 0.308)")
    parser.add_argument("--omega-b0", dest="Ob0", type=float, default=0.0484,
                        help="simulation adopted Omegab0 (default: 0.0484)")
    parser.add_argument("-z", "--z-min", dest="zmin", type=float,
                        required=True, help="minimum/beginning redshift")
    parser.add_argument("-Z", "--z-max", dest="zmax", type=float,
                        required=True, help="maximum/end redshift")
    parser.add_argument("infile", help="input 21cmFAST lighttravel cube")
    parser.add_argument("outfile", help="output FITS lightcone cube")
    args = parser.parse_args()

    if os.path.exists(args.outfile):
        if args.clobber:
            os.remove(args.outfile)
            logger.warning("Removed existing output file: %s" % args.outfile)
        else:
            raise OSError("output file already exists: %s" % args.outfile)

    cosmo = FlatLambdaCDM(H0=args.H0, Om0=args.Om0, Ob0=args.Ob0)
    logger.info("Cosmology: {0}".format(cosmo))
    unit_in = au.Unit(args.unit)
    unit_out = au.Unit(args.unit_out)
    logger.info("Unit: %s (input) -> %s (output)" % (unit_in, unit_out))

    cube_in = np.fromfile(args.infile, dtype=np.dtype(args.data_type))
    cube_in = cube_in.reshape(args.dimension)
    logger.info("Loaded lighttravel cube from file: %s" % args.infile)
    logger.info("Data type: {0}; dimension: {1}".format(
        cube_in.dtype, cube_in.shape))

    cube_in *= unit_in.to(unit_out)
    logger.info("Converted unit to: %s" % unit_out)
    lightcone = LightCone(cube_in, zmin=args.zmin, zmax=args.zmax,
                          unit=unit_out, Lside=args.side_length,
                          cosmo=cosmo, axes_swapped=False)
    lightcone.swap_axes()
    lightcone.write(args.outfile)


if __name__ == "__main__":
    main()
