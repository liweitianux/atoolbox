#!/usr/bin/env python3
#
# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT License
#

"""
Make the light-cone cube of HI 21cm signals along the line-of-sight (LoS)
direction from the simulated deltaTb coeval cubes (brightness temperature,
in units of [K]), e.g., by SimFast21.

The HI reionization simulation gives so-called *coeval cubes*, 3D volumes
of density and HI fraction at the same cosmological redshift.  However, an
observer cannot observe these coeval cubes, but they can be used to create
the observable *light-cone* cubes, utilizing the method outlined in
[datta2012],Sec.(2.3).

NOTE
----
The created light-cone cubes constructed in this way *differ* from the
observational ones in that the field of view (FoV) has a constant comoving
size and not a constant angular size.

References
----------
.. [datta2012]
   Datta et al. 2012, MNRAS, 424, 1877
   "Light-cone effect on the reionization 21-cm power spectrum";
   http://adsabs.harvard.edu/abs/2012MNRAS.424.1877D

.. SimFast21: https://github.com/mariogrs/Simfast21


Sample Configuration (YAML format)
---------------------------------------------------------------------------
# Parameters for the flat ΛCDM cosmology
H0: 71.0
Om0: 0.27

# Whether overwrite existing output file? (default: False)
clobber: False
# Data type of the input deltaTb cubes. (default: float32)
dtype: float32
# Unit of the deltaTb cubes. (default: K)
unit: K

# Redshifts of the input cubes (required)
# (zmin, zmax, dz from SimFast21 configuration file)
zmin:
zmax:
dz:
# Simulation cube side length [cMpc] (required)
# (sim_length from SimFast21 configuration file)
Lside:
# Number of cells at each side of the input cubes (required)
# (N_smoothed from SimFast21 configuration file)
Nside:
# Filename pattern of the input coeval cubes
infiles_pattern: "deltaTb_z{z:05.3f}_N{Nside:d}_L{Lside:.1f}.dat"
# Filename of output light-cone cube (required)
outfile: deltaTb_lightcone.fits
---------------------------------------------------------------------------
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timezone

import yaml
import numpy as np
from scipy import interpolate
import astropy.io.fits as fits
from astropy.cosmology import FlatLambdaCDM
from astropy.wcs import WCS


logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s:%(lineno)d] %(message)s")
logger = logging.getLogger()


class Configs:
    def __init__(self, configfile):
        self.filepath = os.path.abspath(configfile)
        self.configs = yaml.load(open(configfile))
        logger.info("Loaded configurations from file: %s" % configfile)

        for item in [("H0", 71.0),
                     ("Om0", 0.27),
                     ("clobber", False),
                     ("dtype", "float32"),
                     ("unit", "K"),
                     "zmin",
                     "zmax",
                     "dz",
                     "Lside",
                     "Nside",
                     "infiles_pattern",
                     "outfile"]:
            if isinstance(item, tuple):
                option, default = item
                setattr(self, option, self.configs.get(option, default))
            else:
                setattr(self, item, self.configs[item])
        logger.info("Set configurations")

        self.cosmo = FlatLambdaCDM(H0=self.H0, Om0=self.Om0)

    @property
    def redshifts(self):
        return np.arange(self.zmin, self.zmax+self.dz/2, step=self.dz)

    def get_infile(self, z):
        data = {"z": z, "Nside": self.Nside, "Lside": self.Lside}
        filename = self.infiles_pattern.format(**data)
        if os.path.exists(filename):
            return filename
        else:
            raise OSError("requested file does not exists: %s" % filename)

    def get_cubepair(self, z):
        """
        Get the consecutive two cubes enclosing the given redshift.
        """
        redshifts = self.redshifts
        i2 = np.sum(redshifts <= z)
        i1 = i2 - 1
        z1 = redshifts[i1]
        cube1 = self.get_infile(z1)
        try:
            z2 = redshifts[i2]
            cube2 = self.get_infile(z2)
        except IndexError:
            z2 = None
            cube2 = None
        return [(z1, cube1), (z2, cube2)]

    @property
    def Dc_limit(self):
        Dc_min, Dc_max = self.cosmo.comoving_distance(
            [self.zmin, self.zmax]).value  # [Mpc]
        return (Dc_min, Dc_max)

    @property
    def Dc_cell(self):
        """
        Comoving size of a cell.
        """
        return self.Lside / self.Nside

    def Dc_to_redshift(self, Dc):
        """
        Calculate the redshift corresponding to the given comoving distance
        (along LoS) by using interpolation.
        """
        if not hasattr(self, "_Dc_interp"):
            Dc_min, Dc_max = self.Dc_limit
            dDc = self.Dc_cell
            N = int((Dc_max - Dc_min) / dDc)
            z_ = np.linspace(self.zmin, self.zmax, num=N)
            Dc_ = self.cosmo.comoving_distance(z_).value  # [Mpc]
            self._Dc_interp = interpolate.interp1d(Dc_, z_, kind="linear")

        return self._Dc_interp(Dc)


class CubePair:
    """
    A pair of (redshift) consecutive coeval cubes.
    """
    z1 = None
    z2 = None
    infile1 = None
    infile2 = None
    cube1 = None
    cube2 = None

    def __init__(self, Nside, dtype="float32"):
        self.Nside = Nside
        self.dtype = np.dtype(dtype)

    def set(self, cubepair):
        (z1, infile1), (z2, infile2) = cubepair
        if self.infile1 != infile1:
            self.infile1 = infile1
            self.z1 = z1
            self.cube1 = self.load(infile1)
            logger.info("Loaded cube1 [z=%.3f] from: %s" % (z1, infile1))
        if self.infile2 != infile2:
            self.infile2 = infile2
            self.z2 = z2
            self.cube2 = self.load(infile2)
            logger.info("Loaded cube2 [z=%.3f] from: %s" % (z2, infile2))

    def load(self, infile):
        cube = np.fromfile(open(infile, "rb"), dtype=self.dtype)
        cube = cube.reshape((self.Nside, self.Nside, self.Nside))
        return cube

    def get_slice(self, idx, z):
        q = idx % self.Nside
        if abs(z-self.z1) < 1e-5:
            # The last slice, and z2/infile2/cube2 is None
            return self.cube1[:, :, q]
        else:
            if self.z2 is None:
                raise RuntimeError("z2/infile2/cube2 missing!")
            # linear interpolation
            s1 = self.cube1[:, :, q]
            s2 = self.cube2[:, :, q]
            slope = (s2 - s1) / (self.z2 - self.z1)
            return s1 + slope * (z - self.z1)


class LightCone:
    """
    Light-cone cube mimic the observation of HI signal.
    """
    def __init__(self, configs):
        self.configs = configs

        Nside = self.configs.Nside
        Nslice = self.Nslice
        self.cube = np.zeros(shape=(Nslice, Nside, Nside),
                             dtype=np.dtype(self.configs.dtype))
        logger.info("Light-cone cube shape: %dx%d (cells) * %d (slices)" %
                    (Nside, Nside, Nslice))

    def set_slice(self, idx, data):
        self.cube[idx, :, :] = data

    @property
    def slices_Dc(self):
        """
        The slices evenly distributed along the LoS representing with
        comoving distances. [Mpc]
        """
        dDc = self.configs.Dc_cell
        Dc_min, Dc_max = self.configs.Dc_limit
        Dc = np.arange(Dc_min, Dc_max, step=dDc)
        return Dc

    @property
    def Nslice(self):
        return len(self.slices_Dc)

    @property
    def wcs(self):
        Dc_min, __ = self.configs.Dc_limit
        w = WCS(naxis=3)
        w.wcs.ctype = ["pixel", "pixel", "pixel"]
        w.wcs.cunit = ["Mpc", "Mpc", "Mpc"]  # comoving
        w.wcs.crpix = np.array([1.0, 1.0, 1.0])
        w.wcs.crval = np.array([0.0, 0.0, Dc_min])
        w.wcs.cdelt = np.array([self.configs.Dc_cell,
                                self.configs.Dc_cell,
                                self.configs.Dc_cell])
        return w

    @property
    def header(self):
        dDc = self.configs.Dc_cell
        Dc_min, Dc_max = self.configs.Dc_limit
        header = fits.Header()
        header["BUNIT"] = (self.configs.unit, "Data unit")
        header["zmin"] = (self.configs.zmin, "HI simulation minimum redshift")
        header["zmax"] = (self.configs.zmax, "HI simulation maximum redshift")
        header["dz"] = (self.configs.dz, "HI simulation redshift step size")
        header["Dc_min"] = (Dc_min, "[cMpc] comoving distance at zmin")
        header["Dc_max"] = (Dc_max, "[cMpc] comoving distance at zmax")
        header["Dc_step"] = (dDc, "[cMpc] comoving distance between slices")
        header["Lside"] = (self.configs.Lside, "[cMpc] Simulation side length")
        header["Nside"] = (self.configs.Nside, "Number of cells at each side")
        header["DATE"] = (datetime.now(timezone.utc).astimezone().isoformat(),
                          "File creation date")
        header.add_history(" ".join(sys.argv))
        header.extend(self.wcs.to_header(), update=True)
        return header

    def write(self, outfile=None, clobber=None):
        if outfile is None:
            outfile = self.configs.outfile
        if clobber is None:
            clobber = self.configs.clobber

        hdu = fits.PrimaryHDU(data=self.cube, header=self.header)
        try:
            hdu.writeto(outfile, overwrite=clobber)
        except TypeError:
            hdu.writeto(outfile, clobber=clobber)
        logger.info("Wrote light-cone cube to: %s" % outfile)


def main():
    parser = argparse.ArgumentParser(
        description="Create light-cone cube from coeval cubes")
    parser.add_argument("config", help="configuration file")
    args = parser.parse_args()
    configs = Configs(args.config)

    if os.path.exists(configs.outfile) and (not configs.clobber):
        raise OSError("output file already exists: %s" % configs.outfile)

    cubepair = CubePair(Nside=configs.Nside, dtype=configs.dtype)
    lightcone = LightCone(configs)
    for idx, Dc in enumerate(lightcone.slices_Dc):
        z = configs.Dc_to_redshift(Dc)
        logger.info("Slice #%d @ z=%.3f / Dc=%.1f[cMpc] ..." % (idx+1, z, Dc))
        pair = configs.get_cubepair(z)
        cubepair.set(pair)
        data = cubepair.get_slice(idx, z)
        lightcone.set_slice(idx, data)
    lightcone.write()

    logger.warning("*******************************************************")
    logger.warning("The implemented method to make the lightcone cube seems")
    logger.warning("WRONG!  The derived results is VERY DIFFERENT to")
    logger.warning("Mesinger et al. (2016, MNRAS, 459, 2342)!  A careful")
    logger.warning("examination and rewrite is necessary!")
    logger.warning("*******************************************************")


if __name__ == "__main__":
    main()
