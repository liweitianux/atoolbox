#!/usr/bin/env python3
#
# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT License
#

"""
Make/simulate the X-ray photon list from the object's image and
spectral models.

The simulated X-ray photon list will be used to simulate the
Suzaku event observation by ``xissim`` tool.

This script is intended to replace and extend the abilities of the
``mkphlist`` tool.

NOTE
----
The environment variable ``HEADAS`` should be set in order to help
locate the ``PyXspec`` module and XSPEC shared libraries.

References
----------
* mkphlist: https://heasarc.gsfc.nasa.gov/lheasoft/ftools/headas/mkphlist.txt
* xissim: https://heasarc.gsfc.nasa.gov/lheasoft/ftools/headas/xissim.txt
* PyXspec: https://heasarc.gsfc.nasa.gov/xanadu/xspec/python/html/index.html

Example Configuration File
-----------------------------------------------------------------------
# image to determine the photon counts distribution
image: imgbox800_e500-7000_sm.fits
# region (annuli below) center; in "image" coordinate
center: [400, 399]
nh: 0.03  # 1e22 [cm^-2]
redshift: 0.0137
# simulated photon energy range [keV]
erange: [0.3, 10.0]
# number of energy bins (logarithmic)
ebins: 1000
# total photon counts that will be generated
counts: 300000
# exposure [ks]
exposure: 50
# a set of annular regions, with several pie regions inside each
# annulus; each pie region can have a different spectral model.
regions:
    # annulus 1, with 3 pies
    - radius: [0, 100]
      angle: [0, 120, 200]
      temperature: [1.0, 1.5, 2.0]
      abundance: [0.5, 1.0, 1.5]
      weight: [1, 2, 1.5]
    # annulus 2, with 3 pies
    - radius: [100, 200]
      angle: [0, 90, 250]
      temperature: [0.5, 1.0, 1.5]
      abundance: [1.5, 2.0, 1.0]
      weight: [0.5, 1, 1.5]
    # annulus 3, with 4 pies
    - radius: [200, 400]
      angle: [50, 150, 220, 300]
      temperature: [0.8, 1.2, 1.5, 1.3]
      abundance: [1.1, 2.0, 1.5, 1.2]
      weight: [0.2, 1.5, 0.7, 2]
clobber: True
outfiles:
    photons_table: photons.fits
    counts_map: counts_map.fits
    temperature_map: temperature_map.fits
    abundance_map: abundance_map.fits
-----------------------------------------------------------------------
"""

import os
import sys

try:
    headas = os.environ["HEADAS"]
    healib = os.path.join(headas, "lib")
except KeyError:
    raise ValueError("env variable 'HEADAS' not set")

if ("LD_LIBRARY_PATH" not in os.environ) or (
        os.environ["LD_LIBRARY_PATH"].find(healib) < 0):
    os.environ["LD_LIBRARY_PATH"] = ":".join([
        healib, os.environ.get("LD_LIBRARY_PATH", "")
    ])
    try:
        # Hack the ``LD_LIBRARY_PATH`` to import Xspec
        # Credit: https://stackoverflow.com/a/25457751/4856091
        print("sys.argv:", sys.argv)
        os.execv(sys.argv[0], sys.argv)
    except Exception:
        print("ERROR: failed to re-exec with new LD_LIBRARY_PATH")
        raise

sys.path.append(os.path.join(healib, "python"))
import xspec
print("Imported XSPEC!")

import argparse
import logging
from pprint import pprint

import yaml
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s:%(lineno)d] %(message)s")
logger = logging.getLogger()


class Pie:
    """
    Pie region
    """
    def __init__(self, xc, yc, rin, rout, abegin, aend):
        self.xc = xc
        self.yc = yc
        self.rin = rin
        self.rout = rout
        self.abegin = abegin  # [deg] beginning angle
        self.aend = aend  # [deg] ending angle (may be > 360)
        # spectral model parameters
        self._modelpars = {}

    @staticmethod
    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = 180 + np.rad2deg(np.arctan2(y, x))  # 0-360 [deg]
        return (rho, phi)

    def make_mask(self, shape):
        try:
            nrow, ncol = shape
        except TypeError:
            nrow = ncol = shape
        # HACK: to make the masks consistent with ``rand_position()``
        ix = self.xc - np.arange(ncol)
        iy = self.yc - np.arange(nrow)
        mx, my = np.meshgrid(ix, iy)
        rho, phi = self.cart2pol(mx, my)
        mask_rho = (rho >= self.rin) & (rho <= self.rout)
        mask_phi = (phi >= self.abegin) & (phi <= self.aend)
        if self.aend > 360:
            mask_phi |= (phi <= (self.aend-360))
        mask = mask_rho & mask_phi
        return mask

    def rand_position(self, n=None):
        if n is None:
            n = self.modelpar("counts")
        theta = np.random.uniform(low=self.abegin, high=self.aend, size=n)
        r = np.sqrt(np.random.uniform(low=self.rin**2, high=self.rout**2,
                                      size=n))
        x = r * np.cos(np.deg2rad(theta)) + self.xc
        y = r * np.sin(np.deg2rad(theta)) + self.yc
        return (x, y)

    def modelpar(self, key=None, value=None):
        if key is None:
            return self._modelpars
        elif value is None:
            return self._modelpars.get(key)
        else:
            self._modelpars[key] = value

    def set_model(self, nh, redshift):
        model = xspec.Model("wabs*apec")
        model.wabs.nH = nh
        model.apec.Redshift = redshift
        model.apec.kT = self.modelpar("temperature")
        model.apec.Abundanc = self.modelpar("abundance")
        self._model = model

    def rand_photons(self, n=None):
        if n is None:
            n = self.modelpar("counts")
        model = self._model
        mvalues = np.array(model.values(0), dtype=float)  # len: ebins
        p = mvalues / mvalues.sum()
        menergies = np.array(model.energies(0), dtype=float)  # len: ebins+1
        mebins = np.sqrt(menergies[1:] * menergies[:-1])
        photons = np.random.choice(mebins, size=n, p=p)
        return photons  # [keV]


class Regions:
    """
    Configured regions
    """
    def __init__(self, configs):
        self.configs = configs
        self.xc, self.yc = configs["center"]

    @property
    def rmax(self):
        rmax = 0
        for annulus in self.configs["regions"]:
            rin, rout = annulus["radius"]
            if rmax < rout:
                rmax = rout
        return rmax

    def make_mask(self, shape):
        try:
            nrow, ncol = shape
        except TypeError:
            nrow = ncol = shape
        ix = np.arange(ncol) - self.xc
        iy = np.arange(nrow) - self.yc
        mx, my = np.meshgrid(ix, iy)
        rho = np.sqrt(mx**2 + my**2)
        mask = (rho <= self.rmax)
        return mask

    @property
    def regions(self):
        reg_all = []
        for annulus in self.configs["regions"]:
            reg_annulus = []
            rin, rout = annulus["radius"]
            abegin = annulus["angle"]
            aend = abegin[1:] + [abegin[0]+360]
            npie = len(abegin)
            temperature = annulus["temperature"]
            abundance = annulus["abundance"]
            weight = annulus.get("weight", [1]*npie)
            for i in range(npie):
                pie = Pie(xc=self.xc, yc=self.yc, rin=rin, rout=rout,
                          abegin=abegin[i], aend=aend[i])
                pie.modelpar("temperature", temperature[i])
                pie.modelpar("abundance", abundance[i])
                pie.modelpar("weight", weight[i])
                reg_annulus.append(pie)
            reg_all.append(reg_annulus)
        return reg_all


def pixel2world(x, y, wcs):
    pix = np.column_stack([x, y])
    world = wcs.wcs_pix2world(pix, 0)
    ra = world[:, 0]
    dec = world[:, 1]
    return (ra, dec)  # [deg]


def main():
    parser = argparse.ArgumentParser(
        description="Make/simulate X-ray photon list for Suzaku simulation")
    parser.add_argument("config", help="configuration file in YAML format")
    args = parser.parse_args()

    configs = yaml.load(open(args.config))
    logger.info("Load configuration file: %s" % args.config)
    logger.info("Configurations:")
    pprint(configs)

    # Update XSPEC settings
    emin, emax = configs["erange"]  # [keV]
    ebins = configs["ebins"]
    xspec.AllModels.setEnergies("%.1f %.1f %d log" % (emin, emax, ebins))
    logger.info("Energy range: [%.1f, %.1f] [keV]" % (emin, emax))
    logger.info("Energy: %d logarithmic channels" % ebins)

    with fits.open(configs["image"]) as f:
        header = f[0].header
        image = f[0].data
    shape = image.shape
    logger.info("Image size: %dx%d" % (shape[1], shape[0]))

    wcs = WCS(header)
    regions = Regions(configs)
    reg_all = regions.regions
    mask_all = regions.make_mask(shape=shape)
    weight_all = np.sum(image[mask_all])

    counts_all = configs["counts"]
    logger.info("Total counts: %d" % counts_all)

    logger.info("nH: %.4f [1e22 cm^-2]" % configs["nh"])
    logger.info("Redshift: %.5f" % configs["redshift"])
    exposure = configs["exposure"] * 1e3  # [s]
    logger.info("Exposure time: %.1f [s]" % exposure)

    logger.info("Determining photons counts in each region ...")
    counts_sum = 0
    for i, annulus in enumerate(reg_all):
        for j, pie in enumerate(annulus):
            label = "annu#%d/pie#%d" % (i+1, j+1)
            mask = pie.make_mask(shape=shape)
            pixels = np.sum(mask)
            weight = np.sum(image[mask]) * pie.modelpar("weight")
            counts = int(counts_all * weight / weight_all)
            counts_sum += counts
            pie.modelpar("pixels", pixels)
            pie.modelpar("counts", counts)
            logger.info("%s: %d pixels, %d photons" % (label, pixels, counts))

    logger.info("Determined counts sum: %d" % counts_sum)
    logger.info("Adjusting total counts -> %d" % counts_all)
    for i, annulus in enumerate(reg_all):
        for j, pie in enumerate(annulus):
            label = "annu#%d/pie#%d" % (i+1, j+1)
            counts_old = pie.modelpar("counts")
            counts_new = round(counts_old * counts_all / counts_sum)
            pie.modelpar("counts", counts_new)
            logger.info("%s: adjusted photon counts: %d -> %d" %
                        (label, counts_old, counts_new))

    # Output files
    temp_map = np.zeros_like(image)
    abund_map = np.zeros_like(image)
    counts_map = np.zeros_like(image)
    weights_map = np.zeros_like(image)
    photonlist = []

    for i, annulus in enumerate(reg_all):
        for j, pie in enumerate(annulus):
            label = "annu#%d/pie#%d" % (i+1, j+1)
            pie.set_model(nh=configs["nh"], redshift=configs["redshift"])
            mask = pie.make_mask(shape=shape)
            temp = pie.modelpar("temperature")
            abund = pie.modelpar("abundance")
            counts = pie.modelpar("counts")
            logger.info("%s: kT=%.2f, Z=%.2f, %d photons" %
                        (label, temp, abund, counts))

            logger.info("%s: sampling photon positions ..." % label)
            x, y = pie.rand_position(n=counts)
            ra, dec = pixel2world(x, y, wcs=wcs)
            logger.info("%s: sampling photon energies ..." % label)
            energies = pie.rand_photons(n=counts)
            time = np.random.uniform(low=0, high=exposure, size=counts)
            photons = np.column_stack([time, energies, ra, dec])
            photonlist.append(photons)

            logger.info("%s: spatially binning photons ..." % label)
            rbins = np.arange(shape[0]+1, dtype=int)
            cbins = np.arange(shape[1]+1, dtype=int)
            hist2d, __, __ = np.histogram2d(y, x, bins=(rbins, cbins))
            counts_map += hist2d

            temp_map[mask] = temp
            abund_map[mask] = abund
            weights_map[mask] = pie.modelpar("weight")

    logger.info("Creating output FITS header ...")
    header_out = fits.Header()
    header_out.extend(wcs.to_header(), update=True)
    header_out["CREATOR"] = os.path.basename(sys.argv[0])
    header_out.add_history(" ".join(sys.argv))
    logger.info("Creating photons table ...")
    photons = np.row_stack(photonlist)
    photons.sort(axis=0)  # sort by time in place
    hdu = fits.BinTableHDU.from_columns([
        fits.Column(name="PHOTON_TIME", format="D", unit="s",
                    array=photons[:, 0]),
        fits.Column(name="PHOTON_ENERGY", format="E", unit="keV",
                    array=photons[:, 1]),
        fits.Column(name="RA", format="E", unit="deg", array=photons[:, 2]),
        fits.Column(name="DEC", format="E", unit="deg", array=photons[:, 3]),
    ], header=header_out)
    hdu.name = "PHOTON_LIST"
    outfile = configs["outfiles"]["photons_table"]
    hdu.writeto(outfile, overwrite=configs["clobber"])
    logger.info("Wrote photons table to: %s" % outfile)

    data = np.stack([counts_map, weights_map], axis=0)
    hdu = fits.PrimaryHDU(data=data, header=header_out)
    outfile = configs["outfiles"]["counts_map"]
    hdu.writeto(outfile, overwrite=configs["clobber"])
    logger.info("Wrote counts/weights map to: %s" % outfile)
    #
    hdu = fits.PrimaryHDU(data=temp_map, header=header_out)
    outfile = configs["outfiles"]["temperature_map"]
    hdu.writeto(outfile, overwrite=configs["clobber"])
    logger.info("Wrote temperature map to: %s" % outfile)
    #
    hdu = fits.PrimaryHDU(data=abund_map, header=header_out)
    outfile = configs["outfiles"]["abundance_map"]
    hdu.writeto(outfile, overwrite=configs["clobber"])
    logger.info("Wrote abundance map to: %s" % outfile)


if __name__ == "__main__":
    main()
