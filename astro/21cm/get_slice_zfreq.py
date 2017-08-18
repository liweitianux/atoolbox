#!/usr/bin/env python3
#
# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT License
#

"""
Get the slices at the specified redshifts/frequencies from the HI
light-cone cube (created by `make_lightcone.py`), and use linear
interpolation.
"""

import sys
import argparse
import logging
from datetime import datetime, timezone

import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM

from z2freq import z2freq, freq2z


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

cosmo = FlatLambdaCDM(H0=71, Om0=0.27)


class LightCone:
    """
    Light-cone cube mimic the observation of HI signal.
    """
    def __init__(self, infile):
        with fits.open(infile) as f:
            self.data = f[0].data
            self.header = f[0].header
        logger.info("Loaded light-cone cube: %dx%d (cells) * %d (slices)" %
                    (self.Nside, self.Nside, self.Nslice))

    @property
    def Nslice(self):
        ns, __, __ = self.data.shape
        return ns

    @property
    def Nside(self):
        return self.header["Nside"]

    @property
    def slices_Dc(self):
        """
        The comoving distances of each slice in the light-cone cube.
        The slices are evenly distributed along the LoS with equal
        comoving step. [Mpc]
        """
        Dc_step = self.header["Dc_step"]
        Dc_min = self.header["Dc_min"]
        Dc = np.array([Dc_min + Dc_step*i for i in range(self.Nslice)])
        return Dc

    def get_slice(self, z):
        Dc = cosmo.comoving_distance(z).value  # [Mpc]
        slices_Dc = self.slices_Dc
        if Dc < slices_Dc.min() or Dc > slices_Dc.max():
            raise ValueError("requested redshift out of range: %.2f" % z)

        i2 = (slices_Dc <= Dc).sum()
        i1 = i2 - 1
        Dc1, s1 = slices_Dc[i1], self.data[i1, :, :]
        Dc2, s2 = slices_Dc[i2], self.data[i2, :, :]
        slope = (s2 - s1) / (Dc2 - Dc1)
        return s1 + slope * (Dc - Dc1)

    def write_slice(self, outfile, data, z, clobber=False):
        freq = z2freq(z)
        Dc = cosmo.comoving_distance(z).value  # [Mpc]
        header = fits.Header()
        header["BUNIT"] = (self.header["BUNIT"],
                           self.header.comments["BUNIT"])
        header["Lside"] = (self.header["Lside"],
                           self.header.comments["Lside"])
        header["Nside"] = (self.header["Nside"],
                           self.header.comments["Lside"])
        header["REDSHIFT"] = (z, "redshift of this slice")
        header["FREQ"] = (freq, "[MHz] observed HI signal frequency")
        header["Dc"] = (Dc, "[cMpc] comoving distance")
        header["DATE"] = (datetime.now(timezone.utc).astimezone().isoformat(),
                          "File creation date")
        header.add_history(" ".join(sys.argv))
        hdu = fits.PrimaryHDU(data=data, header=header)
        try:
            hdu.writeto(outfile, overwrite=clobber)
        except TypeError:
            hdu.writeto(outfile, clobber=clobber)
        logger.info("Wrote slice to file: %s" % outfile)


def main():
    outfile_pattern = "{prefix}_f{freq:06.2f}_z{z:06.3f}.fits"
    outfile_prefix = "deltaTb"

    parser = argparse.ArgumentParser(
        description="Get slices at requested redshifts/frequencies " +
        "from light-cone cube")
    parser.add_argument("-C", "--clobber", dest="clobber",
                        action="store_true",
                        help="overwrite existing files")
    parser.add_argument("-i", "--infile", dest="infile", required=True,
                        help="input light-cone cube")
    parser.add_argument("-o", "--outfile", dest="outfile",
                        default=outfile_pattern,
                        help="output image slice filename pattern FITS " +
                        "(default: %s)" % outfile_pattern)
    parser.add_argument("-p", "--prefix", dest="prefix",
                        default=outfile_prefix,
                        help="prefix of output slices (default: %s)" %
                        outfile_prefix)
    exgrp = parser.add_mutually_exclusive_group(required=True)
    exgrp.add_argument("-z", "--redshifts", dest="redshifts", nargs="+",
                       help="redshifts where to interpolate slices")
    exgrp.add_argument("-f", "--freqs", dest="freqs", nargs="+",
                       help="21cm frequencies [MHz] to interpolate slices")
    args = parser.parse_args()

    if args.redshifts:
        redshifts = [float(z) for z in args.redshifts]
        freqs = z2freq(redshifts, print_=False)
    else:
        freqs = [float(f) for f in args.freqs]
        redshifts = freq2z(freqs, print_=False)

    lightcone = LightCone(args.infile)
    for z, f in zip(redshifts, freqs):
        outfile = args.outfile.format(prefix=args.prefix, z=z, freq=f)
        logger.info("z=%06.3f, freq=%06.2f MHz : %s ..." % (z, f, outfile))
        data = lightcone.get_slice(z)
        lightcone.write_slice(outfile, data=data, z=z, clobber=args.clobber)


if __name__ == "__main__":
    main()
