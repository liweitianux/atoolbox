#!casa-script
# -*- mode: python -*-
#
# Copyright (c) 2017 Aaron LI <aly@aaronly.me>
# MIT License
#

"""
Add random Gaussian noises to the MeasurementSet DATA.

NOTE: the input MS is modified in place.
"""

import sys
import argparse

import numpy as np

# from casac.ms import ms


def add_noise(msname, sigma):
    print("Opening MS: %s ..." % msname)
    ms.open(msname, nomodify=False)
    ms.selectinit(datadescid=0)
    print("Geting MS data column ...")
    rec = ms.getdata(["data"])
    data = rec["data"]
    print("data mean: (%g, %g)" % (data.real.mean(), data.imag.mean()))
    print("data std: (%g, %g)" % (data.real.std(), data.imag.std()))
    print("Add Gaussian noises sigma=%g ..." % sigma)
    nreal = np.random.normal(scale=sigma, size=data.shape)
    nimag = np.random.normal(scale=sigma, size=data.shape)
    noise = nreal + 1j*nimag
    data += noise
    rec = {"data": data}
    ms.putdata(rec)
    ms.writehistory("Add Gaussian noises: sigma=%g" % sigma)
    ms.close()
    print("Closed MS: %s" % msname)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Add random noise to MS visibilities")
    parser.add_argument("-s", "--sigma", dest="sigma",
                        required=True, type=float,
                        help="sigma of the random Gaussian noises")
    parser.add_argument("-m", "--ms", dest="ms", required=True,
                        help="input MeasurementSet to be modified")
    args = parser.parse_args(argv)

    add_noise(args.ms, sigma=args.sigma)


if __name__ == "__main__":
    argi = sys.argv.index("--") + 1
    argv = sys.argv[argi:]
    print("argv:", argv)
    main(argv)
