#!/usr/bin/env python3
#
# Copyright (c) 2017 Weitian LI <liweitianux@live.com>
# MIT license
#
# 2017-04-08
#

"""
Make images from simulated visibilities using CASA 'clean' task.


Credits
-------
[1] CASA: Common Astronomy Software Applications
    https://casa.nrao.edu/
[2] CASA: clean
    https://casa.nrao.edu/docs/TaskRef/clean-task.html
[2] GitHub: OxfordSKA/EoR - EoR_pipeline/make_images.py
    https://github.com/OxfordSKA/EoR/blob/master/EoR_pipeline/make_images.py
"""

import os
import sys
import argparse
import subprocess

try:
    from configparser import ConfigParser
except ImportError:
    # CASA (4.7) ships Python 2.7
    from ConfigParser import ConfigParser


class Settings:
    """
    Manage settings for imaging.
    """
    def __init__(self, infile):
        self.infile = infile
        # Python 2.7's ConfigParser doesn't have parameter 'interpolation'
        config = ConfigParser()
        config.read(infile)

        DEFAULTS = {
            "casa_bin": "casa",
            "output_ms_fn": "visibility/visibility_{freq:.2f}.ms",
            "output_image_rootname": "image/image_{freq:.2f}",
            "clean_niter": '500',  # int
            "clean_gridmode": "widefield",
            "clean_wprojplanes": '256',  # int
            "clean_weighting": "natural",
            "clean_uvrange": "",
        }

        casa_bin = config.get("my", "casa_bin", vars=DEFAULTS)
        self.casa_bin = os.path.expanduser(casa_bin)

        # Width/X and height/Y of the input FITS image (unit: pixel)
        size = config.get("my", "image_size").split(",")
        self.image_width = int(size[0])
        self.image_height = int(size[1])
        self.image_size = (self.image_width, self.image_height)

        # Pixel size of the input FITS image (unit: arcsec)
        self.image_pixsize = config.getfloat("my", "image_pixsize")

        # String format pattern for the output simulated visibility
        # data in MeasurementSet format.
        self.output_ms_fn = config.get("my", "output_ms_fn", vars=DEFAULTS)

        # String format pattern for the output image rootname created
        # from visibility using CASA 'clean' task.
        self.output_image_rootname = config.get(
            "my", "output_image_rootname", vars=DEFAULTS)

        # Number of iteration over which to clean (i.e., deconvolve the
        # dirty image)
        # NOTE: Python 2.7's .getint() not support 'vars' parameter
        self.clean_niter = int(config.get("my", "clean_niter", vars=DEFAULTS))

        # Apply corrections for non-coplanar effects during imaging
        # using the W-Projection algorithm
        self.clean_gridmode = config.get("my", "clean_gridmode",
                                         vars=DEFAULTS)

        # Number of pre-computed w-planes used for the W-Projection
        # algorithm
        self.clean_wprojplanes = int(config.get("my", "clean_wprojplanes",
                                                vars=DEFAULTS))

        # Decides how much weight is given to uv grid points to allow
        # for different sampling densities
        self.clean_weighting = config.get("my", "clean_weighting",
                                          vars=DEFAULTS)

        # Range of baselines to include when generating the image
        self.clean_uvrange = config.get("my", "clean_uvrange", vars=DEFAULTS)


class Imager:
    """
    Imager using CASA 'clean' task to create image from visibility.
    """
    def __init__(self, ms, rootname):
        self.ms = ms
        self.rootname = rootname

    def make_image(self, settings):
        """
        Make image from visibility using 'clean' task.
        """
        default(clean)
        ret = clean(
            vis=self.ms,
            imagename=self.rootname,
            niter=settings.clean_niter,
            gridmode=settings.clean_gridmode,
            wprojplanes=settings.clean_wprojplanes,
            uvrange=settings.clean_uvrange,
            weighting=settings.clean_weighting,
            imsize=[settings.image_width, settings.image_height],
            cell=[settings.image_pixsize, settings.image_pixsize]
        )
        return ret

    def export_fits(self):
        """
        Export create image & psf into FITS.
        """
        for imgtype in ["image", "psf"]:
            imgfile = "%s.%s" % (self.rootname, imgtype)
            fitsfile = imgfile + ".fits"
            exportfits(imagename=imgfile, fitsimage=fitsfile)


def main_casa():
    imgroot = sys.argv[-1]
    msfile = sys.argv[-2]
    configfile = sys.argv[-3]

    settings = Settings(configfile)
    imager = Imager(msfile, imgroot)
    imager.make_image(settings)
    imager.export_fits()


def main():
    parser = argparse.ArgumentParser(
        description="Make images from visibilities using CASA")
    parser.add_argument("config", help="Configuration file")
    parser.add_argument("frequency", type=float,
                        help="frequency slice to imaging [MHz]")
    args = parser.parse_args()
    settings = Settings(args.config)

    msfile = settings.output_ms_fn.format(freq=args.frequency)
    imgroot = settings.output_image_rootname.format(freq=args.frequency)
    dname = os.path.dirname(imgroot)
    if not os.path.isdir(dname):
        os.makedirs(dname)

    cmd = [
        settings.casa_bin, "--nogui", "--nologger", "--log2term",
        "-c", __file__, args.config, msfile, imgroot
    ]
    print("CMD: %s" % " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    progname = os.path.basename(sys.argv[0])
    if progname in ["casa", "casapy", "casapy.py"]:
        main_casa()
    else:
        main()
