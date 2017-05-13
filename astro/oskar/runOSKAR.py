#!/usr/bin/env python3
#
# Copyright (c) 2017 Weitian LI <liweitianux@live.com>
# MIT license
#
# 2017-04-07
#

"""
Run OSKAR to simulate the visibilities from the sky model specified
by a FITS image.


Credits
-------
[1] GitHub: OxfordSKA/OSKAR
    https://github.com/OxfordSKA/OSKAR
[2] GitHub: OxfordSKA/EoR - Emma_files/sim_tidy.py
    https://github.com/OxfordSKA/EoR/blob/master/Emma_files/sim_tidy.py
"""

import os
import sys
import subprocess
import configparser
import argparse
import logging

import numpy as np
import astropy.io.fits as fits
import astropy.units as au
import astropy.constants as ac
from astropy.wcs import WCS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(os.path.basename(sys.argv[0]))


class Settings:
    """
    OSKAR settings manager.
    """
    def __init__(self, infile):
        self.infile = infile
        self.config = configparser.ConfigParser(interpolation=None)
        self.config.read(infile)
        logger.info("Read in configuration file: %s" % infile)
        self.init_oskar_settings()
        self.update_oskar_settings(self.config)

    @property
    def my_settings(self):
        return self.config["my"]

    @property
    def dryrun(self):
        return self.my_settings.getboolean("dryrun", fallback=False)

    @property
    def clobber(self):
        return self.my_settings.getboolean("clobber", fallback=False)

    @property
    def quiet(self):
        return self.my_settings.getboolean("quiet", fallback=False)

    @property
    def oskar_bin(self):
        oskar = self.my_settings.get("oskar_bin",
                                     fallback="oskar_sim_interferometer")
        return os.path.expanduser(oskar)

    @property
    def output_settings_fn(self):
        """
        String format pattern for the output OSKAR settings file.
        """
        default = "settings/sim_interferometer_{freq:.2f}.ini"
        return self.my_settings.get("output_settings_fn", fallback=default)

    @property
    def output_skymodel_fn(self):
        """
        String format pattern for the output OSKAR sky model file.
        """
        default = "skymodel/skymodel_{freq:.2f}.txt"
        return self.my_settings.get("output_skymodel_fn", fallback=default)

    @property
    def output_skyfits_fn(self):
        """
        String format pattern for the output FITS slice of the sky model.
        """
        default = "skymodel/skymodel_{freq:.2f}.fits"
        return self.my_settings.get("output_skyfits_fn", fallback=default)

    @property
    def output_ms_fn(self):
        """
        String format pattern for the output simulated visibility
        data in MeasurementSet format.
        """
        default = "visibility/visibility_{freq:.2f}.ms"
        return self.my_settings.get("output_ms_fn", fallback=default)

    @property
    def output_vis_fn(self):
        """
        String format pattern for the output simulated visibility
        data in OSKAR binary format.
        """
        default = "visibility/visibility_{freq:.2f}.oskar"
        return self.my_settings.get("output_vis_fn", fallback=default)

    @property
    def telescope_model(self):
        """
        Telescope model used for visibility simulations.
        """
        return self.my_settings["telescope_model"]

    @property
    def input_cube(self):
        """
        Input FITS spectral cube.
        """
        return self.my_settings["input_cube"]

    @property
    def image_size(self):
        """
        Width/X and height/Y of the input FITS image (unit: pixel)
        """
        size = self.my_settings["image_size"].split(",")
        return (int(size[0]), int(size[1]))

    @property
    def image_pixsize(self):
        """
        Pixel size of the input FITS image (unit: arcsec)
        """
        return self.my_settings.getfloat("image_pixsize")

    @property
    def frequency(self):
        """
        Frequency of the input image. (unit: MHz)

        NOTE: required if the above input FITS file is not a cube, but
              a 2D image.
        """
        return self.my_settings.getfloat("frequency")

    @property
    def bandwidth(self):
        """
        Bandwidth of the input image. (unit: MHz)
        """
        return self.my_settings.getfloat("bandwidth")

    @property
    def ra0(self):
        """
        R.A. of the center of the input sky field.
        unit: deg
        """
        return self.my_settings.getfloat("ra0", fallback=0.0)

    @property
    def dec0(self):
        """
        Dec. of the center of the input sky field.
        unit: deg
        """
        return self.my_settings.getfloat("dec0", fallback=-27.0)

    @property
    def use_gpus(self):
        """
        Whether to GPUs
        """
        return self.my_settings.getboolean("use_gpus", fallback=False)

    @property
    def start_time(self):
        """
        Start time of the simulating observation
        """
        # This default time keeps 'EoR0' region above horizon for 12 hours.
        # SKA EoR0 region: (ra, dec) = (0, -27) [deg]
        default = "2000-01-01T03:30:00.000"
        return self.my_settings.getfloat("start_time", fallback=default)

    @property
    def obs_length(self):
        """
        Observation length of time (unit: s).
        """
        default = 12.0 * 3600  # 12 hours
        return self.my_settings.getfloat("obs_length", fallback=default)

    @property
    def obs_interval(self):
        """
        Observation interval providing the number of time steps in the
        output data (unit: s).
        """
        default = 10.0  # [s]
        return self.my_settings.getfloat("obs_interval", fallback=default)

    @property
    def time_average(self):
        """
        Correlator time-average duration to simulate time-averaging smearing
        (unit: s).
        """
        default = 10.0  # [s]
        return self.my_settings.getfloat("time_average", fallback=default)

    def init_oskar_settings(self):
        """
        Initialize a `ConfigParser` instance with the default settings
        for 'oskar_sim_interferometer'.
        """
        settings = configparser.ConfigParser()
        settings.read_dict({
            "General": {
                "app": "oskar_sim_interferometer",
            },
            "simulator": {
                "use_gpus": self.use_gpus,
                "max_sources_per_chunk": 65536,
                "double_precision": "true",
                "keep_log_file": "true",
            },
            "sky": {
                "advanced/apply_horizon_clip": "false",
            },
            "observation": {
                "phase_centre_ra_deg": self.ra0,
                "phase_centre_dec_deg": self.dec0,
                "start_time_utc": self.start_time,
                "length": self.obs_length,
                "num_time_steps":
                    int(np.ceil(self.obs_length/self.obs_interval)),
                "num_channels": 1,
            },
            "telescope": {
                "input_directory": self.telescope_model,
                "pol_mode": "Scalar",
                "normalise_beams_at_phase_centre": "true",
                "allow_station_beam_duplication": "true",
                "aperture_array/array_pattern/enable": "true",
                "aperture_array/element_pattern/functional_type": "Dipole",
                "aperture_array/element_pattern/dipole_length": 0.5,
                "aperture_array/element_pattern/dipole_length_units":
                    "Wavelengths",
                "station_type": "Aperture array",
            },
            "interferometer": {
                "channel_bandwidth_hz": self.bandwidth * 1e6,
                "time_average_sec": self.time_average,
                "uv_filter_min": "min",
                "uv_filter_max": "max",
                "uv_filter_units": "Wavelengths",
            }
        })
        self.oskar_settings = settings
        logger.info("Initialized 'oskar_settings'")

    def update_oskar_settings(self, config):
        """
        Update the OSKAR settings with the loaded user configurations.
        """
        for section in self.oskar_settings.sections():
            if section in config:
                for key, value in config[section].items():
                    self.oskar_settings[section][key] = value
                    logger.info("oskar_settings: [%s]%s = %s" % (
                        section, key, value))
        logger.info("Updated 'oskar_settings'")

    def write_oskar_settings(self, outfile, clobber=False):
        """
        Write the settings file for 'oskar_sim_interferometer'.
        """
        if os.path.exists(outfile) and (not clobber):
            raise OSError("oskar settings file already exists: " % outfile)
        with open(outfile, "w") as fp:
            # NOTE: OSKAR do NOT like space around '='
            self.oskar_settings.write(fp, space_around_delimiters=False)
        logger.info("Wrote oskar settings file: %s" % outfile)


class SpectralCube:
    """
    Manipulate the FITS spectral cube.

    NOTE: The FITS data as `numpy.ndarray` has the opposite index
          ordering, which likes the Fortran style, i.e., fastest
          changing axis last: data[frequency, y, x]
    """
    def __init__(self, infile):
        self.infile = infile
        with fits.open(infile) as hdulist:
            self.header = hdulist[0].header
            self.cube = hdulist[0].data
        self.wcs = WCS(self.header)
        logger.info("Loaded FITS spectral cube: %s" % infile)
        logger.info("Spectral cube: width=%d, height=%d" %
                    (self.width, self.height))
        if not self.is_cube:
            logger.warning("NOT a spectral cube!")
        else:
            logger.info("Number of frequencies: %d" % self.nfreq)

    @property
    def naxis(self):
        return self.header["NAXIS"]

    @property
    def is_cube(self):
        return self.naxis == 3

    @property
    def width(self):
        """
        Width of the image, i.e., X axis.
        """
        return self.header["NAXIS1"]

    @property
    def height(self):
        """
        Height of the image, i.e., Y axis.
        """
        return self.header["NAXIS2"]

    @property
    def nfreq(self):
        return self.header["NAXIS3"]

    @property
    def frequencies(self):
        """
        Frequencies of this cube. (unit: MHz)
        If the input file is not a cube, then return 'None'.
        """
        if not self.is_cube:
            logger.warning("Input FITS file is not a spectral cube: %s" %
                           self.infile)
            return None

        nfreq = self.nfreq
        pix = np.zeros(shape=(nfreq, self.naxis), dtype=np.int)
        pix[:, -1] = np.arange(nfreq)
        world = self.wcs.wcs_pix2world(pix, 0)
        freqMHz = world[:, -1] / 1e6  # Hz -> MHz
        return freqMHz

    def get_slice(self, nth=0):
        """
        Extract the specified nth frequency slice from the cube.
        """
        if not self.is_cube:
            logger.warning("Input FITS file is not a spectral cube: %s" %
                           self.infile)
            return self.cube
        else:
            return self.cube[nth, :, :]


class SkyModel:
    """
    OSKAR sky model.
    """
    def __init__(self, image, freq, pixsize, ra0, dec0):
        self.image = image  # K (brightness temperature)
        self.freq = freq  # MHz
        self.pixsize = pixsize  # arcsec
        self.ra0 = ra0  # deg
        self.dec0 = dec0  # deg
        logger.info("SkyModel: Loaded image @ %.2f [MHz]" % freq)

    @property
    def wcs(self):
        """
        WCS for the given image assuming the 'SIN' projection.
        """
        shape = self.image.shape
        delta = self.pixsize / 3600.0  # deg
        wcs_ = WCS(naxis=2)
        wcs_.wcs.ctype = ["RA---SIN", "DEC--SIN"]
        wcs_.wcs.crval = np.array([self.ra0, self.dec0])
        wcs_.wcs.crpix = np.array([shape[1], shape[0]]) / 2.0 + 1
        wcs_.wcs.cdelt = np.array([delta, delta])
        return wcs_

    @property
    def fits_header(self):
        header = self.wcs.to_header()
        header["BUNIT"] = ("Jy/pixel", "Brightness unit")
        header["FREQ"] = (self.freq, "Frequency [MHz]")
        header["RA0"] = (self.ra0, "Center R.A. [deg]")
        header["DEC0"] = (self.dec0, "Center Dec. [deg]")
        return header

    @property
    def factor_K2JyPixel(self):
        """
        Conversion factor to convert brightness unit from 'K' to 'Jy/pixel'

        http://www.iram.fr/IRAMFR/IS/IS2002/html_1/node187.html
        """
        pixarea = np.deg2rad(self.pixsize/3600.0) ** 2  # [sr]
        kB = ac.k_B.si.value  # Boltzmann constant [J/K]
        c0 = ac.c.si.value  # speed of light in vacuum [m/s]
        freqHz = self.freq * 1e6  # [Hz]
        factor = 2*kB * 1.0e26 * pixarea * (freqHz/c0)**2
        return factor

    @property
    def ra_dec(self):
        """
        Calculate the (ra, dec) of each image pixel using the above WCS.

        NOTE: axis ordering difference between numpy array and FITS
        """
        shape = self.image.shape
        wcs = self.wcs
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        pix = np.column_stack([x.flatten(), y.flatten()])
        world = wcs.wcs_pix2world(pix, 0)
        ra = world[:, 0].reshape(shape)
        dec = world[:, 1].reshape(shape)
        return (ra, dec)

    @property
    def sky(self):
        """
        OSKAR sky model array converted from the input image.

        Columns
        -------
        ra : (J2000) right ascension (deg)
        dec : (J2000) declination (deg)
        flux : source (Stokes I) flux density (Jy)
        """
        ra, dec = self.ra_dec
        ra = ra.flatten()
        dec = dec.flatten()
        flux = self.image.flatten() * self.factor_K2JyPixel
        mask = flux > 1e-40
        sky_ = np.column_stack([ra[mask], dec[mask], flux[mask]])
        return sky_

    def write_sky_model(self, outfile, clobber=False):
        """
        Write the converted sky model for simulation.
        """
        if os.path.exists(outfile) and (not clobber):
            raise OSError("oskar sky model file already exists: " % outfile)
        sky = self.sky
        header=("Frequency = %.3f [MHz]\n" % self.freq +
                "Pixel size = %.2f arcsec\n" % self.pixsize +
                "RA0 = %.4f [deg]\n" % self.ra0 +
                "Dec0 = %.4f [deg]\n" % self.dec0 +
                "Number of sources = %d\n\n" % len(sky) +
                "R.A.[deg]    Dec.[deg]    flux[Jy]")
        np.savetxt(outfile, sky, fmt='%.10e, %.10e, %.10e', header=header)
        logger.info("Wrote oskar sky model file: %s" % outfile)

    def write_fits(self, outfile, oldheader=None, clobber=False):
        if os.path.exists(outfile) and (not clobber):
            raise OSError("Sky FITS already exists: " % outfile)
        if oldheader is not None:
            header = oldheader
            header.extend(self.fits_header, update=True)
        else:
            header = self.fits_header
        image = self.image * self.factor_K2JyPixel
        hdu = fits.PrimaryHDU(data=image, header=header)
        hdu.writeto(outfile, overwrite=True)
        logger.info("Wrote sky FITS to file: %s" % outfile)


class Oskar:
    """
    Run OSKAR simulations
    """
    def __init__(self, settings):
        self.settings = settings

    def run(self, settingsfile, dryrun=False):
        cmd = [self.settings.oskar_bin]
        if self.settings.quiet:
            cmd += ["--quiet"]
        cmd += [settingsfile]
        logger.info("Running OSKAR simulator: CMD: %s" % " ".join(cmd))
        if dryrun:
            logger.info("Dry run!")
        else:
            subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Run OSKAR to simulate visibilities")
    parser.add_argument("config", help="Configuration file")
    args = parser.parse_args()

    settings = Settings(args.config)
    clobber = settings.clobber
    image_cube = SpectralCube(settings.input_cube)
    frequencies = image_cube.frequencies  # [MHz]
    if frequencies is None:
        frequencies = [settings.frequency]
    logger.info("Number of image slices/frequencies: %d" % len(frequencies))

    for nth, freq in enumerate(frequencies):
        logger.info(">>> Processing #%d/%d image slice @ %.2f [MHz] <<<" %
                    (nth+1, len(frequencies), freq))
        settingsfile = settings.output_settings_fn.format(freq=freq)
        skymodelfile = settings.output_skymodel_fn.format(freq=freq)
        skyfitsfile = settings.output_skyfits_fn.format(freq=freq)
        msfile = settings.output_ms_fn.format(freq=freq)
        visfile = settings.output_vis_fn.format(freq=freq)
        for filepath in [settingsfile, skymodelfile, skyfitsfile,
                         msfile, visfile]:
            dname = os.path.dirname(filepath)
            if not os.path.isdir(dname):
                os.makedirs(dname)

        newconfig = configparser.ConfigParser()
        newconfig.read_dict({
            "sky": {
                "oskar_sky_model/file": skymodelfile,
            },
            "observation": {
                "start_frequency_hz": freq * 1e6,
            },
            "interferometer": {
                "oskar_vis_filename": visfile,
                "ms_filename": msfile,
            },
        })
        settings.update_oskar_settings(newconfig)
        settings.write_oskar_settings(outfile=settingsfile, clobber=clobber)

        image_slice = image_cube.get_slice(nth)
        skymodel = SkyModel(image=image_slice, freq=freq,
                            pixsize=settings.image_pixsize,
                            ra0=settings.ra0, dec0=settings.dec0)
        skymodel.write_sky_model(skymodelfile, clobber=clobber)
        skymodel.write_fits(skyfitsfile, oldheader=image_cube.header,
                            clobber=clobber)

        oskar = Oskar(settings)
        oskar.run(settingsfile, dryrun=settings.dryrun)


if __name__ == '__main__':
    main()
