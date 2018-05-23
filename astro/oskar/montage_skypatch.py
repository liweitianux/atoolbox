#!/usr/bin/env python3
#
# Copyright (c) 2018 Aaron LI <aly@aaronly.me>
# MIT License
#

"""
Montage a list of sky patches (e.g., simulated sky maps of radio halos
of 10x10 deg^2) to create a large map (e.g., hemisphere) for OSKAR
simulation to help investigate the far side confusion noise (FSCN).

Configuration file (YAML format)
--------------------------------
nside: 8  # npix=768, pixsize~7.33[deg]

region:
  center: [0.0, -27.0]  # [deg]
  rmin: 10.0  # [deg]
  rmax: 80.0  # [deg]

threshold:
  min: 0.0001  # [K]
  max: null

frequency:
  type: calc
  start: 154.0  # [MHz]
  stop: 162.0
  step: 0.16

input:
  filename: '{dir}/cluster_{freq:06.2f}.fits'
  dirlist: 'dir.list'  # filename of the directory list file
  pixelsize: 20  # [arcsec]

output:
  filename: 'skymodel/cluster_{freq:06.2f}.osm'
  clobber: true
"""

import os
import argparse
import logging
from functools import lru_cache

import yaml
import numpy as np
import healpy as hp
import astropy.units as au
from astropy.io import fits
from astropy.wcs import WCS


logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s:%(lineno)d] %(message)s')
logger = logging.getLogger()


def get_frequencies(config):
    config_freq = config['frequency']
    if config_freq['type'] == 'custom':
        return np.array(config_freq['frequencies'])
    else:
        start = config_freq['start']
        stop = config_freq['stop']
        step = config_freq['step']
        return np.arange(start, stop+step/2, step)


def central_angle(points, p0):
    """
    Calculate the central angles between the points with respect to the
    reference point (p0) on the sphere.

    Input parameters:
        points: (longitude, latitude) [deg]
            point coordinates, two columns
        p0: (longitude, latitude) [deg]
            coordinate of reference point

    Algorithm:
        (radial, azimuthal, polar): (r, theta, phi)
        central_angle: alpha
        longitude: lambda = theta
        latitude: delta = 90 - phi
        colatitude: phi

        Unit vector:
        \hat{r}_1 = (cos(theta1) sin(phi1), sin(theta1) sin(phi1), cos(phi1))
            = (cos(lambda1) cos(delta1), sin(lambda1) cos(delta1), sin(delta1))
        \hat{r}_2 = (cos(theta2) sin(phi2), sin(theta2) sin(phi2), cos(phi2))
            = (cos(lambda2) cos(delta2), sin(lambda2) cos(delta2), sin(delta2))

        Therefore the angle (alpha) between \hat{r}_1 and \hat{r}_2:
        cos(alpha) = \hat{r}_1 \cdot \hat{r}_2
            = cos(delta1) cos(delta2) cos(lambda1-lambda2)
              + sin(delta1) sin(delta2)

    References:
    [1] Spherical Coordinates - Wolfram MathWorld
        http://mathworld.wolfram.com/SphericalCoordinates.html
        Equation (19)
    [2] Great Circle - Wolfram MathWorld
        http://mathworld.wolfram.com/GreatCircle.html
        Equation (1), (2), (4)
    """
    points = np.deg2rad(points)
    if points.ndim == 1:
        lbd1, delta1 = points[0], points[1]
    elif points.ndim == 2:
        lbd1, delta1 = points[:, 0], points[:, 1]
    else:
        raise ValueError('invalid input points')
    lbd0, delta0 = np.deg2rad(p0)
    dotvalue = (np.cos(delta1) * np.cos(delta0) * np.cos(lbd1-lbd0) +
                np.sin(delta1) * np.sin(delta0))
    alpha = np.arccos(dotvalue)
    return np.rad2deg(alpha)  # [deg]


def get_healpix_coords(nside):
    npix = hp.nside2npix(nside)
    lon, lat = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    return (lon, lat)  # [deg]


class Patches:
    def __init__(self, config):
        config_input = config['input']
        self._filename = config_input['filename']
        self._dirlist = [d.strip()
                         for d in open(config_input['dirlist']).readlines()]

    def get_image(self, freq, i):
        i = i % len(self._dirlist)
        fn = self._filename.format(freq=freq, dir=self._dirlist[i])
        logger.info('Load image: %s' % fn)
        with fits.open(fn) as f:
            return (f[0].data, f[0].header)


class SkyModel:
    def __init__(self, image, freq, pixelsize, p0, min_=1e-4, max_=None):
        self._image = image
        self._freq = freq  # [MHz]
        self._pixelsize = pixelsize  # [arcsec]
        self._p0 = p0  # (lon, lat) [deg]
        self._min = min_  # [K]
        self._max = max_  # [K]

    @property
    def K2JyPixel(self):
        pixarea = (self._pixelsize * au.arcsec) ** 2
        equiv = au.brightness_temperature(pixarea, self._freq*au.MHz)
        return au.K.to(au.Jy, equivalencies=equiv)

    @property
    def mask(self):
        _max = self._max or np.inf
        return ((np.abs(self._image) >= self._min) &
                (np.abs(self._image) <= _max))

    @property
    def wcs(self):
        shape = self._image.shape
        delta = self._pixelsize / 3600.0  # [deg]
        wcs = WCS(naxis=2)
        projection = 'SIN'
        wcs.wcs.ctype = ['RA---'+projection, 'DEC--'+projection]
        wcs.wcs.crval = np.array(self._p0)
        wcs.wcs.crpix = np.array([shape[1], shape[0]]) / 2.0 + 1
        wcs.wcs.cdelt = np.array([-delta, delta])
        return wcs

    @property
    def sky(self):
        shape = self._image.shape
        idx = self.mask.flatten()
        wcs = self.wcs
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        pix = np.column_stack([x.flatten()[idx], y.flatten()[idx]])
        world = wcs.wcs_pix2world(pix, 0)
        ra, dec = world[:, 0], world[:, 1]
        flux = self._image.flatten()[idx] * self.K2JyPixel
        return np.column_stack([ra, dec, flux])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Montage sky patches to make a large OSKAR sky model')
    parser.add_argument('config', help='configuration file in YAML format')
    args = parser.parse_args()

    config = yaml.load(open(args.config))
    patches = Patches(config)
    frequencies = get_frequencies(config)  # [MHz]
    nfreq = len(frequencies)
    pcenter = config['region']['center']  # [deg]
    rmin = config['region']['rmin']  # [deg]
    rmax = config['region']['rmax']  # [deg]
    min_ = config['threshold']['min']  # [K]
    max_ = config['threshold']['max']  # [K]
    pixelsize = config['input']['pixelsize']  # [arcsec]
    clobber = config['output']['clobber']

    nside = config['nside']
    logger.info('Nside: %d' % nside)
    resol = hp.nside2resol(nside, arcmin=True)  # [arcmin]
    logger.info('HEALPix resolution: %.2f [arcmin]' % resol)
    imgsize = int(round(resol * 60 / pixelsize))
    logger.info('Image patch size: %d' % imgsize)
    plon, plat = get_healpix_coords(nside)  # [deg]
    npatch = len(plon)

    for i, freq in enumerate(frequencies):
        logger.info('[%d/%d] %.2f[MHz] ...' % (i+1, nfreq, freq))
        outfile = config['output']['filename'].format(freq=freq)
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        if os.path.exists(outfile) and (not clobber):
            raise FileExistsError(outfile)

        results = []
        jj = 0
        for j, p0 in enumerate(zip(plon, plat)):
            logger.info('[%d/%d|%d] patch @ (%.2f, %.2f)' %
                        (j+1, npatch, jj+1, p0[0], p0[1]))
            if central_angle(p0, pcenter) > rmax:
                logger.info('skip')
                continue

            image, header = patches.get_image(freq, jj)
            skymodel = SkyModel(image[:imgsize, :imgsize],
                                freq, pixelsize=pixelsize,
                                p0=p0, min_=min_, max_=max_)
            sky = skymodel.sky
            points = sky[:, 0:2]  # (lon, lat)
            angles = central_angle(points, pcenter)  # [deg]
            idx = ((angles >= rmin) & (angles <= rmax))
            sky = sky[idx, :]
            logger.info('Source counts: %d' % len(sky))
            results.append(sky)
            jj += 1

        sky = np.row_stack(results)
        logger.info('Total source counts: %d' % len(sky))
        header = ('Frequency = %.2f [MHz]\n' % freq +
                  'Source counts = %d\n\n' % len(sky) +
                  'R.A.[deg]    Dec.[deg]    flux[Jy]')
        logger.info('Writing sky model to file: %s ...' % outfile)
        np.savetxt(outfile, sky, fmt='%.10e, %.10e, %.10e', header=header)
        logger.info("Wrote OSKAR sky model to file: %s" % outfile)
