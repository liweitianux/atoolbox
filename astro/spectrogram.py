#!/usr/bin/env python3
#
# Copyright (c) 2018 Aaron LI <aly@aaronly.me>
# MIT license
#

"""
Calculate the spectrograms of the FITS cube.

If the discrete Fourier transform (DFT/FFT) performed, output the real,
imaginary, and magnitude spectrograms.  If the discrete cosine transform
(DCT) performed, then output only the magnitude spectrogram since it's
real numbered.

The transform is applied along the frequency axis, while the two spatial
dimensions are flattened.
"""

import sys
import argparse
import logging

import numpy as np
from scipy import signal
from scipy import fftpack
from astropy.io import fits


logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s:%(lineno)d] %(message)s")
logger = logging.getLogger()


class Spectrogram:
    def __init__(self, cube):
        cube = np.array(cube, dtype=float)  # [Freq, Y, X]
        nfreq, ny, nx = cube.shape
        self.data = cube.reshape((nfreq, ny*nx))
        logger.info('Cube: %d frequencies, %d pixels' % self.data.shape)

    @property
    def nfreq(self):
        return self.data.shape[0]

    @property
    def npix(self):
        return self.data.shape[1]

    def set_window(self, window):
        if window.upper() == 'NONE':
            return
        logger.info('Generating %s window ...' % window)
        wfunc = getattr(signal.windows, window)
        w = wfunc(self.nfreq, sym=False)
        logger.info('Applying window ...')
        self.data *= w[:, np.newaxis]

    def calc_cosine(self):
        logger.info('Calculating DCT ...')
        return fftpack.dct(self.data, axis=0, norm='ortho')

    def calc_fourier(self):
        logger.info('Calculating FFT ...')
        z = fftpack.fftshift(fftpack.fft(self.data, axis=0), axis=0)
        return (np.abs(z), np.real(z), np.imag(z))


def main():
    parser = argparse.ArgumentParser(
        description='Calculate spectrogram of an image cube')
    parser.add_argument('-C', '--clobber', action='store_true')
    parser.add_argument('-w', '--window', required=True,
                        choices=['none', 'nuttall', 'hanning'],
                        help='window function')
    parser.add_argument('-t', '--type', required=True,
                        choices=['cosine', 'fourier'],
                        help='spectrogram type')
    parser.add_argument('-i', '--infile', required=True,
                        help='input image cube')
    parser.add_argument('-M', '--out-mag', dest='outmag', required=True,
                        help='output magnitude spectrogram')
    parser.add_argument('-R', '--out-real', dest='outreal',
                        help='output real spectrogram (type "fourier")')
    parser.add_argument('-I', '--out-imag', dest='outimag',
                        help='output imaginary spectrogram (type "fourier")')
    args = parser.parse_args()

    if args.type == 'fourier':
        assert (args.outreal and args.outimag), '--real and --imag required'

    cube = fits.open(args.infile)[0].data
    logger.info("Cube shape: %dx%dx%d" % cube.shape)

    header = fits.Header()
    header.add_history(' '.join(sys.argv))
    header['SP_TYPE'] = (args.type, 'cosine (DCT) or fourier (FFT)')

    spectrogram = Spectrogram(cube)
    spectrogram.set_window(args.window)
    header['SP_WIND'] = (args.window, 'window function along frequency')

    if args.type == 'cosine':
        sp_mag = spectrogram.calc_cosine()
        header['SP_DATA'] = 'magnitude'
        fits.PrimaryHDU(data=sp_mag, header=header).writeto(
            args.outmag, overwrite=args.clobber)
        logger.info('Wrote magnitude spectrogram to: %s' % args.outmag)
    else:
        sp_mag, sp_real, sp_imag = spectrogram.calc_fourier()
        header['SP_DATA'] = 'magnitude'
        fits.PrimaryHDU(data=sp_mag, header=header).writeto(
            args.outmag, overwrite=args.clobber)
        logger.info('Wrote magnitude spectrogram to: %s' % args.outmag)
        header['SP_DATA'] = 'real'
        fits.PrimaryHDU(data=sp_real, header=header).writeto(
            args.outreal, overwrite=args.clobber)
        logger.info('Wrote real spectrogram to: %s' % args.outreal)
        header['SP_DATA'] = 'imaginary'
        fits.PrimaryHDU(data=sp_imag, header=header).writeto(
            args.outimag, overwrite=args.clobber)
        logger.info('Wrote imaginary spectrogram to: %s' % args.outimag)


if __name__ == '__main__':
    main()
