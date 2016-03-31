#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Squeeze the spectrum according to the grouping specification, then
calculate the statistical errors for each group, and apply error
adjustments (e.g., incorporate the systematic uncertainties).
"""

__version__ = "0.1.0"
__date__ = "2016-01-11"


import sys
import argparse

import numpy as np
from astropy.io import fits


class Spectrum:
    """
    Spectrum class to keep spectrum information and perform manipulations.
    """
    header = None
    channel = None
    counts = None
    grouping = None
    quality = None

    def __init__(self, specfile):
        f = fits.open(specfile)
        spechdu = f['SPECTRUM']
        self.header = spechdu.header
        self.channel = spechdu.data.field('CHANNEL')
        self.counts = spechdu.data.field('COUNTS')
        self.grouping = spechdu.data.field('GROUPING')
        self.quality = spechdu.data.field('QUALITY')
        f.close()

    def squeezeByGrouping(self):
        """
        Squeeze the spectrum according to the grouping specification,
        i.e., sum the counts belonging to the same group, and place the
        sum as the first channel within each group with other channels
        of counts zero's.
        """
        counts_squeezed = []
        cnt_sum = 0
        cnt_num = 0
        first = True
        for grp, cnt in zip(self.grouping, self.counts):
            if first and grp == 1:
                # first group
                cnt_sum = cnt
                cnt_num = 1
                first = False
            elif grp == 1:
                # save previous group
                counts_squeezed.append(cnt_sum)
                counts_squeezed += [ 0 for i in range(cnt_num-1) ]
                # start new group
                cnt_sum = cnt
                cnt_num = 1
            else:
                # group continues
                cnt_sum += cnt
                cnt_num += 1
        # last group
        # save previous group
        counts_squeezed.append(cnt_sum)
        counts_squeezed += [ 0 for i in range(cnt_num-1) ]
        self.counts_squeezed = np.array(counts_squeezed, dtype=np.int32)

    def calcStatErr(self, gehrels=False):
        """
        Calculate the statistical errors for the grouped channels,
        and save as the STAT_ERR column.
        """
        idx_nz = np.nonzero(self.counts_squeezed)
        stat_err = np.zeros(self.counts_squeezed.shape)
        if gehrels:
            # Gehrels
            stat_err[idx_nz] = 1 + np.sqrt(self.counts_squeezed[idx_nz] + 0.75)
        else:
            stat_err[idx_nz] = np.sqrt(self.counts_squeezed[idx_nz])
        self.stat_err = stat_err

    @staticmethod
    def parseSysErr(syserr):
        """
        Parse the string format of syserr supplied in the commandline.
        """
        items = map(str.strip, syserr.split(','))
        syserr_spec = []
        for item in items:
            spec = item.split(':')
            try:
                spec = (int(spec[0]), int(spec[1]), float(spec[2]))
            except:
                raise ValueError("invalid syserr specficiation")
            syserr_spec.append(spec)
        return syserr_spec

    def applySysErr(self, syserr):
        """
        Apply systematic error adjustments to the above calculated
        statistical errors.
        """
        syserr_spec = self.parseSysErr(syserr)
        for lo, hi, se in syserr_spec:
            err_adjusted = self.stat_err[(lo-1):(hi-1)] * np.sqrt(1+se)
        self.stat_err_adjusted = err_adjusted

    def updateHeader(self):
        """
        Update header accordingly.
        """
        # POISSERR
        self.header['POISSERR'] = False

    def write(self, filename, clobber=False):
        """
        Write the updated/modified spectrum block to file.
        """
        channel_col = fits.Column(name='CHANNEL', format='J',
                array=self.channel)
        counts_col = fits.Column(name='COUNTS', format='J',
                array=self.counts_squeezed)
        stat_err_col = fits.Column(name='STAT_ERR', format='D',
                array=self.stat_err_adjusted)
        grouping_col = fits.Column(name='GROUPING', format='I',
                array=self.grouping)
        quality_col = fits.Column(name='QUALITY', format='I',
                array=self.quality)
        spec_cols = fits.ColDefs([channel_col, counts_col, stat_err_col,
                                  grouping_col, quality_col])
        spechdu = fits.BinTableHDU.from_columns(spec_cols, header=self.header)
        spechdu.writeto(filename, clobber=clobber)


def main():
    parser = argparse.ArgumentParser(
            description="Apply systematic error adjustments to spectrum.")
    parser.add_argument("-V", "--version", action="version",
            version="%(prog)s " + "%s (%s)" % (__version__, __date__))
    parser.add_argument("infile", help="input spectrum file")
    parser.add_argument("outfile", help="output adjusted spectrum file")
    parser.add_argument("-e", "--syserr", dest="syserr", required=True,
            help="systematic error specification; " + \
                 "syntax: ch1low:ch1high:syserr1,...")
    parser.add_argument("-C", "--clobber", dest="clobber",
            action="store_true", help="overwrite output file if exists")
    parser.add_argument("-G", "--gehrels", dest="gehrels",
            action="store_true", help="use Gehrels error?")
    args = parser.parse_args()

    spec = Spectrum(args.infile)
    spec.squeezeByGrouping()
    spec.calcStatErr(gehrels=args.gehrels)
    spec.applySysErr(syserr=args.syserr)
    spec.updateHeader()
    spec.write(args.outfile, clobber=args.clobber)


if __name__ == "__main__":
    main()


#  vim: set ts=4 sw=4 tw=0 fenc=utf-8 ft=python: #
