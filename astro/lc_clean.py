#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Aaron LI
# Created: 2016-01-16
# Updated: 2016-01-16
#

"""
Clean the lightcurve by fitting the RATE data with a Gaussian model,
and discard the time bins with RATE beyond [mean-n*sigma, mean+n*sigma].
"""

__version__ = "0.1.0"
__date__ = "2016-01-16"

import sys
import argparse

from astropy.io import fits
import numpy as np


class LightCurve:
    """
    X-ray data light curve class
    """
    def __init__(self, lcfile):
        f = fits.open(lcfile)
        self.lc_data = f[1].data
        self.lc_header = f[1].header
        self.time     = self.lc_data['TIME']
        self.rate     = self.lc_data['RATE']
        self.rate_err = self.lc_data['ERROR']
        self.TSTART   = self.lc_header['TSTART']
        self.TSTOP    = self.lc_header['TSTOP']
        self.TIMEDEL  = self.lc_header['TIMEDEL']
        self.TIMEPIXR = self.lc_header['TIMEPIXR']
        f.close()

    def sigma_clip(self, nsigma=3, maxiter=10):
        """
        Iteratively clip the time bins whose value lie beyond the
        range [mean-n*sigma, mean+n*sigma].
        """
        rate = self.rate
        keep_idx = np.ones(rate.shape, dtype=bool)  # all True's
        keep_num = np.sum(keep_idx)
        keep_num0 = np.inf
        i = 0
        while (keep_num < keep_num0):
            if (i >= maxiter):
                print("WARNING: maximum iteration limit reached",
                        file=sys.stderr)
                break
            keep_num0 = keep_num
            i += 1
            mean  = np.mean(rate[keep_idx])
            sigma = np.std(rate[keep_idx])
            cut_low  = mean - nsigma * sigma
            cut_high = mean + nsigma * sigma
            keep_idx = np.logical_and((rate >= cut_low), (rate <= cut_high))
            keep_num = np.sum(keep_idx)
        # save clip results
        self.niter = i
        self.keep_idx = keep_idx
        self.time_clipped = self.time[keep_idx]
        self.rate_clipped = self.rate[keep_idx]

    def make_gti(self, apply_header=True):
        """
        Make new GTIs (good time intervals) according to the clipped
        time bins.
        """
        frac = 0.01  # TIMEDEL fraction to distingush two time bins
        gti_start = []
        gti_stop  = []
        time_start = self.time_clipped
        time_stop  = time_start + self.TIMEDEL
        # first GTI start time
        gti_start.append(time_start[0])
        for tstart, tstop in zip(time_start[1:], time_stop[:-1]):
            if (np.abs(tstart-tstop) <= frac * self.TIMEDEL):
                # time bin continues
                continue
            else:
                # a new GTI start
                gti_start.append(tstart)
                gti_stop.append(tstop)
        # last GTI stop time
        gti_stop.append(time_stop[-1])
        # convert to numpy array
        gti_start = np.array(gti_start)
        gti_stop  = np.array(gti_stop)
        if apply_header:
            # add TSTART to the time
            gti_start += self.TSTART
            gti_stop  += self.TSTART
        # save results
        self.gti_start = gti_start
        self.gti_stop  = gti_stop

    def write_gti(self, filename=None, header=True):
        """
        Write generated GTIs to file or screen (default)
        """
        if isinstance(filename, str):
            outfile = open(filename, 'w')
        else:
            outfile = sys.stdout
        #
        if header:
            outfile.write('# TSTART\tTSTOP\n')
        outfile.write('\n'.join([ '%s\t%s' % (tstart, tstop) \
                for tstart, tstop in zip(self.gti_start, self.gti_stop) ]))
        #
        if isinstance(filename, str):
            outfile.close()


def main():
    parser = argparse.ArgumentParser(
            description="Clean light curve by sigma clipping")
    parser.add_argument("-V", "--version", action="version",
            version="%(prog)s " + "%s (%s)" % (__version__, __date__))
    parser.add_argument("infile",
            help="input lightcurve file; contains [TIME, RATE] columns")
    parser.add_argument("outfile", nargs='?', default=None,
            help="output text-format GTI file; for XSELECT filter time")
    parser.add_argument("-s", "--nsigma", dest="nsigma", type=float,
            default=2.0, help="sigma clipping significant level")
    parser.add_argument("-H", "--no-header", dest="noheader",
            action="store_true", help="not write header to the output file")
    parser.add_argument("-v", "--verbose", dest="verbose",
            action="store_true", help="show verbose information")
    args = parser.parse_args()

    lc = LightCurve(args.infile)
    lc.sigma_clip(nsigma=args.nsigma)
    lc.make_gti(apply_header=True)
    lc.write_gti(filename=args.outfile, header=(not args.noheader))
    if args.verbose:
        exposure = np.sum(lc.gti_stop - lc.gti_start)
        print("# Total GTI: %.2f (s)" % exposure)


if __name__ == "__main__":
    main()


#  vim: set ts=4 sw=4 tw=0 fenc= ft=python: #
