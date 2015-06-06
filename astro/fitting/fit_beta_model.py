#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Fitting data according to (SBP) beta model.
#
# Aaron LI
# 2015/05/29


import numpy as np
from scipy.optimize import curve_fit

import os
import sys
import getopt
import re


USAGE = """Usage:
    %(prog)s [ -h -c -o outfile ] -i infile

Required arguments:
    -i, --infile
        input data file with the following *four* columns:
            r, rerr, s, serr
    -o, --outfile
        output file to store the fitted data
        if not provided, then print results to screen

Optional arguments:
    -h, --help
        print this usage
    -c, --cut
        accept an integer number
        ignore the inner-most n data points
""" % { 'prog': os.path.basename(sys.argv[0]) }


def usage():
    print(USAGE)


def beta_model(r, s0, rc, beta, c):
    """
    SBP beta model.
    """
    return s0 * np.power((1.0+(r/rc)**2), 0.5-3*beta) + c


def main():
    infile = None
    outfilename= None
    cut = 0

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'c:hi:o:',
                ['cut=', 'help', 'infile=', 'outfile='])
    except getopt.GetoptError as e:
        print(e, file=sys.stderr)
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        elif opt in ('-i', '--infile'):
            infile = arg
        elif opt in ('-o', '--outfile'):
            outfilename = arg
        elif opt in ('-c', '--cut'):
            cut = int(arg)
        else:
            assert False, 'unhandled option'

    if not infile:
        print('ERROR: --infile required', file=sys.stderr)
        sys.exit(11)
    if outfilename:
        outfile = open(outfilename, 'w')
    else:
        outfile = sys.stdout

    # input data list
    r_data = []
    rerr_data = []
    s_data = []
    serr_data = []

    for line in open(infile, 'r'):
        if re.match(r'^\s*#', line):
            continue
        if re.match(r'^\s*$', line):
            continue
        r, rerr, s, serr = map(float, line.split())
        r_data.append(r)
        rerr_data.append(rerr)
        s_data.append(s)
        serr_data.append(serr)

    # convert to numpy array
    r_data = np.array(r_data)
    rerr_data = np.array(rerr_data)
    s_data = np.array(s_data)
    serr_data = np.array(serr_data)

    # initial parameter values for beta model
    s0_0 = 1.0e-7
    rc_0 = 10.0
    beta_0 = 0.6
    c_0 = 0.0
    par_0 = [s0_0, rc_0, beta_0, c_0]

    par_fit, cov_fit = curve_fit(beta_model, r_data[cut:], s_data[cut:],
            p0=par_0, sigma=serr_data[cut:])

    s_fitted = beta_model(r_data, *par_fit)

    print("# beta-model fitting results:", file=outfile)
    print("# s0 * power((1.0+(r/rc)**2), 0.5-3*beta) + c", file=outfile)
    print("# s0 = %g, rc = %g, beta = %g, c = %g" % tuple(par_fit),
            file=outfile)
    print("# radius(input)  brightness(fitted)", file=outfile)
    for i in range(len(s_fitted)):
        print("%g  %g" % (r_data[i], s_fitted[i]), file=outfile)

    if not outfilename:
        outfile.close()


if __name__ == '__main__':
    main()

