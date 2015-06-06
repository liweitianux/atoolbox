#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# To fitting the given SBP data with the following beta model:
#     s = s0 * pow((1.0+(r/rc)^2), 0.5-3*beta) + c
# And this tool supports the following two requirements for the fitting:
#   (1) ignore the specified number of inner-most data points;
#   (2) ignore the data points whose radius value less than the given value.
#
# Aaron LI
# 2015/05/29
#
# Changelogs:
# 2015/05/30:
#   * Added option '-n' / '--no-radius' to ignore radius less than the
#     given value.
#   * Support read model initial parameter values from input file.
#


from __future__ import print_function, division

import numpy as np
from scipy.optimize import curve_fit

import os
import sys
import getopt
import re


# modes of to cut data
CUT_POINT = 'CUT_POINT'
CUT_RADIUS = 'CUT_RADIUS'


def beta_model(r, s0, rc, beta, c):
    """
    SBP beta model, with a constant background.
    """
    return s0 * np.power((1.0+(r/rc)**2), 0.5-3*beta) + c


def fit_beta_model(xdata, ydata, yerrdata, p0,
        cutmode=CUT_POINT, cutvalue=0):
    """
    Fit the provided data with the beta model.

    Arguments:
        p0: initial values for the parameters of beta model
        cutmode: 'point' / 'radius'; ignore data by number of data points,
            or by radius value less than the given value
        cutvalue: the cut limit

    Return:
        [fitted_par, fitted_cov, fitted_model_value]
    """
    if cutmode == CUT_POINT:
        xdata2 = xdata[cutvalue:]
        ydata2 = ydata[cutvalue:]
        yerrdata2 = yerrdata[cutvalue:]
    elif cutmode == CUT_RADIUS:
        ii = xdata >= cutvalue
        xdata2 = xdata[ii]
        ydata2 = ydata[ii]
        yerrdata2 = yerrdata[ii]
    else:
        raise ValueError('Unknown cut mode: %s' % cutmode)
    par_fit, cov_fit = curve_fit(beta_model, xdata2, ydata2,
            p0=p0, sigma=yerrdata2)
    y_fit = beta_model(xdata, *par_fit)
    return (par_fit, cov_fit, y_fit)


def main():
    # options
    infile = None
    outfilename= None
    cutmode = CUT_POINT # ignore data by number of data points
    cutvalue = 0 # do not ignore any data by default
    # initial values for the four parameters of the beta model
    s0_0 = 1.0e-7
    rc_0 = 10.0
    beta_0 = 0.6
    c_0 = 0.0
    # debug
    verbose = False

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'c:hi:n:o:v',
                ['cut-point=', 'help', 'infile=', 'no-radius=',
                 'outfile=', 'verbose'])
    except getopt.GetoptError as e:
        print(e, file=sys.stderr)
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        elif opt in ('-v', '--verbose'):
            verbose = True
        elif opt in ('-i', '--infile'):
            infile = arg
        elif opt in ('-o', '--outfile'):
            outfilename = arg
        elif opt in ('-c', '--cut-point'):
            cutvalue = int(arg)
            cutmode = CUT_POINT
        elif opt in ('-n', '--no-radius'):
            cutvalue = float(arg)
            cutmode = CUT_RADIUS
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

    # regex to match initial parameter names, blank line, and comment line
    re_blank = re.compile(r'^\s*$')
    re_comment = re.compile(r'^\s*#')
    re_s0 = re.compile(r'^\s*#\s*s0_0\s*[:=]')
    re_rc = re.compile(r'^\s*#\s*rc_0\s*[:=]')
    re_beta = re.compile(r'^\s*#\s*beta_0\s*[:=]')
    re_c = re.compile(r'^\s*#\s*c_0\s*[:=]')
    for line in open(infile, 'r'):
        if re_s0.match(line):
            # read 's0_0': initial value for parameter 's0'
            s0_0 = float(re_s0.split(line)[1])
        elif re_rc.match(line):
            # read 'rc_0': initial value for parameter 'rc'
            rc_0 = float(re_rc.split(line)[1])
        elif re_beta.match(line):
            # read 'beta_0': initial value for parameter 'beta'
            beta_0 = float(re_beta.split(line)[1])
        elif re_c.match(line):
            # read 'c_0': initial value for parameter 'c'
            c_0 = float(re_c.split(line)[1])
        elif re_blank.match(line):
            # ignore blank line
            continue
        elif re_comment.match(line):
            # ignore comment line
            continue
        else:
            try:
                r, rerr, s, serr = map(float, line.split())
            except ValueError:
                try:
                    r, s, serr = map(float, line.split())
                    rerr = 0.0
                except ValueError:
                    print('ERROR: unsupported input data format',
                            file=sys.stderr)
                    sys.exit(21)
            r_data.append(r)
            rerr_data.append(rerr)
            s_data.append(s)
            serr_data.append(serr)

    # convert to numpy array
    r_data = np.array(r_data)
    rerr_data = np.array(rerr_data)
    s_data = np.array(s_data)
    serr_data = np.array(serr_data)

    if verbose:
        print('DEBUG: s0_0 = %g, rc_0 = %g, beta_0 = %g, c_0 = %g' % \
                (s0_0, rc_0, beta_0, c_0), file=sys.stderr)

    par_0 = [s0_0, rc_0, beta_0, c_0]

    par_fit, cov_fit, s_fit = fit_beta_model(r_data, s_data, serr_data,
            p0=par_0, cutmode=cutmode, cutvalue=cutvalue)

    print("# beta-model fitting results:", file=outfile)
    print("# s(r) = s0 * pow((1.0+(r/rc)^2), 0.5-3*beta) + c", file=outfile)
    print("# s0 = %g\n# rc = %g\n# beta = %g\n# c = %g" % tuple(par_fit),
            file=outfile)
    print("# radius(input)  brightness(fitted)", file=outfile)
    for i in range(len(s_fit)):
        print("%g  %g" % (r_data[i], s_fit[i]), file=outfile)

    if outfilename:
        outfile.close()


USAGE = """Usage:
    %(prog)s [ -h -c -o outfile ] -i infile

Required arguments:
    -i, --infile
        input data file with the following *four* or *three* columns:
            r, rerr, s, serr
            r, s, serr
        Note: the initial values for beta model paramters can also be
        provided with the following syntax:
            # s0_0 = ??
            # rc_0 = ??
            # beta_0 = ??
            # c_0 = ??
    -o, --outfile
        output file to store the fitted data
        if not provided, then print results to screen

Optional arguments:
    -h, --help
        print this usage
    -c, --cut-point
        accept an integer number (n)
        ignore the inner-most n data points
    -n, --no-radius
        accept a float number (r)
        ignore the data points whose radius is less than r
""" % { 'prog': os.path.basename(sys.argv[0]) }


def usage():
    print(USAGE)


if __name__ == '__main__':
    main()


