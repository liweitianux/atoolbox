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
# v0.5.0, 2015/06/07, Aaron LI
#   * added 'fit_model_bounds' using 'scipy.optimize.minimize' to
#     perform function minimization with bounds
#   * split the data cut section to function 'cut_data'
#   * added argument 'options' to 'fit_model_bounds'
# v0.4.0, 2015/06/06, Aaron LI
#   * replace getopt with 'argparse'
#   * added 'get_parameter' to process model parameter initial value and bounds
#   * support read parameter bounds from input file
#   * added options '--s0', '--rc', '--beta', '--const' to get paramter
#     initial values and bounds
#   * renamed 'fit_beta_model' to 'fit_model', and added argument 'func' to
#     support other models
# v0.3.0, 2015/06/02, Aaron LI
#   * can output chi-squared and dof values
#   * can output one standard deviation errors on the parameters
# v0.2.0, 2015/05/30:
#   * Added option '-n' / '--no-radius' to ignore radius less than the
#     given value.
#   * Support read model initial parameter values from input file.
#
# TODO:
# * calculate fitting parameter's standard deviation
# * to normalize fitting paramters to be the same order of magnitude
#   for better minimization
#   Ref: http://stackoverflow.com/questions/21369139/normalization-for-optimization-in-python
#


from __future__ import print_function, division

__version__ = "0.5.0"
__date__ = "2015/06/07"

import numpy as np
from scipy.optimize import curve_fit, minimize

import os
import sys
import re
import argparse


# modes of to cut data
CUT_POINT = 'CUT_POINT'
CUT_RADIUS = 'CUT_RADIUS'
# default minimize method
MINIMIZE_METHOD = 'L-BFGS-B'


def beta_model(r, s0, rc, beta, c):
    """
    SBP beta model, with a constant background.
    """
    return s0 * np.power((1.0+(r/rc)**2), 0.5-3*beta) + c


def calc_chisq(func, xdata, ydata, yerrdata, *args):
    """
    Calculate the chi-squared values for the given function according
    to the provided data points.

    Arguments:
        xdata: x values of data points
        ydata: y values of data points
        yerrdata: y standard deviation values
        args: additional arguments for 'func'

    Return:
        chi-squared value
    """
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    yerrdata = np.array(yerrdata)
    return np.sum(((ydata - func(xdata, *args)) / yerrdata) ** 2)


def fit_model(func, xdata, ydata, yerrdata, p0):
    """
    Fit the provided data with the beta model.

    Arguments:
        p0: initial values for the parameters of beta model

    Return:
        (popt, infodict)
        popt: optimal values for the parameters
        infodict:
            * fvec: the function evaluated at the output parameters
            * dof: degree of freedom
            * chisq: chi-squared
            * perr: one standard deviation errors on the parameters
    """
    popt, pcov = curve_fit(func, xdata, ydata, p0=p0, sigma=yerrdata)
    # the function evaluated at the output parameters
    fvec = lambda x: func(x, *popt)
    # degree of freedom
    dof = len(xdata) - len(popt)
    # chi squared
    chisq = np.sum(((ydata - fvec(xdata)) / yerrdata) ** 2)
    # one standard deviation errors on the parameters
    perr = np.sqrt(np.diag(pcov))
    infodict = {
            'fvec': fvec,
            'dof': dof,
            'chisq': chisq,
            'perr': perr
    }
    return (popt, infodict)


def fit_model_bounds(func, xdata, ydata, yerrdata, p0=None, bounds=None,
        options=None):
    """
    Fit the provided data with the beta model.

    Arguments:
        p0: initial values for the parameters of beta model
        bounds: (min, max) pairs for each parameter bound
        options: a dict of solver options (=> minimize: options)

    Return:
        (popt, infodict)
        popt: optimal values for the parameters
        infodict:
            * fvec: the function evaluated at the output parameters
            * dof: degree of freedom
            * chisq: chi-squared
            * perr: one standard deviation errors on the parameters
    """
    # objective function to be minimized, required format of 'f(x, *args)'
    f = lambda x: calc_chisq(func, xdata, ydata, yerrdata, *x)
    # minimize the given function using 'scipy.optimize.minimize' with bounds
    res = minimize(f, p0, method=MINIMIZE_METHOD, bounds=bounds,
            options=options)
    popt = res.x
    print("DEBUG: minimization results:\n", res, file=sys.stderr)

    # check minimization results
    if not res.success:
        print("*** WARNING: minimization exited with error: ***\n" + \
                "*** %s ***" % res.message, file=sys.stderr)

    # the function evaluated at the output parameters
    fvec = lambda x: func(x, *popt)
    # degree of freedom
    dof = len(xdata) - len(popt)
    # chi squared
    chisq = res.fun
    # one standard deviation errors on the parameters
    perr = popt * 0.0 # FIXME
    infodict = {
            'fvec': fvec,
            'dof': dof,
            'chisq': chisq,
            'perr': perr
    }
    return (popt, infodict)


def cut_data(xdata, ydata, yerrdata, cutmode=CUT_POINT, cutvalue=0):
    """
    Cut the given data with the provided cutmode and cutvalue,
    return the cut data.

    Arguments:
        xdata, ydata, yerrdata: input data (x, y, yerr)
        cutmode: 'point' / 'radius'; ignore data by number of data points,
            or by radius value less than the given value
        cutvalue: the cut limit

    Return:
        (xdata_cut, ydata_cut, yerrdata_cut)
    """
    if cutmode == CUT_POINT:
        xdata_cut = xdata[cutvalue:]
        ydata_cut = ydata[cutvalue:]
        yerrdata_cut = yerrdata[cutvalue:]
    elif cutmode == CUT_RADIUS:
        ii = xdata >= cutvalue
        xdata_cut = xdata[ii]
        ydata_cut = ydata[ii]
        yerrdata_cut = yerrdata[ii]
    else:
        raise ValueError('Unknown cut mode: %s' % cutmode)
    return (xdata_cut, ydata_cut, yerrdata_cut)


def get_parameter(pstring):
    """
    Process the parameter string of the following format, and return
    the initial value, lower limit, and upper limit.
        "init_value"
        "init_value lower upper"
        "init_value,lower,upper"
    If want to ignore the lower/upper limit, use 'None' (case-insensitive),
    and the None is returned.
    """
    parameters = pstring.replace(',', ' ').split()
    if len(parameters) == 1:
        init_value = float(parameters[0])
        return (init_value, None, None)
    elif len(parameters) == 3:
        init_value = float(parameters[0])
        if parameters[1].upper() == 'NONE':
            lower_value = None
        else:
            lower_value = float(parameters[1])
        if parameters[2].upper() == 'NONE':
            upper_value = None
        else:
            upper_value = float(parameters[2])
        return (init_value, lower_value, upper_value)
    else:
        raise ValueError('Invalid parameter format: %s' % pstring)


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
    # default bounds for the four parameters
    s0_lower, s0_upper = None, None
    rc_lower, rc_upper = None, None
    beta_lower, beta_upper = None, None
    c_lower, c_upper = None, None

    # parser for command line options and arguments
    parser = argparse.ArgumentParser(
            description="Fitting provided data with the beta model.",
            epilog="Version: %s (%s)" % (__version__, __date__))
    parser.add_argument("-v", "--verbose", dest="verbose",
            action="store_true", default=False,
            help="show verbose/debug information (False)")
    parser.add_argument("-V", "--version", action="version",
            version="%(prog)s " + "%s (%s)" % (__version__, __date__))
    parser.add_argument("-i", "--infile",
            dest="infile", required=True,
            help="""input data file with the following 4 or 3 columns:
                 [radius radius_err brightness brightness_err],
                 [radius brightness brightness_err].
                 Note: The initial values and lower/upper limits
                 for the beta models can also be provided with the
                 following syntax:
                 # s0_0 = init_value, lower_limit, upper_limit
                 # rc_0 = init_value, lower_limit, upper_limit
                 # beta_0 = init_value, lower_limit, upper_limit
                 # c_0 = init_value, lower_limit, upper_limit""")
    parser.add_argument("-o", "--outfile", dest="outfilename",
            help="output file to store the fitted data")
    parser.add_argument("-b", "--bounds", dest="bounds",
            action="store_true", default=False,
            help="whether apply paramenter bounds (False)")
    parser.add_argument("-c", "--cut-point",
            dest="cut_point", metavar="N", type=int, default=0,
            help="number of inner-most data points to be ignored")
    parser.add_argument("-n", "--no-radius",
            dest="cut_radius", metavar="RADIUS", type=float, default=0.0,
            help="ignore data points with smaller radius")
    parser.add_argument("--s0", dest="s0",
            metavar="init_value,lower,upper",
            help="initial value and lower/upper limits for parameter s0. " + \
                 "Use 'none' (case-insensitive) to ignore the limit")
    parser.add_argument("--rc", dest="rc",
            metavar="init_value,lower,upper",
            help="initial value and lower/upper limits for parameter rc. " + \
                 "Use 'none' (case-insensitive) to ignore the limit")
    parser.add_argument("--beta", dest="beta",
            metavar="init_value,lower,upper",
            help="initial value and lower/upper limits for parameter beta. " + \
                 "Use 'none' (case-insensitive) to ignore the limit")
    parser.add_argument("--const", dest="const",
            metavar="init_value,lower,upper",
            help="initial value and lower/upper limits for parameter const. " + \
                 "Use 'none' (case-insensitive) to ignore the limit")

    args = parser.parse_args()
    if args.outfilename:
        outfile = open(args.outfilename, 'w')
    else:
        outfile = sys.stdout
    # cut mode and value
    if args.cut_point:
        cutmode = CUT_POINT
        cutvalue = args.cut_point
    elif args.cut_radius:
        cutmode = CUT_RADIUS
        cutvalue = args.cut_radius

    if args.verbose:
        print('DEBUG: apply parameter bounds: %s' % args.bounds,
                file=sys.stderr)
        print("DEBUG: cutmode: %s, cutvalue: %s" % (cutmode, cutvalue))

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
    for line in open(args.infile, 'r'):
        if re_s0.match(line):
            # read 's0_0': initial value for parameter 's0'
            s0_pstring = re_s0.split(line)[1]
            s0_0, s0_lower, s0_upper = get_parameter(s0_pstring)
        elif re_rc.match(line):
            # read 'rc_0': initial value for parameter 'rc'
            rc_pstring = re_rc.split(line)[1]
            rc_0, rc_lower, rc_upper = get_parameter(rc_pstring)
        elif re_beta.match(line):
            # read 'beta_0': initial value for parameter 'beta'
            beta_pstring = re_beta.split(line)[1]
            beta_0, beta_lower, beta_upper = get_parameter(beta_pstring)
        elif re_c.match(line):
            # read 'c_0': initial value for parameter 'c'
            c_pstring = re_c.split(line)[1]
            c_0, c_lower, c_upper = get_parameter(c_pstring)
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

    if args.verbose:
        print('DEBUG: infile: s0_0 = %g (%s, %s)' % \
                (s0_0, s0_lower, s0_upper), file=sys.stderr)
        print('DEBUG: infile: rc_0 = %g (%s, %s)' % \
                (rc_0, rc_lower, rc_upper), file=sys.stderr)
        print('DEBUG: infile: beta_0 = %g (%s, %s)' % \
                (beta_0, beta_lower, beta_upper), file=sys.stderr)
        print('DEBUG: infile: c_0 = %g (%s, %s)' % \
                (c_0, c_lower, c_upper), file=sys.stderr)

    # get parameter initial values and bounds from command line arguments
    if args.s0:
        s0_0, s0_lower, s0_upper = get_parameter(args.s0)
    if args.rc:
        rc_0, rc_lower, rc_upper = get_parameter(args.rc)
    if args.beta:
        beta_0, beta_lower, beta_upper = get_parameter(args.beta)
    if args.const:
        c_0, c_lower, c_upper = get_parameter(args.const)

    if args.verbose:
        print('DEBUG: final: s0_0 = %g (%s, %s)' % \
                (s0_0, s0_lower, s0_upper), file=sys.stderr)
        print('DEBUG: final: rc_0 = %g (%s, %s)' % \
                (rc_0, rc_lower, rc_upper), file=sys.stderr)
        print('DEBUG: final: beta_0 = %g (%s, %s)' % \
                (beta_0, beta_lower, beta_upper), file=sys.stderr)
        print('DEBUG: final: c_0 = %g (%s, %s)' % \
                (c_0, c_lower, c_upper), file=sys.stderr)

    # convert to numpy array
    r_data = np.array(r_data)
    rerr_data = np.array(rerr_data)
    s_data = np.array(s_data)
    serr_data = np.array(serr_data)
    # cut data
    r_data_cut, s_data_cut, serr_data_cut = cut_data(r_data, s_data,
            serr_data, cutmode=cutmode, cutvalue=cutvalue)

    # model parameters
    par_names = ["s0", "rc", "beta", "c"]
    # initial values
    par_0 = np.array([s0_0, rc_0, beta_0, c_0])
    # parameter bounds
    par_bounds = [(s0_lower, s0_upper), (rc_lower, rc_upper),
                  (beta_lower, beta_upper), (c_lower, c_upper)]
    # set eps for the parameters (required for the minimize method,
    # otherwise error 'ABNORMAL_TERMINATION_IN_LNSRCH' occurs, which
    # may due to the different order of magnitude of each parameters)
    par_eps = par_0 * 1e-4

    if args.bounds:
        ## 'fit_model_bounds' to perform fitting with bounds
        par_fit, infodict = fit_model_bounds(beta_model, r_data_cut,
                s_data_cut, serr_data_cut, p0=par_0, bounds=par_bounds,
                options={'eps': par_eps})
    else:
        # 'fit_model' do not support parameter bounds
        par_fit, infodict = fit_model(beta_model, r_data_cut,
                s_data_cut, serr_data_cut, p0=par_0)

    fvec = infodict['fvec']
    dof = infodict['dof']
    chisq = infodict['chisq']
    perr = infodict['perr']

    print("# beta-model fitting results:", file=outfile)
    print("# s(r) = s0 * pow((1.0+(r/rc)^2), 0.5-3*beta) + c", file=outfile)
    for i in range(len(par_names)):
        print("# %s = %g +/- %g" % (par_names[i], par_fit[i], perr[i]),
                file=outfile)
    print("# chisq / dof = %g / %g = %g" % (chisq, dof, chisq/dof),
            file=outfile)
    print("# radius(input)  brightness(fitted)", file=outfile)
    for i in range(len(r_data)):
        print("%g  %g" % (r_data[i], fvec(r_data[i])), file=outfile)

    if args.outfilename:
        outfile.close()


if __name__ == '__main__':
    main()

