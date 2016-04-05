#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Aaron LI
# Created: 2016-03-13
# Updated: 2016-04-05
#
# Changelogs:
# 2016-04-05:
#   * Allow fix parameters
# 2016-03-31:
#   * Remove `ci_report()'
#   * Add `make_results()' to orgnize all results as s Python dictionary
#   * Report results as json string
# 2016-03-28:
#   * Add `main()', `make_model()'
#   * Use `configobj' to handle configurations
#   * Save fit results and plot
#   * Add `ci_report()'
# 2016-03-14:
#   * Refactor classes `FitModelSBeta' and `FitModelDBeta'
#   * Add matplotlib plot support
#   * Add `ignore_data()' and `notice_data()' support
#   * Add classes `FitModelSBetaNorm' and `FitModelDBetaNorm'
#
# TODO:
#   * to allow fit the outer beta component, then fix it, and fit the inner one
#

"""
Fit the surface brightness profile (SBP) with the single-beta model:
  s(r) = s0 * [1.0 + (r/rc)^2] ^ (0.5-3*beta) + bkg
or the double-beta model:
  s(r) = s01 * [1.0 + (r/rc1)^2] ^ (0.5-3*beta1) +
         s02 * [1.0 + (r/rc2)^2] ^ (0.5-3*beta2) + bkg


Sample config file:
-------------------------------------------------
name    = <NAME>
obsid   = <OBSID>

sbpfile = sbprofile.txt

model   = sbeta
outfile = sbpfit_sbeta.txt
imgfile = sbpfit_sbeta.png

#model   = dbeta
#outfile = sbpfit_dbeta.txt
#imgfile = sbpfit_dbeta.png

# data range to be ignored during fitting
#ignore  = 0.0-20.0,

[sbeta]
# name = initial, lower, upper
s0    = 1.0e-8, 0.0, 1.0e-6
rc    = 30.0,   1.0, 1.0e4
beta  = 0.7,    0.3, 1.1
bkg   = 1.0e-9, 0.0, 1.0e-7

[dbeta]
s01   = 1.0e-8, 0.0,  1.0e-6
rc1   = 50.0,   10.0, 1.0e4
beta1 = 0.7,    0.3,  1.1
s02   = 1.0e-8, 0.0,  1.0e-6
rc2   = 30.0,   1.0,  5.0e2
beta2 = 0.7,    0.3,  1.1
bkg   = 1.0e-9, 0.0,  1.0e-7
-------------------------------------------------
"""

__version__ = "0.5.1"
__date__    = "2016-04-05"


import numpy as np
import lmfit
import matplotlib.pyplot as plt

from configobj import ConfigObj
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import os
import sys
import re
import argparse
import json
from collections import OrderedDict


plt.style.use("ggplot")


class FitModel:
    """
    Meta-class of the fitting model.

    The supplied `func' should have the following syntax:
        y = f(x, params)
    where the `params' is the parameters to be fitted,
    and should be provided as well.
    """
    def __init__(self, name=None, func=None, params=lmfit.Parameters()):
        self.name = name
        self.func = func
        self.params = params

    def f(self, x):
        return self.func(x, self.params)

    def get_param(self, name=None):
        """
        Return the requested `Parameter' object or the whole
        `Parameters' object of no name supplied.
        """
        try:
            return self.params[name]
        except KeyError:
            return self.params

    def set_param(self, name, *args, **kwargs):
        """
        Set the properties of the specified parameter.
        """
        param = self.params[name]
        param.set(*args, **kwargs)

    def plot(self, params, xdata, ax):
        """
        Plot the fitted model.
        """
        f_fitted = lambda x: self.func(x, params)
        ydata = f_fitted(xdata)
        ax.plot(xdata, ydata, 'k-')

class FitModelSBeta(FitModel):
    """
    The single-beta model to be fitted.
    Single-beta model, with a constant background.
    """
    params = lmfit.Parameters()
    params.add_many( # (name, value, vary, min, max, expr)
                    ("s0",   1.0e-8, True, 0.0, 1.0e-6, None),
                    ("rc",   30.0,   True, 1.0, 1.0e4,  None),
                    ("beta", 0.7,    True, 0.3, 1.1,    None),
                    ("bkg",  1.0e-9, True, 0.0, 1.0e-7, None))

    @staticmethod
    def sbeta(r, params):
        parvals = params.valuesdict()
        s0   = parvals["s0"]
        rc   = parvals["rc"]
        beta = parvals["beta"]
        bkg  = parvals["bkg"]
        return s0 * np.power((1 + (r/rc)**2), (0.5 - 3*beta)) + bkg

    def __init__(self):
        super(self.__class__, self).__init__(name="Single-beta",
                func=self.sbeta, params=self.params)

    def plot(self, params, xdata, ax):
        """
        Plot the fitted model, as well as the fitted parameters.
        """
        super(self.__class__, self).plot(params, xdata, ax)
        ydata = self.sbeta(xdata, params)
        # fitted paramters
        ax.vlines(x=params["rc"].value, ymin=min(ydata), ymax=max(ydata),
                linestyles="dashed")
        ax.hlines(y=params["bkg"].value, xmin=min(xdata), xmax=max(xdata),
                linestyles="dashed")
        ax.text(x=params["rc"].value, y=min(ydata),
                s="beta: %.2f\nrc: %.2f" % (params["beta"].value,
                    params["rc"].value))
        ax.text(x=min(xdata), y=min(ydata),
                s="bkg: %.3e" % params["bkg"].value,
                verticalalignment="top")


class FitModelDBeta(FitModel):
    """
    The double-beta model to be fitted.
    Double-beta model, with a constant background.

    NOTE:
    the first beta component (s01, rc1, beta1) describes the main and
    outer SBP; while the second beta component (s02, rc2, beta2) accounts
    for the central brightness excess.
    """
    params = lmfit.Parameters()
    params.add("s01",   value=1.0e-8, min=0.0,  max=1.0e-6)
    params.add("rc1",   value=50.0,   min=10.0, max=1.0e4)
    params.add("beta1", value=0.7,    min=0.3,  max=1.1)
    #params.add("df_s0", value=1.0e-8, min=0.0,  max=1.0e-6)
    #params.add("s02",   expr="s01 + df_s0")
    params.add("s02",   value=1.0e-8, min=0.0,  max=1.0e-6)
    #params.add("df_rc", value=30.0,   min=0.0,  max=1.0e4)
    #params.add("rc2",   expr="rc1 - df_rc")
    params.add("rc2",   value=20.0,   min=1.0,  max=5.0e2)
    params.add("beta2", value=0.7,    min=0.3,  max=1.1)
    params.add("bkg",   value=1.0e-9, min=0.0,  max=1.0e-7)

    @staticmethod
    def beta1(r, params):
        """
        This beta component describes the main/outer part of the SBP.
        """
        parvals = params.valuesdict()
        s01   = parvals["s01"]
        rc1   = parvals["rc1"]
        beta1 = parvals["beta1"]
        bkg   = parvals["bkg"]
        return s01 * np.power((1 + (r/rc1)**2), (0.5 - 3*beta1)) + bkg

    @staticmethod
    def beta2(r, params):
        """
        This beta component describes the central/excess part of the SBP.
        """
        parvals = params.valuesdict()
        s02   = parvals["s02"]
        rc2   = parvals["rc2"]
        beta2 = parvals["beta2"]
        return s02 * np.power((1 + (r/rc2)**2), (0.5 - 3*beta2))

    @classmethod
    def dbeta(self, r, params):
        return self.beta1(r, params) + self.beta2(r, params)

    def __init__(self):
        super(self.__class__, self).__init__(name="Double-beta",
                func=self.dbeta, params=self.params)

    def plot(self, params, xdata, ax):
        """
        Plot the fitted model, and each beta component,
        as well as the fitted parameters.
        """
        super(self.__class__, self).plot(params, xdata, ax)
        beta1_ydata = self.beta1(xdata, params)
        beta2_ydata = self.beta2(xdata, params)
        ax.plot(xdata, beta1_ydata, 'b-.')
        ax.plot(xdata, beta2_ydata, 'b-.')
        # fitted paramters
        ydata = beta1_ydata + beta2_ydata
        ax.vlines(x=params["rc1"].value, ymin=min(ydata), ymax=max(ydata),
                linestyles="dashed")
        ax.vlines(x=params["rc2"].value, ymin=min(ydata), ymax=max(ydata),
                linestyles="dashed")
        ax.hlines(y=params["bkg"].value, xmin=min(xdata), xmax=max(xdata),
                linestyles="dashed")
        ax.text(x=params["rc1"].value, y=min(ydata),
                s="beta1: %.2f\nrc1: %.2f" % (params["beta1"].value,
                    params["rc1"].value))
        ax.text(x=params["rc2"].value, y=min(ydata),
                s="beta2: %.2f\nrc2: %.2f" % (params["beta2"].value,
                    params["rc2"].value))
        ax.text(x=min(xdata), y=min(ydata),
                s="bkg: %.3e" % params["bkg"].value,
                verticalalignment="top")


class FitModelSBetaNorm(FitModel):
    """
    The single-beta model to be fitted.
    Single-beta model, with a constant background.
    Normalized the `s0' and `bkg' parameters by take the logarithm.
    """
    params = lmfit.Parameters()
    params.add_many( # (name, value, vary, min, max, expr)
                    ("log10_s0",   -8.0, True, -12.0, -6.0,  None),
                    ("rc",   30.0, True, 1.0,   1.0e4, None),
                    ("beta", 0.7,  True, 0.3,   1.1,   None),
                    ("log10_bkg",  -9.0, True, -12.0, -7.0,  None))

    @staticmethod
    def sbeta(r, params):
        parvals = params.valuesdict()
        s0   = 10 ** parvals["log10_s0"]
        rc   = parvals["rc"]
        beta = parvals["beta"]
        bkg  = 10 ** parvals["log10_bkg"]
        return s0 * np.power((1 + (r/rc)**2), (0.5 - 3*beta)) + bkg

    def __init__(self):
        super(self.__class__, self).__init__(name="Single-beta",
                func=self.sbeta, params=self.params)

    def plot(self, params, xdata, ax):
        """
        Plot the fitted model, as well as the fitted parameters.
        """
        super(self.__class__, self).plot(params, xdata, ax)
        ydata = self.sbeta(xdata, params)
        # fitted paramters
        ax.vlines(x=params["rc"].value, ymin=min(ydata), ymax=max(ydata),
                linestyles="dashed")
        ax.hlines(y=(10 ** params["bkg"].value), xmin=min(xdata),
                xmax=max(xdata), linestyles="dashed")
        ax.text(x=params["rc"].value, y=min(ydata),
                s="beta: %.2f\nrc: %.2f" % (params["beta"].value,
                    params["rc"].value))
        ax.text(x=min(xdata), y=min(ydata),
                s="bkg: %.3e" % (10 ** params["bkg"].value),
                verticalalignment="top")


class FitModelDBetaNorm(FitModel):
    """
    The double-beta model to be fitted.
    Double-beta model, with a constant background.
    Normalized the `s01', `s02' and `bkg' parameters by take the logarithm.

    NOTE:
    the first beta component (s01, rc1, beta1) describes the main and
    outer SBP; while the second beta component (s02, rc2, beta2) accounts
    for the central brightness excess.
    """
    params = lmfit.Parameters()
    params.add("log10_s01",   value=-8.0, min=-12.0, max=-6.0)
    params.add("rc1",   value=50.0, min=10.0,  max=1.0e4)
    params.add("beta1", value=0.7,  min=0.3,   max=1.1)
    #params.add("df_s0", value=1.0e-8, min=0.0, max=1.0e-6)
    #params.add("s02",   expr="s01 + df_s0")
    params.add("log10_s02",   value=-8.0, min=-12.0, max=-6.0)
    #params.add("df_rc", value=30.0, min=0.0,   max=1.0e4)
    #params.add("rc2",   expr="rc1 - df_rc")
    params.add("rc2",   value=20.0, min=1.0,   max=5.0e2)
    params.add("beta2", value=0.7,  min=0.3,   max=1.1)
    params.add("log10_bkg",   value=-9.0, min=-12.0, max=-7.0)

    @staticmethod
    def beta1(r, params):
        """
        This beta component describes the main/outer part of the SBP.
        """
        parvals = params.valuesdict()
        s01   = 10 ** parvals["log10_s01"]
        rc1   = parvals["rc1"]
        beta1 = parvals["beta1"]
        bkg   = 10 ** parvals["log10_bkg"]
        return s01 * np.power((1 + (r/rc1)**2), (0.5 - 3*beta1)) + bkg

    @staticmethod
    def beta2(r, params):
        """
        This beta component describes the central/excess part of the SBP.
        """
        parvals = params.valuesdict()
        s02   = 10 ** parvals["log10_s02"]
        rc2   = parvals["rc2"]
        beta2 = parvals["beta2"]
        return s02 * np.power((1 + (r/rc2)**2), (0.5 - 3*beta2))

    @classmethod
    def dbeta(self, r, params):
        return self.beta1(r, params) + self.beta2(r, params)

    def __init__(self):
        super(self.__class__, self).__init__(name="Double-beta",
                func=self.dbeta, params=self.params)

    def plot(self, params, xdata, ax):
        """
        Plot the fitted model, and each beta component,
        as well as the fitted parameters.
        """
        super(self.__class__, self).plot(params, xdata, ax)
        beta1_ydata = self.beta1(xdata, params)
        beta2_ydata = self.beta2(xdata, params)
        ax.plot(xdata, beta1_ydata, 'b-.')
        ax.plot(xdata, beta2_ydata, 'b-.')
        # fitted paramters
        ydata = beta1_ydata + beta2_ydata
        ax.vlines(x=params["log10_rc1"].value, ymin=min(ydata), ymax=max(ydata),
                linestyles="dashed")
        ax.vlines(x=params["rc2"].value, ymin=min(ydata), ymax=max(ydata),
                linestyles="dashed")
        ax.hlines(y=(10 ** params["bkg"].value), xmin=min(xdata),
                xmax=max(xdata), linestyles="dashed")
        ax.text(x=params["rc1"].value, y=min(ydata),
                s="beta1: %.2f\nrc1: %.2f" % (params["beta1"].value,
                    params["rc1"].value))
        ax.text(x=params["rc2"].value, y=min(ydata),
                s="beta2: %.2f\nrc2: %.2f" % (params["beta2"].value,
                    params["rc2"].value))
        ax.text(x=min(xdata), y=min(ydata),
                s="bkg: %.3e" % (10 ** params["bkg"].value),
                verticalalignment="top")


class SbpFit:
    """
    Class to handle the SBP fitting with single-/double-beta model.
    """
    def __init__(self, model, method="lbfgsb",
            xdata=None, ydata=None, xerr=None, yerr=None,
            name=None, obsid=None):
        self.method = method
        self.model = model
        self.xdata = xdata
        self.ydata = ydata
        self.xerr  = xerr
        self.yerr  = yerr
        if xdata is not None:
            self.mask = np.ones(xdata.shape, dtype=np.bool)
        else:
            self.mask = None
        self.name = name
        self.obsid = obsid

    def set_source(self, name, obsid=None):
        self.name = name
        self.obsid = obsid

    def load_data(self, xdata, ydata, xerr, yerr):
        self.xdata = xdata
        self.ydata = ydata
        self.xerr  = xerr
        self.yerr  = yerr
        self.mask  = np.ones(xdata.shape, dtype=np.bool)

    def ignore_data(self, xmin=None, xmax=None):
        """
        Ignore the data points within range [xmin, xmax].
        If xmin is None, then xmin=min(xdata);
        if xmax is None, then xmax=max(xdata).
        """
        if xmin is None:
            xmin = min(self.xdata)
        if xmax is None:
            xmax = max(self.xdata)
        ignore_idx = np.logical_and(self.xdata >= xmin, self.xdata <= xmax)
        self.mask[ignore_idx] = False
        # reset `f_residual'
        self.f_residual = None

    def notice_data(self, xmin=None, xmax=None):
        """
        Notice the data points within range [xmin, xmax].
        If xmin is None, then xmin=min(xdata);
        if xmax is None, then xmax=max(xdata).
        """
        if xmin is None:
            xmin = min(self.xdata)
        if xmax is None:
            xmax = max(self.xdata)
        notice_idx = np.logical_and(self.xdata >= xmin, self.xdata <= xmax)
        self.mask[notice_idx] = True
        # reset `f_residual'
        self.f_residual = None

    def set_residual(self):
        def f_residual(params):
            if self.yerr is None:
                return self.model.func(self.xdata[self.mask], params) - \
                        self.ydata
            else:
                return (self.model.func(self.xdata[self.mask], params) - \
                        self.ydata[self.mask]) / self.yerr[self.mask]
        self.f_residual = f_residual

    def fit(self, method=None):
        if method is None:
            method = self.method
        if not hasattr(self, "f_residual") or self.f_residual is None:
            self.set_residual()
        self.fitter = lmfit.Minimizer(self.f_residual, self.model.params)
        self.fitted = self.fitter.minimize(method=method)
        self.fitted_model = lambda x: self.model.func(x, self.fitted.params)

    def calc_ci(self, sigmas=[0.68, 0.90]):
        # `conf_interval' requires the fitted results have valid `stderr',
        # so we need to re-fit the model with method `leastsq'.
        fitted = self.fitter.minimize(method="leastsq",
                params=self.fitted.params)
        self.ci, self.trace = lmfit.conf_interval(self.fitter, fitted,
                sigmas=sigmas, trace=True)

    def make_results(self):
        """
        Make the `self.results' dictionary which contains all the fitting
        results as well as the confidence intervals.
        """
        fitted = self.fitted
        self.results = OrderedDict()
        ## fitting results
        self.results.update(
                nfev   = fitted.nfev,
                ndata  = fitted.ndata,
                nvarys = fitted.nvarys,  # number of varible paramters
                nfree  = fitted.nfree,  # degree of freem
                chisqr = fitted.chisqr,
                redchi = fitted.redchi,
                aic    = fitted.aic,
                bic    = fitted.bic)
        params = fitted.params
        pnames = list(params.keys())
        pvalues = OrderedDict()
        for pn in pnames:
            par = params.get(pn)
            pvalues[pn] = [par.value, par.min, par.max, par.vary]
        self.results["params"] = pvalues
        ## confidence intervals
        if hasattr(self, "ci") and self.ci is not None:
            ci = self.ci
            ci_values = OrderedDict()
            ci_sigmas = [ "ci%02d" % (v[0]*100) for v in ci.get(pnames[0]) ]
            ci_names = sorted(list(set(ci_sigmas)))
            ci_idx = { k: [] for k in ci_names }
            for cn, idx in zip(ci_sigmas, range(len(ci_sigmas))):
                ci_idx[cn].append(idx)
            # parameters ci
            for pn in pnames:
                ci_pv = OrderedDict()
                pv = [ v[1] for v in ci.get(pn) ]
                # best
                pv_best = pv[ ci_idx["ci00"][0] ]
                ci_pv["best"] = pv_best
                # ci of each sigma
                pv2 = [ v-pv_best for v in pv ]
                for cn in ci_names[1:]:
                    ci_pv[cn] = [ pv2[idx] for idx in ci_idx[cn] ]
                ci_values[pn] = ci_pv
            self.results["ci"] = ci_values

    def report(self, outfile=sys.stdout):
        if not hasattr(self, "results") or self.results is None:
            self.make_results()
        jd = json.dumps(self.results, indent=2)
        print(jd, file=outfile)

    def plot(self, ax=None, fig=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        # noticed data points
        eb = ax.errorbar(self.xdata[self.mask], self.ydata[self.mask],
                xerr=self.xerr[self.mask], yerr=self.yerr[self.mask],
                fmt="none")
        # ignored data points
        ignore_mask = np.logical_not(self.mask)
        if np.sum(ignore_mask) > 0:
            eb = ax.errorbar(self.xdata[ignore_mask], self.ydata[ignore_mask],
                    xerr=self.xerr[ignore_mask], yerr=self.yerr[ignore_mask],
                    fmt="none")
            eb[-1][0].set_linestyle("-.")
        # fitted model
        xmax = self.xdata[-1] + self.xerr[-1]
        xpred = np.power(10, np.linspace(0, np.log10(xmax), 2*len(self.xdata)))
        ypred = self.fitted_model(xpred)
        ymin = min(min(self.ydata), min(ypred))
        ymax = max(max(self.ydata), max(ypred))
        self.model.plot(params=self.fitted.params, xdata=xpred, ax=ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1.0, xmax)
        ax.set_ylim(ymin/1.2, ymax*1.2)
        name = self.name
        if self.obsid is not None:
            name += "; %s" % self.obsid
        ax.set_title("Fitted Surface Brightness Profile (%s)" % name)
        ax.set_xlabel("Radius (pixel)")
        ax.set_ylabel(r"Surface Brightness (photons/cm$^2$/pixel$^2$/s)")
        ax.text(x=xmax, y=ymax,
                s="redchi: %.2f / %.2f = %.2f" % (self.fitted.chisqr,
                    self.fitted.nfree, self.fitted.chisqr/self.fitted.nfree),
                horizontalalignment="right", verticalalignment="top")
        return (fig, ax)


def make_model(config):
    """
    Make the model with parameters set according to the config.
    """
    modelname = config["model"]
    if modelname == "sbeta":
        # single-beta model
        model = FitModelSBeta()
    elif modelname == "dbeta":
        # double-beta model
        model = FitModelDBeta()
    else:
        raise ValueError("Invalid model")
    # set initial values and bounds for the model parameters
    params = config.get(modelname)
    for p, value in params.items():
        variable = True
        if len(value) == 4 and value[3].upper() in ["FIXED", "FALSE"]:
            variable = False
        model.set_param(name=p, value=float(value[0]),
                min=float(value[1]), max=float(value[2]), vary=variable)
    return model


def main():
    # parser for command line options and arguments
    parser = argparse.ArgumentParser(
            description="Fit surface brightness profile with " + \
                        "single-/double-beta model",
            epilog="Version: %s (%s)" % (__version__, __date__))
    parser.add_argument("-V", "--version", action="version",
            version="%(prog)s " + "%s (%s)" % (__version__, __date__))
    parser.add_argument("config", help="Config file for SBP fitting")
    args = parser.parse_args()

    config = ConfigObj(args.config)

    # fit model
    model = make_model(config)

    # sbp data and fit object
    sbpdata = np.loadtxt(config["sbpfile"])
    sbpfit = SbpFit(model=model, xdata=sbpdata[:, 0], xerr=sbpdata[:, 1],
            ydata=sbpdata[:, 2], yerr=sbpdata[:, 3])
    sbpfit.set_source(config["name"], obsid=config.get("obsid"))

    # apply data range ignorance
    if "ignore" in config.keys():
        for ig in config.as_list("ignore"):
            xmin, xmax = map(float, ig.split("-"))
            sbpfit.ignore_data(xmin=xmin, xmax=xmax)

    # fit and calculate confidence intervals
    sbpfit.fit()
    sbpfit.calc_ci()
    sbpfit.report()
    with open(config["outfile"], "w") as outfile:
        sbpfit.report(outfile=outfile)

    # make and save a plot
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    sbpfit.plot(ax=ax, fig=fig)
    fig.savefig(config["imgfile"])


if __name__ == "__main__":
    main()

#  vim: set ts=4 sw=4 tw=0 fenc=utf-8 ft=python: #
