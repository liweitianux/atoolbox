#!/usr/bin/env python3
#
# Copyright (c) 2017 Aaron LI
# MIT license
#
# Create image from OSKAR simulated visibility data using `WSClean`.
# WSClean: https://sourceforge.net/p/wsclean/
#
# 2017-09-01
#


import argparse
import subprocess
import time


def wsclean(args):
    t1 = time.perf_counter()
    cmd = ["wsclean"] + args
    print("CMD: %s" % " ".join(cmd))
    subprocess.check_call(cmd)
    t2 = time.perf_counter()
    print("Elapsed time: %.1f [min]" % ((t2-t1)/60))


def main():
    parser = argparse.ArgumentParser(description="Run WSClean")
    parser.add_argument("--args", dest="args", nargs="*",
                        help="additional arguments for WSClean")
    parser.add_argument("--dirty", dest="dirty", action="store_true",
                        help="only create dirty images (by setting niter=0)")
    parser.add_argument("--update-model", dest="update_model",
                        action="store_true",
                        help="update the MODEL_DATA column in MS")
    parser.add_argument("--weight-briggs", dest="briggs",
                        type=float, default=0.0,
                        help="Briggs weight parameter")
    parser.add_argument("--niter", dest="niter", type=int, default=100000,
                        help="maximum number of CLEAN iterations")
    parser.add_argument("--gain", dest="gain", type=float, default=0.1,
                        help="CLEAN gain for each minor iteration")
    parser.add_argument("--mgain", dest="mgain", type=float, default=0.85,
                        help="CLEAN gain for major iterations")
    parser.add_argument("--size", dest="size", type=int,
                        required=True,
                        help="output image size (pixel number on a side)")
    parser.add_argument("--pixelsize", dest="pixelsize", type=float,
                        required=True,
                        help="output image pixel size [arcsec]")
    parser.add_argument("--taper-gaussian", dest="taper_gaussian",
                        type=float,
                        help="taper the weights with a Gaussian function " +
                        "to reduce the contribution of long baselines. " +
                        "Gaussian beam size in [arcsec].")
    parser.add_argument("--fit-spec-order", dest="fit_spec_order", type=int,
                        help="do joined-channel CLEAN by fitting the " +
                        "spectra with [order] polynomial in normal-space")
    #
    exgrp = parser.add_mutually_exclusive_group()
    exgrp.add_argument("--threshold-auto", dest="threshold_auto",
                       type=float, default=1.5,
                       help="estimate noise level and stop at <sigma>*<std>")
    exgrp.add_argument("--threshold", dest="threshold", type=float,
                       help="stopping CLEAN threshold [Jy]")
    #
    parser.add_argument("--name", dest="name", required=True,
                        help="filename prefix for the output files")
    parser.add_argument("--ms", nargs="+", help="input visibility MSs")
    args = parser.parse_args()

    nms = len(args.ms)  # i.e., number of MS == number of channels

    cmdargs = [
        "-verbose",
        "-log-time",
        "-pol", "XX",  # OSKAR "Scalar" simulation only give "XX" component
        "-make-psf",  # always make the PSF, even no cleaning performed
    ]

    if args.dirty:
        cmdargs += ["-niter", str(0)]  # make dirty image only
    else:
        cmdargs += ["-niter", str(args.niter)]

    cmdargs += ["-weight", "briggs", str(args.briggs)]
    cmdargs += ["-gain", str(args.gain)]
    cmdargs += ["-mgain", str(args.mgain)]
    cmdargs += ["-size", str(args.size), str(args.size)]
    cmdargs += ["-scale", "{0}asec".format(args.pixelsize)]

    if args.fit_spec_order:
        cmdargs += ["-joinchannels", "-channelsout", str(nms),
                    "-fit-spectral-pol", str(args.fit_spec_order+1)]

    if args.update_model:
        cmdargs += ["-update-model-required"]
    else:
        cmdargs += ["-no-update-model-required"]

    if args.threshold:
        cmdargs += ["-threshold", str(args.threshold)]
    else:
        cmdargs += ["-auto-threshold", str(args.threshold_auto)]

    if args.taper_gaussian:
        cmdargs += ["-taper-gaussian", str(args.taper_gaussian)]

    # additional WSClean arguments
    if args.args:
        cmdargs += args.args

    cmdargs += ["-name", args.name.rstrip("-_")]
    cmdargs += args.ms

    wsclean(cmdargs)


if __name__ == "__main__":
    main()
