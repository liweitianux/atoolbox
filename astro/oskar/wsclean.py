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


import os
import re
import argparse
import subprocess
import time


def wsclean(args, dryrun=False):
    # NOTE: Convert all arguments to strings
    cmd = ["wsclean"] + [str(arg) for arg in args]
    print("CMD: %s" % " ".join(cmd))
    if dryrun:
        print(">>> DRY RUN MODE <<<")
        return

    t1 = time.perf_counter()
    subprocess.check_call(cmd)
    t2 = time.perf_counter()
    print("-----------------------------------------------------------")
    print("WSClean Elapsed time: %.1f [min]" % ((t2-t1)/60))
    print("-----------------------------------------------------------")


def main():
    parser = argparse.ArgumentParser(description="Run WSClean")
    parser.add_argument("-n", "--dry-run", dest="dryrun", action="store_true",
                        help="do not actually run WSClean")
    parser.add_argument("-a", "--args", dest="args",
                        help="additional arguments for WSClean " +
                        "(in a quoted string separated by space)")
    parser.add_argument("-d", "--dirty", dest="dirty", action="store_true",
                        help="only create dirty images (by setting niter=0)")
    parser.add_argument("--update-model", dest="update_model",
                        action="store_true",
                        help="update the MODEL_DATA column in MS")
    parser.add_argument("--save-weights", dest="save_weights",
                        action="store_true",
                        help="save the gridded weights in <name>-weights.fits")
    parser.add_argument("-w", "--weight", dest="weight", default="briggs",
                        choices=["uniform", "natural", "briggs"],
                        help="weighting method (default: 'briggs')")
    parser.add_argument("--briggs", dest="briggs", type=float, default=0.0,
                        help="Briggs robustness parameter (default: 0); " +
                        "-1 (uniform) -> 1 (natural)")
    parser.add_argument("--niter", dest="niter", type=int, default=100000,
                        help="maximum number of CLEAN iterations " +
                        "(default: 100,000)")
    parser.add_argument("--gain", dest="gain", type=float, default=0.1,
                        help="CLEAN gain for each minor iteration")
    parser.add_argument("--mgain", dest="mgain", type=float, default=0.85,
                        help="CLEAN gain for major iterations")
    parser.add_argument("-s", "--size", dest="size", type=int,
                        required=True,
                        help="output image size (pixel number on a side)")
    parser.add_argument("-p", "--pixelsize", dest="pixelsize", type=float,
                        required=True,
                        help="output image pixel size [arcsec]")
    parser.add_argument("--taper-gaus", dest="taper_gaus", type=float,
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
    parser.add_argument("-N", "--name", dest="name", required=True,
                        help="filename prefix for the output files")
    parser.add_argument("-m", "--ms", nargs="+", help="input visibility MSs")
    args = parser.parse_args()

    nms = len(args.ms)  # i.e., number of MS == number of channels

    cmdargs = [
        "-verbose",
        "-log-time",
        "-pol", "XX",  # OSKAR "Scalar" simulation only give "XX" component
        "-make-psf",  # always make the PSF, even no cleaning performed
        "-tempdir", "/tmp",
    ]

    if args.dirty:
        cmdargs += ["-niter", 0]  # make dirty image only
    else:
        cmdargs += ["-niter", args.niter]

    if args.weight == "uniform":
        cmdargs += ["-weight", "uniform",
                    "-weighting-rank-filter", "3"]
    elif args.weight == "briggs":
        cmdargs += ["-weight", "briggs", args.briggs]
    else:
        cmdargs += ["-weight", args.weight]
    cmdargs += ["-gain", args.gain]
    cmdargs += ["-mgain", args.mgain]
    cmdargs += ["-size", args.size, args.size]
    cmdargs += ["-scale", "{0}asec".format(args.pixelsize)]

    if args.fit_spec_order:
        cmdargs += ["-joinchannels", "-channelsout", nms,
                    "-fit-spectral-pol", args.fit_spec_order+1]

    if args.update_model:
        cmdargs += ["-update-model-required"]
    else:
        cmdargs += ["-no-update-model-required"]

    if args.save_weights:
        cmdargs += ["-saveweights"]

    if args.threshold:
        cmdargs += ["-threshold", args.threshold]
    else:
        cmdargs += ["-auto-threshold", args.threshold_auto]

    if args.taper_gaus:
        cmdargs += ["-taper-gaussian", args.taper_gaus]

    # additional WSClean arguments
    if args.args:
        extra_args = re.split(r"\s+", args.args.strip())
        print("Additional WSClean arguments:", extra_args)
        cmdargs += extra_args

    nameprefix = args.name.rstrip("-_")
    cmdargs += ["-name", nameprefix]
    cmdargs += args.ms

    wsclean(cmdargs, dryrun=args.dryrun)

    if args.dirty and not args.dryrun:
        # Remove the output "-image" since it is identical to "-dirty"
        os.remove(nameprefix+"-image.fits")


if __name__ == "__main__":
    main()
