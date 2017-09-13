#!/usr/bin/env python3
#
# Copyright (c) Weitna LI <weitian@aaronly.me>
# MIT License
#
# Run OSKAR interferometer simulator, with corresponding settings
# applied to the common configurations.
#

import os
import re
import argparse
import subprocess
import shutil
from time import time


def run_oskar(configfile, model, freq, vis_ms, vis_oskar=None,
              telescope=None, chunksize=None, dryrun=False):
    """
    Update simulation settings in the configuration file,
    and run the simulator ``oskar_sim_interferometer``.
    """
    if chunksize is not None:
        chunksize = int(chunksize)
    if vis_ms is None and vis_oskar is None:
        raise ValueError("both 'vis_oskar' & 'vis_ms' are missing")

    print("Updating simulation settings ...")
    simulator = "oskar_sim_interferometer"
    for item, value in [
            ("simulator/max_sources_per_chunk", chunksize),
            ("sky/oskar_sky_model/file", model),
            ("telescope/input_directory", telescope),
            ("observation/start_frequency_hz", str(freq*1e6)),
            ("interferometer/oskar_vis_filename", vis_oskar),
            ("interferometer/ms_filename", vis_ms)]:
        if value is not None:
            subprocess.check_call([simulator, "--set",
                                   configfile, item, value])
            print("Updated '%s' -> '%s'" % (item, value))

    print("-------------------------------------------------------------")
    print("Simulating %s @ %.2f [MHz] ..." % (model, freq))
    cmd = [simulator, configfile]
    if dryrun:
        print("Dry run mode!")
        print("CMD: %s" % " ".join(cmd))
    else:
        t1 = time()
        subprocess.check_call(cmd)
        t2 = time()
        print("Elapsed time: %.1f [min]" % ((t2-t1)/60))
    print("-------------------------------------------------------------")


def main():
    default_fconfig = "sim_interferometer.f{freq:06.2f}.ini"
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dry-run", dest="dryrun", action="store_true",
                        help="dry run")
    parser.add_argument("-c", "--config", dest="config", required=True,
                        help="OSKAR base/common configuration file")
    parser.add_argument("-C", "--fconfig", dest="fconfig",
                        default=default_fconfig,
                        help="filename pattern of the configuration files " +
                        "updated for OSKAR usage " +
                        "(default: %s)" % default_fconfig)
    parser.add_argument("-S", "--chunk-size", dest="chunksize", type=float,
                        help="overwrite the chunk size in config file")
    parser.add_argument("--vis-oskar", dest="vis_oskar",
                        action="store_true",
                        help="also save visibility in OSKAR native format")
    parser.add_argument("-t", "--telescope", dest="telescope",
                        help="overwrite the telescope model in config file")
    parser.add_argument("-o", "--outdir", dest="outdir",
                        default="visibility",
                        help="simulated visibility output directory " +
                        "(default: 'visibility/')")
    exgrp = parser.add_mutually_exclusive_group(required=True)
    exgrp.add_argument("-l", "--list", dest="listfile",
                       help="list of frequencies [MHz] and input sky models")
    exgrp.add_argument("-i", "--items", dest="items", nargs="+",
                       help="list of sky model items in format of " +
                       "'<freq[MHz]>:<skymodel-file>'")
    args = parser.parse_args()
    t1 = time()

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
        print("Created output visibility directory: %s" % args.outdir)

    skymodels = []
    if args.listfile:
        for line in open(args.listfile).readlines():
            if re.match(r"^(\s*$|\s*#)", line):
                continue
            freq, skyfile = line.strip().split()
            freq = float(freq)
            skymodels.append((freq, skyfile))
    else:
        for item in args.items:
            freq, skyfile = item.split(":")
            freq = float(freq)
            skymodels.append((freq, skyfile))

    Nosm = len(skymodels)
    print("Number of sky models: %d" % Nosm)

    for i, (freq, skyfile) in enumerate(skymodels):
        print("=============================================================")
        print("[%d/%d] %s @ %.2f [MHz]" % (i+1, Nosm, skyfile, freq))
        print("-------------------------------------------------------------")
        basename = os.path.splitext(os.path.basename(skyfile))[0]
        if args.vis_oskar:
            vis_oskar = os.path.join(args.outdir, basename+".oskar")
        else:
            vis_oskar = None
        vis_ms = os.path.join(args.outdir, basename+".ms")
        configfile = args.fconfig.format(freq=freq)
        shutil.copy(args.config, configfile)
        print("Copied OSKAR configuration file as: %s" % configfile)
        run_oskar(configfile=configfile, freq=freq,
                  model=skyfile, vis_ms=vis_ms, vis_oskar=vis_oskar,
                  telescope=args.telescope, chunksize=args.chunksize,
                  dryrun=args.dryrun)

    t2 = time()
    print("=============================================================")
    print("Total elapsed time: %.1f [min]" % ((t2-t1)/60))


if __name__ == "__main__":
    main()
