#!/usr/bin/env python3
#
# Copyright (c) Weitna LI <weitian@aaronly.me>
# MIT License
#
# Run OSKAR interferometer simulator, with corresponding settings
# applied to the common configurations.
#
# 2017-06-12
#

import os
import re
import argparse
import subprocess
import shutil
from time import time


def run_oskar(configfile, model, freq, vis_oskar, vis_ms, dryrun=False):
    if vis_oskar is None and vis_ms is None:
        raise ValueError("both 'vis_oskar' & 'vis_ms' are missing")
    prog = "oskar_sim_interferometer"
    print("Update settings ...")
    for item, value in [
            ("sky/oskar_sky_model/file", model),
            ("observation/start_frequency_hz", str(freq*1e6)),
            ("interferometer/oskar_vis_filename", vis_oskar),
            ("interferometer/ms_filename", vis_ms)]:
        if value is not None:
            subprocess.check_call([prog, "--set", configfile, item, value])
            print("Set '%s' -> '%s'" % (item, value))
    print("Simulating %s @ %.2f [MHz] ..." % (model, freq))
    cmd = [prog, configfile]
    if dryrun:
        print("Dry run mode!")
        print("CMD: %s" % " ".join(cmd))
    else:
        t1 = time()
        subprocess.check_call(cmd)
        t2 = time()
        print("Elapsed time: %.1f [s]" % (t2-t1))


def main():
    default_fconfig = "sim_interferometer.f{freq:06.2f}.ini"
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dryrun", dest="dryrun", action="store_true",
                        help="dry run")
    parser.add_argument("-c", "--config", dest="config", required=True,
                        help="OSKAR base/common configuration file")
    parser.add_argument("-C", "--fconfig", dest="fconfig",
                        default=default_fconfig,
                        help="filename pattern of the configuration files " +
                        "updated for OSKAR usage " +
                        "(default: %s)" % default_fconfig)
    parser.add_argument("-l", "--list", dest="listfile", required=True,
                        help="List of frequencies [MHz] and input sky models")
    parser.add_argument("-o", "--outdir", dest="outdir",
                        default="visibility",
                        help="simulated visibilities output directory " +
                        "(default: 'visibility/')")
    parser.add_argument("--no-vis-oskar", dest="no_vis_oskar",
                        action="store_true",
                        help="not save visibility in OSKAR native format")
    parser.add_argument("--no-vis-ms", dest="no_vis_ms",
                        action="store_true",
                        help="not save visibility in MeasurementSet format")
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
        print("Created output visibility directory: %s" % args.outdir)

    skymodels = []
    for line in open(args.listfile).readlines():
        if re.match(r"^(\s*$|\s*#)", line):
            continue
        freq, skyfile = line.strip().split()
        skymodels.append((float(freq), skyfile))
        print("Got sky model: %s @ %s [MHz]" % (skyfile, freq))

    for freq, skyfile in skymodels:
        basename = os.path.splitext(os.path.basename(skyfile))[0]
        if args.no_vis_oskar:
            vis_oskar = ""
        else:
            vis_oskar = os.path.join(args.outdir, basename+".oskar")
        if args.no_vis_ms:
            vis_ms = ""
        else:
            vis_ms = os.path.join(args.outdir, basename+".ms")
        configfile = args.fconfig.format(freq=freq)
        shutil.copy(args.config, configfile)
        print("Copied OSKAR configuration file as: %s" % configfile)
        run_oskar(configfile=configfile, freq=freq, model=skyfile,
                  vis_oskar=vis_oskar, vis_ms=vis_ms, dryrun=args.dryrun)


if __name__ == "__main__":
    main()
