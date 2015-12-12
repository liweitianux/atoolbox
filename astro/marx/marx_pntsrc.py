#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Aaron LI
# 2015/06/16

"""
Run MARX simulation on a given list of point sources, merge the
output simulation results, and finally convert into FITS image.
"""

__version__ = "0.1.0"
__date__ = "2015/06/16"

import sys
import argparse
import subprocess
import re
import os


def marx_pntsrc(pfile, ra, dec, flux, outdir):
    """
    Run MARX simulation for the provided point source.
    """
    cmd = "marx @@%(pfile)s SourceRA=%(ra)s " % {"pfile": pfile, "ra": ra} + \
            "SourceDEC=%(dec)s SourceFlux=%(flux)s OutputDir=%(outdir)s" % \
            {"dec": dec, "flux": flux, "outdir": outdir}
    print("CMD: %s" % cmd, file=sys.stderr)
    subprocess.call(cmd, shell=True)


def marxcat(indirs, outdir):
    """
    Concatenate a list of MARX simulation results.

    Note: the number of MARX results to be concatenated at *one* time
          can not be to many, otherwise the 'marxcat' tool will failed.
    """
    if isinstance(indirs, list):
        pass
    elif isinstance(indirs, str):
        indirs = indirs.split()
    else:
        raise ValueError("invalid indirs type: %s" % indirs)
    pid = os.getpid()
    tempdir = "_marx_tmp%d" % pid
    cmd = "cp -a %(marxdir)s %(tempdir)s" % \
            {"marxdir": indirs[0], "tempdir": tempdir}
    print("CMD: %s" % cmd, file=sys.stderr)
    subprocess.call(cmd, shell=True)
    del indirs[0]
    while len(indirs) > 0:
        # concatenated 10 directories each time
        catdirs = indirs[:9]
        del indirs[:9]
        catdirs = tempdir + " " + " ".join(catdirs)
        # concatenate MARX results
        cmd = "marxcat %(catdirs)s %(outdir)s" % \
                {"catdirs": catdirs, "outdir": outdir}
        print("CMD: %s" % cmd, file=sys.stderr)
        subprocess.call(cmd, shell=True)
        # move output results to temporary directory
        cmd = "rm -rf %(tempdir)s && mv %(outdir)s %(tempdir)s" % \
                {"tempdir": tempdir, "outdir": outdir}
        print("CMD: %s" % cmd, file=sys.stderr)
        subprocess.call(cmd, shell=True)
    cmd = "mv %(tempdir)s %(outdir)s" % \
            {"tempdir": tempdir, "outdir": outdir}
    print("CMD: %s" % cmd, file=sys.stderr)
    subprocess.call(cmd, shell=True)


def marx2fits(indir, outfile, params=""):
    """
    Convert the results of MARX simulation into FITS image.
    """
    cmd = "marx2fits %(params)s %(indir)s %(outfile)s" % \
            {"params": params, "indir": indir, "outfile": outfile}
    print("CMD: %s" % cmd, file=sys.stderr)
    subprocess.call(cmd, shell=True)


def main():
    parser = argparse.ArgumentParser(
            description="Run MARX on a given list of point sources")
    parser.add_argument("-V", "--version", action="version",
            version="%(prog)s " + "%s (%s)" % (__version__, __date__))
    parser.add_argument("pfile", help="marx paramter file")
    parser.add_argument("srclist", help="point source list file")
    parser.add_argument("outprefix", help="prefix of output directories")
    args = parser.parse_args()

    outdirs = []
    i = 0
    for line in open(args.srclist, "r"):
        if re.match(r"^\s*$", line):
            # skip blank line
            continue
        elif re.match(r"^\s*#", line):
            # skip comment line
            continue
        i += 1
        ra, dec, flux = map(float, line.split())
        print("INFO: ra = %g, dec = %g, flux = %g" % (ra, dec, flux),
                file=sys.stderr)
        outdir = "%sp%03d" % (args.outprefix, i)
        print("INFO: outdir = %s" % outdir, file=sys.stderr)
        outdirs.append(outdir)
        marx_pntsrc(args.pfile, ra, dec, flux, outdir)
    # merge results
    merged = args.outprefix + "merged"
    marxcat(outdirs, merged)
    # convert to FITS image
    merged_fits = merged + ".fits"
    marx2fits(merged, merged_fits)


if __name__ == "__main__":
    main()

