#!/usr/bin/env python3
#
# Correct the crosstalk effect of XMM spectra by subtracting the
# scattered photons from surrounding regions, and by compensating
# the photons scattered to surrounding regions, according to the
# generated crosstalk ARFs.
#
# Sample config file (in `ConfigObj' syntax):
#----------------------
# fix_negative = True
# verbose = True
# clobber = False
#
# [reg2]
# outfile = cc_reg2.pi
# spec = reg2.pi
# arf = reg2.arf
#   [[cross_in]]
#     [[[in1]]]
#     spec = reg1.pi
#     arf = reg1.arf
#     cross_arf = reg_1-2.arf
#     [[[in2]]]
#     spec = reg3.pi
#     arf = reg3.arf
#     cross_arf = reg_3-2.arf
#   [[cross_out]]
#   cross_arf = reg_2-1.arf, reg_2-3.arf
#----------------------
#
# Weitian LI
# Created: 2016-03-26
# Updated: 2016-03-28
#

from astropy.io import fits
import numpy as np
from configobj import ConfigObj

import sys
import os
import argparse
from datetime import datetime


class ARF:
    """
    Deal with X-ray ARF file (.arf)
    """
    filename = None
    fitsobj = None
    header = None
    energ_lo = None
    energ_hi = None
    specresp = None

    def __init__(self, filename):
        self.filename = filename
        self.fitsobj = fits.open(filename)
        ext_specresp = self.fitsobj["SPECRESP"]
        self.header = ext_specresp.header
        self.energ_lo = ext_specresp.data["ENERG_LO"]
        self.energ_hi = ext_specresp.data["ENERG_HI"]
        self.specresp = ext_specresp.data["SPECRESP"]

    def get_data(self, copy=True):
        if copy:
            return self.specresp.copy()
        else:
            return self.specresp


class Spectrum:
    """
    Deal with X-ray spectrum (.pi)

    NOTE:
    The "COUNTS" column data are converted from "int32" to "float32".
    """
    filename = None
    # FITS object return by `fits.open'
    fitsobj = None
    # header of "SPECTRUM" extension
    header = None
    # "SPECTRUM" extension data
    channel = None
    # name of the column containing the spectrum data, either "COUNTS" or "RATE"
    spec_colname = None
    # spectrum data
    spec_data = None
    # ARF object for this spectrum
    arf = None

    def __init__(self, filename, arffile):
        self.filename = filename
        self.fitsobj = fits.open(filename)
        ext_spec = self.fitsobj["SPECTRUM"]
        self.header = ext_spec.header.copy(strip=True)
        colnames = ext_spec.columns.names
        if "COUNTS" in colnames:
            self.spec_colname = "COUNTS"
        elif "RATE" in colnames:
            self.spec_colname = "RATE"
        else:
            raise ValueError("Invalid spectrum file")
        self.channel = ext_spec.data["CHANNEL"].copy()
        self.spec_data = ext_spec.data.field(self.spec_colname)\
                .astype(np.float32)
        self.arf = ARF(arffile)

    def get_data(self, copy=True):
        if copy:
            return self.spec_data.copy()
        else:
            return self.spec_data

    def get_arf(self, copy=True):
        if self.arf is None:
            return None
        else:
            return self.arf.get_data(copy=copy)

    def subtract(self, spectrum, cross_arf, verbose=False):
        """
        Subtract the photons that originate from the surrounding regions
        but were scattered into this spectrum due to the finite PSF.

        NOTE:
        The crosstalk ARF must be provided, since the `spectrum.arf' is
        required to be its ARF without taking crosstalk into account:
            spec1_new = spec1 - spec2 * (cross_arf_2_to_1 / arf2)
        """
        operation = "  SUBTRACT: %s - (%s/%s) * %s" % (self.filename,
                cross_arf.filename, spectrum.arf.filename, spectrum.filename)
        if verbose:
            print(operation, file=sys.stderr)
        arf_ratio = cross_arf.get_data() / spectrum.get_arf()
        arf_ratio[np.isnan(arf_ratio)] = 0.0
        self.spec_data -= spectrum.get_data() * arf_ratio
        # record history
        self.header.add_history(operation)

    def compensate(self, cross_arf, verbose=False):
        """
        Compensate the photons that originate from this regions but were
        scattered into the surrounding regions due to the finite PSF.

        formula:
            spec1_new = spec1 + spec1 * (cross_arf_1_to_2 / arf1)
        """
        operation = "  COMPENSATE: %s + (%s/%s) * %s" % (self.filename,
                cross_arf.filename, self.arf.filename, self.filename)
        if verbose:
            print(operation, file=sys.stderr)
        arf_ratio = cross_arf.get_data() / self.get_arf()
        arf_ratio[np.isnan(arf_ratio)] = 0.0
        self.spec_data += self.get_data() * arf_ratio
        # record history
        self.header.add_history(operation)

    def fix_negative(self, verbose=False):
        """
        The subtractions may lead to negative counts, it may be necessary
        to fix these channels with negative values.
        """
        neg_counts = self.spec_data < 0
        N = len(neg_counts)
        neg_channels = np.arange(N, dtype=np.int)[neg_counts]
        if len(neg_channels) > 0:
            print("WARNING: %d channels have NEGATIVE counts" % \
                    len(neg_channels), file=sys.stderr)
        i = 0
        while len(neg_channels) > 0:
            i += 1
            if verbose:
                print("*** Fixing negative channels: iteration %d ..." % i,
                        file=sys.stderr)
            for ch in neg_channels:
                neg_val = self.spec_data[ch]
                if ch < N-2:
                    self.spec_data[ch] = 0
                    self.spec_data[(ch+1):(ch+3)] -= 0.5 * np.abs(neg_val)
                else:
                    # just set to zero if it is the last 2 channels
                    self.spec_data[ch] = 0
            # update negative channels indices
            neg_counts = self.spec_data < 0
            neg_channels = np.arange(N, dtype=np.int)[neg_counts]
        if i > 0:
            print("*** Fixed negative channels ***", file=sys.stderr)

    def write(self, filename, clobber=False):
        """
        Create a new "SPECTRUM" table/extension and replace the original
        one, then write to output file.
        """
        ext_spec_cols = fits.ColDefs([
                fits.Column(name="CHANNEL", format="I", array=self.channel),
                fits.Column(name="COUNTS", format="E", unit="count",
                    array=self.spec_data)])
        ext_spec = fits.BinTableHDU.from_columns(ext_spec_cols,
                header=self.header)
        self.fitsobj["SPECTRUM"] = ext_spec
        self.fitsobj.writeto(filename, clobber=clobber, checksum=True)


class Crosstalk:
    """
    Crosstalk correction.
    """
    # `Spectrum' object for the spectrum to be corrected
    spectrum = None
    # XXX: do NOT use list (e.g., []) here, otherwise, all the instances
    # will share these list properties.
    # `Spectrum' and `ARF' objects corresponding to the spectra from which
    # the photons were scattered into this spectrum.
    cross_in_spec = None
    cross_in_arf = None
    # `ARF' objects corresponding to the regions to which the photons of
    # this spectrum were scattered into.
    cross_out_arf = None
    # output filename to which write the corrected spectrum
    outfile = None

    def __init__(self, config):
        """
        `config': a section of the whole config file (`ConfigObj` object).
        """
        self.cross_in_spec = []
        self.cross_in_arf  = []
        self.cross_out_arf = []
        # this spectrum to be corrected
        self.spectrum = Spectrum(config["spec"], config["arf"])
        # spectra and cross arf from which photons were scattered in
        for reg_in in config["cross_in"].values():
            spec = Spectrum(reg_in["spec"], reg_in["arf"])
            self.cross_in_spec.append(spec)
            self.cross_in_arf.append(ARF(reg_in["cross_arf"]))
        # regions into which the photons of this spectrum were scattered into
        if "cross_out" in config.sections:
            cross_arf = config["cross_out"].as_list("cross_arf")
            for arffile in cross_arf:
                self.cross_out_arf.append(ARF(arffile))
        # output filename
        self.outfile = config["outfile"]

    def do_correction(self, fix_negative=False, verbose=False):
        self.spectrum.header.add_history("Crosstalk Correction BEGIN")
        self.spectrum.header.add_history("  TOOL: %s @ %s" % (\
                os.path.basename(sys.argv[0]), datetime.utcnow().isoformat()))
        # subtractions
        if verbose:
            print("INFO: apply subtractions ...", file=sys.stderr)
        for spec, cross_arf in zip(self.cross_in_spec, self.cross_in_arf):
            self.spectrum.subtract(spectrum=spec, cross_arf=cross_arf,
                    verbose=verbose)
        # compensations
        if verbose:
            print("INFO: apply compensations ...", file=sys.stderr)
        for cross_arf in self.cross_out_arf:
            self.spectrum.compensate(cross_arf=cross_arf, verbose=verbose)
        # fix negative values in channels
        if fix_negative:
            if verbose:
                print("INFO: fix negative channel values ...", file=sys.stderr)
            self.spectrum.fix_negative(verbose=verbose)
        self.spectrum.header.add_history("END Crosstalk Correction")

    def write(self, filename=None, clobber=False):
        if filename is None:
            filename = self.outfile
        self.spectrum.write(filename, clobber=clobber)


def main():
    parser = argparse.ArgumentParser(
            description="Correct the crosstalk effects of XMM spectra")
    parser.add_argument("config", help="config file in which describes " +\
            "the crosstalk relations. ('ConfigObj' syntax)")
    parser.add_argument("-N", "--fix-negative", dest="fix_negative",
            action="store_true", help="fix negative channel values")
    parser.add_argument("-C", "--clobber", dest="clobber",
            action="store_true", help="overwrite output file if exists")
    parser.add_argument("-v", "--verbose", dest="verbose",
            action="store_true", help="show verbose information")
    args = parser.parse_args()

    config = ConfigObj(args.config)

    fix_negative = False
    if "fix_negative" in config.keys():
        fix_negative = config.as_bool("fix_negative")
    if args.fix_negative:
        fix_negative = args.fix_negative

    verbose = False
    if "verbose" in config.keys():
        verbose = config.as_bool("verbose")
    if args.verbose:
        verbose = args.verbose

    clobber = False
    if "clobber" in config.keys():
        clobber = config.as_bool("clobber")
    if args.clobber:
        clobber = args.clobber

    for region in config.sections:
        if verbose:
            print("INFO: processing '%s' ..." % region, file=sys.stderr)
        crosstalk = Crosstalk(config.get(region))
        crosstalk.do_correction(fix_negative=fix_negative, verbose=verbose)
        crosstalk.write(clobber=clobber)


if __name__ == "__main__":
    main()

#  vim: set ts=4 sw=4 tw=0 fenc=utf-8 ft=python: #
