#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# References:
# [?] astropy - FITS format code
#     http://docs.astropy.org/en/stable/io/fits/usage/table.html#column-creation
# [?] XSPEC - Spectral Fitting
#     https://heasarc.gsfc.nasa.gov/docs/xanadu/xspec/manual/XspecSpectralFitting.html
#
#
# Weitian LI
# Created: 2016-03-26
# Updated: 2016-04-01
#
# ChangeLog:
# 2016-04-01:
#   * Greatly update the documentations (e.g., description, sample config)
#   * Add class `RMF'
#   * Add method `get_energy()' for class `ARF'
#
# TODO/XXX:
#   * Add background spectrum fields
#   * Subtract background spectrum before correcting crosstalk effects
#   * Estimate channel errors by Monte Carlo simulations
#   * Split classes ARF, RMF, Spectrum to a separate module
#


"""
Correct the crosstalk effect of XMM spectra by subtracting the photons that
scattered from the surrounding regions due to the finite PSF, and by
compensating the photons that scattered to the surrounding regions, according
to the generated crosstalk ARFs by SAS `arfgen'.


Sample config file (in `ConfigObj' syntax):
-----------------------------------------------------------
verbose = True
clobber = False
# whether to fix the negative channel values due to spectral subtractions
fix_negative = True

[...]
...

[reg2]
outfile = cc_reg2.pi
spec = reg2.pi
arf = reg2.arf
rmf = reg2.rmf
bkg = reg2_bkg.pi
  [[cross_in]]
    [[[in1]]]
    spec = reg1.pi
    arf = reg1.arf
    rmf = reg1.rmf
    bkg = reg1_bkg.pi
    cross_arf = reg_1-2.arf
    [[[in2]]]
    spec = reg3.pi
    arf = reg3.arf
    rmf = reg3.rmf
    bkg = reg3_bkg.pi
    cross_arf = reg_3-2.arf
  [[cross_out]]
  cross_arf = reg_2-1.arf, reg_2-3.arf

[...]
...
-----------------------------------------------------------
"""


import numpy as np
import scipy as sp
from astropy.io import fits
from configobj import ConfigObj

import sys
import os
import argparse
from datetime import datetime


class ARF:  # {{{
    """
    Class to handle the ARF (ancillary/auxiliary response file),
    which contains the combined instrumental effective area
    (telescope/filter/detector) and the quantum efficiency (QE) as a
    function of energy averaged over time.
    The effective area is [cm^2], and the QE is [counts/photon]; they are
    multiplied together to create the ARF, resulting in [cm^2 counts/photon].

    **CAVEAT/NOTE**:
    For the XMM EPIC *pn* and Chandra *ACIS*, the generated ARF does NOT have
    the same number of data points to that of spectral channels, i.e., the
    "ENERG_LO" and "ENERG_HI" columns of ARF is different to the "E_MIN" and
    "E_MAX" columns of RMF.
    Therefore it is necessary to interpolate and extrapolate the ARF curve
    in order to match the spectrum (or RMF "EBOUNDS" extension).
    As for the XMM EPIC *MOS1* and *MOS2*, the ARF data points match the
    spectral channels, i.e., the energy positions of each ARF data point and
    spectral channel are consistent.  Thus the interpolation is not needed.

    References:
    [1] CIAO: Auxiliary Response File
        http://cxc.harvard.edu/ciao/dictionary/arf.html
    [2] Definition of RMF and ARF file formats
        https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html
    """
    filename = None
    fitsobj  = None
    # only consider the "SPECTRUM" extension
    header   = None
    energ_lo = None
    energ_hi = None
    specresp = None

    def __init__(self, filename):
        self.filename = filename
        self.fitsobj  = fits.open(filename)
        ext_specresp  = self.fitsobj["SPECRESP"]
        self.header   = ext_specresp.header
        self.energ_lo = ext_specresp.data["ENERG_LO"]
        self.energ_hi = ext_specresp.data["ENERG_HI"]
        self.specresp = ext_specresp.data["SPECRESP"]

    def get_data(self, copy=True):
        if copy:
            return self.specresp.copy()
        else:
            return self.specresp

    def get_energy(self, mean="geometric"):
        """
        Return the mean energy values of the ARF.

        Arguments:
          * mean: type of the mean energy:
                  + "geometric": geometric mean, i.e., e = sqrt(e_min*e_max)
                  + "arithmetic": arithmetic mean, i.e., e = 0.5*(e_min+e_max)
        """
        if mean == "geometric":
            energy = np.sqrt(self.energ_lo * self.energ_hi)
        elif mean == "arithmetic":
            energy = 0.5 * (self.energ_lo + self.energ_hi)
        else:
            raise ValueError("Invalid mean type: %s" % mean)
        return energy

    def interpolate(self, x=None, verbose=False):
        """
        Cubic interpolate the ARF curve using `scipy.interpolate'

        If the requested point is outside of the data range, the
        fill value of *zero* is returned.

        Arguments:
          * x: points at which the interpolation to be calculated.

        Return:
          If x is None, then the interpolated function is returned,
          otherwise, the interpolated data are returned.
        """
        energy = self.get_energy()
        arf = self.get_data(copy=False)
        if verbose:
            print("INFO: ARF interpolating (this may take a while) ...",
                    file=sys.stderr)
        f_interp = sp.interpolate.interp1d(energy, arf, kind="cubic",
                bounds_error=False, fill_value=0.0, assume_sorted=True)
        if x is not None:
            return f_interp(x)
        else:
            return f_interp
# class ARF }}}


class RMF:  # {{{
    """
    Class to handle the RMF (redistribution matrix file),
    which maps from energy space into detector pulse height (or position)
    space.  Since detectors are not perfect, this involves a spreading of
    the observed counts by the detector resolution, which is expressed as
    a matrix multiplication.
    For X-ray spectral analysis, the RMF encodes the probability R(E,p)
    that a detected photon of energy E will be assisgned to a given
    channel value (PHA or PI) of p.

    The standard Legacy format [2] for the RMF uses a binary table in which
    each row contains R(E,p) for a single value of E as a function of p.
    Non-zero sequences of elements of R(E,p) are encoded using a set of
    variable length array columns.  This format is compact but hard to
    manipulate and understand.

    **CAVEAT/NOTE**:
    + See the above ARF CAVEAT/NOTE for the XMM EPIC pn.
    + This class (currently) only deals with the "EBOUNDS" extension, which
      contains the `CHANNEL', `E_MIN' and `E_MAX' columns.  This `CHANNEL'
      is the same as that of a spectrum.  Therefore, the energy values can
      be used to interpolate and extrapolate the ARF curve.
    + The `ENERG_LO' and `ENERG_HI' columns of "MATRIX" extension are the
      same as that of a ARF.

    References:
    [1] CIAO: Redistribution Matrix File
        http://cxc.harvard.edu/ciao/dictionary/rmf.html
    [2] Definition of RMF and ARF file formats
        https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html
    """
    filename    = None
    fitsobj     = None
    ## extension "MATRIX"
    hdr_matrix  = None
    energ_lo    = None
    energ_hi    = None
    n_grp       = None
    f_chan      = None
    n_chan      = None
    # raw squeezed RMF matrix data
    matrix      = None
    ## extension "EBOUNDS"
    hdr_ebounds = None
    channel     = None
    e_min       = None
    e_max       = None
    ## converted 2D RMF matrix/image from the squeezed binary table
    #  size: len(energ_lo) x len(channel)
    rmfimg      = None

    def __init__(self, filename):
        self.filename    = filename
        self.fitsobj     = fits.open(filename)
        ## "MATRIX" extension
        ext_matrix       = self.fitsobj["MATRIX"]
        self.hdr_matrix  = ext_matrix.header
        self.energ_lo    = ext_matrix.data["ENERG_LO"]
        self.energ_hi    = ext_matrix.data["ENERG_HI"]
        self.n_grp       = ext_matrix.data["N_GRP"]
        self.f_chan      = ext_matrix.data["F_CHAN"]
        self.n_chan      = ext_matrix.data["N_CHAN"]
        self.matrix      = ext_matrix.data["MATRIX"]
        ## "EBOUNDS" extension
        ext_ebounds      = self.fitsobj["EBOUNDS"]
        self.hdr_ebounds = ext_ebounds.header
        self.channel     = ext_ebounds.data["CHANNEL"]
        self.e_min       = ext_ebounds.data["E_MIN"]
        self.e_max       = ext_ebounds.data["E_MAX"]

    def get_energy(self, mean="geometric"):
        """
        Return the mean energy values of the RMF "EBOUNDS".

        Arguments:
          * mean: type of the mean energy:
                  + "geometric": geometric mean, i.e., e = sqrt(e_min*e_max)
                  + "arithmetic": arithmetic mean, i.e., e = 0.5*(e_min+e_max)
        """
        if mean == "geometric":
            energy = np.sqrt(self.e_min * self.e_max)
        elif mean == "arithmetic":
            energy = 0.5 * (self.e_min + self.e_max)
        else:
            raise ValueError("Invalid mean type: %s" % mean)
        return energy

    def get_rmfimg(self):
        """
        Convert the RMF data in squeezed binary table (standard Legacy format)
        to a 2D image/matrix.
        """
        def _make_rmfimg_row(n_channel, dtype, f_chan, n_chan, mat_row):
            f_chan -= 1  # FITS indices are 1-based
            idx = np.concatenate([ np.arange(f, f+n) \
                    for f, n in zip(f_chan, n_chan) ])
            rmfrow = np.zeros(n_channel, dtype=dtype)
            rmfrow[idx] = mat_row
            return rmfrow
        #
        if self.rmfimg is None:
            # Make the 2D RMF matrix/image
            n_energy  = len(self.energ_lo)
            n_channel = len(self.channel)
            rmf_dtype = self.matrix[0].dtype
            rmfimg = np.zeros(shape=(n_energy, n_channel), dtype=rmf_dtype)
            for i in range(n_energy):
                rmfimg[i, :] = _make_rmfimg_row(n_channel, rmf_dtype,
                        self.f_chan[i], self.n_chan[i], self.matrix[i])
            self.rmfimg = rmfimg
        return self.rmfimg

    def write_rmfimg(self, outfile, clobber=False):
        rmfimg = self.get_rmfimg()
        # merge headers
        header = self.hdr_matrix.copy(strip=True)
        header.extend(self.hdr_ebounds.copy(strip=True))
        outfits = fits.PrimaryHDU(data=rmfimg, header=header)
        outfits.writeto(outfile, checksum=True, clobber=clobber)
# class RMF }}}


class Spectrum:  # {{{
    """
    Class that deals with the X-ray spectrum file (usually *.pi).

    TODO:
    * to implement the grouping function (and quality columns)
    """
    filename  = None
    # FITS object return by `fits.open()'
    fitsobj   = None
    # header of "SPECTRUM" extension
    header    = None
    # "SPECTRUM" extension data
    channel   = None
    # name of the spectrum data column (i.e., type, "COUNTS" or "RATE")
    spec_type = None
    # unit of the spectrum data ("count" for "COUNTS", "count/s" for "RATE")
    spec_unit = None
    # spectrum data
    spec_data = None
    # several important keywords
    EXPOSURE  = None
    BACKSCAL  = None
    RESPFILE  = None
    ANCRFILE  = None
    BACKFILE  = None
    # numpy dtype and FITS format code of the spectrum data
    spec_dtype       = None
    spec_fits_format = None

    def __init__(self, filename):
        self.filename = filename
        self.fitsobj  = fits.open(filename)
        ext_spec      = self.fitsobj["SPECTRUM"]
        self.header   = ext_spec.header.copy(strip=True)
        colnames      = ext_spec.columns.names
        if "COUNTS" in colnames:
            self.spec_type        = "COUNTS"
            self.spec_unit        = "count"
            self.spec_dtype       = np.int32
            self.spec_fits_format = "J"
        elif "RATE" in colnames:
            self.spec_type        = "RATE"
            self.spec_unit        = "count/s"
            self.spec_dtype       = np.float32
            self.spec_fits_format = "E"
        else:
            raise ValueError("Invalid spectrum file")
        self.channel   = ext_spec.data["CHANNEL"].copy()
        self.spec_data = ext_spec.data.field(self.spec_type)\
                .astype(self.spec_dtype)
        # keywords
        self.EXPOSURE = self.header.get("EXPOSURE")
        self.BACKSCAL = self.header.get("BACKSCAL")
        self.AREASCAL = self.header.get("AREASCAL")
        self.RESPFILE = self.header.get("RESPFILE")
        self.ANCRFILE = self.header.get("ANCRFILE")
        self.BACKFILE = self.header.get("BACKFILE")

    def get_data(self, copy=True):
        if copy:
            return self.spec_data.copy()
        else:
            return self.spec_data

    def get_channel(self, copy=True):
        if copy:
            return self.channel.copy()
        else:
            return self.channel

    def reset_header_keywords(self,
            keywords=["ANCRFILE", "RESPFILE", "BACKFILE"]):
        """
        Reset the keywords to "NONE" to avoid confusion or mistakes.
        """
        for kw in keywords:
            if kw in self.header:
                header[kw] = "NONE"

    def write(self, filename, clobber=False):
        """
        Create a new "SPECTRUM" table/extension and replace the original
        one, then write to output file.
        """
        ext_spec_cols = fits.ColDefs([
                fits.Column(name="CHANNEL", format="I", array=self.channel),
                fits.Column(name=self.spec_type, format=self.spec_fits_format,
                    unit=self.spec_unit, array=self.spec_data)])
        ext_spec = fits.BinTableHDU.from_columns(ext_spec_cols,
                header=self.header)
        self.fitsobj["SPECTRUM"] = ext_spec
        self.fitsobj.writeto(filename, clobber=clobber, checksum=True)
# class Spectrum }}}


class SpectrumSet(Spectrum):  # {{{
    """
    This class handles a set of spectrum, including the source spectrum,
    RMF, ARF, and the background spectrum.

    TODO:
    * Subtract background spectrum before correcting crosstalk effects
    * Estimate channel errors by Monte Carlo simulations

    **NOTE**:
    The "COUNTS" column data are converted from "int32" to "float32",
    since this spectrum will be subtracted/compensated according to the
    ratios of ARFs.
    """
    # ARF object for this spectrum
    arf = None
    # RMF object for this spectrum
    rmf = None
    # background Spectrum object for this spectrum
    bkg = None

    # numpy dtype and FITS format code to which the spectrum data be
    # converted if the data is "COUNTS"
    _spec_dtype       = np.float32
    _spec_fits_format = "E"

    def __init__(self, filename, arffile=None, rmffile=None, bkgfile=None):
        super(self.__class__, self).__init__(filename)
        # convert spectrum data type if necessary
        if self.spec_data.dtype != self._spec_dtype:
            self.spec_data        = self.spec_data.astype(self._spec_dtype)
            self.spec_dtype       = self._spec_dtype
            self.spec_fits_format = self._spec_fits_format
        if arffile is not None:
            self.arf = ARF(arffile)
        if rmffile is not None:
            self.rmf = RMF(rmffile)
        if bkgfile is not None:
            self.bkg = Spectrum(bkgfile)

    def get_energy(self, mean="geometric"):
        """
        Get the energy values of each channel if RMF present.
        """
        if self.rmf is None:
            return None
        else:
            return self.rmf.get_energy(mean=mean)

    def get_arf(self, copy=True):
        """
        Get the corresponding ARF curve data for this spectrum.
        """
        if self.arf is None:
            return None
        else:
            return self.arf.get_data(copy=copy)

    def subtract_bkg(self, inplace=True):
        """
        Subtract the background contribution from the source spectrum.
        The `EXPOSURE' and `BACKSCAL' values are required to calculate
        the fraction/ratio for the background subtraction.

        Arguments:
          * inplace: whether replace the `spec_data' with the background-
                     subtracted spectrum data; If True, the attribute
                     `spec_bkg_subtracted' is also set to `True' when
                     the subtraction finished.

        Return:
          background-subtracted spectrum data
        """
        ratio = (self.EXPOSURE / self.bkg.EXPOSURE) * \
                (self.BACKSCAL / self.bkg.BACKSCAL) * \
                (self.AREASCAL / self.bkg.AREASCAL)
        spec_data_subbkg = self.spec_data - ratio * self.bkg.get_data()
        if inplace:
            self.spec_data = spec_data_subbkg
            self.spec_bkg_subtracted = True
        return spec_data_subbkg

    def subtract(self, spectrum, cross_arf, verbose=False):
        """
        Subtract the photons that originate from the surrounding regions
        but were scattered into this spectrum due to the finite PSF.

        The background of this spectrum and the given spectrum should
        both be subtracted before applying this subtraction for crosstalk
        correction, as well as the below `compensate()' procedure.

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
# class Spectrum }}}


class Crosstalk:  # {{{
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
        Arguments:
          * config: a section of the whole config file (`ConfigObj' object)
        """
        self.cross_in_spec = []
        self.cross_in_arf  = []
        self.cross_out_arf = []
        # this spectrum to be corrected
        self.spectrum = Spectrum(filename=config["spec"],
                arffile=config["arf"], rmffile=config["rmf"],
                bkgfile=config["bkg"])
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
        """
        Perform the crosstalk correction.
        """
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
        self.spectrum.reset_header_keywords(
                keywords=["ANCRFILE", "RESPFILE", "BACKFILE"])
        self.spectrum.write(filename, clobber=clobber)
# class Crosstalk }}}


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
