#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# References:
# [1] Definition of RMF and ARF file formats
#     https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html
# [2] CIAO: Auxiliary Response File
#     http://cxc.harvard.edu/ciao/dictionary/arf.html
# [3] CIAO: Redistribution Matrix File
#     http://cxc.harvard.edu/ciao/dictionary/rmf.html
# [4] astropy - FITS format code
#     http://docs.astropy.org/en/stable/io/fits/usage/table.html#column-creation
# [5] XSPEC - Spectral Fitting
#     https://heasarc.gsfc.nasa.gov/docs/xanadu/xspec/manual/XspecSpectralFitting.html
# [6] Direct X-ray Spectra Deprojection
#     https://www-xray.ast.cam.ac.uk/papers/dsdeproj/
#     Sanders & Fabian 2007, MNRAS, 381, 1381
#
#
# Weitian LI
# Created: 2016-03-26
# Updated: 2016-04-20
#
# ChangeLog:
# 2016-04-20:
#   * Add argument 'add_history' to some methods (to avoid many duplicated
#     histories due to Monte Carlo)
# 2016-04-19:
#   * Ignore numpy error due to division by zero
#   * Update tool description and sample configuration
#   * Add two other main methods: `main_deprojection()' and `main_crosstalk()'
#   * Add argument 'group_squeeze' to some methods for better performance
#   * Rename from 'correct_crosstalk.py' to 'crosstalk_deprojection.py'
# 2016-04-18:
#   * Implement deprojection function: class Deprojection
#   * Support spectral grouping (supply the grouping specification)
#   * Add grouping, estimate_errors, copy, randomize, etc. methods
#   * Utilize the Monte Carlo techniques to estimate the final spectral errors
#   * Collect all ARFs and RMFs within dictionaries
# 2016-04-06:
#   * Fix `RMF: get_rmfimg()' for XMM EPIC RMF
# 2016-04-02:
#   * Interpolate ARF in order to match the spectral channel energies
#   * Add version and date information
#   * Update documentations
#   * Update header history contents
# 2016-04-01:
#   * Greatly update the documentations (e.g., description, sample config)
#   * Add class `RMF'
#   * Add method `get_energy()' for class `ARF'
#   * Split out class `SpectrumSet' from `Spectrum'
#   * Implement background subtraction
#   * Add config `subtract_bkg' and corresponding argument
#
# XXX/FIXME:
#   * Deprojection: account for ARF differences across different regions
#
# TODO:
#   * Split classes ARF, RMF, Spectrum, and SpectrumSet to a separate module
#

__version__ = "0.5.1"
__date__    = "2016-04-19"


"""
Correct the crosstalk effect of XMM spectra by subtracting the photons
that scattered from the surrounding regions due to the finite PSF, and
by compensating the photons that scattered to the surrounding regions,
according to the generated crosstalk ARFs by SAS `arfgen'.

After the crosstalk effect being corrected, the deprojection is performed
to deproject the crosstalk-corrected spectra to derive the spectra with
both the crosstalk effect and projection effect corrected.


Sample config file (in `ConfigObj' syntax):
-----------------------------------------------------------
# operation mode: deprojection, crosstalk, or both (default)
mode = both
# supply a *groupped* spectrum (from which the "GROUPING" and "QUALITY"
# are used to group all the following spectra)
grouping = spec_grp.pi
# whether to subtract the background before crosstalk correction
subtract_bkg = True
# whether to fix the negative channel values due to spectral subtractions
fix_negative = False
# Monte Carlo times for spectral error estimation
mc_times = 5000
# show progress details and verbose information
verbose = True
# overwrite existing files
clobber = False

# NOTE:
# ONLY specifiy ONE set of projected spectra (i.e., from the same detector
# of one observation), since ALL the following specified spectra will be
# used for the deprojection.

[reg1]
...

[reg2]
outfile = deprojcc_reg2.pi
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

WARNING = """
********************************* WARNING ************************************
The generated spectra are substantially modified (e.g., scale, add, subtract),
therefore, take special care when interpretating the fitting results,
especially the metal abundances and normalizations.
******************************************************************************
"""


import sys
import os
import argparse
from datetime import datetime
from copy import copy

import numpy as np
import scipy as sp
import scipy.interpolate
from astropy.io import fits
from configobj import ConfigObj


def group_data(data, grouping):
    """
    Group the data with respect to the supplied `grouping' specification
    (i.e., "GROUPING" columns of a spectrum).  The channel counts of the
    same group are summed up and assigned to the FIRST channel of this
    group, while the OTHRE channels are all set to ZERO.
    """
    data_grp = np.array(data).copy()
    for i in reversed(range(len(data))):
        if grouping[i] == 1:
            # the beginning channel of a group
            continue
        else:
            # other channels of a group
            data_grp[i-1] += data_grp[i]
            data_grp[i]    = 0
    assert np.isclose(sum(data_grp), sum(data))
    return data_grp


class ARF:  # {{{
    """
    Class to handle the ARF (ancillary/auxiliary response file),
    which contains the combined instrumental effective area
    (telescope/filter/detector) and the quantum efficiency (QE) as a
    function of energy averaged over time.
    The effective area is [cm^2], and the QE is [counts/photon]; they are
    multiplied together to create the ARF, resulting in [cm^2 counts/photon].

    **CAVEAT/NOTE**:
    Generally, the "ENERG_LO" and "ENERG_HI" columns of an ARF are *different*
    to the "E_MIN" and "E_MAX" columns of a RMF (which are corresponding
    to the spectrum channel energies).
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
    # function of the interpolated ARF
    f_interp = None
    # energies of the spectral channels
    energy_channel = None
    # spectral channel grouping specification
    grouping       = None
    groupped       = False
    # groupped ARF channels with respect to the grouping
    specresp_grp   = None

    def __init__(self, filename):
        self.filename = filename
        self.fitsobj  = fits.open(filename)
        ext_specresp  = self.fitsobj["SPECRESP"]
        self.header   = ext_specresp.header
        self.energ_lo = ext_specresp.data["ENERG_LO"]
        self.energ_hi = ext_specresp.data["ENERG_HI"]
        self.specresp = ext_specresp.data["SPECRESP"]

    def get_data(self, groupped=False, group_squeeze=False, copy=True):
        if groupped:
            specresp = self.specresp_grp
            if group_squeeze:
                specresp = specresp[self.grouping == 1]
        else:
            specresp = self.specresp
        if copy:
            return specresp.copy()
        else:
            return specresp

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
        if not hasattr(self, "f_interp") or self.f_interp is None:
            energy = self.get_energy()
            arf    = self.get_data(copy=False)
            if verbose:
                print("INFO: interpolating '%s' (this may take a while) ..." \
                        % self.filename, file=sys.stderr)
            f_interp = sp.interpolate.interp1d(energy, arf, kind="cubic",
                    bounds_error=False, fill_value=0.0, assume_sorted=True)
            self.f_interp = f_interp
        if x is not None:
            return self.f_interp(x)
        else:
            return self.f_interp

    def apply_grouping(self, energy_channel, grouping, verbose=False):
        """
        Group the ARF channels (INTERPOLATED with respect to the spectral
        channels) by the supplied grouping specification.

        Arguments:
          * energy_channel: energies of the spectral channel
          * grouping: spectral grouping specification

        Return: `self.specresp_grp'
        """
        if self.groupped:
            return
        if verbose:
            print("INFO: Grouping spectrum '%s' ..." % self.filename,
                    file=sys.stderr)
        self.energy_channel = energy_channel
        self.grouping = grouping
        # interpolate the ARF w.r.t the spectral channel energies
        arf_interp = self.interpolate(x=energy_channel, verbose=verbose)
        self.specresp_grp = group_data(arf_interp, grouping)
        self.groupped = True
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
    + See also the above ARF CAVEAT/NOTE.
    + The "EBOUNDS" extension contains the `CHANNEL', `E_MIN' and `E_MAX'
      columns.  This `CHANNEL' is the same as that of a spectrum.  Therefore,
      the energy values determined from the `E_MIN' and `E_MAX' columns are
      used to interpolate and extrapolate the ARF curve.
    + The `ENERG_LO' and `ENERG_HI' columns of the "MATRIX" extension are
      the same as that of a ARF.

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
            # make sure that `f_chan' and `n_chan' are 1-D numpy array
            f_chan = np.array(f_chan).reshape(-1)
            f_chan -= 1  # FITS indices are 1-based
            n_chan = np.array(n_chan).reshape(-1)
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
            for i in np.arange(n_energy)[self.n_grp > 0]:
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
    # estimated spectral errors for each channel/group
    spec_err  = None
    # statistical errors for each channel/group
    stat_err  = None
    # grouping and quality
    grouping  = None
    quality   = None
    # whether the spectral data being groupped
    groupped  = False
    # several important keywords
    EXPOSURE  = None
    BACKSCAL  = None
    RESPFILE  = None
    ANCRFILE  = None
    BACKFILE  = None
    # numpy dtype and FITS format code of the spectrum data
    spec_dtype       = None
    spec_fits_format = None
    # output filename for writing the spectrum if no filename provided
    outfile = None

    def __init__(self, filename, outfile=None):
        self.filename = filename
        self.fitsobj  = fits.open(filename)
        ext_spec      = self.fitsobj["SPECTRUM"]
        self.header   = ext_spec.header.copy(strip=True)
        colnames      = ext_spec.columns.names
        if "COUNTS" in colnames:
            self.spec_type = "COUNTS"
        elif "RATE" in colnames:
            self.spec_type = "RATE"
        else:
            raise ValueError("Invalid spectrum file")
        self.channel          = ext_spec.data.columns["CHANNEL"].array
        col_spec_data         = ext_spec.data.columns[self.spec_type]
        self.spec_data        = col_spec_data.array.copy()
        self.spec_unit        = col_spec_data.unit
        self.spec_dtype       = col_spec_data.dtype
        self.spec_fits_format = col_spec_data.format
        # grouping and quality
        if "GROUPING" in colnames:
            self.grouping = ext_spec.data.columns["GROUPING"].array
        if "QUALITY"  in colnames:
            self.quality  = ext_spec.data.columns["QUALITY"].array
        # keywords
        self.EXPOSURE = self.header.get("EXPOSURE")
        self.BACKSCAL = self.header.get("BACKSCAL")
        self.AREASCAL = self.header.get("AREASCAL")
        self.RESPFILE = self.header.get("RESPFILE")
        self.ANCRFILE = self.header.get("ANCRFILE")
        self.BACKFILE = self.header.get("BACKFILE")
        # output filename
        self.outfile  = outfile

    def get_data(self, group_squeeze=False, copy=True):
        """
        Get the spectral data (i.e., self.spec_data).

        Arguments:
          * group_squeeze: whether squeeze the spectral data according to
                           the grouping (i.e., exclude the channels that
                           are not the first channel of the group, which
                           also have value of ZERO).
                           This argument is effective only the grouping
                           being applied.
        """
        if group_squeeze and self.groupped:
            spec_data = self.spec_data[self.grouping == 1]
        else:
            spec_data = self.spec_data
        if copy:
            return spec_data.copy()
        else:
            return spec_data

    def get_channel(self, copy=True):
        if copy:
            return self.channel.copy()
        else:
            return self.channel

    def set_data(self, spec_data, group_squeeze=True):
        """
        Set the spectral data of this spectrum to the supplied data.
        """
        if group_squeeze and self.groupped:
            assert sum(self.grouping == 1) == len(spec_data)
            self.spec_data[self.grouping == 1] = spec_data
        else:
            assert len(self.spec_data) == len(spec_data)
            self.spec_data = spec_data.copy()

    def add_stat_err(self, stat_err, group_squeeze=True):
        """
        Add the "STAT_ERR" column as the statistical errors of each spectral
        group, which are estimated by utilizing the Monte Carlo techniques.
        """
        self.stat_err = np.zeros(self.spec_data.shape,
                dtype=self.spec_data.dtype)
        if group_squeeze and self.groupped:
            assert sum(self.grouping == 1) == len(stat_err)
            self.stat_err[self.grouping == 1] = stat_err
        else:
            assert len(self.stat_err) == len(stat_err)
            self.stat_err = stat_err.copy()
        self.header["POISSERR"] = False

    def apply_grouping(self, grouping=None, quality=None):
        """
        Apply the spectral channel grouping specification to the spectrum.

        NOTE:
        * The spectral data (i.e., self.spec_data) is MODIFIED!
        * The spectral data within the same group are summed up.
        * The self grouping is overwritten if `grouping' is supplied, as well
          as the self quality.
        """
        if grouping is not None:
            self.grouping = grouping
        if quality is not None:
            self.quality = quality
        self.spec_data = group_data(self.spec_data, self.grouping)
        self.groupped = True

    def estimate_errors(self, gehrels=True):
        """
        Estimate the statistical errors of each spectral group (after
        applying grouping) for the source spectrum (and background spectrum).

        If `gehrels=True', the statistical error for a spectral group with
        N photons is given by `1 + sqrt(N + 0.75)'; otherwise, the error
        is given by `sqrt(N)'.

        Results: `self.spec_err'
        """
        eps = 1.0e-10
        if gehrels:
            self.spec_err = 1.0 + np.sqrt(self.spec_data + 0.75)
        else:
            self.spec_err = np.sqrt(self.spec_data)
        # replace the zeros with a very small value (because
        # `np.random.normal' requires `scale' > 0)
        self.spec_err[self.spec_err <= 0.0] = eps

    def copy(self):
        """
        Return a copy of this object, with the `np.ndarray' properties are
        copied.
        """
        new = copy(self)
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                setattr(new, k, v.copy())
        return new

    def randomize(self):
        """
        Randomize the spectral data according to the estimated spectral
        group errors by assuming the normal distribution.

        NOTE: this method should be called AFTER the `copy()' method.
        """
        if self.spec_err is None:
            raise ValueError("No valid 'spec_err' presents")
        if self.groupped:
            idx = self.grouping == 1
            self.spec_data[idx] = np.random.normal(self.spec_data[idx],
                    self.spec_err[idx])
        else:
            self.spec_data = np.random.normal(self.spec_data, self.spec_err)
        return self

    def reset_header_keywords(self,
            keywords=["ANCRFILE", "RESPFILE", "BACKFILE"]):
        """
        Reset the keywords to "NONE" to avoid confusion or mistakes.
        """
        for kw in keywords:
            if kw in self.header:
                self.header[kw] = "NONE"

    def write(self, filename=None, clobber=False):
        """
        Create a new "SPECTRUM" table/extension and replace the original
        one, then write to output file.
        """
        if filename is None:
            filename = self.outfile
        columns = [
                fits.Column(name="CHANNEL", format="I", array=self.channel),
                fits.Column(name=self.spec_type, format=self.spec_fits_format,
                            unit=self.spec_unit, array=self.spec_data),
        ]
        if self.grouping is not None:
            columns.append(fits.Column(name="GROUPING",
                                       format="I", array=self.grouping))
        if self.quality  is not None:
            columns.append(fits.Column(name="QUALITY",
                                       format="I", array=self.quality))
        if self.stat_err is not None:
            columns.append(fits.Column(name="STAT_ERR", unit=self.spec_unit,
                                       format=self.spec_fits_format,
                                       array=self.stat_err))
        ext_spec_cols = fits.ColDefs(columns)
        ext_spec = fits.BinTableHDU.from_columns(ext_spec_cols,
                                                 header=self.header)
        self.fitsobj["SPECTRUM"] = ext_spec
        self.fitsobj.writeto(filename, clobber=clobber, checksum=True)
# class Spectrum }}}


class SpectrumSet(Spectrum):  # {{{
    """
    This class handles a set of spectrum, including the source spectrum,
    RMF, ARF, and the background spectrum.

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
    # inner and outer radius of the region from which the spectrum extracted
    radius_inner = None
    radius_outer = None
    # total angular range of the spectral region
    angle = None

    # numpy dtype and FITS format code to which the spectrum data be
    # converted if the data is "COUNTS"
    #_spec_dtype       = np.float32
    #_spec_fits_format = "E"
    _spec_dtype       = np.float64
    _spec_fits_format = "D"

    def __init__(self, filename, outfile=None, arf=None, rmf=None, bkg=None):
        super().__init__(filename, outfile)
        # convert spectrum data type if necessary
        if self.spec_data.dtype != self._spec_dtype:
            self.spec_data        = self.spec_data.astype(self._spec_dtype)
            self.spec_dtype       = self._spec_dtype
            self.spec_fits_format = self._spec_fits_format
        if arf is not None:
            if isinstance(arf, ARF):
                self.arf = arf
            else:
                self.arf = ARF(arf)
        if rmf is not None:
            if isinstance(rmf, RMF):
                self.rmf = rmf
            else:
                self.rmf = RMF(rmf)
        if bkg is not None:
            if isinstance(bkg, Spectrum):
                self.bkg = bkg
            else:
                self.bkg = Spectrum(bkg)
            # convert background spectrum data type if necessary
            if self.bkg.spec_data.dtype != self._spec_dtype:
                self.bkg.spec_data = self.bkg.spec_data.astype(self._spec_dtype)
                self.bkg.spec_dtype = self._spec_dtype
                self.bkg.spec_fits_format = self._spec_fits_format

    def get_energy(self, mean="geometric"):
        """
        Get the energy values of each channel if RMF present.

        NOTE:
        The "E_MIN" and "E_MAX" columns of the RMF is required to calculate
        the spectrum channel energies.
        And the channel energies are generally different to the "ENERG_LO"
        and "ENERG_HI" of the corresponding ARF.
        """
        if self.rmf is None:
            return None
        else:
            return self.rmf.get_energy(mean=mean)

    def get_arf(self, mean="geometric", groupped=True, copy=True):
        """
        Get the interpolated ARF data w.r.t the spectral channel energies
        if the ARF presents.

        Arguments:
          * groupped: (bool) whether to get the groupped ARF

        Return: (groupped) interpolated ARF data
        """
        if self.arf is None:
            return None
        else:
            return self.arf.get_data(groupped=groupped, copy=copy)

    def read_xflt(self):
        """
        Read the XFLT000# keywords from the header, check the validity (e.g.,
        "XFLT0001" should equals "XFLT0002", "XFLT0003" should equals 0).
        Sum all the additional XFLT000# pairs (e.g., ) which describes the
        regions angluar ranges.
        """
        eps = 1.0e-6
        xflt0001 = float(self.header["XFLT0001"])
        xflt0002 = float(self.header["XFLT0002"])
        xflt0003 = float(self.header["XFLT0003"])
        # XFLT000# validity check
        assert np.isclose(xflt0001, xflt0002)
        assert abs(xflt0003) < eps
        # outer radius of the region
        self.radius_outer = xflt0001
        # angular regions
        self.angle = 0.0
        num = 4
        while True:
            try:
                angle_begin = float(self.header["XFLT%04d" % num])
                angle_end   = float(self.header["XFLT%04d" % (num+1)])
                num += 2
            except KeyError:
                break
            self.angle += (angle_end - angle_begin)
        # if NO additional XFLT000# keys exist, assume "annulus" region
        if self.angle < eps:
            self.angle = 360.0

    def scale(self):
        """
        Scale the spectral data (and spectral group errors if present) of
        the source spectrum (and background spectra if present) according
        to the region angular size to make it correspond to the whole annulus
        region (i.e., 360 degrees).

        NOTE: the spectral data and errors (i.e., `self.spec_data', and
        `self.spec_err') is MODIFIED!
        """
        self.spec_data *= (360.0 / self.angle)
        if self.spec_err is not None:
            self.spec_err *= (360.0 / self.angle)
        # also scale the background spectrum if present
        if self.bkg:
            self.bkg.spec_data *= (360.0 / self.angle)
            if self.bkg.spec_err is not None:
                self.bkg.spec_err *= (360.0 / self.angle)

    def apply_grouping(self, grouping=None, quality=None, verbose=False):
        """
        Apply the spectral channel grouping specification to the source
        spectrum, the ARF (which is used during the later spectral
        manipulations), and the background spectrum (if presents).

        NOTE:
        * The spectral data (i.e., self.spec_data) is MODIFIED!
        * The spectral data within the same group are summed up.
        * The self grouping is overwritten if `grouping' is supplied, as well
          as the self quality.
        """
        super().apply_grouping(grouping=grouping, quality=quality)
        # also group the ARF accordingly
        self.arf.apply_grouping(energy_channel=self.get_energy(),
                                grouping=self.grouping, verbose=verbose)
        # group the background spectrum if present
        if self.bkg:
            self.bkg.spec_data = group_data(self.bkg.spec_data, self.grouping)

    def estimate_errors(self, gehrels=True):
        """
        Estimate the statistical errors of each spectral group (after
        applying grouping) for the source spectrum (and background spectrum).

        If `gehrels=True', the statistical error for a spectral group with
        N photons is given by `1 + sqrt(N + 0.75)'; otherwise, the error
        is given by `sqrt(N)'.

        Results: `self.spec_err' (and `self.bkg.spec_err')
        """
        super().estimate_errors(gehrels=gehrels)
        eps = 1.0e-10
        # estimate the errors for background spectrum if present
        if self.bkg:
            if gehrels:
                self.bkg.spec_err = 1.0 + np.sqrt(self.bkg.spec_data + 0.75)
            else:
                self.bkg.spec_err = np.sqrt(self.bkg.spec_data)
            self.bkg.spec_err[self.bkg.spec_err <= 0.0] = eps

    def subtract_bkg(self, inplace=True, add_history=False, verbose=False):
        """
        Subtract the background contribution from the source spectrum.
        The `EXPOSURE' and `BACKSCAL' values are required to calculate
        the fraction/ratio for the background subtraction.

        Arguments:
          * inplace: whether replace the `spec_data' with the background-
                     subtracted spectrum data; If True, the attribute
                     `spec_bkg_subtracted' is also set to `True' when
                     the subtraction finished.
                     The keywords "BACKSCAL" and "AREASCAL" are set to 1.0.

        Return:
          background-subtracted spectrum data
        """
        ratio = (self.EXPOSURE / self.bkg.EXPOSURE) * \
                (self.BACKSCAL / self.bkg.BACKSCAL) * \
                (self.AREASCAL / self.bkg.AREASCAL)
        operation = "  SUBTRACT_BACKGROUND: %s - %s * %s" % \
                (self.filename, ratio, self.bkg.filename)
        if verbose:
            print(operation, file=sys.stderr)
        spec_data_subbkg = self.spec_data - ratio * self.bkg.get_data()
        if inplace:
            self.spec_data = spec_data_subbkg
            self.spec_bkg_subtracted = True
            self.BACKSCAL = 1.0
            self.AREASCAL = 1.0
            # also record history
            if add_history:
                self.header.add_history(operation)
        return spec_data_subbkg

    def subtract(self, spectrumset, cross_arf, groupped=False,
            group_squeeze=False, add_history=False, verbose=False):
        """
        Subtract the photons that originate from the surrounding regions
        but were scattered into this spectrum due to the finite PSF.

        The background of this spectrum and the given spectrum should
        both be subtracted before applying this subtraction for crosstalk
        correction, as well as the below `compensate()' procedure.

        NOTE:
        1. The crosstalk ARF must be provided, since the `spectrumset.arf'
           is required to be its ARF without taking crosstalk into account:
               spec1_new = spec1 - spec2 * (cross_arf_2_to_1 / arf2)
        2. The ARF are interpolated to match the energies of spetral channels.
        """
        operation = "  SUBTRACT: %s - (%s/%s) * %s" % (self.filename,
                cross_arf.filename, spectrumset.arf.filename,
                spectrumset.filename)
        if verbose:
            print(operation, file=sys.stderr)
        energy = self.get_energy()
        if groupped:
            spectrumset.arf.apply_grouping(energy_channel=energy,
                    grouping=self.grouping, verbose=verbose)
            cross_arf.apply_grouping(energy_channel=energy,
                    grouping=self.grouping, verbose=verbose)
            arfresp_spec  = spectrumset.arf.get_data(groupped=True,
                    group_squeeze=group_squeeze)
            arfresp_cross = cross_arf.get_data(groupped=True,
                    group_squeeze=group_squeeze)
        else:
            arfresp_spec  = spectrumset.arf.interpolate(x=energy,
                    verbose=verbose)
            arfresp_cross = cross_arf.interpolate(x=energy, verbose=verbose)
        with np.errstate(divide="ignore", invalid="ignore"):
            arf_ratio = arfresp_cross / arfresp_spec
            # fix nan/inf values due to division by zero
            arf_ratio[ ~ np.isfinite(arf_ratio) ] = 0.0
        spec_data = self.get_data(group_squeeze=group_squeeze) - \
                spectrumset.get_data(group_squeeze=group_squeeze)*arf_ratio
        self.set_data(spec_data, group_squeeze=group_squeeze)
        # record history
        if add_history:
            self.header.add_history(operation)

    def compensate(self, cross_arf, groupped=False, group_squeeze=False,
            add_history=False, verbose=False):
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
        energy = self.get_energy()
        if groupped:
            cross_arf.apply_grouping(energy_channel=energy,
                    grouping=self.grouping, verbose=verbose)
            arfresp_this  = self.arf.get_data(groupped=True,
                    group_squeeze=group_squeeze)
            arfresp_cross = cross_arf.get_data(groupped=True,
                    group_squeeze=group_squeeze)
        else:
            arfresp_this  = self.arf.interpolate(x=energy, verbose=verbose)
            arfresp_cross = cross_arf.interpolate(x=energy, verbose=verbose)
        with np.errstate(divide="ignore", invalid="ignore"):
            arf_ratio = arfresp_cross / arfresp_this
            # fix nan/inf values due to division by zero
            arf_ratio[ ~ np.isfinite(arf_ratio) ] = 0.0
        spec_data = self.get_data(group_squeeze=group_squeeze) + \
                self.get_data(group_squeeze=group_squeeze) * arf_ratio
        self.set_data(spec_data, group_squeeze=group_squeeze)
        # record history
        if add_history:
            self.header.add_history(operation)

    def fix_negative(self, add_history=False, verbose=False):
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
                if i == 1:
                    print("*** Fixing negative channels: iter %d..." % i,
                            end="", file=sys.stderr)
                else:
                    print("%d..." % i, end="", file=sys.stderr)
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
            print("FIXED!", file=sys.stderr)
            # record history
            if add_history:
                self.header.add_history("  FIXED NEGATIVE CHANNELS")

    def set_radius_inner(self, radius_inner):
        """
        Set the inner radius of the spectral region.
        """
        assert radius_inner < self.radius_outer
        self.radius_inner = radius_inner

    def copy(self):
        """
        Return a copy of this object.
        """
        new = super().copy()
        if self.bkg:
            new.bkg = self.bkg.copy()
        return new

    def randomize(self):
        """
        Randomize the source (and background if present) spectral data
        according to the estimated spectral group errors by assuming the
        normal distribution.

        NOTE: this method should be called AFTER the `copy()' method.
        """
        super().randomize()
        if self.bkg:
            self.bkg.spec_data = np.random.normal(self.bkg.spec_data,
                                                  self.bkg.spec_err)
            self.bkg.spec_data[self.grouping == -1] = 0.0
        return self
# class SpectrumSet }}}


class Crosstalk:  # {{{
    """
    XMM-Newton PSF Crosstalk effect correction.
    """
    # `SpectrumSet' object for the spectrum to be corrected
    spectrumset = None
    # NOTE/XXX: do NOT use list (e.g., []) here, otherwise, all the
    #           instances will share these list properties.
    # `SpectrumSet' and `ARF' objects corresponding to the spectra from
    # which the photons were scattered into this spectrum.
    cross_in_specset = None
    cross_in_arf     = None
    # `ARF' objects corresponding to the regions to which the photons of
    # this spectrum were scattered into.
    cross_out_arf = None
    # grouping specification and quality data
    grouping = None
    quality = None
    # whether the spectrum is groupped
    groupped = False

    def __init__(self, config, arf_dict={}, rmf_dict={},
            grouping=None, quality=None):
        """
        Arguments:
          * config: a section of the whole config file (`ConfigObj' object)
        """
        self.cross_in_specset = []
        self.cross_in_arf     = []
        self.cross_out_arf    = []
        # this spectrum to be corrected
        self.spectrumset = SpectrumSet(filename=config["spec"],
                outfile=config["outfile"],
                arf=arf_dict.get(config["arf"], config["arf"]),
                rmf=rmf_dict.get(config.get("rmf"), config.get("rmf")),
                bkg=config.get("bkg"))
        # spectra and cross arf from which photons were scattered in
        for reg_in in config["cross_in"].values():
            specset = SpectrumSet(filename=reg_in["spec"],
                    arf=arf_dict.get(reg_in["arf"], reg_in["arf"]),
                    rmf=rmf_dict.get(reg_in.get("rmf"), reg_in.get("rmf")),
                    bkg=reg_in.get("bkg"))
            self.cross_in_specset.append(specset)
            self.cross_in_arf.append(arf_dict.get(reg_in["cross_arf"],
                                                  ARF(reg_in["cross_arf"])))
        # regions into which the photons of this spectrum were scattered into
        if "cross_out" in config.sections:
            cross_arf = config["cross_out"].as_list("cross_arf")
            for arffile in cross_arf:
                self.cross_out_arf.append(arf_dict.get(arffile, ARF(arffile)))
        # grouping and quality
        self.grouping = grouping
        self.quality = quality

    def apply_grouping(self, verbose=False):
        self.spectrumset.apply_grouping(grouping=self.grouping,
                quality=self.quality, verbose=verbose)
        # also group the related surrounding spectra
        for specset in self.cross_in_specset:
            specset.apply_grouping(grouping=self.grouping,
                quality=self.quality, verbose=verbose)
        self.groupped = True

    def estimate_errors(self, gehrels=True, verbose=False):
        if verbose:
            print("INFO: Estimating spectral errors ...")
        self.spectrumset.estimate_errors(gehrels=gehrels)
        # also estimate errors for the related surrounding spectra
        for specset in self.cross_in_specset:
            specset.estimate_errors(gehrels=gehrels)

    def do_correction(self, subtract_bkg=True, fix_negative=False,
            group_squeeze=True, add_history=False, verbose=False):
        """
        Perform the crosstalk correction.  The background contribution
        for each spectrum is subtracted first if `subtract_bkg' is True.
        The basic correction procedures are recorded to the header.
        """
        if add_history:
            self.spectrumset.header.add_history("Crosstalk Correction BEGIN")
            self.spectrumset.header.add_history("  TOOL: %s (v%s) @ %s" % (\
                    os.path.basename(sys.argv[0]), __version__,
                    datetime.utcnow().isoformat()))
        # background subtraction
        if subtract_bkg:
            if verbose:
                print("INFO: subtract background ...", file=sys.stderr)
            self.spectrumset.subtract_bkg(inplace=True,
                    add_history=add_history, verbose=verbose)
            # also apply background subtraction to the surrounding spectra
            for specset in self.cross_in_specset:
                specset.subtract_bkg(inplace=True,
                        add_history=add_history, verbose=verbose)
        # subtractions
        if verbose:
            print("INFO: apply subtractions ...", file=sys.stderr)
        for specset, cross_arf in zip(self.cross_in_specset,
                self.cross_in_arf):
            self.spectrumset.subtract(spectrumset=specset,
                    cross_arf=cross_arf, groupped=self.groupped,
                    group_squeeze=group_squeeze, add_history=add_history,
                    verbose=verbose)
        # compensations
        if verbose:
            print("INFO: apply compensations ...", file=sys.stderr)
        for cross_arf in self.cross_out_arf:
            self.spectrumset.compensate(cross_arf=cross_arf,
                    groupped=self.groupped, group_squeeze=group_squeeze,
                    add_history=add_history, verbose=verbose)
        # fix negative values in channels
        if fix_negative:
            if verbose:
                print("INFO: fix negative channel values ...", file=sys.stderr)
            self.spectrumset.fix_negative(add_history=add_history,
                    verbose=verbose)
        if add_history:
            self.spectrumset.header.add_history("END Crosstalk Correction")
        # reset header keywords
        self.spectrumset.reset_header_keywords(
                keywords=["ANCRFILE", "BACKFILE"])

    def copy(self):
        new = copy(self)
        # properly handle the copy of spectrumsets
        new.spectrumset = self.spectrumset.copy()
        new.cross_in_specset = [ specset.copy() \
                for specset in self.cross_in_specset ]
        return new

    def randomize(self):
        self.spectrumset.randomize()
        for specset in self.cross_in_specset:
            specset.randomize()
        return self

    def get_spectrum(self, copy=True):
        if copy:
            return self.spectrumset.copy()
        else:
            return self.spectrumset

    def write(self, filename=None, clobber=False):
        self.spectrumset.write(filename=filename, clobber=clobber)
# class Crosstalk }}}


class Deprojection:  # {{{
    """
    Perform the deprojection on a set of PROJECTED spectra with the
    assumption of spherical symmetry of the source object, and produce
    the DEPROJECTED spectra.

    NOTE:
    * Assumption of the spherical symmetry
    * Background should be subtracted before deprojection
    * ARF differences of different regions are taken into account

    Reference & Credit:
    [1] Direct X-ray Spectra Deprojection
        https://www-xray.ast.cam.ac.uk/papers/dsdeproj/
        Sanders & Fabian 2007, MNRAS, 381, 1381
    """
    spectra  = None
    grouping = None
    quality  = None

    def __init__(self, spectra, grouping=None, quality=None, verbose=False):
        """
        Arguments:
          * spectra: a set of spectra from the inner-most to the outer-most
                     regions (e.g., spectra after correcting crosstalk effect)
          * grouping: grouping specification for all the spectra
          * quality: quality column for the spectra
        """
        self.spectra = []
        for spec in spectra:
            if not isinstance(spec, SpectrumSet):
                raise ValueError("Not a 'SpectrumSet' object")
            spec.read_xflt()
            self.spectra.append(spec)
        self.spectra  = spectra
        self.grouping = grouping
        self.quality  = quality
        # sort spectra by `radius_outer'
        self.spectra.sort(key=lambda x: x.radius_outer)
        # set the inner radii
        radii_inner = [0.0] + [ x.radius_outer for x in self.spectra[:-1] ]
        for spec, rin in zip(self.spectra, radii_inner):
            spec.set_radius_inner(rin)
            if verbose:
                print("Deprojection: loaded spectrum: radius: (%s, %s)" % \
                        (spec.radius_inner, spec.radius_outer),
                        file=sys.stderr)
        # check EXPOSURE validity (all spectra must have the same exposures)
        exposures = [ spec.EXPOSURE for spec in self.spectra ]
        assert np.allclose(exposures[:-1], exposures[1:])

    def subtract_bkg(self, verbose=True):
        for spec in self.spectra:
            if not spec.bkg:
                raise ValueError("Spectrum '%s' has NO background" % \
                        spec.filename)
            spec.subtract_bkg(inplace=True, verbose=verbose)

    def apply_grouping(self, verbose=False):
        for spec in self.spectra:
            spec.apply_grouping(grouping=self.grouping, quality=self.quality,
                    verbose=verbose)

    def estimate_errors(self, gehrels=True):
        for spec in self.spectra:
            spec.estimate_errors(gehrels=gehrels)

    def scale(self):
        """
        Scale the spectral data according to the region angular size.
        """
        for spec in self.spectra:
            spec.scale()

    def do_deprojection(self, group_squeeze=True,
            add_history=False, verbose=False):
        #
        # TODO/XXX: How to apply ARF correction here???
        #
        num_spec = len(self.spectra)
        tmp_spec_data = self.spectra[0].get_data(group_squeeze=group_squeeze)
        spec_shape = tmp_spec_data.shape
        spec_dtype = tmp_spec_data.dtype
        spec_per_vol = [None] * num_spec
        #
        for shellnum in reversed(range(num_spec)):
            if verbose:
                print("DEPROJECTION: deprojecting shell %d ..." % shellnum,
                        file=sys.stderr)
            spec = self.spectra[shellnum]
            # calculate projected spectrum of outlying shells
            proj_spec = np.zeros(spec_shape, spec_dtype)
            for outer in range(shellnum+1, num_spec):
                vol = self.projected_volume(
                        r1=self.spectra[outer].radius_inner,
                        r2=self.spectra[outer].radius_outer,
                        R1=spec.radius_inner,
                        R2=spec.radius_outer)
                proj_spec += spec_per_vol[outer] * vol
            #
            this_spec = spec.get_data(group_squeeze=group_squeeze, copy=True)
            deproj_spec = this_spec - proj_spec
            # calculate the volume that this spectrum is from
            this_vol = self.projected_volume(
                    r1=spec.radius_inner, r2=spec.radius_outer,
                    R1=spec.radius_inner, R2=spec.radius_outer)
            # calculate the spectral data per unit volume
            spec_per_vol[shellnum] = deproj_spec / this_vol
        # set the spectral data to these deprojected values
        self.set_spec_data(spec_per_vol, group_squeeze=group_squeeze)
        # add history to header
        if add_history:
            self.add_history()

    def get_spec_data(self, group_squeeze=True, copy=True):
        """
        Extract the spectral data of each spectrum after deprojection
        performed.
        """
        return [ spec.get_data(group_squeeze=group_squeeze, copy=copy)
                 for spec in self.spectra ]

    def set_spec_data(self, spec_data, group_squeeze=True):
        """
        Set `spec_data' for each spectrum to the deprojected spectral data.
        """
        assert len(spec_data) == len(self.spectra)
        for spec, data in zip(self.spectra, spec_data):
            spec.set_data(data, group_squeeze=group_squeeze)

    def add_history(self):
        """
        Append a brief history about this tool to the header.
        """
        history = "Deprojected by %s (v%s) @ %s" % (
                os.path.basename(sys.argv[0]), __version__,
                datetime.utcnow().isoformat())
        for spec in self.spectra:
            spec.header.add_history(history)

    def add_stat_err(self, stat_err, group_squeeze=True):
        """
        Add the "STAT_ERR" column to each spectrum.
        """
        assert len(stat_err) == len(self.spectra)
        for spec, err in zip(self.spectra, stat_err):
            spec.add_stat_err(err, group_squeeze=group_squeeze)

    def write(self, filenames=[], clobber=False):
        """
        Write the deprojected spectra to output file.
        """
        if filenames == []:
            filenames = [ spec.outfile for spec in self.spectra ]
        for spec, outfile in zip(self.spectra, filenames):
            spec.write(filename=outfile, clobber=clobber)

    @staticmethod
    def projected_volume(r1, r2, R1, R2):
        """
        Calculate the projected volume of a spherical shell of radii r1 -> r2
        onto an annulus on the sky of radius R1 -> R2.

        This volume is the integral:
          Int(R=R1,R2) Int(x=sqrt(r1^2-R^2),sqrt(r2^2-R^2)) 2*pi*R dx dR
          =
          Int(R=R1,R2) 2*pi*R * (sqrt(r2^2-R^2) - sqrt(r1^2-R^2)) dR

        Note that the above integral is only half the total volume
        (i.e., front only).
        """
        def sqrt_trunc(x):
            if x > 0:
                return np.sqrt(x)
            else:
                return 0.0
        #
        p1 = sqrt_trunc(r1**2 - R2**2)
        p2 = sqrt_trunc(r1**2 - R1**2)
        p3 = sqrt_trunc(r2**2 - R2**2)
        p4 = sqrt_trunc(r2**2 - R1**2)
        return 2.0 * (2.0/3.0) * np.pi * ((p1**3 - p2**3) + (p4**3 - p3**3))
# class Deprojection }}}


# Helper functions {{{
def calc_median_errors(results):
    """
    Calculate the median and errors for the spectral data gathered
    through Monte Carlo simulations.

    TODO: investigate the errors calculation approach used here!
    """
    results = np.array(results)
    # `results' now has shape: (mc_times, num_spec, num_channel)
    # sort by the Monte Carlo simulation axis
    results.sort(0)
    mc_times = results.shape[0]
    medians  = results[ int(mc_times * 0.5) ]
    lowerpcs = results[ int(mc_times * 0.1585) ]
    upperpcs = results[ int(mc_times * 0.8415) ]
    errors   = np.sqrt(0.5 * ((medians-lowerpcs)**2 + (upperpcs-medians)**2))
    return (medians, errors)


def set_argument(name, default, cmdargs, config):
    value = default
    if name in config.keys():
        value = config.as_bool(name)
    value_cmd = vars(cmdargs)[name]
    if value_cmd != default:
        value = value_cmd  # command arguments overwrite others
    return value
# helper functions }}}


# main routine {{{
def main(config, subtract_bkg, fix_negative, mc_times,
        verbose=False, clobber=False):
    # collect ARFs and RMFs into dictionaries (avoid interpolation every time)
    arf_files = set()
    rmf_files = set()
    for region in config.sections:
        config_reg = config[region]
        arf_files.add(config_reg.get("arf"))
        rmf_files.add(config_reg.get("rmf"))
        for reg_in in config_reg["cross_in"].values():
            arf_files.add(reg_in.get("arf"))
            arf_files.add(reg_in.get("cross_arf"))
        if "cross_out" in config_reg.sections:
            for arf in config_reg["cross_out"].as_list("cross_arf"):
                arf_files.add(arf)
    arf_files = arf_files - set([None])
    arf_dict  = { arf: ARF(arf) for arf in arf_files }
    rmf_files = rmf_files - set([None])
    rmf_dict  = { rmf: RMF(rmf) for rmf in rmf_files }
    if verbose:
        print("INFO: arf_files:", arf_files, file=sys.stderr)
        print("INFO: rmf_files:", rmf_files, file=sys.stderr)

    # get the GROUPING and QUALITY data
    grouping_fits = fits.open(config["grouping"])
    grouping = grouping_fits["SPECTRUM"].data.columns["GROUPING"].array
    quality  = grouping_fits["SPECTRUM"].data.columns["QUALITY"].array
    # squeeze the groupped spectral data, etc.
    group_squeeze = True

    # crosstalk objects (BEFORE background subtraction)
    crosstalks_cleancopy = []
    # crosstalk-corrected spectra
    cc_spectra = []

    # correct crosstalk effects for each region first
    for region in config.sections:
        if verbose:
            print("INFO: processing '%s' ..." % region, file=sys.stderr)
        crosstalk = Crosstalk(config.get(region),
                arf_dict=arf_dict, rmf_dict=rmf_dict,
                grouping=grouping, quality=quality)
        crosstalk.apply_grouping(verbose=verbose)
        crosstalk.estimate_errors(verbose=verbose)
        # keep a (almost) clean copy of the crosstalk object
        crosstalks_cleancopy.append(crosstalk.copy())
        if verbose:
            print("INFO: doing crosstalk correction ...", file=sys.stderr)
        crosstalk.do_correction(subtract_bkg=subtract_bkg,
                fix_negative=fix_negative, group_squeeze=group_squeeze,
                add_history=True, verbose=verbose)
        cc_spectra.append(crosstalk.get_spectrum(copy=True))

    # load back the crosstalk-corrected spectra for deprojection
    if verbose:
        print("INFO: preparing spectra for deprojection ...", file=sys.stderr)
    deprojection = Deprojection(spectra=cc_spectra, grouping=grouping,
            quality=quality, verbose=verbose)
    if verbose:
        print("INFO: scaling spectra according the region angular size...",
                file=sys.stderr)
    deprojection.scale()
    if verbose:
        print("INFO: doing deprojection ...", file=sys.stderr)
    deprojection.do_deprojection(add_history=True, verbose=verbose)
    deproj_results = [ deprojection.get_spec_data(
        group_squeeze=group_squeeze, copy=True) ]

    # Monte Carlo for spectral group error estimation
    print("INFO: Monte Carlo to estimate spectral errors (%d times) ..." % \
            mc_times, file=sys.stderr)
    for i in range(mc_times):
        if i % 100 == 0:
            print("%d..." % i, end="", flush=True, file=sys.stderr)
        # correct crosstalk effects
        cc_spectra_copy = []
        for crosstalk in crosstalks_cleancopy:
            # copy and randomize
            crosstalk_copy = crosstalk.copy().randomize()
            crosstalk_copy.do_correction(subtract_bkg=subtract_bkg,
                    fix_negative=fix_negative, group_squeeze=group_squeeze,
                    add_history=False, verbose=False)
            cc_spectra_copy.append(crosstalk_copy.get_spectrum(copy=True))
        # deproject spectra
        deprojection_copy = Deprojection(spectra=cc_spectra_copy,
                grouping=grouping, quality=quality, verbose=False)
        deprojection_copy.scale()
        deprojection_copy.do_deprojection(add_history=False, verbose=False)
        deproj_results.append(deprojection_copy.get_spec_data(
            group_squeeze=group_squeeze, copy=True))
    print("DONE!", flush=True, file=sys.stderr)

    if verbose:
        print("INFO: Calculating the median and errors for each spectrum ...",
                file=sys.stderr)
    medians, errors = calc_median_errors(deproj_results)
    deprojection.set_spec_data(medians, group_squeeze=group_squeeze)
    deprojection.add_stat_err(errors, group_squeeze=group_squeeze)
    if verbose:
        print("INFO: Writing the crosstalk-corrected and deprojected " + \
                "spectra with estimated statistical errors ...",
                file=sys.stderr)
    deprojection.write(clobber=clobber)
# main routine }}}


# main_deprojection routine {{{
def main_deprojection(config, mc_times, verbose=False, clobber=False):
    """
    Only perform the spectral deprojection.
    """
    # collect ARFs and RMFs into dictionaries (avoid interpolation every time)
    arf_files = set()
    rmf_files = set()
    for region in config.sections:
        config_reg = config[region]
        arf_files.add(config_reg.get("arf"))
        rmf_files.add(config_reg.get("rmf"))
    arf_files = arf_files - set([None])
    arf_dict  = { arf: ARF(arf) for arf in arf_files }
    rmf_files = rmf_files - set([None])
    rmf_dict  = { rmf: RMF(rmf) for rmf in rmf_files }
    if verbose:
        print("INFO: arf_files:", arf_files, file=sys.stderr)
        print("INFO: rmf_files:", rmf_files, file=sys.stderr)

    # get the GROUPING and QUALITY data
    grouping_fits = fits.open(config["grouping"])
    grouping = grouping_fits["SPECTRUM"].data.columns["GROUPING"].array
    quality  = grouping_fits["SPECTRUM"].data.columns["QUALITY"].array
    # squeeze the groupped spectral data, etc.
    group_squeeze = True

    # load spectra for deprojection
    if verbose:
        print("INFO: preparing spectra for deprojection ...", file=sys.stderr)
    proj_spectra = []
    for region in config.sections:
        config_reg = config[region]
        specset = SpectrumSet(filename=config_reg["spec"],
                outfile=config_reg["outfile"],
                arf=arf_dict.get(config_reg["arf"], config_reg["arf"]),
                rmf=rmf_dict.get(config_reg["rmf"], config_reg["rmf"]),
                bkg=config_reg["bkg"])
        proj_spectra.append(specset)

    deprojection = Deprojection(spectra=proj_spectra, grouping=grouping,
            quality=quality, verbose=verbose)
    deprojection.apply_grouping(verbose=verbose)
    deprojection.estimate_errors()
    if verbose:
        print("INFO: scaling spectra according the region angular size ...",
                file=sys.stderr)
    deprojection.scale()

    # keep a (almost) clean copy of the input projected spectra
    proj_spectra_cleancopy = [ spec.copy() for spec in proj_spectra ]

    if verbose:
        print("INFO: subtract the background ...", file=sys.stderr)
    deprojection.subtract_bkg(verbose=verbose)
    if verbose:
        print("INFO: doing deprojection ...", file=sys.stderr)
    deprojection.do_deprojection(add_history=True, verbose=verbose)
    deproj_results = [ deprojection.get_spec_data(
        group_squeeze=group_squeeze, copy=True) ]

    # Monte Carlo for spectral group error estimation
    print("INFO: Monte Carlo to estimate spectral errors (%d times) ..." % \
            mc_times, file=sys.stderr)
    for i in range(mc_times):
        if i % 100 == 0:
            print("%d..." % i, end="", flush=True, file=sys.stderr)
        # copy and randomize the input projected spectra
        proj_spectra_copy = [ spec.copy().randomize()
                              for spec in proj_spectra_cleancopy ]
        # deproject spectra
        deprojection_copy = Deprojection(spectra=proj_spectra_copy,
                grouping=grouping, quality=quality, verbose=False)
        deprojection_copy.subtract_bkg(verbose=False)
        deprojection_copy.do_deprojection(add_history=False, verbose=False)
        deproj_results.append(deprojection_copy.get_spec_data(
            group_squeeze=group_squeeze, copy=True))
    print("DONE!", flush=True, file=sys.stderr)

    if verbose:
        print("INFO: Calculating the median and errors for each spectrum ...",
                file=sys.stderr)
    medians, errors = calc_median_errors(deproj_results)
    deprojection.set_spec_data(medians, group_squeeze=group_squeeze)
    deprojection.add_stat_err(errors, group_squeeze=group_squeeze)
    if verbose:
        print("INFO: Writing the deprojected spectra " + \
                "with estimated statistical errors ...",
                file=sys.stderr)
    deprojection.write(clobber=clobber)
# main_deprojection routine }}}


# main_crosstalk routine {{{
def main_crosstalk(config, subtract_bkg, fix_negative, mc_times,
        verbose=False, clobber=False):
    """
    Only perform the crosstalk correction.
    """
    # collect ARFs and RMFs into dictionaries (avoid interpolation every time)
    arf_files = set()
    rmf_files = set()
    for region in config.sections:
        config_reg = config[region]
        arf_files.add(config_reg.get("arf"))
        rmf_files.add(config_reg.get("rmf"))
        for reg_in in config_reg["cross_in"].values():
            arf_files.add(reg_in.get("arf"))
            arf_files.add(reg_in.get("cross_arf"))
        if "cross_out" in config_reg.sections:
            for arf in config_reg["cross_out"].as_list("cross_arf"):
                arf_files.add(arf)
    arf_files = arf_files - set([None])
    arf_dict  = { arf: ARF(arf) for arf in arf_files }
    rmf_files = rmf_files - set([None])
    rmf_dict  = { rmf: RMF(rmf) for rmf in rmf_files }
    if verbose:
        print("INFO: arf_files:", arf_files, file=sys.stderr)
        print("INFO: rmf_files:", rmf_files, file=sys.stderr)

    # get the GROUPING and QUALITY data
    if "grouping" in config.keys():
        grouping_fits = fits.open(config["grouping"])
        grouping = grouping_fits["SPECTRUM"].data.columns["GROUPING"].array
        quality  = grouping_fits["SPECTRUM"].data.columns["QUALITY"].array
        group_squeeze = True
    else:
        grouping = None
        quality  = None
        group_squeeze = False

    # crosstalk objects (BEFORE background subtraction)
    crosstalks_cleancopy = []
    # crosstalk-corrected spectra
    cc_spectra = []

    # correct crosstalk effects for each region first
    for region in config.sections:
        if verbose:
            print("INFO: processing '%s' ..." % region, file=sys.stderr)
        crosstalk = Crosstalk(config.get(region),
                arf_dict=arf_dict, rmf_dict=rmf_dict,
                grouping=grouping, quality=quality)
        if grouping is not None:
            crosstalk.apply_grouping(verbose=verbose)
        crosstalk.estimate_errors(verbose=verbose)
        # keep a (almost) clean copy of the crosstalk object
        crosstalks_cleancopy.append(crosstalk.copy())
        if verbose:
            print("INFO: doing crosstalk correction ...", file=sys.stderr)
        crosstalk.do_correction(subtract_bkg=subtract_bkg,
                fix_negative=fix_negative, group_squeeze=group_squeeze,
                add_history=True, verbose=verbose)
        cc_spectra.append(crosstalk.get_spectrum(copy=True))

    # spectral data of the crosstalk-corrected spectra
    cc_results = []
    cc_results.append([ spec.get_data(group_squeeze=group_squeeze, copy=True)
                        for spec in cc_spectra ])

    # Monte Carlo for spectral group error estimation
    print("INFO: Monte Carlo to estimate spectral errors (%d times) ..." % \
            mc_times, file=sys.stderr)
    for i in range(mc_times):
        if i % 100 == 0:
            print("%d..." % i, end="", flush=True, file=sys.stderr)
        # correct crosstalk effects
        cc_spectra_copy = []
        for crosstalk in crosstalks_cleancopy:
            # copy and randomize
            crosstalk_copy = crosstalk.copy().randomize()
            crosstalk_copy.do_correction(subtract_bkg=subtract_bkg,
                    fix_negative=fix_negative, group_squeeze=group_squeeze,
                    add_history=False, verbose=False)
            cc_spectra_copy.append(crosstalk_copy.get_spectrum(copy=True))
        cc_results.append([ spec.get_data(group_squeeze=group_squeeze,
                                          copy=True)
                            for spec in cc_spectra_copy ])
    print("DONE!", flush=True, file=sys.stderr)

    if verbose:
        print("INFO: Calculating the median and errors for each spectrum ...",
                file=sys.stderr)
    medians, errors = calc_median_errors(cc_results)
    if verbose:
        print("INFO: Writing the crosstalk-corrected spectra " + \
                "with estimated statistical errors ...",
                file=sys.stderr)
    for spec, data, err in zip(cc_spectra, medians, errors):
        spec.set_data(data, group_squeeze=group_squeeze)
        spec.add_stat_err(err, group_squeeze=group_squeeze)
        spec.write(clobber=clobber)
# main_crosstalk routine }}}


if __name__ == "__main__":
    # arguments' default values
    default_mode = "both"
    default_mc_times = 5000
    # commandline arguments parser
    parser = argparse.ArgumentParser(
            description="Correct the crosstalk effects for XMM EPIC spectra",
            epilog="Version: %s (%s)" % (__version__, __date__))
    parser.add_argument("config", help="config file in which describes " +\
            "the crosstalk relations ('ConfigObj' syntax)")
    parser.add_argument("-m", "--mode", dest="mode", default=default_mode,
            help="operation mode (both | crosstalk | deprojection)")
    parser.add_argument("-B", "--no-subtract-bkg", dest="subtract_bkg",
            action="store_false", help="do NOT subtract background first")
    parser.add_argument("-N", "--fix-negative", dest="fix_negative",
            action="store_true", help="fix negative channel values")
    parser.add_argument("-M", "--mc-times", dest="mc_times",
            type=int, default=default_mc_times,
            help="Monte Carlo times for error estimation")
    parser.add_argument("-C", "--clobber", dest="clobber",
            action="store_true", help="overwrite output file if exists")
    parser.add_argument("-v", "--verbose", dest="verbose",
            action="store_true", help="show verbose information")
    args = parser.parse_args()
    # merge commandline arguments and config
    config       = ConfigObj(args.config)
    subtract_bkg = set_argument("subtract_bkg", True,  args, config)
    fix_negative = set_argument("fix_negative", False, args, config)
    verbose      = set_argument("verbose",      False, args, config)
    clobber      = set_argument("clobber",      False, args, config)
    # operation mode
    mode = config.get("mode", default_mode)
    if args.mode != default_mode:
        mode = args.mode
    # Monte Carlo times
    mc_times = config.as_int("mc_times")
    if args.mc_times != default_mc_times:
        mc_times = args.mc_times

    if mode.lower() == "both":
        print("MODE: CROSSTALK + DEPROJECTION", file=sys.stderr)
        main(config, subtract_bkg=subtract_bkg, fix_negative=fix_negative,
                mc_times=mc_times, verbose=verbose, clobber=clobber)
    elif mode.lower() == "deprojection":
        print("MODE: DEPROJECTION", file=sys.stderr)
        main_deprojection(config, mc_times=mc_times,
                verbose=verbose, clobber=clobber)
    elif mode.lower() == "crosstalk":
        print("MODE: CROSSTALK", file=sys.stderr)
        main_crosstalk(config, subtract_bkg=subtract_bkg,
                fix_negative=fix_negative, mc_times=mc_times,
                verbose=verbose, clobber=clobber)
    else:
        raise ValueError("Invalid operation mode: %s" % mode)
    print(WARNING)

#  vim: set ts=4 sw=4 tw=0 fenc=utf-8 ft=python: #
