#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# References:
# [1] Jean-Luc Starck, Fionn Murtagh & Jalal M. Fadili
#     Sparse Image and Signal Processing: Wavelets, Curvelets, Morphological Diversity
#     Section 3.5, 6.6
#
# Credits:
# [1] https://github.com/abrazhe/image-funcut/blob/master/imfun/atrous.py
#
# Aaron LI
# Created: 2016-03-17
# Updated: 2016-04-22
#
# ChangeLog:
# 2016-04-22:
#   * Show more verbose information/details
#   * Fix a bug about "p_cutoff" when "comp" contains ALL False's
#   * Add argument "start-scale" to specifiy the start denoising scale
# 2016-04-20:
#   * Add argparse and main() for scripting
#

"""
Starlet wavelet transform, i.e., isotropic undecimated wavelet transform
(IUWT), or à trous wavelet transform.
And multi-scale variance stabling transform (MS-VST), which can be used
to effectively remove the Poisson noises.
"""

__version__ = "0.2.3"
__date__    = "2016-04-22"


import sys
import os
import argparse
from datetime import datetime

import numpy as np
import scipy as sp
from scipy import signal
from astropy.io import fits


class B3Spline:  # {{{
    """
    B3-spline wavelet.
    """
    # scaling function (phi)
    dec_lo = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16
    dec_hi = np.array([-1.0, -4.0, 10.0, -4.0, -1.0]) / 16
    rec_lo = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    rec_hi = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
# B3Spline }}}


class IUWT:  # {{{
    """
    Isotropic undecimated wavelet transform.
    """
    ## Decomposition filters list:
    # a_{scale} = convole(a_0, filters[scale])
    # Note: the zero-th scale filter (i.e., delta function) is the first
    # element, thus the array index is the same as the decomposition scale.
    filters = []

    phi = None              # wavelet scaling function (2D)
    level = 0               # number of transform level
    decomposition = None    # decomposed coefficients/images
    reconstruction = None   # reconstructed image

    # convolution boundary condition
    boundary = "symm"

    def __init__(self, phi=B3Spline.dec_lo, level=None, boundary="symm",
            data=None):
        self.set_wavelet(phi=phi)
        self.level = level
        self.boundary = boundary
        self.data = np.array(data)

    def reset(self):
        """
        Reset the object attributes.
        """
        self.data = None
        self.phi = None
        self.decomposition = None
        self.reconstruction = None
        self.level = 0
        self.filters = []
        self.boundary = "symm"

    def load_data(self, data):
        self.reset()
        self.data = np.array(data)

    def set_wavelet(self, phi):
        self.reset()
        phi = np.array(phi)
        if phi.ndim == 1:
            phi_ = phi.reshape(1, -1)
            self.phi = np.dot(phi_.T, phi_)
        elif phi.ndim == 2:
            self.phi = phi
        else:
            raise ValueError("Invalid phi dimension")

    def calc_filters(self):
        """
        Calculate the convolution filters of each scale.
        Note: the zero-th scale filter (i.e., delta function) is the first
        element, thus the array index is the same as the decomposition scale.
        """
        self.filters = []
        # scale 0: delta function
        h = np.array([[1]])  # NOTE: 2D
        self.filters.append(h)
        # scale 1
        h = self.phi[::-1, ::-1]
        self.filters.append(h)
        for scale in range(2, self.level+1):
            h_up = self.zupsample(self.phi, order=scale-1)
            h2 = signal.convolve2d(h_up[::-1, ::-1], h, mode="same",
                    boundary=self.boundary)
            self.filters.append(h2)

    def transform(self, data, scale, boundary="symm"):
        """
        Perform only one scale wavelet transform for the given data.

        return:
            [ approx, detail ]
        """
        self.decomposition = []
        approx = signal.convolve2d(data, self.filters[scale],
                mode="same", boundary=self.boundary)
        detail = data - approx
        return [approx, detail]

    def decompose(self, level, boundary="symm"):
        """
        Perform IUWT decomposition in the plain loop way.
        The filters of each scale/level are calculated first, then the
        approximations of each scale/level are calculated by convolving the
        raw/finest image with these filters.

        return:
            [ W_1, W_2, ..., W_n, A_n ]
            n = level
            W: wavelet details
            A: approximation
        """
        self.boundary = boundary
        if self.level != level or self.filters == []:
            self.level = level
            self.calc_filters()
        self.decomposition = []
        approx = self.data
        for scale in range(1, level+1):
            # approximation:
            approx2 = signal.convolve2d(self.data, self.filters[scale],
                    mode="same", boundary=self.boundary)
            # wavelet details:
            w = approx - approx2
            self.decomposition.append(w)
            if scale == level:
                self.decomposition.append(approx2)
            approx = approx2
        return self.decomposition

    def decompose_recursive(self, level, boundary="symm"):
        """
        Perform the IUWT decomposition in the recursive way.

        return:
            [ W_1, W_2, ..., W_n, A_n ]
            n = level
            W: wavelet details
            A: approximation
        """
        self.level = level
        self.boundary = boundary
        self.decomposition = self.__decompose(self.data, self.phi, level=level)
        return self.decomposition

    def __decompose(self, data, phi, level):
        """
        2D IUWT decomposition (or stationary wavelet transform).

        This is a convolution version, where kernel is zero-upsampled
        explicitly. Not fast.

        Parameters:
        - level : level of decomposition
        - phi : low-pass filter kernel
        - boundary : boundary conditions (passed to scipy.signal.convolve2d,
                     'symm' by default)

        Returns:
            list of wavelet details + last approximation. Each element in
            the list is an image of the same size as the input image. 
        """
        if level <= 0:
            return data
        shapecheck = map(lambda a,b:a>b, data.shape, phi.shape)
        assert np.all(shapecheck)
        # approximation:
        approx = signal.convolve2d(data, phi[::-1, ::-1], mode="same",
                boundary=self.boundary)
        # wavelet details:
        w = data - approx
        phi_up = self.zupsample(phi, order=1)
        shapecheck = map(lambda a,b:a>b, data.shape, phi_up.shape)
        if level == 1:
            return [w, approx]
        elif not np.all(shapecheck):
            print("Maximum allowed decomposition level reached",
                    file=sys.stderr)
            return [w, approx]
        else:
            return [w] + self.__decompose(approx, phi_up, level-1)

    @staticmethod
    def zupsample(data, order=1):
        """
        Upsample data array by interleaving it with zero's.

        h{up_order: n}[l] = (1) h[l], if l % 2^n == 0;
                            (2) 0, otherwise
        """
        shape = data.shape
        new_shape = [ (2**order * (n-1) + 1) for n in shape ]
        output = np.zeros(new_shape, dtype=data.dtype)
        output[[ slice(None, None, 2**order) for d in shape ]] = data
        return output

    def reconstruct(self, decomposition=None):
        if decomposition is not None:
            reconstruction = np.sum(decomposition, axis=0)
            return reconstruction
        else:
            self.reconstruction = np.sum(self.decomposition, axis=0)

    def get_detail(self, scale):
        """
        Get the wavelet detail coefficients of given scale.
        Note: 1 <= scale <= level
        """
        if scale < 1 or scale > self.level:
            raise ValueError("Invalid scale")
        return self.decomposition[scale-1]

    def get_approx(self):
        """
        Get the approximation coefficients of the largest scale.
        """
        return self.decomposition[-1]
# IUWT }}}


class IUWT_VST(IUWT):  # {{{
    """
    IUWT with Multi-scale variance stabling transform.

    Refernce:
    [1] Bo Zhang, Jalal M. Fadili & Jean-Luc Starck,
        IEEE Trans. Image Processing, 17, 17, 2008
    """
    # VST coefficients and the corresponding asymptotic standard deviation
    # of each scale.
    vst_coef = []

    def reset(self):
        super(self.__class__, self).reset()
        vst_coef = []

    def __decompose(self):
        raise AttributeError("No '__decompose' attribute")

    @staticmethod
    def soft_threshold(data, threshold):
        if isinstance(data, np.ndarray):
            data_th = data.copy()
            data_th[np.abs(data) <= threshold] = 0.0
            data_th[data > threshold] -= threshold
            data_th[data < -threshold] += threshold
        else:
            data_th = data
            if np.abs(data) <= threshold:
                data_th = 0.0
            elif data > threshold:
                data_th -= threshold
            else:
                data_th += threshold
        return data_th

    def tau(self, k, scale):
        """
        Helper function used in VST coefficients calculation.
        """
        return np.sum(np.power(self.filters[scale], k))

    def filters_product(self, scale1, scale2):
        """
        Calculate the scalar product of the filters of two scales,
        considering only the overlapped part.
        Helper function used in VST coefficients calculation.
        """
        if scale1 > scale2:
            filter_big   = self.filters[scale1]
            filter_small = self.filters[scale2]
        else:
            filter_big   = self.filters[scale2]
            filter_small = self.filters[scale1]
        # crop the big filter to match the size of the small filter
        size_big = filter_big.shape
        size_small = filter_small.shape
        size_diff2 = list(map(lambda a,b: (a-b)//2, size_big, size_small))
        filter_big_crop = filter_big[
                size_diff2[0]:(size_big[0]-size_diff2[0]),
                size_diff2[1]:(size_big[1]-size_diff2[1])]
        assert(np.all(list(map(lambda a,b: a==b,
                size_small, filter_big_crop.shape))))
        product = np.sum(filter_small * filter_big_crop)
        return product

    def calc_vst_coef(self):
        """
        Calculate the VST coefficients and the corresponding
        asymptotic standard deviation of each scale, according to the
        calculated filters of each scale/level.
        """
        self.vst_coef = []
        for scale in range(self.level+1):
            b = 2 * np.sqrt(np.abs(self.tau(1, scale)) / self.tau(2, scale))
            c = 7.0*self.tau(2, scale) / (8.0*self.tau(1, scale)) - \
                    self.tau(3, scale) / (2.0*self.tau(2, scale))
            if scale == 0:
                std = -1.0
            else:
                std = np.sqrt((self.tau(2, scale-1) / \
                        (4 * self.tau(1, scale-1)**2)) + \
                        (self.tau(2, scale) / (4 * self.tau(1, scale)**2)) - \
                        (self.filters_product(scale-1, scale) / \
                        (2 * self.tau(1, scale-1) * self.tau(1, scale))))
            self.vst_coef.append({ "b": b, "c": c, "std": std })

    def vst(self, data, scale, coupled=True):
        """
        Perform variance stabling transform

        XXX: parameter `coupled' why??
        Credit: MSVST-V1.0/src/libmsvst/B3VSTAtrous.h
        """
        self.vst_coupled = coupled
        if self.vst_coef == []:
            self.calc_vst_coef()
        if coupled:
            b = 1.0
        else:
            b = self.vst_coef[scale]["b"]
        data_vst = b * np.sqrt(np.abs(data + self.vst_coef[scale]["c"]))
        return data_vst

    def ivst(self, data, scale, cbias=True):
        """
        Inverse variance stabling transform
        NOTE: assuming that `a_{j} + c^{j}' are all positive.

        XXX: parameter `cbias' why??
             `bias correction' is recommended while reconstruct the data
             after estimation
        Credit: MSVST-V1.0/src/libmsvst/B3VSTAtrous.h
        """
        self.vst_cbias = cbias
        if cbias:
            cb = 1.0 / (self.vst_coef[scale]["b"] ** 2)
        else:
            cb = 0.0
        data_ivst = data ** 2 + cb - self.vst_coef[scale]["c"]
        return data_ivst

    def is_significant(self, scale, fdr=0.1, independent=False, verbose=False):
        """
        Multiple hypothesis testing with false discovery rate (FDR) control.

        `independent': whether the test statistics of all the null
        hypotheses are independent.
        If `independent=True': FDR <= (m0/m) * q
        otherwise: FDR <= (m0/m) * q * (1 + 1/2 + 1/3 + ... + 1/m)

        References:
        [1] False discovery rate - Wikipedia
            https://en.wikipedia.org/wiki/False_discovery_rate
        """
        coef = self.get_detail(scale)
        std = self.vst_coef[scale]["std"]
        pvalues = 2.0 * (1.0 - sp.stats.norm.cdf(np.abs(coef) / std))
        p_sorted = pvalues.flatten()
        p_sorted.sort()
        N = len(p_sorted)
        if independent:
            cn = 1.0
        else:
            cn = np.sum(1.0 / np.arange(1, N+1))
        p_comp = fdr * np.arange(N) / (N * cn)
        comp = (p_sorted < p_comp)
        if np.sum(comp) == 0:
            # `comp' contains ALL False
            p_cutoff = 0.0
        else:
            # cutoff p-value after FDR control/correction
            p_cutoff = np.max(p_sorted[comp])
        sig = (pvalues <= p_cutoff)
        if verbose:
            print("std/sigma: %g, p_cutoff: %g" % (std, p_cutoff),
                    flush=True, file=sys.stderr)
        return (sig, p_cutoff)

    def denoise(self, fdr=0.1, fdr_independent=False, start_scale=1,
            verbose=False):
        """
        Denoise the wavelet coefficients by controlling FDR.
        """
        self.fdr = fdr
        self.fdr_indepent = fdr_independent
        self.denoised = []
        # supports of significant coefficients of each scale
        self.sig_supports = [None]  # make index match the scale
        self.p_cutoff = [None]
        if verbose:
            print("MSVST denosing ...", flush=True, file=sys.stderr)
        for scale in range(1, self.level+1):
            coef = self.get_detail(scale)
            if verbose:
                print("\tScale %d: " % scale, end="",
                        flush=True, file=sys.stderr)
            if scale < start_scale:
                if verbose:
                    print("skipped", flush=True, file=sys.stderr)
                sig, p_cutoff = None, None
            else:
                sig, p_cutoff = self.is_significant(scale, fdr=fdr,
                        independent=fdr_independent, verbose=verbose)
                coef[np.logical_not(sig)] = 0.0
            #
            self.denoised.append(coef)
            self.sig_supports.append(sig)
            self.p_cutoff.append(p_cutoff)
        # append the last approximation
        self.denoised.append(self.get_approx())

    def decompose(self, level=5, boundary="symm", verbose=False):
        """
        2D IUWT decomposition with VST.
        """
        self.boundary = boundary
        if self.level != level or self.filters == []:
            self.level = level
            self.calc_filters()
            self.calc_vst_coef()
        self.decomposition = []
        approx = self.data
        if verbose:
            print("IUWT decomposing (%d levels): " % level,
                    end="", flush=True, file=sys.stderr)
        for scale in range(1, level+1):
            if verbose:
                print("%d..." % scale, end="", flush=True, file=sys.stderr)
            # approximation:
            approx2 = signal.convolve2d(self.data, self.filters[scale],
                    mode="same", boundary=self.boundary)
            # wavelet details:
            w = self.vst(approx, scale=scale-1) - self.vst(approx2, scale=scale)
            self.decomposition.append(w)
            if scale == level:
                self.decomposition.append(approx2)
            approx = approx2
        if verbose:
            print("DONE!", flush=True, file=sys.stderr)
        return self.decomposition

    def reconstruct_ivst(self, denoised=True, positive_project=True):
        """
        Reconstruct the original image from the *un-denoised* decomposition
        by applying the inverse VST.

        This reconstruction result is also used as the `initial condition'
        for the below `iterative reconstruction' algorithm.

        arguments:
        * denoised: whether use th denoised data or the direct decomposition
        * positive_project: whether replace negative values with zeros
        """
        if denoised:
            decomposition = self.denoised
        else:
            decomposition = self.decomposition
        self.positive_project = positive_project
        details = np.sum(decomposition[:-1], axis=0)
        approx = self.vst(decomposition[-1], scale=self.level)
        reconstruction = self.ivst(approx+details, scale=0)
        if positive_project:
            reconstruction[reconstruction < 0.0] = 0.0
        self.reconstruction = reconstruction
        return reconstruction

    def reconstruct(self, denoised=True, niter=10, verbose=False):
        """
        Reconstruct the original image using iterative method with
        L1 regularization, because the denoising violates the exact inverse
        procedure.

        arguments:
        * denoised: whether use the denoised coefficients
        * niter: number of iterations
        """
        if denoised:
            decomposition = self.denoised
        else:
            decomposition = self.decomposition
        # L1 regularization
        lbd = 1.0
        delta = lbd / (niter - 1)
        # initial solution
        solution = self.reconstruct_ivst(denoised=denoised,
                positive_project=True)
        #
        iuwt = IUWT(level=self.level)
        iuwt.calc_filters()
        # iterative reconstruction
        if verbose:
            print("Iteratively reconstructing (%d times): " % niter,
                    end="", flush=True, file=sys.stderr)
        for i in range(niter):
            if verbose:
                print("%d..." % i, end="", flush=True, file=sys.stderr)
            tempd = self.data.copy()
            solution_decomp = []
            for scale in range(1, self.level+1):
                approx, detail = iuwt.transform(tempd, scale)
                approx_sol, detail_sol = iuwt.transform(solution, scale)
                # Update coefficients according to the significant supports,
                # which are acquired during the denosing precodure with FDR.
                sig = self.sig_supports[scale]
                detail_sol[sig] = detail[sig]
                detail_sol = self.soft_threshold(detail_sol, threshold=lbd)
                #
                solution_decomp.append(detail_sol)
                tempd = approx.copy()
                solution = approx_sol.copy()
            # last approximation (the two are the same)
            solution_decomp.append(approx)
            # reconstruct
            solution = iuwt.reconstruct(decomposition=solution_decomp)
            # discard all negative values
            solution[solution < 0] = 0.0
            #
            lbd -= delta
        if verbose:
            print("DONE!", flush=True, file=sys.stderr)
        #
        self.reconstruction = solution
        return self.reconstruction
# IUWT_VST }}}


def main():
    # commandline arguments parser
    parser = argparse.ArgumentParser(
            description="Poisson Noise Removal with Multi-scale Variance " + \
                        "Stabling Transform and Wavelet Transform",
            epilog="Version: %s (%s)" % (__version__, __date__))
    parser.add_argument("-l", "--level", dest="level",
            type=int, default=5,
            help="level of the IUWT decomposition")
    parser.add_argument("-r", "--fdr", dest="fdr",
            type=float, default=0.1,
            help="false discovery rate")
    parser.add_argument("-I", "--fdr-independent", dest="fdr_independent",
            action="store_true", default=False,
            help="whether the FDR null hypotheses are independent")
    parser.add_argument("-s", "--start-scale", dest="start_scale",
            type=int, default=1,
            help="which scale to start the denoising")
    parser.add_argument("-n", "--niter", dest="niter",
            type=int, default=10,
            help="number of iterations for reconstruction")
    parser.add_argument("-v", "--verbose", dest="verbose",
            action="store_true", default=False,
            help="show verbose progress")
    parser.add_argument("-C", "--clobber", dest="clobber",
            action="store_true", default=False,
            help="overwrite output file if exists")
    parser.add_argument("infile", help="input image with Poisson noises")
    parser.add_argument("outfile", help="output denoised image")
    args = parser.parse_args()

    if args.verbose:
        print("infile: '%s'" % args.infile, file=sys.stderr)
        print("outfile: '%s'" % args.outfile, file=sys.stderr)
        print("level: %d" % args.level, file=sys.stderr)
        print("fdr: %.2f" % args.fdr, file=sys.stderr)
        print("fdr_independent: %s" % args.fdr_independent, file=sys.stderr)
        print("start_scale: %d" % args.start_scale, file=sys.stderr)
        print("niter: %d\n" % args.niter, flush=True, file=sys.stderr)

    imgfits = fits.open(args.infile)
    img = imgfits[0].data
    # Remove Poisson noises
    msvst = IUWT_VST(data=img)
    msvst.decompose(level=args.level, verbose=args.verbose)
    msvst.denoise(fdr=args.fdr, fdr_independent=args.fdr_independent,
            start_scale=args.start_scale, verbose=args.verbose)
    msvst.reconstruct(denoised=True, niter=args.niter, verbose=args.verbose)
    img_denoised = msvst.reconstruction
    # Output
    imgfits[0].data = img_denoised
    imgfits[0].header.add_history("%s: Removed Poisson Noises @ %s" % (
                os.path.basename(sys.argv[0]), datetime.utcnow().isoformat()))
    imgfits[0].header.add_history("  TOOL: %s (v%s)" % (
                os.path.basename(sys.argv[0]), __version__))
    imgfits[0].header.add_history("  PARAM: %s" % " ".join(sys.argv[1:]))
    imgfits.writeto(args.outfile, checksum=True, clobber=args.clobber)


if __name__ == "__main__":
    main()

