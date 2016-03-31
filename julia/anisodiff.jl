# -*- coding: utf-8 -*-
#
# ANISODIFF - Anisotropic diffusion
#
# Usage:
#   diff = anisodiff(img, niter, kappa, lambda, option)
#
# Arguments:
# | img    - input image (2D grayscale)
# | niter  - number of iterations
# | kappa  - conduction coefficient (gradient modulus threshold)
# |          This parameter controls conduction as a function of gradient.
# |          If kappa is low, small intensity gradients are able to block
# |          conduction and hence diffusion across step edges.  A large value
# |          reduces the influence of intensity gradients on conduction.
# | lambda - integration constant for stability (0 <= lambda <= 0.25)
# |          This parameter controls the diffusion speed, and you
# |          usually want it at the maximum value of 0.25.
# |          default value: 0.25
# | option - conduction coefficient functions proposed by Perona & Malik:
# |          1: c(x, y, t) = exp(-(nablaI/kappa).^2)
# |             privileges high-contrast edges over low-contrast ones
# |          2: c(x, y, t) = 1 ./ (1 + (nablaI/kappa).^2)
# |             privileges wide regions over smaller ones
# |          default value: 1
#
# Returns:
# | diff   - anisotropic diffused image
#
# Reference:
# [1] P. Perona and J. Malik.
#     Scale-space and edge detection using ansotropic diffusion.
#     IEEE Transactions on Pattern Analysis and Machine Intelligence,
#     12(7):629-639, July 1990.
#     https://dx.doi.org/10.1109%2F34.56205
#
# Credits:
# [1] Peter Kovesi
#     pk@peterkovesi.com
#     MATLAB and Octave Functions for Computer Vision and Image Processing
#     http://www.peterkovesi.com/matlabfns/Spatial/anisodiff.m
#     --
#     June 2000  original version
#     March 2002 corrected diffusion eqn No 2.
# [2] Daniel Lopes
#     Anisotropic Diffusion (Perona & Malik)
#     http://www.mathworks.com/matlabcentral/fileexchange/14995-anisotropic-diffusion--perona---malik-
#
#
# Aaron LI <aaronly.me@gmail.com>
# 2015/07/17
#

include("calc_k_percentile.jl");

function anisodiff(img, niter, k=calc_k_percentile, lambda=0.25, option=1)
    diff = float(img)
    rows, cols = size(diff)

    for i = 1:niter
        println("anisodiff - iteration: ", i)

        # Construct diffl which is the same as diff but
        # has an extra padding of zeros around it.
        diffl = zeros(rows+2, cols+2)
        diffl[2:rows+1, 2:cols+1] = diff

        # North, South, East and West differences
        deltaN = diffl[1:rows,   2:cols+1] - diff
        deltaS = diffl[3:rows+2, 2:cols+1] - diff
        deltaE = diffl[2:rows+1, 3:cols+2] - diff
        deltaW = diffl[2:rows+1, 1:cols]   - diff

        # Calculate the kappa
        if isa(k, Function)
            kappa = k(diff)
        else
            kappa = k
        end

        println("  kappa: ", kappa)

        # Conduction
        if option == 1
            cN = exp(-(deltaN/kappa).^2)
            cS = exp(-(deltaS/kappa).^2)
            cE = exp(-(deltaE/kappa).^2)
            cW = exp(-(deltaW/kappa).^2)
        elseif option == 2
            cN = 1 ./ (1 + (deltaN/kappa).^2)
            cS = 1 ./ (1 + (deltaS/kappa).^2)
            cE = 1 ./ (1 + (deltaE/kappa).^2)
            cW = 1 ./ (1 + (deltaW/kappa).^2)
        end

        diff += lambda * (cN.*deltaN + cS.*deltaS + cE.*deltaE + cW.*deltaW)
    end

    return diff
end

