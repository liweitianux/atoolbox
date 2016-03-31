# -*- coding: utf-8 -*-
#
# Calculate the percentile of the gradient image, which
# used as the 'kappa' parameter of the anisotropic diffusion.
#
# Credits:
# [1] KAZE: nldiffusion_functions.cpp / compute_k_percentile()
#
# Aaron LI
# 2015/07/20
#

include("scharr.jl");

function calc_k_percentile(img, percent=0.7, nbins=300)
    rows, cols = size(img)
    # derivatives of the image
    img_gx = scharr(img, 1, 0)
    img_gy = scharr(img, 0, 1)
    img_modg = sqrt(img_gx.^2 + img_gy.^2)
    # histogram
    hmax = maximum(img_modg)
    hist_e, hist_counts = hist(reshape(img_modg, length(img_modg)), nbins)
    hist_cum = cumsum(hist_counts)
    # find the percent of the histogram percentile
    npoints = sum(img_modg .> 0.0)
    nthreshold = npoints * percent
    k = sum(hist_cum .<= nthreshold)
    kperc = (k == length(hist_cum)) ? 0.03 : (hmax * k / nbins)
    return kperc
end

