# -*- coding: utf-8 -*-
#
# Kullback-Leibler or Jensen-Shannon divergence between two distributions
#
# The Kullback-Leibler divergence is given by:
#   D_{KL}(P(x), Q(x)) = sum[ P(x) * log(P(x) / Q(x)) ]
# where P(x) is the underground true distribution, and Q(x) the approximation
# distribution.  Thus KL divergence measures the information lost when Q is
# used to approximate P.
#
# The Jensen-Shannon divergence is given by:
#   D_{JS}(P, Q) = 0.5 * D_{KL}(P, M) + 0.5 * D_{KL}(Q, M);  M = (P+Q)/2
# This is a symmetrised divergence, and is equal to 1/2 the so-called
# Jeffrey divergence.
#
# Credits:
# [1] Wikipedia - Kullback-Leibler divergence
#     https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
# [2] David Fass, KLDIV
#     http://www.mathworks.com/matlabcentral/fileexchange/13089-kldiv/content//kldiv.m
#
# Aaron LI
# 2015/09/04
#


# Calculate the entropy of the probability mass distribution.
# The zeros are ignored.
#
# Arguments:
#   x - probability mass distribution
#
# Return:
#   entropy in unit "bits"
#
calc.entropy <- function(x) {
    x.gt0 <- x[x>0]
    return(sum(-x.gt0 * log2(x.gt0)))
}


# Calculate the KL divergence of distribution P from Q, or the JS divergence
# between the P and Q distributions.
#
# TODO:
# * to add other methods to deal with zero probabilities:
#   - add eps to p.m.f and renormalize
#   - Byesian prior
#   - smoothing
#
# Credits:
# [1] Calculate the Kullback-Leibler Divergence in practice?
#     http://stats.stackexchange.com/questions/97938/calculate-the-kullback-leibler-divergence-in-practice
# [2] How to compute KL-divergence when PMF contains 0s?
#     http://mathoverflow.net/questions/72668/how-to-compute-kl-divergence-when-pmf-contains-0s
#
# Arguments:
#   p     - probabilities representing the distribution P (underground true)
#   q     - probabilities representing the distribution Q (approximation)
#   type  - which type of divergence to be calculated
#           + "kl": (default) Kullback-Leibler divergence
#           + "klsym": symmetric variant of the Kullback-Leibler divergence,
#                      which given by (KL(p, q) + KL(q, p))/2
#           + "js": Jensen-Shannon divergence
#   zeros - how to deal with the zeros in each distribution probabilities
#           + "ignore": just ignore the data points with probability of zero
#
#   Note that the vectors p and q must have the same length, and the
#   sum of probabilities p and q must be 1 +/- 1e-5
#
# Return:
#   calculate divergence value in unit "bits"
#
kldiv <- function(p, q, type="kl", zeros="ignore") {
    # check length of vectors
    stopifnot(length(p) == length(q))
    # validate probabilities
    eps_prob <- 1e-5
    stopifnot(abs(sum(p) - 1) <= eps_prob, abs(sum(q) - 1) <= eps_prob)
    # how to deal with zero probabilities
    if (zeros == "ignore") {
        # just ignore the zeros in probabilities
        nonzeros <- (p > 0) & (q > 0)
        p <- p[nonzeros]
        q <- q[nonzeros]
    } else {
        stop(paste("Unsupported parameter value zeros=", zeros, "\n", sep=""))
    }
    # check divergence type
    if (type == "kl") {
        # Kullback-Leibler divergence
        div <- sum(p * (log2(p) - log2(q)))
    } else if (type == "klsym") {
        # symmetric variant KL divergence
        div <- 0.5 * (sum(p * (log2(p) - log2(q))) +
                      sum(q * (log2(q) - log2(p))))
    } else if (type == "js") {
        # Jensen-Shannon divergence
        m <- (p + q) / 2
        div <- 0.5 * (sum(p * (log2(p) - log2(m))) +
                      sum(q * (log2(q) - log2(m))))
    } else {
        stop(paste("Unsupported parameter value type=", type, "\n", sep=""))
    }
    return(div)
}


# Estimate the probability mass distribution for the observation data,
# using "density()".
# The range of output coordinates of points is set to be:
#   from: min(x) - cut*bw
#   to:   max(x) + cut*bw
# And the probability mass distribution is normalized.
#
# Arguments:
#   x      - input observation data
#   n      - number of equally spaced points at which the probability mass is
#            to be estimated.
#   bw     - bandwidth to be used
#   kernel - the smoothing kernel
#   from   - coordinate of the left-most point
#   to     - coordinate of the right-most point
#   cut    - c(left, right). Number of bandwidths beyond the left and right
#            extremes of the input data.
#            This allows the estimated density to drop to approximately zero
#            at the extremes.
#            If "from" and "to" specified, then "cut" is ignored.
#
# Returns:
#   list with following components:
#   x      - the coordinates of the points where probability mass estimated
#   y      - the estimated probability mass
#   bw     - bandwidth used
#   kernel - kernel used
#   n      - sample size
#   cut    - left and right cut used
#   from   - coordinate of the left-most point used
#   to     - coordinate of the right-most point used
#
estimate.prob.mass <- function(x, bw="nrd0", kernel="gaussian", n=512,
                               from=NULL, to=NULL, cut=c(3,3)) {
    data <- x[!is.na(x)]
    # calculate the bandwidth
    bw <- get(paste("bw.", bw, sep=""))(data)
    # determine the estimation range
    if (is.null(from)) {
        from <- min(data) - cut[1] * bw
    }
    if (is.null(to)) {
        to <- max(data) + cut[2] * bw
    }
    # estimate with "density()"
    d <- density(data, bw=bw, kernel=kernel, n=n, from=from, to=to)
    # renormalize the probability mass distribution
    pmass <- d$y / sum(d$y)
    prob.mass <- list(x=d$x, y=pmass, bw=bw, kernel=kernel,
                      n=n, from=from, to=to, cut=cut)
    return(prob.mass)
}


# Estimate the probability mass distribution for the source and corresponding
# background data using 'estimate.prob.mass()'.
#
# The coordinates at which the probability masses are estimated are the same
# for the source and corresponding background probability mass distributions.
# Therefore we can calculate the KL divergence between these two distributions.
#
# Argument:
#   srcdata - raw counts data drawn from the source region
#   bkgdata - raw counts data drawn from the background region
#
# Return:
#   data.frame with the following components:
#   x   - the coordinates of the points where probability mass estimated
#   src - the estimated probability masses of the source data
#   bkg - the estimated probability masses of the background data
#
pm.src.bkg <- function(srcdata, bkgdata) {
    # compare the data ranges
    if (max(srcdata) > max(bkgdata)) {
        pm.src <- estimate.prob.mass(srcdata)
        from <- pm.src$from
        to <- pm.src$to
        pm.bkg <- estimate.prob.mass(bkgdata, from=from, to=to)
    } else {
        pm.bkg <- estimate.prob.mass(bkgdata)
        from <- pm.bkg$from
        to <- pm.bkg$to
        pm.src <- estimate.prob.mass(srcdata, from=from, to=to)
    }
    df <- data.frame(x=pm.src$x, src=pm.src$y, bkg=pm.bkg$y)
    return(df)
}


# Calculate the entropies and KL/JS divergences of the source and background
# probability mass distribution group.
#
# Arguments:
#   pmdf - data.frame of the probability mass distribution
#   comp - components to be calculated
#          + "entropy": entropy of the source and background
#          + "kl": KL divergences from source to background and vice versa
#          + "klsym": symmetric variant of KL divergence
#          + "js": JS divergence
#
# Return:
#   list with following components:
#   entropy.src - entropy of the source distribution
#   entropy.bkg - entropy of the background distribution
#   kl.src2bkg  - KL divergence from source to background
#   kl.bkg2src  - KL divergence from background to source
#   klsym       - symmetric variant KL divergence
#   js          - JS divergence
info.src.bkg <- function(pmdf, comp=c("entropy", "kl", "klsym", "js")) {
    pm.src <- pmdf$src
    pm.bkg <- pmdf$bkg
    entropy.src <- NULL
    entropy.bkg <- NULL
    kl.src2bkg  <- NULL
    kl.bkg2src  <- NULL
    klsym       <- NULL
    js          <- NULL
    if ("entropy" %in% comp) {
        entropy.src <- calc.entropy(pm.src)
        entropy.bkg <- calc.entropy(pm.bkg)
    }
    if ("kl" %in% comp) {
        kl.src2bkg <- kldiv(pm.src, pm.bkg, type="kl")
        kl.bkg2src <- kldiv(pm.bkg, pm.src, type="kl")
    }
    if ("klsym" %in% comp) {
        klsym <- kldiv(pm.src, pm.bkg, type="klsym")
    }
    if ("js" %in% comp) {
        js <- kldiv(pm.src, pm.bkg, type="js")
    }
    return(list(entropy.src=entropy.src, entropy.bkg=entropy.bkg,
                kl.src2bkg=kl.src2bkg, kl.bkg2src=kl.bkg2src,
                klsym=klsym, js=js))
}


# Calculate the entropies and KL/JS divergences of the source density
# histogram with respect to the corresponding background data which
# drawn from the estimated Poisson mass distribution.
#
# Arguments:
#   src  - raw counts data of the source region
#   comp - components to be calculated
#          + "entropy": entropy of the source and background
#          + "kl": KL divergences from source to background and vice versa
#          + "klsym": symmetric variant of KL divergence
#          + "js": JS divergence
#
# Return:
#   list with following components:
#   entropy.src - entropy of the source distribution
#   entropy.bkg - entropy of the background distribution
#   kl.src2bkg  - KL divergence from source to background
#   kl.bkg2src  - KL divergence from background to source
#   klsym       - symmetric variant KL divergence
#   js          - JS divergence
#
info.src.pois <- function(src, comp=c("entropy", "kl", "klsym", "js")) {
    # make the density histogram of the source counts data
    hist.src <- hist(src, breaks=(min(src):(max(src)+1)-0.5), plot=FALSE)
    x <- hist.src$mids
    pm.src <- hist.src$density
    # calculate the corresponding theoretical Poisson density/mass distribution
    # as the estimated background
    lambda <- mean(src)
    pm.pois <- dpois(x, lambda)
    pm.pois <- pm.pois / sum(pm.pois)
    # calculate the entropy, KL/JS divergences
    entropy.src <- NULL
    entropy.bkg <- NULL
    kl.src2bkg  <- NULL
    kl.bkg2src  <- NULL
    klsym       <- NULL
    js          <- NULL
    if ("entropy" %in% comp) {
        entropy.src <- calc.entropy(pm.src)
        entropy.bkg <- calc.entropy(pm.pois)
    }
    if ("kl" %in% comp) {
        kl.src2bkg <- kldiv(pm.src, pm.pois, type="kl")
        kl.bkg2src <- kldiv(pm.pois, pm.src, type="kl")
    }
    if ("klsym" %in% comp) {
        klsym <- kldiv(pm.src, pm.pois, type="klsym")
    }
    if ("js" %in% comp) {
        js <- kldiv(pm.src, pm.pois, type="js")
    }
    return(list(entropy.src=entropy.src, entropy.bkg=entropy.bkg,
                kl.src2bkg=kl.src2bkg, kl.bkg2src=kl.bkg2src,
                klsym=klsym, js=js))
}


# Calculate the information (e.g., entropy, divergences) for each group of
# region data.
# If the background data are not provided, then the background is estimated
# with a Poisson density/mass distribution.
info.reglist <- function(srcdatalist, bkgdatalist=NULL) {
    if (is.null(bkgdatalist)) {
        infofunc <- "info.src.pois"
    } else {
        infofunc <- "info.src.bkg"
        stopifnot(length(srcdatalist) == length(bkgdatalist))
    }
    l <- length(srcdatalist)
    infodf <- data.frame(entropy.src=numeric(l), entropy.bkg=numeric(l),
                         kl.src2bkg=numeric(l), kl.bkg2src=numeric(l),
                         klsym=numeric(l), js=numeric(l))
    for (i in 1:length(srcdatalist)) {
        #cat(i, "\n")
        if (is.null(bkgdatalist)) {
            if (sum(srcdatalist[[i]]) == 0) {
                # srcdata all zeros
                cat(i, ": WARNING: srcdata are all zeros!\n")
                info <- list(entropy.src=NA, entropy.bkg=NA,
                             kl.src2bkg=NA, kl.bkg2src=NA,
                             klsym=NA, js=NA)
            } else {
                info <- get(infofunc)(srcdatalist[[i]])
            }
        } else {
            if (sum(srcdatalist[[i]]) == 0 || sum(bkgdatalist[[i]]) == 0) {
                # srcdata / bkgdata all zeros
                cat(i, ": WARNING: srcdata/bkgdata are all zeros!\n")
                info <- list(entropy.src=NA, entropy.bkg=NA,
                             kl.src2bkg=NA, kl.bkg2src=NA,
                             klsym=NA, js=NA)
            } else {
                pmdf <- pm.src.bkg(srcdatalist[[i]], bkgdatalist[[i]])
                info <- get(infofunc)(pmdf)
            }
        }
        infodf[i, ] <- info
    }
    return(infodf)
}


# vim: set ts=8 sw=4 tw=0 fenc=utf-8 ft=r: #
