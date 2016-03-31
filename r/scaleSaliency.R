# -*- coding: utf-8 -*-
#
# Scale Saliency algorithm
#
# Reference:
# [1] T. Kadir & M. Brady, Saliency, Scale and Image Description.
#     2001, International Journal of Computer Vision, 45(2), 83-105
#
# Aaron LI
# 2015/07/29
#


# Calculate Shannon entropy from histogram of probability densities.
# Arguments:
#   phist - histogram of probability density
# Return:
#   entropy value
calcShannonEntropy <- function(phist) {
    # Helper function to calculate the entropy
    # Arguments:
    #   p - probability density of each bin
    # Return:
    #   p * log(p) if p > 0, otherwise 0
    plogp <- function(p) {
        if (p < 1e-10) {
            return(0.0)
        } else {
            return(p * log(p))
        }
    }
    # calculate the entropy
    entropy <- -sum(sapply(phist, plogp))
    return(entropy)
}


# Generate the template circular regions for each scale.
# Arguments:
#   scale_min - minimum scale (radius of circle)
#   scale_max - maximum scale
# Return:
#   list of matrix which represent the regions of interest (with value of TRUE)
circleROI <- function(scale_min, scale_max) {
    rows <- 2 * scale_max + 1
    cols <- rows
    rc   <- (rows + 1) / 2  # central row
    cc   <- (cols + 1) / 2  # central col
    roi  <- list()
    for (s in scale_min:scale_max) {
        radius2 <- s^2
        m <- matrix(0, nrow=rows, ncol=cols)
        roi[[paste("scale", s, sep="")]] <-
                ifelse(((row(m)-rc)^2 + (col(m)-cc)^2) <= radius2,
                       TRUE, FALSE)
    }
    return(roi)
}


# Calculate the scale saliencies for the 1D case: scalar image
# Arguments:
#   img         - input *scalar* image
#   scale_min   - minimum scale (pixels of radius of circle)
#   scale_max   - maximum scale (NOTE: scale_max >= scale_min+2)
#   nbins       - how many bins used for histogram
#   progressbar - whether to display the progress bar
# Return:
#   6-column data.frame contains the scale saliencies results
calcScaleSaliency1D <- function(img, scale_min, scale_max, nbins,
                                progressbar=TRUE) {
    # check scale range first: must have at least 3 scales
    stopifnot(scale_max >= scale_min+2)
    # get number of rows and columns
    rows <- nrow(img)
    cols <- ncol(img)
    # determine the saliency calculation region of the image
    # FIXME: how to deal with the boundaries???
    row_begin <- scale_max + 1
    col_begin <- scale_max + 1
    row_end   <- rows - scale_max
    col_end   <- cols - scale_max
    # templates of regions for each scale
    roi <- circleROI(scale_min, scale_max)
    # R data frame to store the saliency results
    scaleSaliency <- data.frame(row=numeric(0), col=numeric(0),
                                scale=numeric(0), entropy=numeric(0),
                                disimilarity=numeric(0), saliency=numeric(0))
    # determine the breakpoints for histogram
    hist_breaks <- (0:nbins) * (max(img) - min(img))/nbins + min(img)
    if (progressbar) {
        # progress bar
        pb <- txtProgressBar(min=row_begin, max=row_end, style=3)
    }
    for (ri in row_begin:row_end) {
        if (progressbar) {
            # update progress bar
            setTxtProgressBar(pb, ri)
        }
        for (ci in col_begin:col_end) {
            # filter out the required size of image block, which is
            # used to calculate its histogram, entropy, etc.
            imgROI <- img[(ri-scale_max):(ri+scale_max),
                          (ci-scale_max):(ci+scale_max)]
            # vectors to store entropies and distances
            entropy <- numeric(scale_max-scale_min+1)
            distance <- numeric(scale_max-scale_min+1)
            # initial probability density for scale of 's-1'
            scaleHistPr0 <- rep(0, nbins)
            for (s in scale_min:scale_max) {
                scaleROI <- roi[[paste("scale", s, sep="")]]
                # NOTE: do not use 'breaks=nbins', since the number is a
                # suggestion only and breakpoints will be set to 'prtty'
                # values in this case.
                scaleHist <- hist(imgROI[scaleROI],
                                  breaks=hist_breaks, plot=FALSE)
                scaleHistPr <- scaleHist$counts / sum(scaleHist$counts)
                # calculate Shannon entropy
                entropy[s-scale_min+1] <- calcShannonEntropy(scaleHistPr)
                # FIXME: calculate distance of scales???
                distance[s-scale_min+1] <- sum(abs(scaleHistPr-scaleHistPr0))
                # save the probability density of current scale 's'
                scaleHistPr0 <- scaleHistPr
            }
            # smooth the 'distance' vector to reduce the impacts of noise
            distance1 <- c(distance[1], distance[1:(length(distance)-1)])
            distance2 <- c(distance[2:length(distance)],
                           distance[length(distance)])
            distance  <- (distance1 + distance + distance2) / 3
            # find the peaks of entropy, and the corresponding scales
            peakScale <- c(FALSE,
                           ((entropy[2:(length(entropy)-1)] >
                             entropy[1:(length(entropy)-2)]) &
                            (entropy[2:(length(entropy)-1)] >
                             entropy[3:length(entropy)])),
                           FALSE)
            #cat("peakScale:", peakScale, "\n")
            # calculate the inter-scale saliencies for each entropy peaks
            for (s in (scale_min:scale_max)[peakScale]) {
                scaleNorm    <- s*s / (2*s - 1)
                scaleEntropy <- entropy[s-scale_min+1]
                disimilarity <- scaleNorm * distance[s-scale_min+1]
                saliency     <- scaleEntropy * disimilarity
                scaleSaliency[nrow(scaleSaliency)+1, ] <- list(ri, ci, s,
                                                               scaleEntropy,
                                                               disimilarity,
                                                               saliency)
            }
        }
    }
    if (progressbar) {
        # close progress bar
        close(pb)
    }
    return(scaleSaliency)
}


# Simple greedy clustering algorithm to filter out salient regions.
# Arguments:
#   ssaliency       - saliency results from 'calcScaleSaliency*'
#   ssaliency_th    - inter-scale saliency threshold
#   disimilarity_th - disimilarity threshold
# Return:
#   clustered & filtered saliency regions
greedyCluster <- function(ssaliency, ssaliency_th, disimilarity_th) {
    # filter by global saliency & inter-scale saliency threshold
    ssaliency <- ssaliency[((ssaliency$saliency > ssaliency_th) &
                            (ssaliency$disimilarity > disimilarity_th)), ]
    # sort in descending inter-scale saliency
    ssaliency <- ssaliency[order(-ssaliency$saliency), ]
    # cluster salienct points
    clusteredSaliency <- ssaliency[NULL, ]
    while (nrow(ssaliency) > 0) {
        ss <- ssaliency[1, ]
        clusteredSaliency[nrow(clusteredSaliency)+1, ] <- ss
        distance2 <- (ssaliency$row - ss$row)^2 + (ssaliency$col - ss$col)^2
        # filter out the points inside the current salient circle
        ssaliency <- ssaliency[(distance2 > ss$scale^2), ]
    }
    return(clusteredSaliency)
}


# Plot the image and salient regions with ggplot2
# Arguments:
#   img      - input image
#   saliency - saliency restults by clusteredSaliency()
plotSalientReg <- function(img, saliency) {
    require(reshape2)
    require(ggplot2)
    plotCircle <- function(xc, yc, radius) {
        theta <- seq(0, 2*pi, length.out=100)
        gcircle <- annotate("path",
                            x=xc+radius*cos(theta),
                            y=yc+radius*sin(theta),
                            colour="green")
        return(gcircle)
    }
    # plot the image
    gp <- ggplot(melt(img), aes(Var2, -Var1, fill=value)) + geom_raster()
    # add circles
    for (i in 1:nrow(saliency)) {
        ss <- saliency[i, ]
        gcircle <- plotCircle(ss$col, -ss$row, ss$scale)
        gp <- gp + gcircle
    }
    return(gp)
}


# Convert the scale saliency information to DS9 regions.
#
# NOTE:
#   However, the rows and columns of the FITS matrix in R correspond
#   to the X and Y axes in DS9, which is *swapped*.
#   Thus the region width and height correspond to the row range and
#   column range, respectively.
#
# Arguments:
#   saliency - saliency restults by clusteredSaliency()
# Return:
#   vector of DS9 region strings
saliency2region <- function(saliency) {
    regions <- with(saliency,
                    paste("circle(", row, ",", col, ",", scale, ")",
                          sep=""))
    return(regions)
}


# Write DS9 region to file with appropriate header information.
#
# Arguments:
#   filename - output region file
#   region   - vector/list of region strings
save.region <- function(filename, region) {
    rf <- file(filename, "w")
    region.hdr <- c("# Region file format: DS9 version 4.1", "image")
    writeLines(region.hdr, rf)
    writeLines(region, rf)
    close(rf)
}


#  vim: set ts=8 sw=4 tw=0 fenc=utf-8 ft=r: #
