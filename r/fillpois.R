# -*- coding: utf-8 -*-
#
# Identify the abnormal pixels (with relatively larger values) in the
# given X-ray count image, and replace their values with random Poisson
# values whose parameter lambda is determined by the neighboring pixels.
#
#
# Aaron LI
# 2015/09/01
# Updated: 2015/09/02
#

# Fill a vector of a row/column of X-ray image with Poisson distribution.
#
# The parameter lambda of the Poisson distribution is determined by the
# mean value of the left n (default: 3) elements.
#
# If the value of one element is greater than or equal to (>=)
# qpois(prob, lambda), then its value is replaced with rpois(1, lambda).
#
# Arguments:
#   vec  - input data vector
#   n    - number of elements used to calculate the lambda (default: 3)
#   prob - quantile probability (default: 95%)
#
# Return:
#   a vector of the same length with abnormal values replaced
fill.pois.vec <- function(vec, n=3, prob=0.95) {
    # Select the given index of element from the vector
    # Arguments:
    #   bc - boundary condition:
    #        + "cyclic": vec[0, -1] := vec[length(vec), length(vec)-1]
    #        + "symmetric": vec[0, -1] := vec[1, 2]
    elem <- function(vec, index, bc="cyclic") {
        if (index <= 0) {
            if (bc == "cyclic") {
                index <- length(vec) + index
            } else if (bc == "symmetric") {
                index <- 1 - index
            } else {
                stop(paste("Invalid boundary condition:", bc, "\n"))
            }
        }
        return(vec[index])
    }
    # Calculate the mean value of left n elements for the given element.
    mean.left <- function(vec, index, n=3, bc="cyclic") {
        elements <- get(class(vec))(n)
        for (i in 1:n) {
            elements[i] <- elem(vec, index-i)
        }
        return(mean(elements))
    }
    #
    vec.filled <- vec
    for (i in 1:length(vec)) {
        lambda <- mean.left(vec, i, n)
        if (vec[i] >= qpois(prob, lambda)) {
            vec.filled[i] <- rpois(1, lambda)
        }
    }
    return(vec.filled)
}


# Fill a matrix of the X-ray image with Poisson distribution by row or column.
#
# For more details, see 'fill.pois.vec()'.
#
# Arguments:
#   mat   - input image matrix
#   byrow - where Poisson fill the matrix by row (default by column)
#   n     - number of elements used to calculate the lambda (default: 3)
#   prob  - quantile probability (default: 95%)
#
# Return:
#   a matrix of the same size with abnormal values replaced
fill.pois.mat <- function(mat, byrow=FALSE, n=3, prob=0.95) {
    mat.filled <- mat
    if (byrow) {
        # process by row
        rows <- nrow(mat)
        for (r in 1:rows) {
            vec <- mat[r, ]
            vec.filled <- fill.pois.vec(vec, n, prob)
            mat.filled[r, ] <- vec.filled
        }
    } else {
        # process by column
        cols <- ncol(mat)
        for (c in 1:cols) {
            vec <- mat[, c]
            vec.filled <- fill.pois.vec(vec, n, prob)
            mat.filled[, c] <- vec.filled
        }
    }
    return(mat.filled)
}


# Identify the abnormal pixels (with relatively larger values) in the
# given X-ray count image, and replace their values with random Poisson
# values whose parameter lambda is determined by the neighboring pixels.
#
# The refilled image is the average of the two images, which are the
# original image processed with 'fill.pois.mat()' by row and column,
# respectively.
#
# TODO: to verify???
# The two-step procedure is employed to avoid the grid-like pattern/structure
# in the refilled image.
#
# For more details, see 'fill.pois.vec()'.
#
# Arguments:
#   img   - input image (a matrix)
#   n     - number of elements used to calculate the lambda (default: 3)
#   prob  - quantile probability (default: 95%)
#
# Return:
#   a matrix of the same size with abnormal values replaced
fill.pois.img <- function(img, n=3, prob=0.95) {
    img.fillbycol <- fill.pois.mat(img, byrow=FALSE, n=n, prob=prob)
    img.fillbyrow <- fill.pois.mat(img, byrow=TRUE, n=n, prob=prob)
    img.filled <- (img.fillbycol + img.fillbyrow) / 2
    return(img.filled)
}


#  vim: set ts=8 sw=4 tw=0 fenc=utf-8 ft=r: #
