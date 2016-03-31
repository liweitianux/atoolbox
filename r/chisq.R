# -*- coding: utf-8 -*-
#
# Calculate the chi-squared values between the given data and model values.
#

calc.chisq <- function(value, error=NULL, model=NULL) {
    if (is.data.frame(value)) {
        df <- value
        value <- df$value
        error <- df$error
        model <- df$model
    }
    chisq <- (value - model)^2
    if (! is.null(error)) {
        weights <- error ^ (-2)
        chisq <- chisq * weights
    }
    return(sum(chisq))
}

#  vim: set ts=8 sw=4 tw=0 fenc=utf-8 ft=r: #
