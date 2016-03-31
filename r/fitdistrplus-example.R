n <- 50
m <- 50
set.seed(1)
mu <- -0.4
sig <- 0.12
x <- matrix(data=rlnorm(n*m, mu, sig), nrow=m)

library(fitdistrplus)
## Fit a log-normal distribution to the 50 random data set
f <- apply(x, 2,  fitdist, "lnorm")

## Plot the results 
for(i in 1:n)
plot(f[[i]])

## Save plot in an animated GIF-file
library(animation)
saveGIF({for(i in 1:n) plot(f[[i]])})

apply((sapply(f, "[[", "estimate")),1, summary)
#         meanlog   sdlog
# Min.    -0.4347 0.09876
# 1st Qu. -0.4140 0.11480
# Median  -0.4010 0.12110
# Mean    -0.4011 0.12270
# 3rd Qu. -0.3899 0.12950
# Max.    -0.3522 0.14780


## How much variance can we expect in the mean and std?
## Expeted mean
ExpectedMean <- function(mu, sig) exp(mu+ sig^2/2)
## Expected std
ExpectedStd <- function(mu, sig) sqrt((exp(sig^2)-1)*exp(2*mu + sig^2))

summary(apply(sapply(f, "[[", "estimate"), 2, function(x) ExpectedMean(x[1], x[2])))
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.6529  0.6665  0.6747  0.6748  0.6819  0.7087 
summary(apply(sapply(f, "[[", "estimate"), 2, function(x) ExpectedStd(x[1], x[2])))
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.06604 0.07880 0.08212 0.08316 0.08794 0.10170 

## Let's look at the goodness of fit statistics to get an
## idea how much variance we can expect there:
gof.ln <- lapply(f, gofstat)
gof.test <- lapply(gof.ln, function(x) data.frame(x[c("chisqpvalue", "cvm", "ad", "ks")]))
apply(do.call("rbind", gof.test), 2, summary)
#         chisqpvalue     cvm     ad      ks
# Min.      0.0002673 0.02117 0.1537 0.05438
# 1st Qu.   0.1394000 0.03755 0.2708 0.07488
# Median    0.3578000 0.04841 0.3216 0.08054
# Mean      0.3814000 0.05489 0.3564 0.08431
# 3rd Qu.   0.6409000 0.06913 0.4358 0.09436
# Max.      0.9245000 0.13220 0.7395 0.12570