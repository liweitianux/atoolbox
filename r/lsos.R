# -*- encoding: utf-8 -*-

# Tricks to manage the available memory in an R session
# http://stackoverflow.com/q/1358003/4856091 

# improved list of objects
.ls.objects <- function(pos=1, pattern, order.by,
                        decreasing=FALSE, pretty.size=FALSE,
                        head=FALSE, n=10) {
    napply <- function(names, fn) {
        sapply(names, function(x) fn(get(x, pos=pos)))
    }
    names <- ls(pos=pos, pattern=pattern)
    obj.class <- napply(names, function(x) as.character(class(x))[1])
    obj.mode <- napply(names, mode)
    obj.type <- ifelse(is.na(obj.class), obj.mode, obj.class)
    obj.size.bytes <- napply(names, object.size)
    if (pretty.size) {
        obj.size <- napply(names, function(x) {
                               format(object.size(x), units="auto")
                           })
    } else {
        obj.size <- obj.size.bytes
    }
    obj.dim <- t(napply(names, function(x) as.numeric(dim(x))[1:2]))
    vec <- is.na(obj.dim)[, 1] & (obj.type != "function")
    obj.dim[vec, 1] <- napply(names, length)[vec]
    out <- data.frame(obj.type, obj.size, obj.dim)
    names(out) <- c("Type", "Size", "Rows", "Columns")
    if (! missing(order.by))
        if (order.by == "Size") {
            out <- out[order(obj.size.bytes, decreasing=decreasing), ]
        } else {
            out <- out[order(out[[order.by]], decreasing=decreasing), ]
        }
    if (head)
        out <- head(out, n)
    out
}
# shorthand
lsobjs <- function(..., n=10) {
    .ls.objects(..., order.by="Size", decreasing=TRUE,
                pretty.size=TRUE, head=TRUE, n=n)
}

