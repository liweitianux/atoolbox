# -*- coding: utf -*-
#
# Calculate the "force field transform" of the image, using the
# specified *cell* size.
#
# The image is padded with the mirrored boundary condition.
#
# NOTE:TODO:
# The transformation output strengths image is NOT normalized!
#
#
# Credit:
# [1] TODO:
#     Hurley et al., 2002, 2005
#
#
# Aaron LI
# 2015/08/28
#


# The attractive force between two points on the image.
# NOTE: the coefficient is ignored
#
# Arguments:
#   p0, p1 - (r, c, value), the row and column number of the point position,
#            and the value of that point
#
# Return:
#   the force vector (f_r, f_c):
#     'f_r': force along the row direction, positive goes downside
#     'f_c': force along the column direction, positive goes to the right
#   Note that this is the force that 'p1' act on 'p0', and is directed
#   to point 'p1'.
p2p.force <- function(p0, p1) {
    r0 = p0[1]
    c0 = p0[2]
    r1 = p1[1]
    c1 = p1[2]
    f_r = p0[3]*p1[3] * (r1-r0) / ((r1-r0)^2 + (c1-c0)^2)^1.5
    f_c = p0[3]*p1[3] * (c1-c0) / ((r1-r0)^2 + (c1-c0)^2)^1.5
    return(c(f_r, f_c))
}


#  vim: set ts=8 sw=4 tw=0 fenc=utf-8 ft=r: #
