# -*- coding: utf-8 -*-
#
# Calculate the derivatives of an image using the Scharr operator
# of kernal size 3x3.
#
# References:
# [1] https://en.wikipedia.org/wiki/Sobel_operator
# [2] http://docs.opencv.org/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html
#
# Aaron LI
# 2015/07/20
#

# Calculate the derivatives of the image using the Scharr operator
# img - input image
# dx  - order of the derivative x
# dy  - order of the derivative y
function scharr(img, dx, dy)
    rows, cols = size(img)
    img_d = float(img)
    (isa(dx, Int) && dx >= 0) || error("dx should be an integer >= 0")
    (isa(dy, Int) && dy >= 0) || error("dy should be an integer >= 0")
    # Scharr operator
    Gy = [-3.0 -10.0 -3.0; 0.0 0.0 0.0; 3.0 10.0 3.0];
    Gx = Gy'
    # calculate the derivatives using convolution
    for i = 1:dx
        img_d = conv2(img_d, Gx)
    end
    for i = 1:dy
        img_d = conv2(img_d, Gy)
    end
    # FIXME: 'conv2' will increase the image size
    rows_d, cols_d = size(img_d)
    return img_d[(div(rows_d-rows, 2)+1):(div(rows_d-rows, 2)+rows), (div(cols_d-cols, 2)+1):(div(cols_d-cols, 2)+cols)]
end

