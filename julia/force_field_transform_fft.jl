# -*- coding: utf-8 -*-
#
# To do force field transform using FFT
#
# Aaron LI
# 2015/07/16
#

function forcefieldtransform_fft(img)
    rows, cols = size(img)
    pic = zeros(3*rows, 3*cols)
    pic[1:rows, 1:cols] = img
    # unit force field
    unit_ff = complex(zeros(3*rows, 3*cols))
    for r = 1:(2*rows-1)
        for c = 1:(2*cols)
            d = (rows+cols*im) - (r+c*im)
            if (r, c) == (rows, cols)
                unit_ff[r, c] = 0 + 0im
            else
                unit_ff[r, c] = d / abs(d)^3
            end
        end
    end
    # FIXME matrix sizes
    ff = sqrt(rows*cols) * ifft(fft(pic) .* fft(unit_ff))
    #ff_crop = ff[rows:2*rows, cols:2*cols]
end

