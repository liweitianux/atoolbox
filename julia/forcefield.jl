# -*- coding: utf-8 -*-
#
# Force field transform with specified size of mask.
#
# Aaron LI
# 2015/07/19
#

# Make the specified sized force field mask.
# NOTE: the number of rows and cols must be odd.
function ff_mask(rows=5, cols=5)
    rows % 2 == 1 || error("rows must be odd number")
    cols % 2 == 1 || error("cols must be odd number")
    mask = complex(zeros(rows, cols))
    for r = range(-div(rows, 2), rows)
        for c = range(-div(cols, 2), cols)
            i, j = r + div(rows+1, 2), c + div(cols+1, 2)
            #@printf("(r,c) = (%d,%d); (i,j) = (%d,%d)\n", r, c, i, j)
            d = c + r*im
            if abs(d) < 1e-8
                mask[i, j] = 0.0
            else
                mask[i, j] = d / abs(d)^3
            end
        end
    end
    return mask / sum(abs(mask))
end


# Padding image by specified number of rows and cols.
# Default padding mode: mirror
function pad_image(img, pad_rows, pad_cols, mode="mirror")
    rows, cols = size(img)
    rows_new, cols_new = rows + 2*pad_rows, cols + 2*pad_cols
    img_pad = zeros(rows_new, cols_new)
    img_pad[(pad_rows+1):(pad_rows+rows), (pad_cols+1):(pad_cols+cols)] = img
    for r = 1:rows_new
        for c = 1:cols_new
            if mode == "mirror"
                if r <= pad_rows
                    r_mirror = 2*(pad_rows+1) - r
                elseif r <= pad_rows+rows
                    r_mirror = r
                else
                    r_mirror = 2*(pad_rows+rows) - r
                end
                if c <= pad_cols
                    c_mirror = 2*(pad_cols+1) - c
                elseif c <= pad_cols+cols
                    c_mirror = c
                else
                    c_mirror = 2*(pad_cols+cols) - c
                end
                if (r_mirror, c_mirror) != (r, c)
                    #@printf("(%d,%d) <= (%d,%d)\n", r, c, r_mirror, c_mirror)
                    img_pad[r, c] = img_pad[r_mirror, c_mirror]
                end
            else
                error("mode not supported")
            end
        end
    end
    return img_pad
end


# Perform force field transform for the image.
function ff_transform(img, mask, mode="mirror")
    rows, cols = size(img)
    mask_rows, mask_cols = size(mask)
    pad_rows, pad_cols = div(mask_rows, 2), div(mask_cols, 2)
    img_pad = pad_image(img, pad_rows, pad_cols)
    # result images
    ff_amplitudes = zeros(rows, cols)
    ff_angles = zeros(rows, cols)
    # calculate transformed values
    for r = (pad_rows+1):(pad_rows+rows)
        for c = (pad_cols+1):(pad_cols+cols)
            force = sum(img_pad[r, c] * img_pad[(r-pad_rows):(r+pad_rows), (c-pad_cols):(c+pad_cols)] .* mask)
            ff_amplitudes[r-pad_rows, c-pad_cols] = abs(force)
            ff_angles[r-pad_rows, c-pad_cols] = angle(force)
        end
    end
    return ff_amplitudes, ff_angles
end

