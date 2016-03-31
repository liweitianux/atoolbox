#!/usr/bin/env julia
# -*- coding: utf-8 -*-
#
# Force field transform
#
# Aaron LI
# 2015/07/14
#

using FITSIO;
#include("../julia/ndgrid.jl");

@everywhere function meshgrid(vx, vy)
    m, n = length(vy), length(vx)
    vx = reshape(vx, 1, n)
    vy = reshape(vy, m, 1)
    (repmat(vx, m, 1), repmat(vy, 1, n))
end


# Calculate the forces between the specified point with respect to the image.
@everywhere function force(p0, img)
    img = copy(img);
    x0, y0 = p0;
    v0 = img[y0, x0];
    img[y0, x0] = 0.0;
    rows, cols = size(img);
    x, y = meshgrid(1:cols, 1:rows);
    x[y0, x0] = -1;
    y[y0, x0] = -1;
    f_x = v0 .* img .* (x-x0) ./ ((x-x0).^2 + (y-y0).^2).^1.5;
    f_y = v0 .* img .* (y-y0) ./ ((x-x0).^2 + (y-y0).^2).^1.5;
    #return (f_x, f_y);
    return (sum(f_x), sum(f_y));
end


# Perform the "force field transform" for the input image.
#
# Return:
#   (amplitudes, angles)
#   amplitudes: the amplitudes of the resulting forces of each pixel
#   angles: the directions of the resulting forces of each pixel,
#           in unit radian.
@everywhere function force_field_transform_serial(img, rowstart=1, rowend="end")
    rows, cols = size(img)
    if rowend == "end"
        rowend = rows
    end
    amplitudes = zeros(rows, cols)
    angles = zeros(rows, cols)
    t0 = time()
    t_p = t0 + 30  # in 30 seconds
    for y = rowstart:rowend
        for x = 1:cols
            t1 = time()
            if (t1 >= t_p)
                percent = 100*((y-rowstart)*cols + x+1) / ((rowend-rowstart+1)*cols)
                @printf("Worker #%d: progress: %.3f%%; %.1f min\n",
                        myid(), percent, (t1-t0)/60.0)
                t_p += 30  # in 30 seconds
            end
            F_x, F_y = force((x, y), img)
            #@printf("F_x, F_y = (%f, %f)\n", F_x, F_y);
            amplitudes[y, x] = sqrt(F_x^2 + F_y^2)
            angles[y, x] = atan2(F_y, F_x)
        end
    end
    t1 = time()
    @printf("Worker #%d: finished in %.1f min!\n", myid(), (t1-t0)/60.0)
    return (amplitudes, angles)
end


# parallel-capable
function force_field_transform(img)
    t0 = time()
    rows, cols = size(img)
    np = nprocs()
    amplitudes = cell(np)
    angles = cell(np)
    # split rows for each process
    rows_chunk = div(rows, np)
    rowstart = cell(np)
    rowend = cell(np)
    @sync begin
        for p = 1:np
            rowstart[p] = 1 + rows_chunk * (p-1)
            if p == np
                rowend[p] = rows
            else
                rowend[p] = rowstart[p] + rows_chunk - 1
            end
            # perform transform
            @async begin
                amplitudes[p], angles[p] = remotecall_fetch(p,
                        force_field_transform_serial,
                        img, rowstart[p], rowend[p])
            end
        end
    end
    t1 = time()
    @printf("Finished in %.1f min!\n", (t1-t0)/60.0)
    return (sum(amplitudes), sum(angles))
end


# arguments
#println(ARGS);
if length(ARGS) != 3
    println("Usage: PROG <input_fits_img> <out_fits_amplitudes> <out_fits_angles>");
    exit(1);
end

infile = ARGS[1];
outfile_ampl = ARGS[2];
outfile_angles = ARGS[3];

fits_img = FITS(infile);
img = read(fits_img[1]);
header = read_header(fits_img[1]);

# perform force field transform
ampl, angles = force_field_transform(img);

outfits_ampl = FITS(outfile_ampl, "w");
outfits_angles = FITS(outfile_angles, "w");
write(outfits_ampl, ampl; header=header);
write(outfits_angles, angles; header=header);

close(fits_img);
close(outfits_ampl);
close(outfits_angles);

#= vim: set ts=8 sw=4 tw=0 fenc=utf-8 ft=julia: =#
