#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Aaron LI <aaronly.me@gmail.com>
# 2015/04/22
#

"""
Computes the radially averaged power spectral density (power spectrum).
"""


import numpy as np
from scipy import fftpack


def PSD2d( img, normalize=True ):
    """
    Computes the 2D power spectrum of the given image.

    Reference:
    [1] raPsd2d.m by Evan Ruzanski
        Radially averaged power spectrum of 2D real-valued matrix
        https://www.mathworks.com/matlabcentral/fileexchange/23636-radially-averaged-power-spectrum-of-2d-real-valued-matrix
    """
    img = np.array( img )
    rows, cols = img.shape
    ## Compute power spectrum
    # Perform the Fourier transform and shift the zero-frequency
    # component to the center of the spectrum.
    imgf = fftpack.fftshift( fftpack.fft2( img ) )
    if normalize:
        norm = rows * cols
    else:
        norm = 1.0  # Do not normalize
    psd2d = ( np.abs( imgf ) / norm ) ** 2
    return psd2d


def radialPSD( psd2d ):
    """
    Computes the radially averaged power spectral density (power spectrum)
    from the provided 2D power spectrum.

    Reference:
    [1] raPsd2d.m by Evan Ruzanski
        Radially averaged power spectrum of 2D real-valued matrix
        https://www.mathworks.com/matlabcentral/fileexchange/23636-radially-averaged-power-spectrum-of-2d-real-valued-matrix
    """
    psd2d = np.array( psd2d )
    rows, cols = psd2d.shape
    ## Adjust the PSD array size
    dim_diff = np.abs( rows - cols )
    dim_max = max( rows, cols )
    # Pad the PSD array to be sqaure
    if rows > cols:
        # pad columns
        if np.mod( dim_diff, 2 ) == 0:
            cols_left = np.zeros( (rows, dim_diff/2) )
            cols_left[:] = np.nan
            cols_right = np.zeros( (rows, dim_diff/2) )
            cols_right[:] = np.nan
            psd2d = np.hstack( (cols_left, psd2d, cols_right) )
        else:
            cols_left = np.zeros( (rows, np.floor(dim_diff/2)) )
            cols_left[:] = np.nan
            cols_right = np.zeros( (rows, np.floor(dim_diff/2)+1) )
            cols_right[:] = np.nan
            psd2d = np.hstack( (cols_left, psd2d, cols_right) )
    elif rows < cols:
        # pad rows
        if np.mod( dim_diff, 2 ) == 0:
            rows_top = np.zeros( (dim_diff/2, cols) )
            rows_top[:] = np.nan
            rows_bottom = np.zeros( (dim_diff/2, cols) )
            rows_bottom[:] = np.nan
            psd2d = np.vstack( (rows_top, psd2d, rows_bottom) )
        else:
            rows_top = np.zeros( (np.floor(dim_diff/2), cols) )
            rows_top[:] = np.nan
            rows_bottom = np.zeros( (np.floor(dim_diff/2)+1, cols) )
            rows_bottom[:] = np.nan
            psd2d = np.vstack( (rows_top, psd2d, rows_bottom) )
    ## Compute radially average power spectrum
    px = np.arange( -dim_max/2, dim_max/2 )
    x, y = np.meshgrid( px, px )
    rho, phi = cart2pol( x, y )
    rho = np.around( rho ).astype(int)
    dim_half = np.floor( dim_max/2 ) + 1
    radial_psd = np.zeros( dim_half )
    radial_psd_err = np.zeros( dim_half ) # standard error
    for r in np.arange( dim_half, dtype=int ):
        # Get the indices of the elements satisfying rho[i,j]==r
        ii, jj = (rho == r).nonzero()
        # Calculate the mean value at a given radii
        data = psd2d[ii, jj]
        radial_psd[r] = np.nanmean( data )
        radial_psd_err[r] = np.nanstd( data )
    # Calculate frequencies
    f = fftpack.fftfreq( dim_max, d=1 ) # sample spacing: set to 1 pixel
    freqs = np.abs( f[:dim_half] )
    #
    return (freqs, radial_psd, radial_psd_err)


def plotRadialPSD( freqs, radial_psd, radial_psd_err=None ):
    """
    Make a plot of the radial 1D PSD with matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        import sys
        print( "Error: matplotlib.pyplot cannot be imported!",
                file=sys.stderr )
        sys.exit( 30 )
    dim_half = radial_psd.size
    # plot
    plt.figure()
    plt.loglog( freqs, radial_psd )
    plt.title( "Radially averaged power spectrum" )
    plt.xlabel( "k (/pixel)" )
    plt.ylabel( "Power" )
    plt.show()


def cart2pol( x, y ):
    """
    Convert Cartesian coordinates to polar coordinates.
    """
    rho = np.sqrt( x**2 + y**2 )
    phi = np.arctan2( y, x )
    return (rho, phi)

def pol2cart( rho, phi ):
    """
    Convert polar coordinates to Cartesian coordinates.
    """
    x = rho * np.cos( phi )
    y = rho * np.sin( phi )
    return (x, y)


def loadData( filename, ftype="fits" ):
    """
    Load data from file into numpy array.
    """
    if ftype == "fits":
        try:
            from astropy.io import fits
        except ImportError:
            import sys
            print( "Error: astropy.io.fits cannot be imported!",
                    file=sys.stderr )
            sys.exit( 20 )
        ffile = fits.open( filename )
        data = ffile[0].data.astype( float )
        ffile.close()
    else:
        print( "Error: not implemented yet!",
                file=sys.stderr )
        sys.exit( 10 )
    #
    return data


def main():
    pass


if __name__ == "__main__":
    main()

