"""
Utility script to create a synthetic FITS source image
Run this if you don't have a source FITS file
"""

import numpy as np
from astropy.io import fits
from astropy.modeling.functional_models import Gaussian2D
import os

def create_synthetic_source(filename='source_image.fits', size=300, pixel_scale=0.01):
    """
    Create a synthetic Gaussian source image as a FITS file
    
    Parameters:
    -----------
    filename : str
        Output FITS filename
    size : int
        Image size in pixels (size x size)
    pixel_scale : float
        Pixel scale in arcseconds
    """
    
    # Create coordinate grids
    extent = size * pixel_scale / 2
    y, x = np.mgrid[-extent:extent:complex(0, size), 
                    -extent:extent:complex(0, size)]
    
    # Create a Gaussian source
    gaussian = Gaussian2D(
        amplitude=1.0, 
        x_mean=0.0, 
        y_mean=0.0, 
        x_stddev=0.3, 
        y_stddev=0.3
    )
    
    data = gaussian(x, y)
    
    # Create FITS file
    hdu = fits.PrimaryHDU(data)
    hdu.header['CDELT1'] = pixel_scale
    hdu.header['CDELT2'] = pixel_scale
    hdu.header['CRPIX1'] = size / 2
    hdu.header['CRPIX2'] = size / 2
    hdu.header['CRVAL1'] = 0.0
    hdu.header['CRVAL2'] = 0.0
    hdu.header['CTYPE1'] = 'RA---TAN'
    hdu.header['CTYPE2'] = 'DEC--TAN'
    hdu.header['CUNIT1'] = 'deg'
    hdu.header['CUNIT2'] = 'deg'
    hdu.header['BUNIT'] = 'Jy/beam'
    
    hdu.writeto(filename, overwrite=True)
    print(f"Created synthetic source image: {filename}")
    print(f"Image size: {size}x{size} pixels")
    print(f"Pixel scale: {pixel_scale} arcsec/pixel")
    print(f"Field of view: {extent*2:.2f} x {extent*2:.2f} arcsec")

if __name__ == '__main__':
    # Check if source image already exists
    if os.path.exists('source_image.fits'):
        response = input("source_image.fits already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            exit()
    
    create_synthetic_source()
