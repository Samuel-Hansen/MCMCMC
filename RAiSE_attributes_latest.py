# Ross version
# add comments here

import numpy as np
#import pandas as pd

import RAiSEHD4 as RAiSE
import ellipse_fitting_LATEST as ef
import matplotlib.pyplot as plt
import pandas as pd 

from scipy import ndimage
#from scipy.ndimage import map_coordinates, binary_dilation
#from scipy.optimize import minimize, least_squares
#from astropy.cosmology import FlatLambdaCDM
#from astropy import constants as const
#from astropy import units as u
#from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans,convolve_fft
#import scipy as sp
#import numbers

#import jax.numpy as jnp
#from jax import jit

# Define function for finding the attributes of AGN lobe: wrapper for observed sources
max_radius = 1.5

def RAiSE_attributes(image_data, cdelts, beam_params, beam_rms, core_coords, lobe_coords, cont_level=5, flux_error=0.02, using_RAiSE = False, suffix = None):
    """
    docstrings
    returns the attributes (luminosity, length, width, extent) for a single frequency observation
    """
    
    # image preparation: translate and rotate image to centre core and align jet along negative x-axis
    rotated_data, cdelts_r, rotation, core_coords_r, lobe_coords_r = __rotate_image(image_data, cdelts, core_coords, lobe_coords)

    # plt.close()
    # plt.pcolor(rotated_data.T)
    # plt.savefig('rot.pdf')    
    beam_params = beam_params.copy()
    beam_params[2] = beam_params[2] + rotation*180./np.pi
      
    # luminosity attributes: flux density
    flux_density = get_brightness_attributes(rotated_data, cdelts_r, beam_params, beam_rms, cont_level=cont_level, flux_error=flux_error)

    # spatial attributes: size, width, extent
    ring_coords, fit_coords, results, size_data, centre_points, ring_coords_t, core_coords_t = get_spatial_attributes(rotated_data, cdelts_r, beam_params, beam_rms, core_coords_r, lobe_coords_r, cont_level=5, using_RAiSE = using_RAiSE, suffix = suffix)

    ring_coords_p = ring_coords
    # reverse operations: back rotate points and remove translation
    ring_coords = __back_rotate_points(ring_coords, rotation, core_coords, core_coords_r)
    fit_coords = __back_rotate_points(fit_coords, rotation, core_coords, core_coords_r)

    centre_points_r = []
    for points in centre_points:
        points_r = __back_rotate_points(points, rotation, core_coords, core_coords_r)
        centre_points_r.append(points_r)
    results = np.vstack([results, flux_density])
    return ring_coords, fit_coords, size_data, results, [core_coords_r, lobe_coords_r], ring_coords_t, core_coords_t


def __rotate_image(image_data, cdelts, core_coords, lobe_coords, axis_ratio=2):
    
    # calculate angle of rotation required to align jet axis with negative x-axis
    jet_x, jet_y = lobe_coords[0] - core_coords[0], lobe_coords[1] - core_coords[1]
    rotation = np.pi - np.arctan2(jet_y, jet_x)
    
    # image preparation: padding and translation to ensure rotated jet fits
    pad_size = max_radius*np.sqrt((lobe_coords[0] - core_coords[0])**2 + (lobe_coords[1] - core_coords[1])**2)
    pad_x1 = max(int(pad_size - image_data.shape[0]/2), 0)
    pad_y1 = max(int(pad_size - image_data.shape[1]/2), 0)
    pivot_coords = [int(core_coords[0] + 0.5), int(core_coords[1] + 0.5)]

    # image preparation: padding and translation to centre the core at the pivot point of the image
    pad_x = __pivot_padding(image_data, pivot_coords, pad_x1, axis=0)
    pad_y = __pivot_padding(image_data, pivot_coords, pad_y1, axis=1)

    # image preparation: pad image
    padded_data = np.pad(image_data, (pad_x, pad_y), 'constant', constant_values=0)
    # plt.close()
    # plt.pcolor(padded_data.T)
    # plt.scatter(core_coords[0], core_coords[1], c = 'red', label = str(core_coords[0]) + ',' + str(core_coords[1]))
    # plt.scatter(pivot_coords[0], pivot_coords[1], c = 'green', label = str(pivot_coords[0]) +',' + str(pivot_coords[1]))
    # plt.scatter(lobe_coords[0], lobe_coords[1], c = 'blue', label = str(lobe_coords[0]) + ',' +str(lobe_coords[1]))   
    # plt.text(20, 20, str(image_data.shape), c = 'red')
    # plt.text(50, 50, str(padded_data.shape), c = 'red')
    # plt.text(75, 75, str(pad_x) + "_" + str(pad_y), c = 'red')
    # plt.text(100, 100, str(pad_x1) + "_" + str(pad_y1), c = 'red')

    # plt.legend()
    # plt.savefig('pad1.pdf') 
    # rotate image: rotate image to align jet axis with negative x-axis
    rotated_data = ndimage.rotate(padded_data, (rotation)*180/np.pi, reshape=False, cval=0, mode='constant')
    # rotate image: update lobe and core coordinates
    core_coords_r = [rotated_data.shape[0]//2, rotated_data.shape[1]//2]
    lobe_coords_r = [jet_x*np.cos(rotation) - jet_y*np.sin(rotation) + core_coords_r[0], jet_x*np.sin(rotation) + jet_y*np.cos(rotation) + core_coords_r[1]]
    # plt.close()
    # plt.pcolor(rotated_data.T)
    # plt.scatter(core_coords_r[0], core_coords_r[1])
    # plt.scatter(lobe_coords_r[0], lobe_coords_r[1])   
    # plt.scatter(lobe_coords_r[0], lobe_coords_r[1])    
    # plt.savefig('rot1.pdf') 
    # image finalisation: cropping and translation to remove excess data
    crop_size = max_radius*np.sqrt((lobe_coords_r[0] - core_coords_r[0])**2 + (lobe_coords_r[1] - core_coords_r[1])**2)
    crop_x = max(int(image_data.shape[0]/2 - crop_size), 0)
    crop_y = max(int(image_data.shape[1]/2 - crop_size/axis_ratio), 0) # typical axis ratio of 2

    # image finalisation: crop image
    final_data = rotated_data[crop_x:rotated_data.shape[0]-crop_x, crop_y:rotated_data.shape[1]-crop_y]
    # image finalisation: update lobe and core coordinates
    core_coords_r[0], core_coords_r[1] = core_coords_r[0] - crop_x, core_coords_r[1] - crop_y
    lobe_coords_r[0], lobe_coords_r[1] = lobe_coords_r[0] - crop_x, lobe_coords_r[1] - crop_y
    
    # image finalisation: update cdelts if applicable
    cdelts_r = cdelts

    return final_data, cdelts_r, rotation, core_coords_r, lobe_coords_r #, [pad_x[0], pad_y[0]], [crop_x, crop_y]

    

def __pivot_padding(image_data, pivot_coords, pad, axis=0):

    M = image_data.shape[axis]
    N = pivot_coords[axis]

    total_size = max(M + 2*pad, 2 * max(N, M - N - 1) + 2*pad + 1)

    centre = (total_size - 1) // 2

    pad_before = centre - N
    pad_after  = total_size - M - pad_before

    return [pad_before, pad_after]

# def __pivot_padding(image_data, pivot_coords, pad, axis=0):

#     # add padding to centre core at pivot point in image with odd number of pixels
#     if image_data.shape[axis] % 2 == 1:
#         # length is odd
#         if pivot_coords[axis] > (image_data.shape[axis] + 2*pad)//2:
#             pad = [pad, 2*abs((image_data.shape[axis] + 2*pad)//2 - pivot_coords[axis])]
#         elif pivot_coords[axis] < image_data.shape[axis]//2:
#             pad = [2*abs((image_data.shape[axis] + 2*pad)//2 - pivot_coords[axis]), pad]
#         else:
#             pad = [pad, pad]
#     else:
#         # length is even
#         if pivot_coords[axis] > image_data.shape[axis]//2:
#             pad = [pad, 2*abs((image_data.shape[axis] + 2*pad)//2 - pivot_coords[axis]) + 1]
#         elif pivot_coords[axis] < image_data.shape[axis]//2:
#             pad = [2*abs((image_data.shape[axis] + 2*pad)//2 - pivot_coords[axis]) - 1, pad]
#         else:
#             pad = [pad, pad + 1]
    
#     return pad
    

def __back_rotate_points(ring_coords, rotation, core_coords, core_coords_r):

    # reverse operations: remove cropping
    _xs = ring_coords[0] - core_coords_r[0]
    _ys = ring_coords[1] - core_coords_r[1]
    
    # reverse operations: back rotate points and remove translation
    xs = _xs*np.cos(-rotation) - _ys*np.sin(-rotation) + core_coords[0]
    ys = _xs*np.sin(-rotation) + _ys*np.cos(-rotation) + core_coords[1]

    return [xs, ys]
                

def get_brightness_attributes(image_data, cdelts, beam_params, beam_rms, cont_level=5, flux_error=0.05):
    """
    docstrings
    """

    # image preparation: cut image in half along the x-axis
    image_data = np.array_split(image_data, 2, axis=0)[0]
    # image preparation: apply 'electron density' function
    luminosity_data = image_data * __electron_density_cdf(image_data / (beam_rms * cont_level))
    luminosity_data[image_data > beam_rms * cont_level] = image_data[image_data > beam_rms * cont_level]

    # luminosity calculation: conversion from 'per beam' to 'per pixel'
    beam_pixels = __beam_to_pixels(cdelts, beam_params)
    pixels_per_beam = np.pi*beam_pixels[0]*beam_pixels[1] / (4 * np.log(2))
    luminosity_data = luminosity_data / pixels_per_beam

    #np.savetxt("luminosity_data.csv", luminosity_data, delimiter=",")

    # luminosity calculation: calculate total flux density in image and uncertainity
    flux_density = max(np.nansum(luminosity_data), 2 * cont_level * beam_rms)
    #flux_density_err = np.sqrt((max(1, np.sqrt(np.nansum(image_data > beam_rms * cont_level) / pixels_per_beam)) * beam_rms)**2 + (flux_density * flux_error)**2)
    flux_density_err = max(np.sqrt((max(1, np.sqrt(np.nansum(image_data > beam_rms * cont_level) / pixels_per_beam)) * beam_rms)**2 + (flux_density * flux_error)**2), 1e-10 * flux_density)

    if flux_density_err > flux_density:
        for i in range(5):
            print("flux err")
        print(flux_density_err)
        print(flux_density)
        print(np.nansum(luminosity_data))
        print(np.sqrt((max(1, np.sqrt(np.nansum(image_data > beam_rms * cont_level) / pixels_per_beam)) * beam_rms)**2))
        print((flux_density * flux_error)**2)
        print(1e-10 * flux_density)
            
    return [flux_density, flux_density_err]


def __beam_to_pixels(cdelts, beam_params):
    
    # convert beam (in arcseconds) to number of pixels (cdelts in arcseconds)
    bpa = np.deg2rad(90.0 - beam_params[2])
    bmajor = np.sqrt((beam_params[0]*np.cos(bpa)/(cdelts[0]))**2 + (beam_params[0]*np.sin(bpa)/(cdelts[1]))**2)
    bminor = np.sqrt((beam_params[1]*np.cos(bpa + np.pi/2)/(cdelts[0]))**2 + (beam_params[1]*np.sin(bpa + np.pi/2)/(cdelts[1]))**2)

    return [bmajor, bminor, bpa]
    
    
def get_spatial_attributes(image_data, cdelts, beam_params, beam_rms, core_coords, lobe_coords, cont_level=5, using_RAiSE = False, suffix = None):
    """
    docstrings
    """
    
    # flux ring finding: extract the (x,y) coordinates of the flux ring
    ring_coords, image_bounds, size_data, centre_points, core_coords, ring_coords_t, core_coords_t = __get_flux_ring(image_data, beam_rms, core_coords, lobe_coords,cdelts,  cont_level=cont_level, using_RAiSE = using_RAiSE, suffix = suffix)

    #ring_coords = _remove_outliers(ring_coords[0], ring_coords[1])
    # flux ring fitting: fit an ellipse (or similar) to the flux ring
    results = np.array([(0, 0), \
                       (0,0),\
                       (0,0)])
    fit_coords = [[np.nan] * 10, [np.nan] * 10]
    if len(ring_coords) > 0:
        fit_coords, results_fit = ef.skewed_ellipse_fit(ring_coords_t, image_bounds)
        
        if not np.all(np.array(results_fit) == 0):
            geom_parms = ef._geometric_parameters(results_fit,image_bounds, BH_x = core_coords_t[0])
        
            results = np.array([(np.array(geom_parms[0]) * cdelts[1]), \
                               (np.array(geom_parms[1]) * cdelts[0]),\
                               (np.array(geom_parms[2]) * cdelts[0])])
    if not using_RAiSE:
        fit_coords, results_fit = ef.skewed_ellipse_fit(ring_coords, image_bounds)

    if suffix == 0:
        savename = "fit_coords.csv"

        np.savetxt(savename, fit_coords, delimiter=",")

    
    # plt.scatter(ring_coords[0], ring_coords[1])
    # plt.scatter(ring_coords_t[0], ring_coords_t[1])
    # plt.scatter(fit_coords[0], fit_coords[1])
    # plt.show()
 
    results = np.maximum(results, np.zeros((3,2)) + np.array([0.1,0.01]) * np.array(cdelts))
    return ring_coords,fit_coords, results, size_data, centre_points, ring_coords_t, core_coords_t






def __get_flux_ring(image_data, beam_rms, core_coords, lobe_coords, cdelts, cont_level=5, using_RAiSE = False, suffix = None):
    """
    The number of angles should be based on the radius! There is no point have 1000 angles if the radius is 2 pixels
    """
    # image preparation: cut image in half along the x-axis, and repeat final column to ensure flux ring fits to correct shape
    image_data = np.array_split(image_data, 2, axis=0)[0]

    if using_RAiSE:
        extended_data = np.vstack((image_data, np.repeat(image_data[-2, :].reshape(-1, 1), image_data.shape[0]//2, axis=1).T))
    else:
        extended_data = np.vstack((image_data, np.repeat(image_data[-1, :].reshape(-1, 1), image_data.shape[0]//2, axis=1).T))
    # image preparation: apply 'electron density' function
    size_data = __electron_density_pdf(extended_data / (beam_rms * cont_level))
    if suffix == 0:
        np.savetxt("size_data1.csv", size_data, delimiter=",")

    #size_data = extended_data / (beam_rms * cont_level)  
    ccore_coords = np.array(core_coords)# * cdelts[0]
    clobe_coords = np.array(lobe_coords)# * cdelts[0]
    size_data = __lobe_kernel_cdf(size_data, ccore_coords, clobe_coords)
    if suffix == 0:

        np.savetxt("ccore_coords.csv", ccore_coords, delimiter=",")
        np.savetxt("clobe_coords.csv", clobe_coords, delimiter=",")
        
        np.savetxt("size_data2.csv", size_data, delimiter=",")

    #get ring coords for different centres
    #centre_lst = [[lobe_coords[0], lobe_coords[1]]] +  [[lobe_coords[0] + (core_coords[0] - lobe_coords[0]) //(i+2),lobe_coords[1]]for i in range(5)]
    centre_lst = [[lobe_coords[0] + i*(core_coords[0] - lobe_coords[0]) /(2*(5-1)),lobe_coords[1]]for i in range(5)]
    if suffix == 0:
    
        np.savetxt("centre_lst.csv", centre_lst, delimiter=",")

    x_lst = []
    y_lst = []
    
    for centre_coords in centre_lst:

        ring_coords_n = _peak_points(centre_coords, core_coords, size_data)
        x_ring, y_ring = zip(ring_coords_n)
        x_lst = list(x_lst) + list(x_ring[0])
        y_lst = list(y_lst) + list(y_ring[0])

    ring_coords = [np.array(x_lst), np.array(y_lst)]
    if suffix == 0:

        np.savetxt("ring_coords.csv", ring_coords, delimiter=",")

    ring_coords = _remove_outliers(ring_coords)     
    if suffix == 0:

        np.savetxt("ring_coords_outliers.csv", ring_coords, delimiter=",")   
    if using_RAiSE == True and len(ring_coords) == 2:
    #makes symmetric
        x_lst = np.r_[ring_coords[0], ring_coords[0]]
        y_lst = np.r_[ring_coords[1], 2 * core_coords[1] - ring_coords[1]]
        pairs = set(zip(x_lst, y_lst))
        x_unique, y_unique = map(list, zip(*pairs))
        ring_coords = [x_unique, y_unique]
    image_bounds = [[-(image_data.shape[0] - 1)/2, (image_data.shape[0] - 1)/2], [- (image_data.shape[1] - 1)/2, (image_data.shape[1] - 1)/2]]
    if len(ring_coords) == 2:
        ring_coords_t = np.array([np.array(ring_coords[0]) - image_data.shape[0]/2, np.array(ring_coords[1]) - image_data.shape[1]/2])
        core_coords_t = np.array([np.array(core_coords[0]) - image_data.shape[0]/2, np.array(core_coords[1]) - image_data.shape[1]/2])
        mask = ring_coords_t[0] <= image_data.shape[0] / 2 - 1
        ring_coords_t = ring_coords_t[:, mask]
    else:
        ring_coords_t = ring_coords
        core_coords_t = core_coords
    if suffix == 0:
        np.savetxt("ring_coords_t.csv", [image_data.shape[0]/2, image_data.shape[1]/2], delimiter=",")
    
    return ring_coords, image_bounds, size_data, centre_lst, core_coords, ring_coords_t, core_coords_t
    
def _peak_points(lobe_coords, core_coords, size_data):
    # flux ring finding: define arrays of radii and angles
    a0_radius = np.sqrt((lobe_coords[0] - core_coords[0])**2 + (lobe_coords[1] - core_coords[1])**2)
    radii = np.arange(0, max_radius * a0_radius, 0.5)
    thetas = np.arange(0, 2*np.pi, np.pi/180)

    # flux ring finding: convert to nearest cartesian pixel
    xs = np.round(lobe_coords[0] + radii[:, None] * np.cos(thetas[None, :])).astype(int)
    ys = np.round(lobe_coords[1] + radii[:, None] * np.sin(thetas[None, :])).astype(int)

    # flux ring finding: remove points outside bounds of image
    image_mask = (xs >= 0) & (xs < size_data.shape[0]) & (ys >= 0) & (ys < size_data.shape[1])

    # flux ring finding: find peak of ring along each radial line
    peak_indices = np.zeros_like(thetas).astype(int)
    for i in range(0, len(thetas)):
        # find radius of peak of the ring pdf weighted by the radius pdf
        if np.sum(image_mask[:, i]) > 0:
            peak_index = np.argmax(size_data[xs[:, i][image_mask[:, i]], ys[:, i][image_mask[:, i]]]) #* np.sqrt(__electron_density_pdf(radii[:][image_mask[:, i]] / a0_radius)))

            # add ring location if not in repeated region
            if xs[peak_index, i] < size_data.shape[0]:
                peak_indices[i] = peak_index
            else:
                peak_indices[i] = -1
        else:
            peak_indices[i] = -1
    
    # array finalisation: convert indices into arrays of cartesian coordinates
    ring_coords = np.column_stack((xs[peak_indices[peak_indices >= 0], np.arange(xs.shape[1])[peak_indices >= 0]], ys[peak_indices[peak_indices >= 0], np.arange(ys.shape[1])[peak_indices >= 0]]))
    ring_coords = np.unique(ring_coords, axis=0).T

    return ring_coords
def __electron_density_cdf(x):

    # 1s orbital
    c = 0.25
    cdf = (1 -  np.exp(-2*x/c) * (1 + 2*x/c + 2*(x/c)**2))
    if isinstance(x, (list, np.ndarray)):
        cdf[x > 1] = 1
        cdf[x < 1] = 0
    else:
        if x > 1:
            cdf = 1
        elif x < 0:
            cdf = 0
    return cdf
    
def __electron_density_pdf(x):

    # 1s orbital
    pdf = x**2 * np.exp(2 - 2 * np.abs(x))
    if isinstance(x, (list, np.ndarray)):
        pdf[x < 0] = 0
    else:
        if x < 0:
            pdf = 0
    return pdf
    
def __lobe_kernel_cdf(x, core_coords, lobe_coords, axis_ratio=2):

    # gaussian kernel along identifed lobe
    cdf = x # null value
    sigma = np.abs(lobe_coords[0] - core_coords[0])
    if isinstance(x, (list, np.ndarray)):
        # gentle taper of width axis ratio starting at jet axis
        cdf = cdf * np.exp(- np.minimum(0, np.arange(x.shape[0])[:, None] - lobe_coords[0])**2/(2*sigma**2) - (np.arange(x.shape[1])[None, :] - lobe_coords[1])**2/(2*(sigma/axis_ratio)**2))
        # sharp taper starting at jet axis plus axis ratio
        #cdf = cdf * np.exp(-np.minimum(0, np.arange(x.shape[0])[:, None] - lobe_coords[0] + sigma)**2/(2*(0.1*sigma)**2) - np.minimum(0, -np.abs(np.arange(x.shape[1])[None, :] - lobe_coords[1]) + sigma)**2/(2*(0.1*sigma)**2))
    return cdf
    
def _remove_outliers(points):

    xs = np.asarray(points[0])
    ys = np.asarray(points[1])

    img = np.zeros((xs.max() + 1, ys.max() + 1), dtype=bool)
    img[xs, ys] = True
    #include all pixels that touch, even diagonally
    structure = np.ones((3, 3), dtype=int)
    # Label connected components
    labels, n_components = ndimage.label(img, structure = structure)

    sizes = ndimage.sum(img, labels, range(1, n_components+1))
    #find the largest element
    largest_label = np.argmax(sizes) + 1
    largest_region = np.isin(labels, np.where(sizes > 5)[0] + 1).astype(int)
    
    y_coords, x_coords = np.where(largest_region == 1)
    # Zip them together to get a list of (x, y) tuples
    large_region_coords = np.array(list(zip(y_coords, x_coords))).T
    return large_region_coords

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans,convolve_fft

def _convolve_raise(fixed_image, cdelts, beam_diameter,rms):
    #takes fixed_image, an array and convolves it with a beam with diameter beam_diameter and rms, rms

    ang_pix_width = cdelts[0] #angular pixel width in arc seconds
    ang_pix_height = cdelts[1]#angular pixel height in arc seconds
    
    pixel_per_beam_width = beam_diameter/ang_pix_width #fwhm x
    pixel_per_beam_height = beam_diameter/ang_pix_height #fwhm y
            
    ###creates a beam from values given
    
    beam = Gaussian2DKernel (x_stddev = pixel_per_beam_width/(2 * np.sqrt(2 * np.log(2))),y_stddev = pixel_per_beam_height/(2 * np.sqrt(2 * np.log(2))))
    
    ###convolves image with beam
    fixed_image = convolve_fft(fixed_image, beam, allow_huge=True)  
    return fixed_image        
    
def _prep_raise_arcsec_jy(image_data, beam_rms, redshift, beam_fwhm_arcsec , H0 = 70):
        # takes RAiSE output in W/Hz/Pixel, converts to Jy, convolves with the same beam used in observations, returns raise image in Jy/beam
        
        #from RAiSE in W/Hz/pixel
        
        # Convert W/Hz to Jy/Beam
        cosmo = FlatLambdaCDM(H0=H0 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3) ##cosmology
        dL = cosmo.luminosity_distance(redshift).to(u.m).value 

        dA = dL/(1+redshift)**2 ###angular diameter in m
        raise_pixel_size_kpc = [image_data.columns[1] - image_data.columns[0],
                                image_data.index[1] - image_data.index[0]]

        # Pixel sizes in m to radians
        raise_cdelts = np.array([raise_pixel_size_kpc[0]*3.086e19, raise_pixel_size_kpc[1]*3.086e19]) / dA
        
        # Convert radians to arcsec
        raise_cdelts_arcsec = raise_cdelts * (180/np.pi) * 3600
        
        # Flux density in Jy/pixel
              
        image_data_per_pixel = image_data / (1e-26 * 4*np.pi*dL**2)
        
        # Pixels per beam
        pixels_per_beam = (np.pi/(4*np.log(2))) * (beam_fwhm_arcsec**2 / (raise_cdelts_arcsec[0] * raise_cdelts_arcsec[1]))
        
        # Convolve
        image_data_per_pixel = _convolve_raise(image_data_per_pixel, raise_cdelts_arcsec, beam_fwhm_arcsec, beam_rms)
        # Convert to Jy/beam
        image_data_per_beam = image_data_per_pixel * pixels_per_beam
        image_data_per_beam = np.pad(image_data_per_beam, (np.shape(image_data)[1]//2+1, np.shape(image_data)[1]//2+1), 'constant',constant_values=(0))    
  
        return image_data_per_beam, dA, pixels_per_beam, raise_cdelts_arcsec

