import matplotlib.pyplot as plt
from astropy.io import fits

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact


# clickable_array.py
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

# interactive_click_plot.py
import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from radio_beam import Beam # pip install radio-beam

import numpy as np
#import pandas as pd

import ellipse_fitting_LATEST as ef

from scipy import ndimage
from scipy.optimize import minimize, least_squares
import scipy as sp
import RAiSE_attributes_latest as ra
import traceback

def interactive_jet(image_file, maxes_only = True, cont = 5):
    """
    docstirngs
    """

    # add comment
    image_data, wcs, cdelts, beam_params, beam_rms = get_fits_parameters(image_file)
    
    # add comment
    x_values = np.arange(0, image_data.shape[0])
    y_values = np.arange(0, image_data.shape[1])
    X, Y = np.meshgrid(x_values, y_values, indexing = 'ij')

    # add comment
    fig = plt.figure(1)
    ax = plt.subplot(projection=wcs, slices=('x', 'y'))
    ax.coords[0].set_ticklabel(exclude_overlapping=True)
    
    # add comment
    ax.pcolor(X, Y, np.log10(np.maximum(image_data, beam_rms)))
    #ax.pcolor(X, Y, image_data)
    ax.grid(color='white', ls='solid', lw=0)        # lw=0, does the color matter?

    ax.coords[0].set_axislabel('Right Ascension', fontsize=13)
    ax.coords[1].set_axislabel('Declination', fontsize=13)
    plt.tight_layout()
    # add comment
    text = ax.text(ax.get_xlim()[0],ax.get_ylim()[0], r'Select AGN Core', va="bottom", ha="left", color='white')
    
    # add comment
    count = 0
    lobe_coords, core_coords = [], []
    #n_lst, s_lst = 0, 0
    results_lst = []
    s_lst, n_lst = [], []
    beam_lst = [cdelts[0], cdelts[1], beam_params, beam_rms]
        
# #    ring_coords, rotated_data = ra.RAiSE_attributes(image_data, cdelts, beam_params, beam_rms, [260, 253], [276, 338], cont_level=5)
# #    ring_coords, rotated_data = ra.RAiSE_attributes(image_data, cdelts, beam_params, beam_rms, [260, 253], [240, 177], cont_level=5)
# , ring_coords_p    #x_values = np.arange(0, len(rotated_data[:,0]), 1)
#     #y_values = np.arange(0, len(rotated_data[0,:]), 1)
#     #X, Y = np.meshgrid(x_values, y_values, indexing = 'ij')
#     #ax.pcolor(X, Y, np.log10(np.maximum(rotated_data, beam_rms)))

#     #ax.scatter(ring_coords[0], ring_coords[1], s=1, c = 'orangered', zorder=10000)
            
    #%matplotlib ipympl
    # enable interactivity if running in a normal Python shell
    plt.ion()
    
    # add comment           # not that fond of a function definition in a function, but likely will not work otherwise
    def onclick(event):
        nonlocal count # use count variable outside function

        # add comment
        ix, iy = event.xdata, event.ydata
        
        # select AGN core
        if count == 0:
            # fit beam to flux near point
            ax.plot(ix, iy, 'x', c='cyan')
            core_coords.append([ix, iy])
            # next text
            text.set_text('Select North Lobe')
            text.set_text((ix, iy))
        # select north lobe
        elif count == 1:
            # fit ellipse to flux near point
            ax.plot(ix, iy, 'x', c='black')
            lobe_coords.append([ix, iy])

            try:
                ring_coords,fit_coords, rotated_data, results_n,  centre_points, ring_coords_t, core_coords_t = ra.RAiSE_attributes(image_data, cdelts, beam_params, beam_rms, core_coords[-1], lobe_coords[-1], cont_level=5)
                n_lst.append(results_n)
                results_lst.append([image_data, cdelts, beam_params, beam_rms])
                #x_values = np.arange(0, len(rotated_data[:,0]), 1)
                #y_values = np.arange(0, len(rotated_data[0,:]), 1)
                #X, Y = np.meshgrid(x_values, y_values, indexing = 'ij')
                #ax.pcolor(X, Y, np.log10(np.maximum(rotated_data, beam_rms)))
    
                ax.scatter(ring_coords[0], ring_coords[1], s=1, c = 'orangered', zorder=10000)
                ax.scatter(fit_coords[0], fit_coords[1], s=1, c = 'yellow', zorder=10000)
                # for points in centre_points:
                #     ax.scatter(points[0], points[1], c = 'green')
                    

            except Exception as e:
               print("womp womp")
               text.set_text(str(e))
    
            # next text
            #text.set_text('Select South Lobe')
            #text.set_text((ix, iy))

        # select south lobe
        elif count == 2:
            # fit ellipse to flux near point
            ax.plot(ix, iy, 'x', c='black')

            #ax.plot(ix, iy, 'x', c='orange')
            lobe_coords.append([ix,iy])   

            #try:
            ring_coords,fit_coords, rotated_data, results_s,  centre_points, ring_coords_t, core_coords_t = ra.RAiSE_attributes(image_data, cdelts, beam_params, beam_rms, core_coords[-1], lobe_coords[-1], cont_level=5)
            s_lst.append(results_s)
             
            ax.scatter(ring_coords[0], ring_coords[1], s=1, c = 'orangered', zorder=10000)
            ax.scatter(fit_coords[0], fit_coords[1], s=1, c = 'yellow', zorder=10000)
            # for points in centre_points:
            #     ax.scatter(points[0], points[1], c = 'green')
            #ax.fill(ring_coords[0], ring_coords[1], c = 'blue', alpha = 0.5)
            #except Exception as e:
            #    print("womp womp")
            #    text.set_text(str(e))
      
            # next text
            text.set_text(r' Add to Catalogue')
        else:
            # confirm text        
            text.set_text(r'Done ...')
    
            fig.canvas.mpl_disconnect(cid)
        
        count = count + 1



    
    coords = [core_coords, lobe_coords]
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    return n_lst, s_lst, results_lst, beam_lst, coords

def get_fits_parameters(fits_image):

    # open fits image
    hdu = fits.open(fits_image)[0]
    
    # ensure FITS image data is in correct format
    naxis = len(np.shape(hdu.data))
    if naxis == 2:
        image_data = hdu.data.T # output as ra/dec order
    elif naxis == 3:
        image_data = hdu.data[0, :, :].T
    elif naxis == 4:
        image_data = hdu.data[0, 0, :, :].T
    elif naxis == 5:
        image_data = hdu.data[0, 0, 0, :, :].T
    else:
        raise Exception('Invalid number of axes in FITS image.')
        
    # find beam parameters and image projection
    wcs = WCS(hdu.header, naxis=2)
    cdelts = [np.abs(hdu.header['CDELT1'])*3600, hdu.header['CDELT2']*3600]
    beam_params = __get_beam_parameters(hdu.header)
    # find standard deviation to generate noise floor in images
    beam_rms = np.std(np.append(image_data[image_data < 0], -image_data[image_data < 0]))
    return image_data, wcs, cdelts, beam_params, beam_rms
    
def __get_beam_parameters(header, bmin = None, bmaj = None, pa = None):
    # find beam size in arcseconds using package (to avoid issues with different formats)
    try:
        image_beam = Beam.from_fits_header(header) # gives in degrees by default
        bmaj = image_beam.major.value
        bmin = image_beam.minor.value
        bpa = image_beam.pa.value        
    except:
        #try:
        bmaj, bmin = _get_beam_hist(header)
        bpa = 0
        #except:
            #print("No BMIN, BMAJ or PA found in header. Using given values")
    bmaj = bmaj * 3600
    bmin = bmin * 3600
    bpa = bpa * 3600
    
    return [bmaj, bmin, bpa] # convert from FWHM to sigma


# HISTORY cards may be returned as a list-like object
def _get_beam_hist(hdu):
    import re

    history = hdu.get('HISTORY', [])
    if isinstance(history, str):
        history = [history]
    
    for line in history:
        if "Resolution in RA, Dec" in line:
            match = re.search(
                r"Resolution in RA,\s*Dec\s+([0-9.]+)\s+([0-9.]+)\s+arcsec",
                line
            )
            if match:
                ra_res = float(match.group(1))
                dec_res = float(match.group(2))
    
                bmaj = max(ra_res, dec_res)
                bmin = min(ra_res, dec_res)
    return bmaj, bmin
plt.close()