import numpy as np
import pandas as pd

import RAiSEHD4 as RAiSE

from scipy import ndimage
from scipy.ndimage import map_coordinates, binary_dilation
from scipy.optimize import minimize, least_squares
from astropy.cosmology import FlatLambdaCDM
from astropy import constants as const
from astropy import units as u
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans,convolve_fft
import scipy as sp
import numbers
from scipy.optimize import minimize
from scipy.optimize import newton
from scipy.special import log_ndtr
import matplotlib.pyplot as plt


def skewed_ellipse_fit(ring_coords, image_bounds):
    """
    Finds the best fit x,y co-ordinates of the skewed ellipse to given ring of pixels

    ----------
    Parameters
    x - list of x co-ordinates of the ring
    y - list of y co-ordinates of the ring
    image_bounds - list of bounds of the image. Of the form [[x_max, x_min],[y_min,y_max]]
    
    ---------
    Returns
    [x_skew, y_skew]
    x_skew - list of x co-ordinates of the skewed ellipse
    y_skew - list of y co-ordinates of the skewed ellipse
    results - list of tuples that fully describe the skewed ellipse. Six parameters, in order are;
        x0 is the x co-ordinate of the centre of the ellipse
        y0 is the y co-ordinate of the centre of the ellipse
        sigx is the standard deviation of the ellipse in the x plane
        sigy is the standard deviation of the ellipse in the y plane
        alpha is the measure of skewness
        theta is the angle that the ellipse is rotated from the y axis

        Each tuple is also with their uncertainty dparameter; results should be of the form;
        [[x0, unc_x0], [y0, dy0], [sigx, dsigx], [sigy, dsigy], [alpha, dalpha], [theta, dtheta]]        
    """

    #define the radius to be 1/sqrt(2)
    x, y = ring_coords[0], ring_coords[1] #observed/given x,y co-ordinates which are going to be fitted
    
    #test to make sure the data is fine to run on
    if len(x) == 0 or len(x) != len(y):
        return [np.zeros(10), np.zeros(10)], np.zeros(7)    

    #define the lower and upper bounds and the initial value for the fit
    x_low, x_upp, x_init = np.quantile(x, 0.2) -  1e-9, np.quantile(x, 0.8) +  1e-9, np.quantile(x, 0.5)

    #y values from quantiles
    y_low, y_upp, y_init = np.quantile(y, 0.2) -  1e-9, np.quantile(y, 0.8) +  1e-9, np.quantile(y, 0.5)
    sigx_low, sigx_upp, sigx_init = 1e-6, x_upp - x_low, max((x_upp - x_low)/2,1e-5)
    sigy_low, sigy_upp, sigy_init = 1e-6, y_upp - y_low, max((y_upp - y_low)/2,1e-5)
    alpha_low, alpha_upp , alpha_init =  0,5,0.1
    theta_low, theta_upp, theta_init = -0.1,  0.1, 0.05
    R_low, R_upp, R_init = 0, 10, 1/np.sqrt(2)

    xa, ya = _weights_by_repeat(x + max(abs(x)), y)
    xa = xa - max(abs(x))
    
    try:
        f_min = least_squares(skew_ellipse_minimisation,[x_init,y_init,sigx_init,sigy_init,alpha_init, theta_init, R_init],\
           bounds = ([x_low,y_low,sigx_low,sigy_low,alpha_low, theta_low, R_low],[x_upp,y_upp,sigx_upp,sigy_upp,alpha_upp, theta_upp, R_upp]),args = (xa,ya))
    except ValueError:
        return [np.zeros(10), np.zeros(10)], np.zeros(10)      

    #retrieve the results
    x0 = f_min.x[0]
    y0 = f_min.x[1]
    sigx = f_min.x[2]
    sigy = f_min.x[3]
    alpha = f_min.x[4]
    theta = f_min.x[5]
    R = f_min.x[6]
    #get the corresponding x-y co-ordinates of the ellipse
    x_skew,y_skew = _skew_ellipse_plot(x,x0,y0,sigx,sigy,alpha,theta,R)


    results = [x0,y0,sigx,sigy,alpha,theta,R]
    return [x_skew, y_skew], results


def skew_ellipse_minimisation(parm, x,y):
        
    #equation to be minimised
    #give parm, a list of parameters that described an ellipse
    #x, y the co-ordinates of the ellipse for which you want the best fit ellipse
    x0 = parm[0] #x co-ordinate for centre of ellipse
    y0 = parm[1] # y co-ordinate for centre of ellipse
    sigx = parm[2] #sigma for x-direction Gaussian
    sigy = parm[3] #sigma for y-direction Gaussian
    alpha = parm[4] #skew term
    theta = parm[5] #rotation term
    R = parm[6] #radius
    #apply rotation
    xn = (x-x0)*np.cos(-theta) - (y-y0)*np.sin(-theta) + x0
    yn = (x-x0)*np.sin(-theta) + (y-y0)*np.cos(-theta) + y0
    

    xcomp = -(xn - x0)**2/(2*sigx**2) #x component of the ellipse
    ecomp = sp.special.log_ndtr(alpha * (xn - x0) / sigx) + np.log(2) #skewed component
    with np.errstate(divide='ignore', invalid='ignore'):
        ycomp = -(yn - y0)**2/(2*sigy**2) #y component
    result = (ycomp + xcomp + ecomp + R**2) #equation that will be minimised

    
    return result

def _weights_by_repeat(x, y):
    #takes co-ordinates x, y and repeats the left-most points to add extra weight
    #this encourages the fit to go through the front of the jet
    # Quantile setup
    quantile_lst = np.linspace(0, 1, 100)
    ax = np.abs(x)
    
    # Quantile bin edges
    edges = np.quantile(ax, quantile_lst)
    
    # Assign each x to a quantile bin
    bins = np.digitize(ax, edges) - 1
    bins = np.clip(bins, 0, len(quantile_lst) - 2)
    
    # Sigmoid weights per bin
    #9, 35, -5
    sigmoid = (9 / (1 + np.exp(35*quantile_lst[:-1] - 5)) + 1).astype(int)
    
    # Replication count per point
    rep = sigmoid[bins]
    
    # Repeat once
    xa = np.repeat(x, rep)
    ya = np.repeat(y, rep)
    return xa, ya

def _skew_ellipse_points(x,x0,y0,sigx,sigy,alpha,R):
    """
    Calculates the y components of a skewed ellipse from given parameters (x0, y0, sigx, sigy and alpha) over given x values and their uncertainty values

    -----
    Parameters
    x - xmax or the average of the Gaussian in the x plane
    x0 - the centre of the ellipse in the x direction
    y0 - the centre of the ellipse in the y direction
    sigx - the standard deviation of the Gaussian in the x-direction
    sigy - the standard deviation of the Gaussian in the y-direction
    alpha - measure of the skewness in the ellipse

    -----
    Returns
    y1 - y co-ordinates of the top of the ellipse
    y2 - y co-ordinates of the bottom of the ellipse
    
    """
    #calculates the y components of a skewed ellipse from given parameters (x0, y0, sigx, sigy and alpha) over given x values
    try:
        xcomp = -(x - x0)**2/(2*sigx**2)
    except TypeError:
        print("type error")
        print(x, x0, sigx, sigy, alpha)
    ecomp = sp.special.log_ndtr(alpha * (x - x0) / sigx) + np.log(2)
    
    q = -R**2 - xcomp - ecomp
    with np.errstate(divide='ignore', invalid='ignore'):
        y1 = y0 + np.sqrt(-2*sigy**2*q)
        y2 = y0 - np.sqrt(-2*sigy**2*q)
    
    return y1,y2

def _skew_ellipse_plot(x,x0,y0,sigx,sigy,alpha,theta,R):

    """
    Gives the x, y co-ordinates of the skewed ellipse for plotting based on parameters of a skewed ellipse

    -----
    Parameters
    x - list of x co-ordinates of the ellipse
    x0 - the centre of the ellipse in the x direction
    y0 - the centre of the ellipse in the y direction
    sigx - the standard deviation of the Gaussian in the x-direction
    sigy - the standard deviation of the Gaussian in the y-direction
    alpha - measure of the skewness in the ellipse

    -----
    Returns
    X - list of co-ordinates of the ellipse in the x-direction
    Y - list of co-ordinates of the ellipse in the y-direction
    """
    #the best piece of code in the whole damn package
    #takes the parameters that define a skewed ellipse and returns the X, Y co-ordinates for that ellipse

    x = np.linspace(x0 - 10*sigx, x0 +10*sigx, 1000)
    
    #R = 1/np.sqrt(2)
    xcomp = -(x - x0)**2/(2*sigx**2) #x-component
    ecomp = sp.special.log_ndtr(alpha * (x - x0) / sigx) + np.log(2) 
    
    q = -R**2 - xcomp - ecomp
    #np.seterr(all=None, divide=None, over=None, under=None, invalid='ignore')
    with np.errstate(invalid='ignore'):
        y = y0 + np.sqrt(-2*sigy**2*q) #y-component

    X = np.hstack((x, np.flipud(x)))
    with np.errstate(invalid='ignore'):
        Y = np.hstack((y, y0 - np.sqrt(-2*sigy**2*np.flipud(q))))

    x_rot = (X-x0)*np.cos(theta) - (Y-y0)*np.sin(theta) + x0
    y_rot = (X-x0)*np.sin(theta) + (Y-y0)*np.cos(theta) + y0

    return x_rot,y_rot

def _geometric_parameters(results,image_bounds, BH_x = 0):
    """
    Calculates the parameters that describe the parameters of the skewed ellipse and therefore the geometric parameters of the lobe.

    -----
    Parameters
    results - measurements that fully describe the skewed ellipse and their uncertainty. [[x0, unc_x0], [y0, dy0], [sigx, dsigx], [sigy, dsigy], [alpha, dalpha], [theta, dtheta]]        
    image_bounds - minimums and maximums in the x and y direction of the image
    
    BH_x - x co-ordinate location of the Black Hole (Default 0)

    Returns
    results - list of tuples that contain the geometric parameter and their uncertainty
    [[width, u_width],[length, u_length],[extent, u_extent]]

    """
    #calculates the measurements of the skewed ellipse, based on the parameters of the ellipse. This is done analytically if possible.
    #Returns the maximum and minimum y co-ordinates,  maximum x co-oordinates, width of the ellipse in the y and x direction and estimation of the x width.
    x0 = results[0]
    y0 = results[1]
    sigx = results[2]
    sigy = results[3]
    alpha = results[4]
    R = results[6]
    #approximates the location of the mode of the skewed ellipse
    xmax = x0
    #the mode of the non-skewed ellipse
    ymax = y0
    #calculates the location of the maximum y-values which occur at xmax
    
    ymax_1,ymax_2 = _skew_ellipse_points(xmax,x0,y0,sigx,sigy,alpha,R)
    ymax_l = min([ymax_1, ymax_2])
    ymax_u = max([ymax_1, ymax_2])
    
    width = abs(ymax_u - ymax_l)
    #if abs(alpha) < 0.5: ### if -0.5 < alpha < 0.5, calculate using quadratic formula
        # calculates the location of the x maxs and x min using the quadratic formula
        # quadratic formula arises from the first 3 terms of the Taylor series approximation of the ln(1/2 + erf()) function
    x_1, x_2 = _approx_roots(results)
        # x_1 = (alpha*np.sqrt(2*np.pi)*sigx - np.sqrt(np.pi)*np.sqrt((4*alpha**2+np.pi)/sigx**2)*sigx**2 + 2*alpha**2*x0 + np.pi * x0)/(2*alpha**2+np.pi)
        # x_2 = (alpha*np.sqrt(2*np.pi)*sigx + np.sqrt(np.pi)*np.sqrt((4*alpha**2+np.pi)/sigx**2)*sigx**2 + 2*alpha**2*x0 + np.pi * x0)/(2*alpha**2+np.pi)
        
    # else:
    #     # if alpha is too big, the difference between the actual skewed ellipse and the Taylor Series approximiation can be large. Instead the x width is calculated using a (computationally slow) maximum method. This function creates a new skewed ellipse with high resolution and finds the maximum x point in the created ellipse.
    #     try:
            
    #         xs_min = image_bounds[0][0]
    #         xs_max = image_bounds[0][1]
    #         x_temp = np.linspace(xs_min-abs(xs_min)*0.1,xs_max+abs(xs_max)*0.1,1000)
    #         y1,y2 = _skew_ellipse_points(x_temp,x0,y0,sigx,sigy,alpha,R) #creates the ellipse points
    #         ind = np.isnan(y1)
    #         y1 = y1[~ind]
    #         y2 = y2[~ind]
    #         x_temp = x_temp[~ind]
    #         x_w = abs(max(x_temp) - min(x_temp)) #x_width
    #         x_1 = max(x_temp)
    #         x_2 = min(x_temp)
    #     except ValueError:
    #         print("results")
    #         print(results)
    
    extent = abs(x_1 - x_2)
    length = abs(x_1 - BH_x)
    u_width, u_length, u_extent = np.sqrt(1**2 + 1**2), np.sqrt(1**2 + 1**2), np.sqrt(1**2 + 1**2)

    results = [[width, u_width], [length, u_length], [extent, u_extent]]
    return results


def q(x, x0, sigx, alpha, R):
    xcomp = -(x - x0)**2/(2*sigx**2)
    ecomp = log_ndtr(alpha*(x - x0)/sigx) + np.log(2)
    return -R**2 - xcomp - ecomp

def _approx_roots(results):
    x0 = results[0]
    y0 = results[1]
    sigx = results[2]
    sigy = results[3]
    alpha = results[4]
    R = results[6]
    
    guess_left  = x0 - np.sqrt(2)*sigx*R
    guess_right = x0 + np.sqrt(2)*sigx*R
    
    x_left  = newton(lambda x: q(x,x0,sigx,alpha,R), guess_left)
    x_right = newton(lambda x: q(x,x0,sigx,alpha,R), guess_right)

    return x_left, x_right