import numpy as np
import timeit
import matplotlib.pyplot as plt
import pandas as pd
import RAiSEHD4 as RAiSE


def get_initial_jets(n_chains, n_temps, bins_arr, hist_arr, bounds, fit_dic, parms_const, freq, beam_params_lst, beam_rms_lst,posterior,\
                    resolution = '__512',cont = 5, redshift = 0.01, geometric_idx = 0, H0 = 70, particle_data_gamma = None, seed = 12345):

    jet_lst = []

    np.random.seed(seed)
    
    while len(jet_lst) < n_chains:
        jets = get_jets(n_temps, bins_arr, hist_arr, bounds, fit_dic, parms_const, freq, beam_params_lst, beam_rms_lst,posterior,\
                    resolution = '__512',cont = 5, redshift = 0.01, geometric_idx = 0, H0 = 70, particle_data_gamma = None)
        jet_lst.append(jets)

    return jet_lst
    


def get_jets(n_jets, bins_arr, hist_arr, bounds, fit_dic, parms_const, freq, beam_params_lst, beam_rms_lst,posterior,\
                    resolution = '__512',cont = 5, redshift = 0.01, geometric_idx = 0, H0 = 70, particle_data_gamma = None):
    obs_dic = {'flux': [0], 'geom': [0]}
    
    obj = posterior(obs_dic, cont, freq, fit_dic, parms_const,redshift,  beam_rms_lst, beam_params_lst, resolution, particle_data_gamma, geometric_idx = geometric_idx, H0 = H0)
    
    particle_data_gamma = None    
    mask = np.array(list(fit_dic.values()), dtype=bool)
    hist_select = [h for h, m in zip(hist_arr, mask) if m]
    bins_select = [h for h, m in zip(bins_arr, mask) if m]
    bounds_select = bounds[mask]
    jets = []
    fit_lst = np.nonzero(list(fit_dic.values()))[0]
    fit_idx = list(fit_dic.values())

    hists_trans = hist_select #jet_power_hist_trans - gauss_finder(jp[0],jet_power_bins, jet_power_hist_trans, jet_power_bounds)
    
    while len(jets) < n_jets:
        parms_const = np.array(parms_const)


        first_jet = _make_first_guess(bins_select, hists_trans, bounds_select)
        parms_const[fit_idx] = first_jet
        
        
        jet_power = parms_const[0]
        source_age = parms_const[1]
        halo_mass = parms_const[2]
        equipartition = parms_const[3]
        spectral_index = parms_const[4]
        axis_ratio = parms_const[5]


        lum_data = []
        beam_test = []
        df, dg_gamma, particle_data_gamma = RAiSE.RAiSE_run(freq, redshift, axis_ratio, jet_power=jet_power, source_age=source_age, \
                    halo_mass=halo_mass, angle=0.,\
                    resolution=resolution,equipartition = equipartition,spectral_index = spectral_index,\
                                                                particle_data = particle_data_gamma)
        if len(dg_gamma) == len(freq):
            cols = list(dg_gamma[0].columns.tolist())
            rows = list(dg_gamma[0].index)
            pixel_size_kpc = ((cols[1] - cols[0]) * (rows[1] - rows[0]))*3.086e+19
            R = rows[0]
            d = -df[0]['Size (kpc)'][0]/2
            dr = rows[1] - rows[0]
            
            n = (R - d)/dr               
            unc_lst = []
            geometric_idx = 2
            k = 0
    
            
            parms_const = [jet_power, source_age, halo_mass, equipartition, spectral_index, axis_ratio, 0]
            for image_data in dg_gamma:      
            
                
                
                particle_data = particle_data_gamma
                beam_params  = beam_params_lst[k]
                beam_rms  = beam_rms_lst[k]
    
                beam_fwhm_arcsec = beam_params[0]
                
                maxes_only = True
            
                #image_data = df
                #raise_pixel_size_kpc = [(cols[1] - cols[0]),  (rows[1] - rows[0])]
                #image_data = np.pad(image_data, (np.shape(df)[1]//2+1, np.shape(df)[1]//2+1), 'constant', constant_values=(0))    
                noise = np.random.normal(0, beam_rms, np.shape(image_data))
                
                image_data = image_data + (noise)
           
                image_data, dA, beam_area_pixels,raise_cdelts = obj._prep_raise_arcsec_jy(image_data, beam_rms, beam_fwhm_arcsec) 
                beam_test.append(np.max(image_data)/beam_rms)
                
                if k == geometric_idx:
                    geometric_results =  obj._get_geometric_attributes(image_data, raise_cdelts, beam_params, beam_rms,cont,  using_RAiSE = True)
            
                lum_value = obj._process_flux(image_data, raise_cdelts, beam_params, beam_rms, using_RAiSE = True)
                lum_data.append(lum_value)

                
                k += 1
            you_shall_not_pass = False
            example_dic = {'flux': lum_data, 'geom': [geometric_results]}
            # if jet exists and it can be seen:
            import os
    
            lum_data = np.array(lum_data)
    
            print("All jets exist at all frequencies: " + str(len(lum_data) == 4))
            print("All images are brighter than 5 times the beam rms: " + str(np.all(np.array(beam_test) > 5)))
            print("There is some geometry: " + str(len(np.shape(geometric_results)) == 2))
            print("The width is more than 10 pixels across: " + str(geometric_results[0][0] > 10 * raise_cdelts[0]))
            if len(lum_data) == 4 and np.all(np.array(beam_test) > 5) and len(np.shape(geometric_results)) == 2 and (geometric_results[0][0] > 10 * raise_cdelts[0]):
                jets.append(parms_const)

                #update hists
            fidx = 0
            for parm_dx in range(len(parms_const)):
                if fit_idx[parm_dx]:
                    hists_trans[fidx]= hists_trans[fidx] -\
                    _gauss_finder(parms_const[parm_dx],bins_select[fidx], hists_trans[fidx], bounds_select[fidx])
                    fidx += 1

    return jets

def _gauss(x, mean, sigma, height = 1):
    return height * np.exp(-(x-mean)**2/(2*sigma**2))

def _gauss_finder(xn,bins, hist, bounds):
    
    yn = _height_finder(xn, bins, hist)

    bounds_l, bounds_u = bounds[0], bounds[1]

    sigma = (bounds_u - bounds_l) * 0.1

    
    return _gauss(bins, xn, sigma, height = yn/2)

def _height_finder(xn, bins, hist):
    idx = np.searchsorted(bins, xn, side='left', sorter=None)

    try:
        x1, x2 = bins[idx - 1], bins[idx]
        y1, y2 = hist[idx - 1], hist[idx]    
    except IndexError:
        print(idx)
        print(xn)
        print(len(bins), len(hist))

        print(bins, hist)

    yn = linear_interpolator(x1, x2, y1, y2, xn)

    return yn

def linear_interpolator(x1, x2, y1, y2, xn):
    return abs((y2 - y1)) * abs((xn - x1))/abs((x2 - x1)) + y1


def _make_first_guess(bins_arr, hist_arr, bounds):
    #make first guess of the MCMCM
    #returns first guess
    # bounds = self.bounds
    # hist_arr = self.hist_arr
    # bins_arr = self.bins_arr
    #calls all parameters from uniform distribution
    theta_p = np.random.uniform(bounds[:,0], bounds[:,1])
    
    for i, hist in enumerate(hist_arr):    
        #see if there's a prior
        if type(hist) == list or isinstance(hist, np.ndarray):
            # a prior has been provided
            # adjust value
            bins = bins_arr[i]

            #ensure minimum is 0

            hist_sum = hist + abs(np.min(hist))
            #normalise histogram
            p = hist_sum[:-1] / np.sum(hist_sum[:-1])
            # Select a bin according to its probability
            idx = np.random.choice(len(p), p=p)
            
            # Calculate its left and right edges
            if idx == 0:
                left = bins[0] - (bins[1] - bins[0]) / 2
            else:
                left = (bins[idx - 1] + bins[idx]) / 2
            
            if idx == len(bins) - 1:
                right = bins[-1] + (bins[-1] - bins[-2]) / 2
            else:
                right = (bins[idx] + bins[idx + 1]) / 2
            
            # Sample continuously within the selected bin
            theta_p[i] = np.random.uniform(left, right)
    return theta_p
    