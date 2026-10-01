import RAiSE_attributes_latest as ra
import RAiSEHD4 as RAiSE
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import pandas as pd
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans,convolve_fft
import numpy as np
import matplotlib.pyplot as plt
import timeit

res_error_dic = {'__2' :0.06970931707328415, '__8':0.05653804853596847, '__32':0.016018968689669552}
res_error_dic = {'__2' :0.0, '__8':0.0, '__32':0.0}

class RAiSE_posterior():

    def __init__(self,  obs_dic, cont, frequency, parm_dic, parms_const, redshift,\
                 beam_rms_lst, beam_params_lst, resolution, particle_data, geometric_idx = 0, H0 = 70):

        """Creates a posterior function that returns the AIC when compared to RAiSE
    
        Parameters
        ----------
        obs_dic - a dictionary of observed parameters. Requires keys "flux" or "geom" with values in a list
        cont - float of contour value used to find the ellipse and flux
        frequency - list of radio frequencies to be called in RAiSE
        parm_dic - dictionary of parameters to be varied. Requires keys of parameters put into RAiSE and boolean values
        redshift - float of redshift to be put into RAiSE
        beam_rms - list of beam_rms' used in the observations. Order to be the same as frequency
        resolution - list of strings of the resolutions to be used in RAiSE for each layer
        particle_data - array (?) of particle data to be used by RAiSE
        geometric_idx - integer that indicates the index of the lobe to use
        H0 - float used as Hubble constant
        """
        #define parameters
        self.H0 = H0
        self.beam_params_lst = beam_params_lst
        self.obs_dic = obs_dic
        self.frequency = frequency
        self.geometric_idx = geometric_idx
        self.cont = cont
        self.parm_dic = parm_dic
        self.parms_const = parms_const
        self.beam_rms_lst = beam_rms_lst
        self.redshift = redshift
        self.resolution = resolution
        self.particle_data = particle_data
        self.mask = np.array(list(parm_dic.values()), dtype=bool)
        self.min_unc = 0.01

        #sets handlers for processing different sets of parameters
        self.handlers = {"geom": self._process_geom, "flux": self._process_flux}

            
    def create_posterior_function(self, resolution, parameters):

        print("min unc")
        print(self.min_unc)

        print("freq")
        print(self.frequency)
        #posterior function creating function
        #takes resolution and parameters with which to create a posterior function
        #returns a posterior function based on RAiSE
        #creates jet function based on resolution
        if np.all(parameters == 'geom'):
            frequency = [self.frequency[self.geometric_idx]]
            beam_rms_lst = [self.beam_rms_lst[self.geometric_idx]]
            beam_params_lst = [self.beam_params_lst[self.geometric_idx]]
        else:
            frequency = self.frequency
            beam_rms_lst = self.beam_rms_lst
            beam_params_lst = self.beam_params_lst
        # try:
        #     if isinstance(resolution, str):
        #         self.geom_err = res_error_dic[resolution]
        #     else:
        #         self.geom_err = res_error_dic[resolution[0]]
        # except KeyError:
        
        self.geom_err = 0
        
        jet_func = self._make_jet_function(resolution, frequency)  
        #sets the constant values
        # beam_params = self.beam_params
        # beam_rms = self.beam_rms
        cont = self.cont
        prep = self._prep_raise_arcsec_jy
        obs_fit_lst = self._get_fit_dic(self.obs_dic)
        #checks that parameters are in a list. If not, naively puts them into a list

        
        if not isinstance(parameters, (list, np.ndarray)):
            parameters = [parameters]
        #defines the posterior function
        def posterior_function(theta, using_RAiSE = False):
            error_jitter = 0
            #gets image data from jet function at provided theta
            
            df, image_data = jet_func(theta)


            raise_data_dic = {}
            idx = 0
            for img in image_data:
                try:
                    beam_rms = beam_rms_lst[idx]
                except IndexError:
                    print("beam rms")
                    print(beam_rms_lst)
                    print(idx)
                    print(len(image_data))
                    print(parameters)
                    print(self.geometric_idx)
                    print(frequency)
                    print(image_data)
                beam_params = beam_params_lst[idx]
                beam_fwhm_arcsec = beam_params[0]
                freq = frequency[idx]
                if beam_rms > 0 and beam_fwhm_arcsec > 0:
                    #preps RAiSE output to be in Jy/arcsecs
                    image_data_per_beam, dA, pixels_per_beam, raise_cdelts = prep(img,beam_rms,beam_fwhm_arcsec)
                else:
                    image_data_per_beam, dA, pixels_per_beam, raise_cdelts = 0,0,0,0
                #get the AIC value for the parameter being called
                for parm in parameters:
                    if parm not in raise_data_dic:
                        raise_data_dic[parm] = []   
                    # Directly call the correct pre-bound handler
                    if len(parameters) == 1 or parm == 'flux' or idx == self.geometric_idx:

                        value = self.handlers[parm](image_data_per_beam, raise_cdelts,  beam_params, beam_rms, using_RAiSE)

                        # Append to the correct storage list
                        raise_data_dic[parm].append(value)
                idx += 1
            #transform values into ratios to be used in AIC caluclation 
            raise_lst = self._get_fit_dic(raise_data_dic)
            #calculate AIC
            AIC = self._calculate_AIC(obs_fit_lst, raise_lst, error_jitter = error_jitter)        
            return AIC
        return posterior_function
        
    def _make_jet_function(self, resolution, frequency):
        #creates RAiSE jet function with constant values already applied
        consts = np.array(self.parms_const.copy())    # pure python list/array
        mask = self.mask                   # no attribute lookups inside fn
        redshift = self.redshift
        particle_data_old = self.particle_data

        def produce_jet(theta):
            theta_jet = consts.copy()
            theta_jet[mask] = np.array(theta)
            #read parameters
            jet_power = theta_jet[0]
            source_age = theta_jet[1]
            halo_mass = theta_jet[2]
            equipartition = theta_jet[3]
            spectral_index = theta_jet[4]
            axis_ratio = theta_jet[5]

            if jet_power == 38.063541010790225 and source_age == 8.284015280343239 and halo_mass == 16.001973060179132:
                print("wack")
                jet_power = 38
  
            #run raise
            df, dg, particle_data = RAiSE.RAiSE_run(frequency, redshift, axis_ratio, jet_power=jet_power, source_age=source_age, \
                    halo_mass=halo_mass,\
                    angle=0., resolution=resolution,equipartition = equipartition,spectral_index = spectral_index,\
                    particle_data = particle_data_old, pair_plasma = False)
            #sometimes RAiSE can freak out
            if len(dg) > 0 and (np.shape(dg[0])[0] > 1e4 or np.shape(dg[0])[1] > 1e4):
                
                dg = [pd.DataFrame(np.zeros((5, 5))) for _ in range(len(frequency))]
            return df, dg
        return produce_jet

        
    def _create_cont_lst(self, cont, frequency):
        # Checks contour
        if isinstance(cont, (int, float)):
            return np.full(len(frequency), cont)
    
        # cont is already an array/list
        return np.asarray(cont)
            
    def _get_fit_lst(self, fit_dic, parms_const, theta):
        #takes dictionary of what to fit for, constant parameters and current theta
        #returns theta with constant values in place to be read by other functions
        mask = np.array(list(fit_dic.values()), dtype=bool)
        
        raise_parms = parms_const.copy()
        raise_parms[mask] = theta[0] 
        return theta


        
    def _convolve_raise(self, fixed_image, cdelts, beam_diameter,rms):
        #takes fixed_image, an array and convolves it with a beam with diameter beam_diameter and rms, rms
    
        ang_pix_width = cdelts[0] #angular pixel width in arc seconds
        ang_pix_height = cdelts[1]#angular pixel height in arc seconds
        
        pixel_per_beam_width = beam_diameter/ang_pix_width #fwhm x
        pixel_per_beam_height = beam_diameter/ang_pix_height #fwhm y
                
        ###creates a beam from values given
        beam = Gaussian2DKernel (x_stddev = pixel_per_beam_width/(2 * np.sqrt(2 * np.log(2))),y_stddev = pixel_per_beam_height/(2 * np.sqrt(2 * np.log(2))))

        fixed_image = fixed_image.astype("float32")
        beam = beam.array.astype("float32")
        ###convolves image with beam

        fixed_image = convolve_fft(fixed_image, beam, allow_huge=True)  
        return fixed_image        
        
    def _prep_raise_arcsec_jy(self, image_data, beam_rms, beam_fwhm_arcsec):
        # takes RAiSE output in W/Hz/Pixel, converts to Jy, convolves with the same beam used in observations, returns raise image in Jy/beam
        
        #from RAiSE in W/Hz/pixel
        
        # Convert W/Hz to Jy/Beam
        #beam_rms = self.beam_rms
        redshift = self.redshift

        
        H0 = self.H0
        cosmo = FlatLambdaCDM(H0=H0 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3) ##cosmology
        dL = cosmo.luminosity_distance(redshift).to(u.m).value 

        dA = dL/(1+redshift)**2 ###angular diameter in m
        raise_pixel_size_kpc = [image_data.index[1] - image_data.index[0],
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
        image_data_per_pixel = self._convolve_raise(image_data_per_pixel, raise_cdelts_arcsec, beam_fwhm_arcsec, beam_rms)
        
        # Convert to Jy/beam
        image_data_per_beam = image_data_per_pixel * pixels_per_beam
        image_data_per_beam = np.pad(image_data_per_beam, (np.shape(image_data)[1]//2+1, np.shape(image_data)[1]//2+1), 'constant',constant_values=(0))    
    
        return image_data_per_beam, dA, pixels_per_beam, raise_cdelts_arcsec
    
    def _get_fit_dic(self, data_dic):
        #takes in dictionary of data with keys 'flux' and/or 'geom'
        #applies the dictionary transformations of this data and returns a dictionary containing that
        fit_dic = {key: [] for key in data_dic}
        for name, arg in data_dic.items():
            #finds function that matches parameter
            method = getattr(self, f"{name}_fit_dic")
            #applies data to function
            results = method(arg)
            #puts results into dictionary
            fit_dic[name] = results
        return fit_dic
    
    ###flux functions###

    def _process_flux(self, img, raise_cdelts, beam_params, beam_rms, using_RAiSE = False):
        
        #takes in RAiSE image and RAiSE cdelts and returns flux parameters
        
        self.cdelts = raise_cdelts

        lum_value = ra.get_brightness_attributes(
            img,
            raise_cdelts,
            beam_params,
            beam_rms,
            cont_level=self.cont,
            flux_error=self.min_unc
        )


        return lum_value
    
    def flux_fit_dic(self, lum_data):
        #takes in a list of luminosity data
        #lum data should be [[lum1, dlum1], [lum2, dlum2]... [lumn, dlumn] for n frequencies where dlum is the uncertainty associated with lum
        #returns the fit dic for flux data      
        frequency = self.frequency
        lum_data = np.array(lum_data).T
        lum_lst = lum_data[0]
        dlum_lst = lum_data[1]
        results = []

        # Map length to function
        func_map = {
            1: self.luminosity_and_error,
            2: self._alpha_calc,
            3: self._alpha_calc
        }

        slice_map = {
            1: slice(0, 1),       # first element
            2: slice(0, 2),       # first two elements
        }    
        # For each length from 1 to len(arr), call the corresponding function
        # builds luminosioty dictionary dynamically
        for leng in range(1, len(frequency)+1):
            #gets slice function. This determines which luminosities to use
            slice_func = slice_map.get(leng, slice(-2,None))
            # gets correct function to apply
            func = func_map.get(leng)
            if func is not None:
                # Pass only the first 'length' elements to the function
                # applies function to luminosities selected
                results.append(func(frequency[slice_func], lum_lst[slice_func], dlum_lst[slice_func]))
        # returns list of results
        return results

        
    def _alpha_calc(self,frequency, lum_lst, dlum_lst):
        #takes lumiosities and frequencies and their uncertaianties and returns alpha low, alpha high and their uncertainties
        lum1, lum2 = lum_lst
        delta_lum1, delta_lum2 = dlum_lst
        freq1, freq2 = frequency
        if lum1 == 0 or lum2 == 0:
            return [0,0]
        with np.errstate(divide='ignore', invalid='ignore'):
            alpha = np.log10(lum1/lum2)/np.log10(10**freq1/10**freq2)
        a = lum1/lum2
        b = freq1/freq2
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                delta_a = a*(delta_lum1/lum1 + delta_lum2/lum2)
        except ZeroDivisionError:
            delta_a = np.nan
    
        c = np.log10(a)
        d = np.log10(b)
    
        delta_c = 1/np.log(10) * (delta_a/a)
        with np.errstate(divide='ignore', invalid='ignore'):
            delta_alpha = alpha*(delta_c/c)
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                delta_alpha = alpha*(np.sqrt((delta_lum1/lum1)**2 + (delta_lum2/lum2)**2))/np.log(lum1/lum2)
        except ZeroDivisionError:
            delta_alpha = np.nan
        return [alpha, delta_alpha]
    
    def luminosity_and_error(self, frequency, lum_lst, dlum_lst):
        #simple function that returns the log10 luminosity and its error of the lobe of the lowest frequency
        lum = lum_lst[0]
        dlum = np.max([dlum_lst[0], lum * self.min_unc])

        
        
        with np.errstate(divide='ignore', invalid='ignore'):
            return [np.log10(lum),abs(np.log10(lum  - dlum) - np.log10(lum  + dlum))/2]

    ###geometric functions###
        
    def _process_geom(self, img, raise_cdelts,  beam_params, beam_rms, using_RAiSE = False):
        # takes in RAiSE image and returns geometric parameters according to geometric attributes
        geom_value = self._get_geometric_attributes(img, raise_cdelts,beam_params, beam_rms, self.cont, using_RAiSE = using_RAiSE)
        return geom_value
        
    def _get_geometric_attributes(self, image_data, cdelts, beam_params, beam_rms, cont,  using_RAiSE = False):
        #takes image_data and returns geometric attributes frolm RAiSE_attributes
        try:
            y_axis = len(image_data[0,:])/2
        except: print(image_data)
        # x point is 1/2 the distance from the edge of the jet to the BH
        x_axis = len(image_data[:,0])/2
        avg = 1*np.shape(image_data)[0]/5
    
        core_coords = [x_axis, y_axis]
        lobe_coords = [avg, y_axis]
        image_bounds = [[0,np.shape(image_data)[0]],[0,np.shape(image_data)[1]]]
        ring_coords, fit_coords, results, size_data, centre_points, ring_coords_t, core_coords_t = ra.get_spatial_attributes(image_data, cdelts, beam_params, beam_rms, core_coords, lobe_coords, cont_level=5, using_RAiSE = using_RAiSE)
        

        return results  
        
    #Prepare data that has been provided
    def geom_fit_dic(self, geom_data):
 
        geom_data = geom_data[0]
        #geometric_idx = self.geometric_idx
        #define variables from geometric idx
        width = geom_data[0][0]#- 0.4 * geom_data[0][0]
        dwidth = np.max([np.sqrt(geom_data[0][1]**2 + (self.geom_err * width)**2), width * self.min_unc])
        length = geom_data[1][0]# - 0.1 * geom_data[1][0]
        dlength = np.max([np.sqrt(geom_data[1][1]**2 + (self.geom_err * length)**2), length * self.min_unc])
        extent = geom_data[2][0]# - 0.1 * geom_data[2][0]
        dextent = np.max([np.sqrt(geom_data[2][1]**2 + (self.geom_err * extent)**2), extent * self.min_unc])
        if width == 0 or length == 0 or extent == 0:
            return [[0,0],[0,0],[0,0]]

        #define geometric ratios
        r1 = width/length
        dr1 = r1*np.sqrt((dwidth/width)**2 + (dlength/length)**2)
        
        r2 = length/extent
        dr2 = r2*np.sqrt((dextent/extent)**2 + (dlength/length)**2)
        gamma_parms = [ [r1,dr1] ,\
                        [r2,dr2],\
                        [np.log10(length),abs(np.log10(length  - dlength) - np.log10(length  + dlength))/2]]
        return gamma_parms
            

    def _calculate_AIC(self, obs_dic, ras_dic, error_jitter = 0):
        #calcualtes AIC between observations and RAiSE
        #sees what parameters are shared between observations and RAiSE
        shared_keys = obs_dic.keys() & ras_dic.keys()
        # matches those keys
        obs_lst = [obs_dic[k] for k in shared_keys]
        ras_lst = [ras_dic[k] for k in shared_keys]
        # Extract columns
        parameter_observed = np.array([v for group in obs_lst for (v, dv) in group])
        error_observed = abs(np.array([dv for group in obs_lst for (v, dv) in group]))
        parameter_proposed = np.array([v for group in ras_lst for (v, dv) in group])
        error_proposed = abs(np.array([dv for group in ras_lst for (v, dv) in group]))

        # Combined variance term
        var = (error_proposed**2 + error_observed**2)
        var = var + max(error_jitter, 0)
        likelihood = np.nan
        # Vectorized log-likelihood
        # Test to make sure all proposed parameters exist
        likelihood = 0
        if all(np.isfinite(parameter_proposed)) and len(parameter_proposed) > 0:
            
            likelihood = (
                np.log(1.0 / (np.sqrt(2*np.pi) * error_observed))
                - (parameter_proposed - parameter_observed)**2 / (2*var)
            ).sum()
        else:
            likelihood = np.nan
        k = parameter_observed.shape[0]  # number of parameters
        AIC = np.nan_to_num(2*k - 2*likelihood, nan = 1e11)
        return AIC