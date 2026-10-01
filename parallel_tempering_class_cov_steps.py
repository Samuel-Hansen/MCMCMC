import RAiSE_attributes_latest as ra
import RAiSEHD4 as RAiSE
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import pandas as pd
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans,convolve_fft
import numpy as np
class MCMCMC():

    def __init__(self, nsamples, nburns, posterior, hist_arr, bins_arr,bounds, **kwargs):

        """Creates the MCMCMC function.
    
        Parameters
        ----------
        nsamples - integer of number of samples to run
        nburns - integer of number of burns to run
        posterior - class of posterior to use
        hist_arr - array of histograms to use as priors (y axis of prior - probabilities)
        bins_arr - array of buns to use as priors (x axis of prior)
        kwargs - key word arguments needed for posterior
        """
        self.nsamples = nsamples
        self.nburns = nburns
        #self.theta = theta
        self.sigma = 0.02#0.1*((0.1*2.2)**(1/3))/2.2 #0.2*((0.1*2.2)**(1/len(bounds)))/2.2
        self.dtheta = abs(self.sigma*(bounds[:,1] - bounds[:,0]))
        self.hist_arr = hist_arr
        self.bins_arr = bins_arr
        self.posterior = posterior
        self.bounds = bounds
        self.forward_kwargs = kwargs
        self.posterior_class = self.posterior(**self.forward_kwargs)
        #boolean array which contains location of parameters with priors
        self.has_prior = np.array([isinstance(h, (list, np.ndarray)) for h in hist_arr],dtype=bool)

        self.sample_cov = np.diag(self.dtheta)


        
        
        # # Overall tuning factor
        # scale = 0.15
        
        # # Standard random-walk Metropolis scaling
        # base_scale = 2.38**2 / len(bounds)
        
        self.proposal_cov = (
            # scale**2
            # * base_scale
            self.sample_cov
        )
        
        
            
    def run(self, res_lst, parameters, temps = [1], seed = 0, first_guess = None, sigma = None):

        if sigma != None:
            self.dtheta = abs(sigma*(self.bounds[:,1] - self.bounds[:,0]))    

            self.sigma = sigma
        self.seed = seed

        print("sigma = " + str(self.sigma))

        #get posterior functions
        self.post_funcs = self._make_posterior_functions(res_lst, parameters)
        #get next step functions

        self.propose_func = self._next_step_functions(self.dtheta, self.hist_arr, self.bins_arr)
        #get first guess
        aic_lst = np.array([1e11]*len(res_lst))
        np.random.seed(self.seed)

        aic_lst_lst = [np.array([1e11]*len(res_lst))]*len(temps)
        val_lst = [[None] * len(res_lst)]*len(temps)

        accept_k = 0
        theta_temp = []
        
        self.covariances = np.array(np.broadcast_to(self.proposal_cov, (len(temps), *self.proposal_cov.shape)))

        
        if first_guess.any() == None or len(first_guess) != len(temps):
            print("Guessing for first guess")
            try:
                print(len(first_guess), len(temps))
            except:
                print("wag")
            for kdx in list(range(len(temps))):
                aic_lst = aic_lst_lst[kdx]
                while not (np.all(aic_lst < 1e10) and np.all(aic_lst != 0)):
                    #runs until each parameter has an AIC
                    theta_p = self._make_first_guess()      
                    theta_p = self._check_theta(theta_p)
                    for idx in list(range(len(res_lst))):
                        #for each resoltution and parameter find the AIC. 
                        aic_lst[idx] = self.post_funcs[f"layer_{idx}"](theta_p, using_RAiSE = True)
    
                theta_temp.append(theta_p)
        else:
            theta_temp = first_guess

        theta_temp = list(theta_temp)
        #original step    
        self.theta = theta_temp[0]
        #next step
        # for kdx in range(nchains):
        #     theta = theta_temp[kdx]
        #     temp = temps[kdx]

        #     theta_p = self.propose_func(theta, temp)

            
        #calcualte the new aic
        j = 0
        samples = []
        aics = []
        swap_interval = 20
        if len(temps) == 1:
            swap_interval = 2*(self.nsamples + self.nburns + 1)
        nchains = len(temps)

        cold_idx = 0#temps.index(1)
        accept_temps = [[] for _ in range(len(temps))]
        k_accept = 0
        ks = np.zeros(len(temps))
        acceptance_rate_tracker = 0


        burn_samples = np.empty((self.nburns + 1, len(temps), len(self.bounds) ))
        
        while j < self.nsamples + self.nburns + 1:
            for kdx in range(nchains):
                accept_lst = []
                temp = temps[kdx]
                theta = theta_temp[kdx]
                aic_lst = aic_lst_lst[kdx]

                sample_cov = self.covariances[kdx]
                theta_p = self.propose_theta(theta, sample_cov, temp = temp)
                theta_p = self._check_theta(theta_p)
        
                aic_lst_temp = []
                fully_accepted = True

                logprior_old = self.log_prior(theta)
            
                logprior_new = self.log_prior(theta_p)

                # print("log priors")
                # print(logprior_old, logprior_new)
                # print(theta, theta_p)
                for idx, aic_current in enumerate(aic_lst):
                    aic_proposed = self.post_funcs[f"layer_{idx}"](theta_p, using_RAiSE = True)

                    # print(aic_current, aic_proposed)
     
                    with np.errstate(over='ignore', invalid='ignore'):
                        prob = np.exp((aic_current - aic_proposed) / (2 * 1))
                        log_alpha = (-0.5 * (aic_proposed - aic_current)/ temp + logprior_new - logprior_old)
        
                    aic_lst_temp.append(aic_proposed)
                    prob = min(np.nan_to_num(prob, nan=0.0), 1.0)
                    test = np.log10(np.random.uniform())

                    # print("test, alpha")
                    # print(test, log_alpha)
                    if test >= log_alpha:
                        # print("not accepted")
                        fully_accepted = False
                        accept_lst.append(fully_accepted)
                        break

                    accept_lst.append(fully_accepted)
                accept_temps[kdx].append(accept_lst)
                
                if fully_accepted:

                    theta_temp[kdx] = theta_p
                    aic_lst_lst[kdx] = aic_lst_temp
                    if kdx == 0:
                        acceptance_rate_tracker += 1
               
            j += 1
        
            if j > self.nburns:
                samples.append(theta_temp[cold_idx])
                aics.append(aic_lst_lst[cold_idx])
            else:
                burn_samples[j] = theta_temp


                
            if j % 100 == 0:
                print(j)
                print(theta_temp[0])
                print(100 * acceptance_rate_tracker / j)
                print(k_accept / ((j / swap_interval ) * (max(nchains - 1,1))))
                print(np.array(ks)/ ((j / swap_interval) * (max(nchains - 1,1))))



            
            #adjust covariance matrix
            if j >= self.nburns // 2 and j % 20 == 0 and j < self.nburns:
                #update for each temperature
                for tempdx in range(nchains):
                
                    sample_cov = np.cov(
                        burn_samples[self.nburns // 2:j, tempdx , :],
                        rowvar=False
                    )
                    
                    sample_cov += 1e-8 * np.eye(len(self.bounds))   
                    
                    self.covariances[tempdx] = sample_cov


                
            if j % swap_interval == 0:
                for i in range(nchains - 1):
                    swap_temps = True
                    T_i, T_j = temps[i], temps[i + 1]
                    beta_i, beta_j = 1.0 / T_i, 1.0 / T_j
        
                    # total "energy" = sum of AIC components
                    # E_i = sum(aic_lst_lst[i])
                    # E_j = sum(aic_lst_lst[i + 1])
                    for kdx in range(len(aic_lst_lst[i])):
                        aic_i = aic_lst_lst[i][kdx]
                        aic_j = aic_lst_lst[i + 1][kdx]
                        swap_prob = np.exp((beta_i - beta_j) * (aic_i - aic_j) / 2)
                        if np.random.uniform() >= swap_prob:
                            swap_temps = False
                            break
                        
                    # for aic_i in aic_lst_lst[i]:
                    #     aic_j = aic_lst_lst[i + 1]
                    # # with np.errstate(over='ignore', invalid='ignore'):
                    # #     swap_prob = np.exp((beta_i - beta_j) * (E_j - E_i) / 2)
        
                    # swap_prob = min(np.nan_to_num(swap_prob, nan=0.0), 1.0)
        
                    if swap_temps == True:

                        theta_temp[i], theta_temp[i + 1] = theta_temp[i + 1], theta_temp[i]
                        aic_lst_lst[i], aic_lst_lst[i + 1] = aic_lst_lst[i + 1], aic_lst_lst[i]
                        ks[i] += 1
                        k_accept += 1
                        print(f"Swapped chains {i} and {i+1} at iteration {j}")
                            
        return samples, aics, accept_temps, k_accept, ks, acceptance_rate_tracker, self.covariances


    def _make_posterior_functions(self, res_lst, parameters):
        #posterior function creating function based on resolution and parameters
        #returns a dictionary of posterior functions

        post_funcs = {}

        idx = 0
        #get posterior functions
        for idx in list(range(len(res_lst))):
            #define posterior functions to self
            def make_f(idx):
                return self.posterior_class.create_posterior_function(res_lst[idx], parameters[idx])
            post_funcs[f"layer_{idx}"] = make_f(idx)
        return post_funcs

        
    def _make_first_guess(self):
        #make first guess of the MCMCM
        #returns first guess
        bounds = self.bounds
        hist_arr = self.hist_arr
        bins_arr = self.bins_arr
        #calls all parameters from uniform distribution
        theta_p = np.random.uniform(bounds[:,0], bounds[:,1])
        
        for i, hist in enumerate(hist_arr):    
            #see if there's a prior
            if type(hist) == list or isinstance(hist, np.ndarray):
                # a prior has been provided
                # adjust value
                bins = self.bins_arr[i]
                hist_sum = hist / hist.sum()       # ensure normalization
                # draw one value
                theta_p[i] = np.random.choice(bins, p=hist_sum)
        return theta_p


    def propose_theta(self, theta, sample_cov, temp = 1):
        """
        Draw a correlated proposal using the covariance
        estimated from the MCMC samples.
        """
    
        ndim = len(theta)
    
        proposal_cov = (
            self.sigma**2
            * temp 
            * (2.38**2 / ndim)
            * sample_cov
        )
        
        step = np.random.multivariate_normal(
            mean=np.zeros(ndim),
            cov=proposal_cov
        )
    
        return theta + step              
        
    
    def _no_priors(self, theta,dtheta):
        # no priors so don't apply one
        return theta + np.random.normal(0, dtheta)
    
    def _adjust_priors(self, theta, dtheta, hist, bins):
        #change name
        #if there are priors, select initial guess from given distribution
        hist = hist * np.exp((-(bins - theta)**2)/(2*dtheta**2))
        hist = hist/np.sum(hist) #normalised prob dens function
        cum_sum = np.cumsum(hist) #cumulative distribution
        x01 = np.random.uniform(0, 1)
        ldx = np.nonzero(x01 <= cum_sum)[0][0]
        if ldx == 0:
            topx, botx = bins[ldx], 2* bins[ldx] - bins[ldx+1]
            top_cumsum, bot_cumsum = cum_sum[ldx], 0          
        else:
            topx, botx = bins[ldx], bins[ldx-1]
            top_cumsum, bot_cumsum = cum_sum[ldx], cum_sum[ldx-1]
        d_cumsum = (x01 - bot_cumsum)/(top_cumsum - bot_cumsum)
        theta_x = botx + d_cumsum*(topx-botx)
        return theta_x
    
    def _next_step_functions(self, dtheta, hist_arr, bins_arr):
        # next step function creating function
        # takes step value (dtheta), hist and bins of priors
        # returns correct step function for taking the next step in the MCMCMC
        dtheta = np.asarray(dtheta, dtype=float)
        has_prior = np.asarray(self.has_prior, dtype=bool)
    
        # Pre-select indices for prior vs no-prior
        idx_prior = np.where(has_prior)[0]
        idx_free  = np.where(~has_prior)[0]
    
        # Pre-extract arrays for faster closures
        hist_arr  = list(hist_arr)
        bins_arr  = list(bins_arr)
        #create next step function
        def step(theta, temp):
            new_theta = np.empty_like(theta, dtype=float)
            dtheta = self.dtheta * np.sqrt(temp)


            
            # For parameters with priors
            # for i in idx_prior:
            #     new_theta[i] = self._adjust_priors(theta[i],dtheta[i],hist_arr[i],bins_arr[i])
    
            # # For parameters without priors
            # for i in idx_free:
            #     new_theta[i] = self._no_priors(theta[i],dtheta[i])
    
            return new_theta
    
        return step  
        
    def log_prior_hist(self, x, hist, bins):
        """
        Evaluate a 1D prior stored as a histogram / sampled PDF.
        """
        if np.isscalar(hist) and np.isscalar(bins):
            if hist == 0 and bins == 0:
                return 0.0
        p = np.interp(x,bins,hist,left=0.0,right=0.0)
    
        if p <= 0:
            return -np.inf
    
        return np.log(p)
    
    
    def log_prior(self, theta):
        """
        Total log-prior for all six parameters.
        Assumes independent priors.
        """
        prior_hists = self.hist_arr
        prior_bins = self.bins_arr
        lp = 0.0
    
        for i in range(len(theta)):
    
            lp_i = self.log_prior_hist(theta[i],prior_hists[i],prior_bins[i])
    
            if not np.isfinite(lp_i):
                return -np.inf
    
            lp += lp_i
    
        return lp
        
    def _check_theta(self, theta_p):
        #checks to see if proposed theta is within the bounds
        bounds_arr = self.bounds
        low, high = theta_p < bounds_arr[:,0], theta_p > bounds_arr[:,1]
        #if no, reflects theta
        theta_p[low], theta_p[high] = np.minimum(2*bounds_arr[low,0] - theta_p[low], bounds_arr[low,1]), \
                        np.maximum(2*bounds_arr[high,1] - theta_p[high], bounds_arr[high,0])
        return theta_p