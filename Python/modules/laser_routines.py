#!/usr/bin/env python

import FPI
import numpy as np
from lmfit import Minimizer, Parameters, Parameter, minimize


def fit_laser(instrument, annuli, laser_spectra, laser_sigma, N0 = 0, N1 = 500):

    ####### Find good initial guesses for the parameters ######

    # Magnification parameter by using geometry (assuming square binning)
    alpha = instrument['pix_size']/instrument['focal_length'] * instrument['XBinning']
    
    # Intensity and background by looking at fringes
    I = laser_spectra.max() - laser_spectra.min()
    B = laser_spectra.min()

    I = np.percentile(laser_spectra,99) - np.percentile(laser_spectra,2)
    B = np.percentile(laser_spectra,2)

    lam_laser = instrument['lam_laser']
    last_t = instrument['nominal_t']
    ESTIMATE_BLUR = True
    
    laser_params = Parameters()
    laser_params.add('n',     value = 1.0,       vary = False)
    laser_params.add('t',     value = None,      vary = False) # We will search for this
    laser_params.add('lam',   value = lam_laser, vary = False)
    laser_params.add('R',     value = 0.5,        vary = False)
    laser_params.add('alpha', value = alpha,     vary = False)
    laser_params.add('I',     value = I,         vary = False)
    laser_params.add('B',     value = B,         vary = False)
    laser_params.add('a1',    value = -0.1,      vary = False)
    laser_params.add('a2',    value = 0.0,       vary = False)
    laser_params.add('b0',    value = 0.5,       vary = False)
    laser_params.add('b1',    value = 0.0,       vary = False)
    laser_params.add('b2',    value = 0.0,       vary = False)

    # To find a good initial guess for "t", we'll need to do a grid search.  Search
    # over 1 FSR around the last solved-for value (or the nominal value, if this is first trial).
    # TODO: make this grid search a separate general function.
    def goodness(t):
        # return the correlation between the fringe with this t and the data
        laser_params['t'].value = t
        fringe = FPI.Laser_FringeModel(laser_params, annuli['r'])
        return np.dot(fringe, laser_spectra)/(np.linalg.norm(fringe)*np.linalg.norm(laser_spectra))
    t_candidates = np.linspace(last_t - lam_laser/4, last_t + lam_laser/4, 50)
    corrvec = np.array([goodness(t) for t in t_candidates])
    best_t = t_candidates[corrvec.argmax()]
    laser_params['t'].value = best_t

    ####### Inversion of laser image ##########

    # Now do least-squares fit, but in stages, varying only certain parameters at a time, according to:
    order = [
        ['alpha'],\
        ['t','alpha'],\
        ['B','I'],\
        ['R'],\
        ['t','alpha','B','I','R','a1','a2'], \
    ]
    if ESTIMATE_BLUR:
        order.append(['t','alpha','B','I','R','a1','a2','b0','b1','b2'])

    data = laser_spectra[N0:N1]
    sigma = laser_sigma[N0:N1]

    for group in order: # set all the params in this group to "vary=True", and then run inversion
        for param in list(laser_params.keys()):
            if param in group:
                laser_params[param].vary = True
            else:
                laser_params[param].vary = False
        # Set falloff terms to false, if this instrument only has a couple fringes
        if not instrument['many_fringes']:
            for param in ['a2', 'b1', 'b2']:
                laser_params[param].vary=False
                laser_params[param].value=0.0

        laser_fit = Minimizer(FPI.Laser_Residual, laser_params, \
                              fcn_args=(annuli['r'][N0:N1],), fcn_kws={'data': data, 'sigma': sigma}, \
                              scale_covar = True)
        result = laser_fit.leastsq()
        laser_params = result.params

    return result

def find_optimum_N0(laser_spectra, laser_sigma, instrument, annuli, initial_guess = 0, finalN = 500):

    diffSave = 1e32
    N_0 = -1
    nStartTest = initial_guess
    nMax = 25
    keepGoing = True
    firstHit = False

    peak2valley = np.percentile(laser_spectra,95) - np.percentile(laser_spectra,5)
    print(' -> Laser Spectra Brightness difference : ', peak2valley)
    
    while (keepGoing):
        # Perform the fit to the best fringe pattern parameters:
        result = fit_laser(instrument, annuli, laser_spectra, laser_sigma, N0 = nStartTest, N1 = finalN)
        laser_params = result.params

        # Create a fringe pattern based on the results:
        fringe = FPI.Laser_FringeModel(laser_params, annuli['r'])

        diff = np.sqrt(np.mean((fringe[nStartTest:-1] - laser_spectra[nStartTest:-1])**2))

        if (diff < diffSave):
            N_0 = nStartTest
            print('  -> new N0 found! N0, old diff, new diff : ', N_0, diffSave, diff)
            diffSave = diff
            # Reset first hit if we find another value lower!
            firstHit = False
        else:
            # Let the first one slide, but stop on the second one
            # where the error is larger:
            if (firstHit):
                keepGoing = False
            else:
                firstHit = True

        if (nStartTest >= nMax):
            keepGoing = False
            
        nStartTest += 1

    return N_0, diffSave

    
