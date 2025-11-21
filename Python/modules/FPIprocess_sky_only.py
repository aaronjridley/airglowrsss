#!/usr/bin/env python3

import matplotlib
matplotlib.use('AGG')

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import os
import glob
import datetime
import fpiinfo
import argparse
import FPI

from helper_functions import *
from laser_routines import *
from center_functions import *
from pyitm.fileio import logfile

# ----------------------------------------------------------------------
# Function to parse input arguments
# ----------------------------------------------------------------------

def get_args_fpi():

    parser = argparse.ArgumentParser(description =
                                     'Process FPI data ring pattern image')

    parser.add_argument('sky', nargs='+', \
                        help = 'list of sky files for processing')

    parser.add_argument('-laser', default = 'none', \
                        help = 'laser log file with parameters')
    
    args = parser.parse_args()

    return args

# ----------------------------------------------------------------------
# We need to make cubic splines of the parameters
# ----------------------------------------------------------------------

def cubic_spline_all_params(laserData):

    exclude = ['Year',
               'Month',
               'Day',
               'Hour',
               'Minute',
               'Second',
               'times',
               'rMin',
               'rMax',
               'nPoints',
               'ExposureTime',
               'CC_Temperature']

    vars = laserData['vars']

    nTimes = len(laserData['times'])

    # Grad the roughly center time and calculate dtLasers for cubic spline:
    nMid = int(nTimes/2)
    tMid = laserData['times'][nMid]
    dtLasers = np.zeros(nTimes)
    for iTime in range(nTimes):
        print(iTime)
        dtLasers[iTime] = (laserData['times'][iTime] - tMid).total_seconds()
    
    laserSplines = {'tMid': tMid,
                    'dtLasers': dtLasers}

    for var in vars:
        if (not(var in exclude)):
            p = laserData[var]
            # This should be the error, but we don't have that:
            w = np.ones(np.shape(p))
            s = nTimes
            # this is not a result at times, but a method for getting
            # the value given a dt:
            splineFit = interpolate.UnivariateSpline(dtLasers, p, w=w, s=s)
            laserSplines[var] = splineFit

            # Need some of the raw measurements also:
            laserSplines['raw_' + var] = laserData[var]
            
    return laserSplines


# ----------------------------------------------------------------------
# Provide the radius array
# ----------------------------------------------------------------------

def calculate_radius(laserData):

    nPts = int(laserData['nPoints'][iTime])
    rMin2 = laserData['rMin'][iTime]**2
    rMax2 = laserData['rMax'][iTime]**2
    dr2 = (rMax2 - rMin2) / (nPts - 1)
    radius = np.sqrt(np.linspace(rMin2, rMax2, num = nPts))

    return radius


# ----------------------------------------------------------------------
# Move fits into params
# ----------------------------------------------------------------------

def move_fits_into_params(laserSplines, dt):
    
    # get the laser fit parameters at the given time:
    centerX = laserSplines['CenterX'](dt)
    centerY = laserSplines['CenterY'](dt)
    # The thickness is in two parts, since the delta is so small:
    thickness = laserSplines['Thickness'](dt)
    dThickness = laserSplines['DeltaThickness'](dt)
    t = thickness + dThickness
    lam = laserSplines['Wavelength'](dt)
    R = laserSplines['Reflectivity'](dt)
    alpha = laserSplines['alpha'](dt)
    I = laserSplines['Intensity'](dt)
    B = laserSplines['Baseline'](dt)
    a1 = laserSplines['a1'](dt)
    a2 = laserSplines['a2'](dt)
    b0 = laserSplines['b0'](dt)
    b1 = laserSplines['b1'](dt)
    b2 = laserSplines['b2'](dt)

    # Set the fitting parameter guesses
    sky_params = Parameters()
    sky_params.add('t', value = t, vary = False)
    sky_params.add('lam', value = lam, vary = False)
    sky_params.add('R', value = R, vary = False)
    sky_params.add('alpha', value = alpha, vary = False)
    sky_params.add('I', value = I, vary = False)
    sky_params.add('B', value = B, vary = False)
    sky_params.add('a1', value = a1, vary = False)
    sky_params.add('a2', value = a2, vary = False)
    sky_params.add('b0', value = b0, vary = False)
    sky_params.add('b1', value = b1, vary = False)
    sky_params.add('b2', value = b2, vary = False)
    sky_params.add('n', value = 1.0, vary = False)

    return sky_params, centerX, centerY
    
    
# ----------------------------------------------------------------------
# Fit a single sky image
# ----------------------------------------------------------------------

def fit_sky_image(skyData, laserSplines, instrument):

    c = 299792458.
    k = 1.3806503e-23
    m = 16/6.0221367e26

    timeReference = laserSplines['tMid']
    dtSky = (skyData.info['LocalTime'] - timeReference).total_seconds()

    sky_params, centerX, centerY = move_fits_into_params(laserSplines, dtSky)
    # But change the sky wavelength to lam0
    sky_params['lam'].value = instrument['lam0']
    
    skyImage = np.asarray(skyData)
    flags = []

    if skyData.info['CCDTemperature'] > MAX_CCD_TEMP:
        flags.append[iFlag_CCD_Over_Temp_]

    # Calculate the annuli to use for this time
    annuli = FPI.FindEqualAreas(skyImage, centerX, centerY, N)

    # Perform the annular summation
    sky_spectra, sky_sigma = FPI.AnnularSum(skyImage, annuli, 0)

    print('  --> Intesity diff on sky spectra : ', \
          np.max(sky_spectra) - np.min(sky_spectra))
    print('  --> Total sky sigma (Variability of Image) : ', \
          np.sum(np.abs(sky_sigma)))

    N0 = instrument['N0']
    N1 = instrument['N1']
    
    # Set up the forward model
    L = 301
    lam0 = instrument['lam0']
    FRINGEFACTOR = 0.8     
    last_lamc = lam0 # nominal value of line center
    
    A_1D, lamvec = FPI.get_conv_matrix_1D(sky_params, \
                                          annuli['r'][N0:N1], \
                                          L, \
                                          lam0)

    # forward model works along the edges of the image
    sky_sigma[N0:N0+5] = sky_sigma[N0:N0+5]*10
    sky_sigma[N1-5:N1] = sky_sigma[N1-5:N1]*10

    # Come up with good initial guesses for the sky parameters
    skyI_guess = sky_spectra[N0:N1].max() - sky_spectra[N0:N1].min()
    skyB_guess = 0.0
    ccdB_guess = sky_spectra[N0:N1].min()

    # We'll do a grid search to find a good value for lamc
    sky_params.add('lamc', value = lam0, vary = True)
    sky_params.add('T', value = 1000, vary = True, min = 20., max = 5000.0)
    sky_params.add('skyI', value = skyI_guess, vary = True, min = 0.0)
    sky_params.add('skyB', value = skyB_guess, vary = True)
    # Don't try to estimate skym (or should we?)
    sky_params.add('skym', value = 0.0, vary = False) 
    sky_params.add('ccdB', value = ccdB_guess, vary = True)
    # This is determined by chemistry so it can't change.
    sky_params.add('lam0', value = lam0, vary = False)

    # Do a grid search to find a good starting value for lamc.
    # The instrument bias might mean that lam0 is a really bad
    # value to start with.  Search over one FSR around the
    # last solved-for value (or the nominal value, if it's the
    # first run).  It's not really important to sort by
    # direction, since the instrument bias dominates.
    def goodness(lamc):
        # return the correlation between the fringe with this
        # lamc and the data
        sky_params['lamc'].value = lamc
        fringe = FPI.Sky_FringeModel(sky_params, \
                                     annuli['r'][N0:N1], \
                                     lamvec, \
                                     A_1D)
        top = np.dot(fringe, sky_spectra[N0:N1])
        bot = (np.linalg.norm(fringe)*np.linalg.norm(sky_spectra[N0:N1]))
        return top/bot
    FSR = lam0**2/(2*sky_params['t'].value)

    lamc_candidates = np.linspace(last_lamc - FSR/2, last_lamc + FSR/2, 50)
    corrvec = np.array([goodness(lamc) for lamc in lamc_candidates])
    #print('Searching for best lam')
    best_lamc = lamc_candidates[corrvec.argmax()]
    sky_params['lamc'].value = best_lamc

    # Take the inversion in steps
    order = [
        ['skyI'],\
        ['skyI','ccdB'],\
        ['skyI', 'ccdB', 'skyB'], \
        ['skyI', 'ccdB', 'skyB', 'lamc', 'T'], \
    ]

    # set all the params in this group to "vary=True", and then run
    # inversion
    for group in order: 
        for param in list(sky_params.keys()):
            if param in group:
                sky_params[param].vary = True
            else:
                sky_params[param].vary = False

        sky_fit = Minimizer(FPI.Sky_Residual, \
                            sky_params, \
                            fcn_args = (annuli['r'][N0:N1], \
                                        lamvec, \
                                        A_1D), \
                            fcn_kws = {'data': sky_spectra[N0:N1], \
                                       'sigma': sky_sigma[N0:N1]}, \
                            scale_covar = True)
        sky_fit.prepare_fit()
        result = sky_fit.leastsq()
        sky_params = result.params

    # Redo the fit using only points near the spectral peak
    # (determined by fringefactor)
    if FRINGEFACTOR < 1.0:
        alpha = sky_params['alpha'].value
        t = sky_params['t'].value
        n = sky_params['n'].value
        lamc = sky_params['lamc'].value

        r = annuli['r'][N0:N1]
        m = 2 * n * t / lamc * np.cos(alpha * r)

        idx = abs(m - m.round()) < FRINGEFACTOR/2

        r2 = r[idx]
        temp = sky_spectra[N0:N1]
        sky_spectra2 = temp[idx]
        temp = sky_sigma[N0:N1]
        sky_sigma2 = temp[idx]

        A_1D2, lamvec2 = FPI.get_conv_matrix_1D(sky_params, r2, L, lam0)

        sky_fit = Minimizer(FPI.Sky_Residual, \
                            sky_params, \
                            fcn_args = (r2, lamvec2, A_1D2), \
                            fcn_kws = {'data' : sky_spectra2, \
                                       'sigma': sky_sigma2})
        sky_fit.prepare_fit()
        sky_fit.scale_covar = True
        result = sky_fit.leastsq()
        sky_params = result.params

    lamc = sky_params['lamc'].value
    sigma_lamc = sky_params['lamc'].stderr
        
    # convert Doppler shift to velocity
    losV = c * (lamc/lam0 - 1)
    temperature = result.params['T'].value

    rawBaseThickness = laserSplines['raw_Thickness']
    rawDeltaThickness = laserSplines['raw_DeltaThickness']
    rawThickness = rawBaseThickness + rawDeltaThickness
    sigmaCalV = calc_wind_calib_error(laserSplines['dtLasers'], \
                                      rawThickness, \
                                      dtSky)
    # Need some errors from laser fit!
    #sigma_lamc_gap = np.median(laser_stderr['t']) * \
    #    lam0 / np.median(rawThickness)

    # Fit error on LOS wind
    sigmaFitV = c * sigma_lamc / lam0

    sigmaLOSwind = sqrt(sigmaFitV**2 + sigmaCalV**2)
    sigmaTemperature = result.params['T'].stderr

    
    fit = {'los': losV,
           'temperature': temperature}
    
    return fit
    

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Main Code
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

args = get_args_fpi()

doDebug = True

site, instrument = get_station_info_from_directory(doDebug)

if (doDebug):
    print(' -> site and instrument : ', \
          site['Abbreviation'], \
          instrument['Abbreviation'])

station = site['Abbreviation']

skyFiles = args.sky
laserLog = args.laser
if (laserLog == 'none'):
    laserLog = glob.glob(station + '_lasers_*.txt')
    laserLog = laserLog[0]

if (len(laserLog) < 1):
    print('Cant seem to find laser log file! Does it exist?')
    exit()

laserData = logfile.read_logfile(logfilename = laserLog)

laserSplines = cubic_spline_all_params(laserData)

timeReference = laserSplines['tMid']

MAX_CCD_TEMP = -55.0 # C
iFlag_CCD_Over_Temp_ = 1
N = 500

for iFile, fname in enumerate(skyFiles):

    ignore_this_one = False

    # Read in the sky image
    print('Processing sky image : ', fname)
    skyData = FPI.ReadIMG(fname)
    fit = fit_sky_image(skyData, laserSplines, instrument)
    print('LOS V: ', fit['los'])
    print('Temp: ', fit['temperature'])
    


    
