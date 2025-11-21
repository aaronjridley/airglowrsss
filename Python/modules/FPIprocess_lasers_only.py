#!/usr/bin/env python3

import matplotlib
matplotlib.use('AGG')

import matplotlib.pyplot as plt
import numpy as np
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

#
# This code is designed to do a bunch of different things with the
# laser images:
# 1. Simply read and make image(s)
# 2. Find the center of the image(s)
# 3. Plot out the fringes from the radial integrals
# 4. Fit the fringes with an instrument model
#

# ----------------------------------------------------------------------
# Function to parse input arguments
# ----------------------------------------------------------------------

def get_args_fpi():

    parser = argparse.ArgumentParser(description =
                                     'Process FPI data ring pattern image')

    parser.add_argument('lasers', nargs='+', \
                        help = 'list of laser files for processing')

    parser.add_argument('-subtract', \
                        help = 'subtract background', \
                        action = 'store_true')
    
    parser.add_argument('-last',  default = False,
                        action = 'store_true',
                        help = 'Only use the last laser image')
    
    parser.add_argument('-brightness',  default = False,
                        action = 'store_true',
                        help = 'Make Brightness Summary Plot')
    
    parser.add_argument('-datadir', default = '/backup/Data/Fpi', \
                        help = 'base directory for data')
    
    args = parser.parse_args()

    return args


args = get_args_fpi()

doDebug = True

site, instrument = get_station_info_from_directory(doDebug)

if (doDebug):
    print(' -> site and instrument : ', \
          site['Abbreviation'], \
          instrument['Abbreviation'])

data_dir = './'
results_dir = './'

laserFiles = args.lasers

# These define the points to start and stop the analysis on the
# laser fringe pattern:
N = instrument['N']
N0 = instrument['N0']
N1 = instrument['N1']

# Reflectivity
R = instrument['default_params']['R']

# In Harding et al, these are I1 and I2:
a1 = instrument['default_params']['a1']
a2 = instrument['default_params']['a2']

# In Harding et al, these are image blur:
b0 = instrument['default_params']['b0']
b1 = instrument['default_params']['b1']
b2 = instrument['default_params']['b2']

# I don't know what this is, but it is true!
ESTIMATE_BLUR = True

# wavelength of the laser:
lam_laser = instrument['lam_laser']

# nominal instrument thickness:
last_t = instrument['nominal_t']

nTimes = len(laserFiles)
logdata = {'times': [],
           'Thickness' : np.zeros(nTimes),
           'DeltaThickness' : np.zeros(nTimes),
           'alpha' : np.zeros(nTimes),
           'Wavelength' : np.zeros(nTimes),
           'Reflectivity' : np.zeros(nTimes),
           'Intensity' : np.zeros(nTimes),
           'Baseline' : np.zeros(nTimes),
           'CenterX' : np.zeros(nTimes),
           'CenterY' : np.zeros(nTimes),
           'PeakToValley' : np.zeros(nTimes),
           'a1' : np.zeros(nTimes),
           'a2' : np.zeros(nTimes),
           'b0' : np.zeros(nTimes),
           'b1' : np.zeros(nTimes),
           'b2' : np.zeros(nTimes),
           'chi2' : np.zeros(nTimes),
           'rMin' : np.zeros(nTimes),
           'rMax' : np.zeros(nTimes),
           'nPoints' : np.zeros(nTimes),
           'ExposureTime' : np.zeros(nTimes),
           'CCD_Temperature' : np.zeros(nTimes)}

for iTime, laserFile in enumerate(laserFiles):
    print('Processing Laser : ', laserFile)
    data = FPI.ReadIMG(laserFile)
    image = np.asarray(data)
    (cx,cy) = FPI.FindCenter(image, max_r = 250)
    print('  -> center x, y : ', cx, cy)

    exposureTime = data.info['ExposureTime']
    ccdTemperature = data.info['CCDTemperature']

    annuli = FPI.FindEqualAreas(image, cx, cy, N)

    # Perform annular summation
    laser_spectra, laser_sigma = FPI.AnnularSum(image, annuli, 0)

    # Magnification Constant:
    alpha = \
        instrument['pix_size'] / \
        instrument['focal_length'] * \
        data.info['XBinning']
    # Intensity and background by looking at fringes
    I = laser_spectra.max() - laser_spectra.min()
    peakToValley = I
    print('  -> Laser peak to valley intensity : ', I)

    # CCD bias (background intensity):
    B = laser_spectra.min()
    laser_params = Parameters()
    laser_params.add('n',     value = 1.0,       vary = False)
    laser_params.add('t',     value = None,      vary = False) 
    laser_params.add('lam',   value = lam_laser, vary = False)
    laser_params.add('R',     value = R,       vary = False)
    laser_params.add('alpha', value = alpha,     vary = False)
    laser_params.add('I',     value = I,         vary = False)
    laser_params.add('B',     value = B,         vary = False)
    laser_params.add('a1',    value = a1,      vary = False)
    laser_params.add('a2',    value = a2,       vary = False)
    laser_params.add('b0',    value = b0,       vary = False)
    laser_params.add('b1',    value = b1,       vary = False)
    laser_params.add('b2',    value = b2,       vary = False)
    
    # To find a good initial guess for "t", we'll need to
    # do a grid search.  Search over 1 FSR around the last
    # solved-for value (or the nominal value, if this is
    # first trial).  TODO: make this grid search a
    # separate general function.
    def goodness(t):
        # return the correlation between the fringe with
        # this t and the data
        laser_params['t'].value = t
        fringe = FPI.Laser_FringeModel(laser_params, annuli['r'])
        top = np.dot(fringe, laser_spectra)
        bot = np.linalg.norm(fringe) * np.linalg.norm(laser_spectra)
        return top/bot

    nC = 51

    nTries = 4
    div = 2.0
    best_t = last_t
    #for iTry in range(nTries):
    iTry = 0
    while (div < 32.0):
        t_candidates = np.zeros(nC)
        dl = lam_laser / div / (nC-1)
        for i in range(nC):
            t_candidates[i] = best_t + (i - (nC-1)/2) * dl
        corrvec = np.array([goodness(t) for t in t_candidates])
        index = corrvec.argmax()
        best_t = t_candidates[index]

        if ((index < 5) or (index > nC*0.9)):
            div = div / 2
        else:
            div = div * 2
                
    laser_params['t'].value = best_t
    print('  -> Best (fit) thickness : ', best_t)

    order = [
        ['alpha'],\
        ['t','alpha'],\
        ['B','I'],\
        ['R'],\
        ['t','alpha','B','I','R','a1','a2'], \
    ]
    
    #order = [
    #    ['alpha'],\
    #    ['t','alpha'],\
    #    ['B','I'],\
    #    ['a1','a2'],\
    #    ['R'],\
    #    ['B','I'],\
    #    ['a1','a2'],\
    #    ['t','alpha'],\
    #    ['t','alpha','B','I','R','a1','a2'], \
    #]
    if ESTIMATE_BLUR:
        order.append(['t','alpha','B','I','R','a1','a2','b0','b1','b2'])
    
    measuredFringes = laser_spectra[N0:N1]
    sigma = laser_sigma[N0:N1]

    # set all the params in this group to "vary=True", and
    # then run inversion
                
    for group in order: 
        for param in list(laser_params.keys()):
            if param in group:
                laser_params[param].vary = True
            else:
                laser_params[param].vary = False
        # Set falloff terms to false, if this instrument
        # only has a couple fringes
        laser_fit = Minimizer(FPI.Laser_Residual, \
                              laser_params, \
                              fcn_args=(annuli['r'][N0:N1],), \
                              fcn_kws = {'data': measuredFringes, \
                                         'sigma': sigma}, \
                              scale_covar = True)
        result = laser_fit.leastsq()
        laser_params = result.params

    fringe = FPI.Laser_FringeModel(laser_params, annuli['r'])

    print('  -> chi2 : ', result.redchi)
    for param in result.params.keys():
        print('  ->  ', param, ' : ', result.params[param].value)
    fig = plt.figure(figsize=(6, 10))
    plt.rcParams.update({'font.size': 14})
    axB = fig.add_axes([0.1, 0.05, 0.85, 0.37])
    axB.plot(annuli['r']**2, fringe)
    axB.plot(annuli['r']**2, laser_spectra)
    axB.axvline(annuli['r'][N0]**2)

    ax1 = fig.add_axes([0.1, 0.45, 0.7, 0.7])

    m = np.mean(image)
    s = np.std(image)
    cax1 = ax1.pcolor(image, cmap = 'magma', vmin = m-3*s, vmax = m+3*s)
    cbar1 = fig.colorbar(cax1, ax = ax1, shrink = 0.5, pad=0.01)
    ax1.set_aspect(1.0)
    ax1.axhline(cy)
    ax1.axvline(cx)

    mini = np.min([np.min(fringe), np.min(laser_spectra)])
    maxi = np.max([np.max(fringe), np.max(laser_spectra)])
    
    dx = (maxi-mini) / 14.0
    iy = 0.0
    nX = 0
    maxNx = 7
    ix = maxi + dx * maxNx
    for param in result.params.keys():
        f = '%f'
        if (param == 'alpha'):
            f = '%10.3e'
        if (param == 'lam'):
            f = '%10.3e'
        if (param == 't'):
            f = '%12.10f'
        sParam = param + ': ' + f % result.params[param].value
        axB.text(iy, ix, sParam)
        ix = ix - dx
        nX = nX + 1
        if (nX == maxNx):
            ix = maxi + dx * maxNx
            nX = 0
            iy = np.max(annuli['r']**2)/2.0

    sParam = 'chi2: %f' % result.redchi
    axB.text(iy, ix, sParam)
            
    plotFile = laserFile + '.png'
    print('  --> Outputting plotfile : ', plotFile)
    fig.savefig(plotFile)
    plt.close(fig)

    logdata['times'].append(data.info['LocalTime'])
    logdata['Thickness'][iTime] = result.params['t'].value
    logdata['DeltaThickness'][iTime] = \
        result.params['t'].value - \
        np.floor(result.params['t'].value * 1000.0)/1000.0
    logdata['alpha'][iTime] = result.params['alpha'].value
    logdata['Wavelength'][iTime] = result.params['lam'].value
    logdata['Reflectivity'][iTime] = result.params['R'].value
    logdata['Intensity'][iTime] = result.params['I'].value
    logdata['Baseline'][iTime] = result.params['B'].value
    logdata['CenterX'][iTime] = cx
    logdata['CenterY'][iTime] = cy
    logdata['PeakToValley'][iTime] = peakToValley
    logdata['a1'][iTime] = result.params['a1'].value
    logdata['a2'][iTime] = result.params['a2'].value
    logdata['b0'][iTime] = result.params['b0'].value
    logdata['b1'][iTime] = result.params['b1'].value
    logdata['b2'][iTime] = result.params['b2'].value
    logdata['rMin'][iTime] = annuli['r'][0]
    logdata['rMax'][iTime] = annuli['r'][-1]
    logdata['nPoints'][iTime] = len(annuli['r'])
    logdata['chi2'][iTime] = result.redchi
    logdata['ExposureTime'][iTime] = exposureTime
    logdata['CCD_Temperature'][iTime] = ccdTemperature

message = os.getcwd()
logfile.write_log(logdata, \
                  fileHeader = site['Abbreviation'] + '_lasers', \
                  message = '')
