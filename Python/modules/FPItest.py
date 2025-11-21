#!/usr/bin/env python3

# Functions to process FPI data on remote2

#import matplotlib
#matplotlib.use('AGG')

#import FPI
import fpiinfo
import datetime
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import argparse
import h5py

##from optparse import OptionParser
##import BoltwoodSensor
##import X300Sensor
#import FPIDisplay
#import pytz
#import multiprocessing
#import subprocess
#import traceback
#from matplotlib import dates
#import shutil

from laser_routines import *

# ----------------------------------------------------------------------
# Function to parse input arguments
# ----------------------------------------------------------------------

def get_args_fpi():

    parser = argparse.ArgumentParser(description =
                                     'Process FPI data ring pattern image')

    parser.add_argument('filelist', nargs='+', \
                        help = 'list files to use for generating plots')

    parser.add_argument('-subtract', \
                        help='subtract background', \
                        action="store_true")
    
    parser.add_argument('-save',  default = False,
                        action = "store_true",
                        help = 'Save background-subtracted data')

    parser.add_argument('-x', default = 255, type = int, \
                        help = 'center x position')
    
    parser.add_argument('-y', default = 255, type = int, \
                        help = 'center y position')
    
    parser.add_argument('-s', default = 5, type = int, \
                        help = 'search positions')
    
    parser.add_argument('-n0', default = -1, type = int, \
                        help = 'search positions')
    
    args = parser.parse_args()

    return args

# ----------------------------------------------------------------------
# save all of the data in a new directory
# ----------------------------------------------------------------------

def save_new_image(oldfilename, dirIn, imageData):

    completeFilename = dirIn + '/' + oldfilename
    print('Saving copy here : ', completeFilename)
    data_file = h5py.File(completeFilename, 'w')
    img = np.asarray(imageData)
    f = data_file.create_dataset("image", data = img)
    for key in imageData.attrs.keys():
        f.attrs[key] = imageData.attrs[keys]
        print(key)
    data_file.close()

    return

# ----------------------------------------------------------------------
# save the background.  This is normalized by the exposure!
# ----------------------------------------------------------------------

def save_background_image(site, image, time, directory):
    
    stime = time.strftime('_BG_%Y%m%d_%H%M%S')
    fileout = directory + '/' + site + stime + '.hdf5'
    data_file = h5py.File(fileout, 'w')
    f = data_file.create_dataset("image", data = image)
    f.attrs['year'] = time.year
    f.attrs['month'] = time.month
    f.attrs['day'] = time.day
    f.attrs['hour'] = time.hour
    f.attrs['minute'] = time.minute
    f.attrs['second'] = time.second
    data_file.close()
    return

# ----------------------------------------------------------------------
# create radial array - used in center finding
# ----------------------------------------------------------------------

def create_radial_array(image, cx, cy):
    nx, ny = np.shape(image)
    x = np.linspace(0, nx, nx)
    y = np.linspace(0, ny, ny)
    x2d, y2d = np.meshgrid(x, y)
    r = np.sqrt((x2d-cx)**2 + (y2d-cy)**2)
    return r

# hw = half-width in pixels
def integrate_radially(image, cx, cy, hw):
    r = np.round(create_radial_array(image, cx, cy))
    nx, ny = np.shape(image)

    nr = int(nx/2)
    integral = np.zeros((nr)) + np.mean(image)
    rMean = np.zeros((nr))
    for i in range(hw+1,nr):
        integral[i] = np.mean(image[(r >= i-hw) & (r <= i+hw) ])
        rMean[i] = np.mean(r[(r >= i-hw) & (r <= i+hw) ])

    return integral, rMean


def find_peak_and_valley(array1d):

    iMax_ = np.argmax(array1d)

    if (iMax_ > len(array1d) - 10):
        iMax_ = len(array1d) - 1
        iMin_ = 1
        fwhm = iMax_/2
        return iMax_, iMin_, fwhm
    
    iMin_ = iMax_ + 5
    while (array1d[iMin_] < array1d[iMin_ - 1]):
        iMin_ += 1

    iMax2_ = np.argmax(array1d[iMin_:]) + iMin_

    # find real min:
    iMin_ = iMax_ + np.argmin(array1d[iMax_:iMax2_])

    # Find full width at half max:
    halfMax = array1d[iMax_] - 0.6 * (array1d[iMax_] - array1d[iMin_])
    iRight_ = iMax_ + 1
    iLeft_ = iMax_ - 1
    while (array1d[iRight_] > halfMax):
        iRight_ += 1
    while (array1d[iLeft_] > halfMax):
        iLeft_ -= 1
    fwhm = iRight_ - iLeft_

    return iMax_, iMin_, fwhm


def find_center_ajr(array, cCx, cCy, delta):

    iCx_ = cCx
    iCy_ = cCy
    dyMax = -100.0
    fwhm_ = 1000.0
    isVerbose = True

    peak2valley = np.percentile(array,98) - np.percentile(array,2)
    if (peak2valley < 20.0):
        return iCx_, iCy_
    
    for cx in np.arange(cCx - delta, cCx + delta + 1):
        for cy in np.arange(cCy - delta, cCy + delta + 1):
            integral, rdummy = integrate_radially(array, cx, cy, 0)
            iMax_, iMin_, fwhm = find_peak_and_valley(integral)
            dx = (iMin_ - iMax_)
            dy = (integral[iMax_] - integral[iMin_])
            if (dy > dyMax):
                dyMax = dy
                iCx_ = cx
                iCy_ = cy
                fwhm_ = fwhm
                if (isVerbose):
                    print('  -> Found better center : ', cx, cy, dx, dy, fwhm)

    # Refine Further!
    iCxS_ = iCx_
    iCyS_ = iCy_
    for cx in np.arange(iCxS_ - 0.5, iCxS_ + 0.5, 0.1):
        for cy in np.arange(iCyS_ - 0.5, iCyS_ + 0.5, 0.1):
            integral, rdummy = integrate_radially(array, cx, cy, 0)
            iMax_, iMin_, fwhm = find_peak_and_valley(integral)
            dx = (iMin_ - iMax_)
            dy = (integral[iMax_] - integral[iMin_])
            if (dy > dyMax):
                dyMax = dy
                iCx_ = cx
                iCy_ = cy
                fwhm_ = fwhm
                if (isVerbose):
                    print('  -> Found better center : ', \
                          cx, cy, dx, dy, fwhm)

    print(' -> Final center : ', iCx_, iCy_, dyMax, fwhm)

    return iCx_, iCy_

# -------------------------------------------------------------------
# Make a label that shows the whole time range
# -------------------------------------------------------------------

def get_label_time_range(times):
    startTime = times[0]
    endTime = times[-1]
    label = startTime.strftime('%b %d, %Y %H:%M UT') + ' - ' + \
        endTime.strftime('%b %d, %Y %H:%M UT (Hours)')
    return label

# -------------------------------------------------------------------
# This gives the axes for multi-row, but single column plots.
# Inputs:
#    - figIn: this is the figure (e.g., fig = plt.figure(figsize=(10, 10)))
#    - nPlots: number of plots in total
#    - yBottom: space desired below the bottom plot
#    - yTop: space desired above the top plot
#    - yBuffer: space desired between each plot
# Optional Inputs:
#    - xLeft: space desired to the left of the plots
#    - xRight: space desired to the right of the plots
# Outputs:
#    - ax: array of axes
# -------------------------------------------------------------------
    
def get_axes_one_column(figIn,
                        nPlots,
                        yBottom,
                        yTop,
                        yBuffer,
                        xLeft = 0.11,
                        xRight = 0.02):
    
    # plot size in y direction:
    ySize = (1.0 - yBottom - yTop) / nPlots - yBuffer * (nPlots - 1) / nPlots
    xSize = 1.0 - xLeft - xRight
    
    ax = []
    for iPlot in range(nPlots):
        # I want to reverse this, so that 0 is the top plot:
        i = nPlots - iPlot - 1
        ax.append(fig.add_axes([xLeft,
                                yBottom + i * (ySize + yBuffer),
                                xSize,
                                ySize]))
    
    return ax


args = get_args_fpi()

filelist = args.filelist

nominal_dt = datetime.datetime.now()

isFound = False
stat = ''

cwd = os.getcwd()
m = re.match(r'.*/(.*)/\d*',cwd)
if m:
    stat = m.group(1)
    isFound = True

if (isFound):
    isFound = False
    if (stat == 'ann02'):
        instr_name = 'minime28'
        site_name = 'an2'
        isFound = True

    if (stat == 'ann03'):
        instr_name = 'minime08'
        site_name = 'ann'
        isFound = True

    if (stat == 'aak'):
        instr_name = 'minime08'
        site_name = 'aak'
        isFound = True

isFound = True
instr_name = 'minime08'
site_name = 'abi'
instr_name = 'minime28'
site_name = 'kev'
subtractBackground = False
if (args.subtract):
    subtractBackground = True

iFilesStart_ = 0
nFilesMax = iFilesStart_ + 10

if (isFound):
    print('From directory : ', stat, '; site_name = ', site_name)

else:
    print('Directory is : ->', cwd)
    print('Can not file the station name in this.')
    exit()

# Import the site information
site_name = fpiinfo.get_site_of(instr_name, nominal_dt)
site = fpiinfo.get_site_info(site_name, nominal_dt)
# Import the instrument information
instrument = fpiinfo.get_instr_info(instr_name, nominal_dt)

nFiles = len(filelist)

nSave = []
rmss = []
brightnesses = []
times = []
cxs = []
cys = []
N = 500
slopes = []
inters = []

maxi = -1.0
mini = 1e32
minis = []
maxis = []
means = []
temps = []
skyTemps = []
groundTemps = []

def convert_string_to_datetime(sTime):
    ye = int(sTime[0:4])
    mo = int(sTime[5:7])
    da = int(sTime[8:10])
    hr = int(sTime[11:13])
    mi = int(sTime[14:16])
    se = int(sTime[17:19])
    time = datetime.datetime(ye, mo, da, hr, mi, se)
    return(time)
    

for iFile, file in enumerate(filelist):

    print('Grabbing max and min from : ', file)
    d = FPI.ReadIMG(file)
    skyTemps.append(d.attrs['OutsideTemperature (C)'])
    groundTemps.append(d.attrs['AmbientTemperature (C)'])
    temps.append(d.attrs['CCDTemperature'])

    img = np.asarray(d)
    peak2valley = np.percentile(img,98) - np.percentile(img,2)
    minis.append(np.percentile(img, 2))
    maxis.append(np.percentile(img, 99))
    means.append(np.mean(img))
    
    maxi = np.max([maxi, np.percentile(img,99) * 1.1])
    mini = np.min([mini, np.percentile(img,2) * 0.9])
    times.append(d.info['LocalTime'])
    
    dir = 'preprocessed'
    if (args.save):
        save_new_image(file, dir, d)

fig = plt.figure(figsize=(10, 10))
plt.rcParams.update({'font.size': 14})

nPlots = 2
y0 = 0.07
yT = 0.01
yB = 0.04
ax = get_axes_one_column(fig, nPlots, y0, yT, yB)

ax[0].plot(times, temps, label = 'CCD Temp')
ax[0].plot(times, skyTemps, label = 'Sky Temp')
ax[0].plot(times, groundTemps, label = 'Ground Temp')
ax[0].set_ylabel('Temperature (C)')
ax[0].legend()

ax[1].plot(times, minis, label = 'Min')
ax[1].plot(times, means, label = 'Mean')
ax[1].plot(times, maxis, label = 'Max')
ax[1].legend()
ax[1].set_ylabel('Image Intensity Limits')

xLabel = get_label_time_range(times)
ax[1].set_xlabel(xLabel)

plotFile = 'summary.png'
print('  --> Outputting plotfile : ', plotFile)
fig.savefig(plotFile)
plt.close(fig)

cx = args.x
cy = args.y
delta = args.s

for iFile, file in enumerate(filelist):

    print('Analyzing File : ', file, '(number : %i of %i)' % (iFile, nFiles))

    d = FPI.ReadIMG(file)
    img = np.asarray(d)
    
    #(cx, cy) = FPI.FindCenter(img, max_r = 250, \
    #                          circle_fit_method = 'geometric')
    #mcx = cx
    #mcy = cy
    mcx, mcy = find_center_ajr(img, cx, cy, delta)

    if (subtractBackground):
        img1d, r1d = integrate_radially(img, mcx, mcy, 0)
        r1d2 = r1d * r1d
        iMax_, iMin_, fwhm = find_peak_and_valley(img1d)
        valInner = np.mean(img1d[iMin_ - 2 : iMin_ + 3])
        rInner = np.mean(r1d2[iMin_ - 2 : iMin_ + 3])
        valOuter = np.mean(img1d[-5:])
        rOuter = np.mean(r1d2[-5:])

        slope = (valOuter - valInner) / (rOuter - rInner)
        inter = valOuter - slope * rOuter

        slopes.append(slope)
        inters.append(inter)

        r2d = create_radial_array(img, mcx, mcy)
        r2d2 = r2d * r2d
        background = r2d2 * slope + inter

        img = img - background

        normImage = background / d.attrs['ExposureTime']

        directory = 'backgrounds'
        if (not os.path.exists(directory)):
            command = 'mkdir ' + directory
            print('Running command : ', command)
            os.system(command)
        save_background_image(site_name, \
                              normImage, \
                              times[iFile], \
                              directory)

    # use current center as guess for next:
    cx = mcx
    cy = mcy
    
    #print('old, new x : ', cx, mcx)
    #print('old, new y : ', cy, mcy)
    
    cxs.append(mcx)
    cys.append(mcy)
    print(' -> Center located : ', mcx, mcy)
    peak2valley = np.percentile(img,98) - np.percentile(img,2)
    print(' -> Brightness difference : ', peak2valley)
    fringeFile = times[iFile].strftime('fringe_%Y%m%d_%H%M%S.png')

    annuli = FPI.FindEqualAreas(img, mcx, mcy, N)
    # Perform annular summation
    laser_spectra, laser_sigma = FPI.AnnularSum(img, annuli, 0)

    # Add this so we can calculate Alpha in the functions called below:
    instrument['XBinning'] = d.info['XBinning']

    if (args.n0 > 0):
        N_0 = args.n0
        rms = 0.0
        N_0, rms = find_optimum_N0(laser_spectra, \
                                   laser_sigma, \
                                   instrument, \
                                   annuli, \
                                   initial_guess = args.n0, \
                                   finalN = -1)
    else:
        if (len(nSave) > 0):
            if (nSave[-1] < 5):
                N0 = 0
            else:
                N0 = int(nSave[-1] * 0.75)
        else:
            N0 = 25
        N_0, rms = find_optimum_N0(laser_spectra, \
                                   laser_sigma, \
                                   instrument, \
                                   annuli, \
                                   initial_guess = N0, \
                                   finalN = N)
        
    nSave.append(N_0)
    rmss.append(rms)
    brightnesses.append(peak2valley)

    # Perform the fit to the best fringe pattern parameters:
    result = fit_laser(instrument, annuli, laser_spectra, laser_sigma, N0 = N_0, N1 = N)
    laser_params = result.params

    # Create a fringe pattern based on the results:
    fringe = FPI.Laser_FringeModel(laser_params, annuli['r'])
    diff = np.sqrt(np.mean((fringe[N_0:-1] - laser_spectra[N_0:-1])**2))
    
    fig = plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 14})
    ax1 = fig.add_axes([0.09,0.63,0.9,0.34])
    ax2 = fig.add_axes([0.09,0.08,0.5,0.5])
    
    ax1.plot(laser_spectra)
    ax1.plot(fringe)
    ax1.set_title('Filename : ' + file)
    ax1.text(N_0 + 2, mini + 0.05 * (maxi - mini), 'RMS Diff: %5.1f, N0: %i' % (diff, N_0))
    #ax1.set_ylim(mini, maxi)
    ax1.axvline(N_0)

    m = np.mean(img)
    s = np.std(img)
    print(m, s)
    cax = ax2.pcolor(img, cmap = 'bwr', vmin = m-3*s, vmax = m+3*s)
    ax2.set_aspect(1.0)
    ax2.axhline(mcy)
    ax2.axvline(mcx)

    
    print('  --> Outputting plotfile : ', fringeFile)
    fig.savefig(fringeFile)
    plt.close(fig)
    

N_0 = int(np.median(nSave))
print('Mean N_0 : ', N_0)

nSave = np.array(nSave)
rmss = np.array(rmss)
brightnesses = np.array(brightnesses)

# Perform the fit to the best fringe pattern parameters:
result = fit_laser(instrument, annuli, laser_spectra, laser_sigma, N0 = N_0, N1 = N)
laser_params = result.params

# Create a fringe pattern based on the results:
fringe = FPI.Laser_FringeModel(laser_params, annuli['r'])

diff = np.sqrt(np.mean((fringe[N_0:-1] - laser_spectra[N_0:-1])**2))
        
fig = plt.figure(figsize=(10, 10))
plt.rcParams.update({'font.size': 14})

y0 = 0.04
yS = 0.27
yB = 0.04
if (subtractBackground):
    nPlots = 5
else:
    nPlots = 3
ax = get_axes_one_column(fig, nPlots, y0, yT, yB)

ax[0].plot(times, rmss / brightnesses * 100.0 * 10, label = 'RMS * 10 (norm)')
ax[0].plot(times, nSave, label = 'N0')
ax[0].set_ylabel('RMS + N0')
ax[0].legend()

ax[1].plot(times, brightnesses)
ax[1].set_ylabel('Laser Brightness')

ax[2].plot(times, cxs, label = 'CX')
ax[2].plot(times, cys, label = 'CY')
ax[2].set_ylabel('Centers')
ax[2].legend()

if (subtractBackground):
    ax[3].plot(times, slopes)
    ax[3].set_ylabel('Slopes')

    ax[4].plot(times, inters)
    ax[4].set_ylabel('Intercepts')

plotFile = 'test.png'
print('  --> Outputting plotfile : ', plotFile)
fig.savefig(plotFile)
plt.close(fig)

