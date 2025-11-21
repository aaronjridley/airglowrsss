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

    parser.add_argument('-lasers', nargs='+', \
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
    
    parser.add_argument('-images',  default = False,
                        action = 'store_true',
                        help = 'Save images of raw lasers')

    parser.add_argument('-save',  default = False,
                        action = 'store_true',
                        help = 'Save background-subtracted data')

    parser.add_argument('-x', default = 255, type = int, \
                        help = 'center x position guess')
    
    parser.add_argument('-y', default = 255, type = int, \
                        help = 'center y position guess')
    
    parser.add_argument('-s', default = 5, type = int, \
                        help = 'search positions for the center')
    
    parser.add_argument('-n0', default = 45, type = int, \
                        help = 'n0 for the instrument fit')
    
    parser.add_argument('-datadir', default = '/backup/Data/Fpi', \
                        help = 'base directory for data')
    
    args = parser.parse_args()

    return args



# --------------------------------------------------------------------
# Main Code:
#  
# --------------------------------------------------------------------

args = get_args_fpi()

doDebug = True

site, instrument = get_station_info_from_directory(doDebug)

print('site and instrument : ', site['Abbreviation'], instrument['Abbreviation'])

#db = load_database(args.datadir, site)

laserFiles = args.lasers

if (laserFiles == None):
    laserFiles = get_all_laser_images()
else:
    if (len(laserFiles) == 1):
        # hmmmm.... this may not be right
        if (not os.path.exists(laserFiles[0])):
            print('Can not file file : ', laserFiles)
            exit()

if (args.last):
    laserFiles = [laserFiles[-1]]

cx = args.x
cy = args.y
delta = args.s

times = []
centerXs = []
centerYs = []
nXs = []
nYs = []
slopes = []
intercepts = []
brightnessFromHist = []
brightnessFromFringe = []
nFiles = len(laserFiles)

for iFile, file in enumerate(laserFiles):

    print('Analyzing File : ', file, \
          '(number : %i of %i)' % (iFile + 1, nFiles))

    d = FPI.ReadIMG(file)
    times.append(d.info['LocalTime'])
    img = np.asarray(d)
    nx, ny = np.shape(img)

    cleanImage = remove_hotspots(img)
    
    mcx, mcy = find_center_ajr(cleanImage, cx, cy, delta)
    (tcx,tcy) = FPI.FindCenter(img, max_r = 250)
    print(' -> Center comparison, x/x/dx, y/y/dy: ', \
          int(mcx), int(tcx), mcx-tcx, \
          int(mcy), int(tcy), mcy-tcy)

    centerXs.append(mcx)
    centerYs.append(mcy)
    nXs.append(nx)
    nYs.append(ny)
    
    img1d, r1d = integrate_radially(cleanImage, mcx, mcy, 0)
    r1d2 = r1d * r1d
    iMax_, iMin_, fwhm = find_peak_and_valley(img1d)
    valInner = np.mean(img1d[iMin_ - 2 : iMin_ + 3])
    rInner = np.mean(r1d2[iMin_ - 2 : iMin_ + 3])
    valOuter = np.mean(img1d[-5:])
    rOuter = np.mean(r1d2[-5:])

    slopes.append((valOuter - valInner) / (rOuter - rInner))
    intercepts.append(valOuter - slopes[-1] * rOuter)

    peak2valley = np.percentile(cleanImage,98) - np.percentile(cleanImage,2)
    brightnessFromHist.append(peak2valley)
    brightnessFromFringe.append(img1d[iMax_] - img1d[iMin_])

    print(' -> Brightness values : ', \
          brightnessFromHist[-1], brightnessFromFringe[-1])
    
    if (args.images):
        fig = plt.figure(figsize=(6, 10))
        plt.rcParams.update({'font.size': 14})
        ax1 = fig.add_axes([0.1,0.05,0.9,0.5])
        ax2 = fig.add_axes([0.1,0.6,0.85,0.35])

        m = np.mean(cleanImage)
        s = np.std(cleanImage)
        cax1 = ax1.pcolor(cleanImage, cmap = 'magma', vmin = m-3*s, vmax = m+3*s)
        cbar1 = fig.colorbar(cax1, ax = ax1, shrink = 0.5, pad=0.01)
        ax1.set_aspect(1.0)
        ax1.axhline(mcy)
        ax1.axvline(mcx)

        ax2.plot(r1d, img1d)
        x = [iMin_, iMax_]
        y = img1d[x]
        ax2.scatter(x, y)
        
        fringeFile = times[-1].strftime('fringe_%Y%m%d_%H%M%S.png')
        print('  --> Outputting plotfile : ', fringeFile)
        fig.savefig(fringeFile)
        plt.close(fig)
    
        
if (args.brightness):
    
    fig = plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})
    ax = fig.add_axes([0.1,0.1,0.85,0.85])
    ax.plot(times, brightnessFromHist, label = 'From Hists')
    ax.plot(times, brightnessFromFringe, label = 'From Fringes')
    ax.set_ylabel('Brightness')
    sTime = times[0].strftime('%b %d, %Y %H:%M UT') + ' - ' + \
        times[-1].strftime('%b %d, %Y %H:%M UT Hours')
    ax.set_xlabel(sTime)
    ax.legend()
    brightFile = times[-1].strftime('bright_%Y%m%d.png')
    print('  --> Outputting plotfile : ', brightFile)
    fig.savefig(brightFile)
    plt.close(fig)
    
