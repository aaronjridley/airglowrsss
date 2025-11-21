#!/usr/bin/env python3

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import h5py
import numpy as np
import glob
import argparse
import FPI

# ----------------------------------------------------------------------
# Function to parse input arguments
# ----------------------------------------------------------------------

def get_args_fpi():

    parser = argparse.ArgumentParser(description =
                                     'Process FPI data ring pattern image')

    parser.add_argument('-files', nargs='+', \
                        help = 'list of files to process')
    
    parser.add_argument('-brightness',  default = False,
                        action = 'store_true',
                        help = 'Make Brightness Summary Plot')
    
    parser.add_argument('-x', default = 255, type = int, \
                        help = 'center x position guess')
    
    parser.add_argument('-y', default = 255, type = int, \
                        help = 'center y position guess')
    
    parser.add_argument('-s', default = 5, type = int, \
                        help = 'search positions for the center')
    
    args = parser.parse_args()

    return args

def remove_hotspots(image):

    newImage = image

    std = np.std(newImage)
    med = np.median(newImage)
    accMax = med + 10.0 * std
    actMax = np.max(newImage)
    print('Acceptable max : ',accMax)
    print('Actual max     : ',np.max(newImage))

    if (accMax < actMax):
        print('Replacing hot spot pixels...')
        newImage[newImage > accMax] = med

    print(np.min(newImage), np.mean(newImage), np.median(newImage), np.max(newImage))

    return newImage

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
    for i in range(hw+1,nr):
        integral[i] = np.mean(image[(r >= i-hw) & (r <= i+hw) ])

    return integral

def find_peak_and_valley(array1d):

    iMax_ = np.argmax(array1d)

    iMin_ = iMax_ + 1
    if (iMin_ < len(array1d)-3):
        while ((array1d[iMin_] < array1d[iMin_ - 1]) and (iMin_ < len(array1d))):
            iMin_ += 1

    iMax2_ = np.argmax(array1d[iMin_:]) + iMin_

    # find real min:
    iMin_ = iMax_ + np.argmin(array1d[iMax_:iMax2_])

    # this equation causes the gradient to be positive:
    # grad = (array1d[iMax_] - array1d[iMin_]) / (iMin_ - iMax_)

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


# --------------------------------------------------------------------
# Main Code:
#  
# --------------------------------------------------------------------

args = get_args_fpi()
filelist = args.files

times = []
peak_to_valley = []

for file in filelist:

    print('Processing File : ', file)
    if ('_L_' in file):
        type = 'laser'
    else:
        type = 'sky'
    data = FPI.ReadIMG(file)

    print('  -> Outside Temp : ', data.info['OutsideTemperature'])
    
    time = data.info['LocalTime']
    times.append(time)
    array = np.asarray(data)
    better = remove_hotspots(array)

    fig = plt.figure(figsize = (8,10))
    ax1 = fig.add_axes([0.2, 0.02, 0.65, 0.65])
    ax2 = fig.add_axes([0.07, 0.80, 0.87, 0.18])
    ax3 = fig.add_axes([0.07, 0.60, 0.87, 0.18])

    m = np.mean(better)
    s = np.std(better)
    cax = ax1.pcolor(better, cmap = 'plasma', vmin = m-2*s, vmax = m+2*s)
    ax1.set_aspect(1.0)

    cCx = args.x
    cCy = args.y
    delta = args.s
    dyMax = -100.0
    iCx_ = -1
    iCy_ = -1

    for cx in range(cCx - delta, cCx + delta + 1):
        for cy in range(cCy - delta, cCy + delta + 1):
            integral = integrate_radially(array, cx, cy, 0)
            #ax2.plot(integral, linewidth = 0.5, alpha = 0.2, color = 'grey')
            iMax_, iMin_, fwhm = find_peak_and_valley(integral)
            #print('Peak :', iMax_, integral[iMax_])
            #print('Valley :', iMin_, integral[iMin_])
            dx = (iMin_ - iMax_)
            dy = (integral[iMax_] - integral[iMin_])
            if (dy > dyMax):
                dyMax = dy
                iCx_ = cx
                iCy_ = cy
                print('   --> Found better center : ', cx, cy, dx, dy, fwhm)


    print(' -> Final center : ', iCx_, iCy_, dyMax)
    peak_to_valley.append(np.percentile(array, 98) - np.percentile(array, 2))
    integral = integrate_radially(array, iCx_, iCy_, 0)
    r = create_radial_array(array, iCx_, iCy_)
    ax2.plot(integral, linewidth = 1.5, color = 'k')

    ax3.plot(r[iCx_:,iCy_], array[iCx_:,iCy_], linewidth = 1.5, color = 'r')
    ax3.plot(np.flip(r[:iCx_+1,iCy_]), np.flip(array[:iCx_+1,iCy_]), linewidth = 1.5, color = 'r')

    ax3.plot(r[iCx_,iCy_:], array[iCx_,iCy_:], linewidth = 1.5, color = 'b')

    ax1.axhline(iCy_)
    ax1.axvline(iCx_)

    ymdhms = time.strftime('%Y%m%d_%H%M%S')
    outfile = type + '_' + ymdhms + '.png'
    print('Writing file : ' + outfile)
    plt.savefig(outfile)
    plt.close()

fig = plt.figure(figsize = (10,8))
ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax1.plot(times, peak_to_valley)
outfile = type + '_summary.png'
print('Writing file : ' + outfile)
plt.savefig(outfile)
plt.close()
                          
