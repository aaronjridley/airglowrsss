#!/usr/bin/env python

# Subtract a background and average lasers

import matplotlib
matplotlib.use('AGG')

import FPI
import fpiinfo
import argparse
import datetime
import numpy as np
import os
import h5py

# ----------------------------------------------------------------------
# Function to parse input arguments
# ----------------------------------------------------------------------

def get_args_fpi():

    parser = argparse.ArgumentParser(description =
                                     'Process FPI data ring pattern image')

    parser.add_argument('filelist', nargs='+', \
                        help = 'list files to use for generating plots')

    parser.add_argument('-x', default = 255, type = int, \
                        help = 'center x position')
    
    parser.add_argument('-y', default = 255, type = int, \
                        help = 'center y position')
    
    parser.add_argument('-s', default = 5, type = int, \
                        help = 'search positions')
    
    parser.add_argument('-smooth', default = 5, type = int, \
                        help = 'number of lasers to smooth over')
    
    args = parser.parse_args()

    return args

# ----------------------------------------------------------------------
# save all of the data in a new directory
# ----------------------------------------------------------------------

def save_new_image(oldfilename, dirIn, imageData, newImage):

    completeFilename = dirIn + '/' + oldfilename
    print('Saving copy here : ', completeFilename)
    data_file = h5py.File(completeFilename, 'w')
    f = data_file.create_dataset("image", data = newImage)
    for key in imageData.attrs.keys():
        f.attrs[key] = imageData.attrs[key]
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

# ----------------------------------------------------------------------
# 
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# 
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# 
# ----------------------------------------------------------------------

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

# ----------------------------------------------------------------------
# 
# ----------------------------------------------------------------------

args = get_args_fpi()

filelist = args.filelist

if (args.smooth % 2 == 0):
    print('Smooth needs to be odd.')
    exit()
nSmooth = int((args.smooth - 1) / 2)

nominal_dt = datetime.datetime.now()
site_name = 'aak'
instr_name = 'minime08'
# Import the site information
site_name = fpiinfo.get_site_of(instr_name, nominal_dt)
site = fpiinfo.get_site_info(site_name, nominal_dt)
# Import the instrument information
instrument = fpiinfo.get_instr_info(instr_name, nominal_dt)

nFiles = len(filelist)

cx = args.x
cy = args.y
delta = args.s

allCompleteImages = []
allCenters = []
allLaserImages = []
allTimes = []
allExposures = []
for file in filelist:

    print('Reading and storing file : ', file)

    d = FPI.ReadIMG(file)
    allCompleteImages.append(d)
    img = np.asarray(d)
    allLaserImages.append(img)
    allTimes.append(d.info['LocalTime'])
    allExposures.append(d.info['ExposureTime'])

directory = 'integrated_lasers'
if (not os.path.exists(directory)):
    command = 'mkdir ' + directory
    print('Running command : ', command)
    os.system(command)

fpCenters = open(directory + '/centers.csv', 'w')
fpCenters.write('file,year,month,day,hour,minute,second,cx,cy,slope,intercept\n')
    
allIntegrated = []
for iImage in range(nFiles):
    iStart_ = iImage - nSmooth
    if (iStart_ < 0):
        iStart_ = 0
    iEnd_ = iStart_ + (nSmooth * 2 + 1)
    if (iEnd_ >= nFiles):
        iEnd_ = nFiles-1
        iStart_ = iEnd_ - (nSmooth * 2 + 1)
    print(iStart_, iEnd_, nFiles)
    exposure = 0.0
    for i in range(iStart_, iEnd_+1):
        exposure += allExposures[i]
        if (i == iStart_):
            allIntegrated.append(allLaserImages[i])
        else:
            allIntegrated[-1] = \
                allIntegrated[-1] + \
                allLaserImages[i]
    img = allIntegrated[-1]
    mcx, mcy = find_center_ajr(img, cx, cy, delta)
    cx = mcx
    cy = mcy
    sTime = allTimes[iImage].strftime('%Y,%m,%d,%H,%M,%S,')
    fpCenters.write(filelist[iImage] + ',' + sTime)
    sCenters = '%5.2f,%5.2f\n' % (cx, cy)
    fpCenters.write(sCenters)


    img1d, r1d = integrate_radially(img, mcx, mcy, 0)
    r1d2 = r1d * r1d
    iMax_, iMin_, fwhm = find_peak_and_valley(img1d)
    valInner = np.mean(img1d[iMin_ - 2 : iMin_ + 3])
    rInner = np.mean(r1d2[iMin_ - 2 : iMin_ + 3])
    valOuter = np.mean(img1d[-5:])
    rOuter = np.mean(r1d2[-5:])

    slope = (valOuter - valInner) / (rOuter - rInner)
    inter = valOuter - slope * rOuter

    r2d = create_radial_array(img, mcx, mcy)
    r2d2 = r2d * r2d
    background = r2d2 * slope + inter

    img = img - background

    normImage = background / exposure

    directory = 'preprocessed'
    if (not os.path.exists(directory)):
        command = 'mkdir ' + directory
        print('Running command : ', command)
        os.system(command)

    imageData = allCompleteImages[iImage]
    save_new_image(filelist[iImage], directory, imageData, img)

    directory = 'backgrounds'
    if (not os.path.exists(directory)):
        command = 'mkdir ' + directory
        print('Running command : ', command)
        os.system(command)
    save_background_image(site_name, \
                          normImage, \
                          allTimes[iImage], \
                          directory)

    
fpCenters.close()
