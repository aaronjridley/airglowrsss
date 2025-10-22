#!/usr/bin/env python

import numpy as np

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
# create radial array from nX and nY - used in center finding
# ----------------------------------------------------------------------

def create_radial_array_nxny(cx, cy, nX, nY):
    x = np.linspace(0, nX, nX)
    y = np.linspace(0, nY, nY)
    x2d, y2d = np.meshgrid(x, y)
    r = np.sqrt((x2d-cx)**2 + (y2d-cy)**2)
    return r


# ----------------------------------------------------------------------
# Find Mean Values image of each radial distance from a center point
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
# Given an array, find the first peak and the subsequent valley
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
# This function finds the center of an image by:
# 1. guessing the center
# 2. Creating mean values of the image as a function of radius
# 3. Finding the peak to valley difference
# 4. Refining the center until the max peak-to-valley difference is reached
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
    if (delta > 0):
        delta = 0.5

    for cx in np.arange(iCxS_ - delta, iCxS_ + delta, 0.1):
        for cy in np.arange(iCyS_ - delta, iCyS_ + delta, 0.1):
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


    integral, rdummy = integrate_radially(array, iCx_, iCy_, 0)
    iMax_, iMin_, fwhm = find_peak_and_valley(integral)
    dy = (integral[iMax_] - integral[iMin_])
                    
    print(' -> Final center : ', iCx_, iCy_, dyMax, fwhm)

    return iCx_, iCy_
