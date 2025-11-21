#!/usr/bin/env python

#import matplotlib
#matplotlib.use('AGG')

import numpy as np
import os
import datetime
import fpiinfo
import glob

# ----------------------------------------------------------------------
# This is for removing hotspots from the image, not used by default
# ----------------------------------------------------------------------

def remove_hotspots(image):

    newImage = image

    std = np.std(newImage)
    med = np.median(newImage)
    accMax = med + 10.0 * std
    actMax = np.max(newImage)

    if (accMax < actMax):
        print(' -> Removing hot pixels: ')
        print('   -> Acceptable max : ',accMax)
        print('   -> Actual max     : ',np.max(newImage))
        print('   -> replaced %d pixels' % len(newImage[newImage > accMax]))
        newImage[newImage > accMax] = med

    return newImage

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

# --------------------------------------------------------------------
# convert YYYYMMDD_HHMMSS to a datetime
# --------------------------------------------------------------------

def convert_string_to_datetime(sTime):
    ye = int(sTime[0:4])
    mo = int(sTime[5:7])
    da = int(sTime[8:10])
    hr = int(sTime[11:13])
    mi = int(sTime[14:16])
    se = int(sTime[17:19])
    time = datetime.datetime(ye, mo, da, hr, mi, se)
    return(time)
    
# --------------------------------------------------------------------
# Get the station and instrument descriptions from the current
# working directory.
# This assumes a directory name that is: /something/stat/YYYYMMDD
# if this is changed, this code needs to change.
# --------------------------------------------------------------------

def get_station_info_from_directory(doDebug = False):

    cwd = os.getcwd()

    if (doDebug):
        print(cwd)
    cwdArray = cwd.split('/')
    ymd = cwdArray[-1]
    site_name = cwdArray[-2]
    date = datetime.datetime(int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8]))
    
    # Import the site information
    site = fpiinfo.get_site_info(site_name, date)

    # These are UM systems. Need to edit this:
    if (site_name == 'abi'):
        instr_name = 'minime08'
    if (site_name == 'aak'):
        instr_name = 'minime08'
    if (site_name == 'kev'):
        instr_name = 'minime28'
    
    # Import the instrument information
    instrument = fpiinfo.get_instr_info(instr_name, date)

    return site, instrument

# --------------------------------------------------------------------
# Get a list of all of the laser images in a given directory
# --------------------------------------------------------------------

def get_all_laser_images(direc = '.', doDebug = False):
    '''
    Return all laser images in the specified directory, as a list of strings.
    Laser images are those of the following forms:

    L20091103001.img
    UAO_L_20091103_210000_001.img
    UAO_L_20221129_074314.hdf5

    Return empty list if none are found.
    '''

    fns_1 = glob.glob(direc + '/L[0-9]*.img')
    fns_2 = glob.glob(direc + '/*_L_*.img')
    fns_3 = glob.glob(direc + '/*_L_*.hdf5')

    return sorted(fns_1 + fns_2 + fns_3)


# --------------------------------------------------------------------
# Load the database for the given station
# --------------------------------------------------------------------

def load_database(datadir, site):
    
    dbFile = datadir + '/' + site + '/database/' + site + '_db.nc'
    if (not os.path.exists(dbFile)):
        print('Can not find database file. Creating empty DB.')
        lasers = {'times': [],
                  'cx' : [],
                  'cy' : []}
    else:
        # going to have to make a reader function here:
        db = {}
    return db


