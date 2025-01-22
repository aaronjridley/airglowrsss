#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import argparse

# ----------------------------------------------------------------------
# Function to parse input arguments
# ----------------------------------------------------------------------

def get_args_fpi():

    parser = argparse.ArgumentParser(description =
                                     'Make a simple summary plot of FPI data')
    
    parser.add_argument('filelist', nargs='+', \
                        help = 'list files to use for generating plots')
    
    #parser.add_argument('-x', default = 255, type = int, \
    #                    help = 'center x position')
    #
    #parser.add_argument('-log', \
    #                    help='make a log file of centers', \
    #                    action="store_true")

    args = parser.parse_args()

    return args


args = get_args_fpi()

files = args.filelist

for npzname in files:

    # Load the results
    npzfile = np.load(npzname, allow_pickle=True)
    FPI_Results = npzfile['FPI_Results']
    FPI_Results = FPI_Results.reshape(-1)[0]
    site = npzfile['site']
    site = site.reshape(-1)[0]
    abb = site['Abbreviation']
    siteName = site['Name'] + ' (' + abb + ')'
    del npzfile.f # http://stackoverflow.com/questions/9244397/memory-overflow-when-using-numpy-load-in-a-loop
    npzfile.close()

    losTimes = FPI_Results['sky_times']
    losWind = FPI_Results['LOSwind']
    sigWindCal = FPI_Results['sigma_cal_LOSwind']
    sigWindFit = FPI_Results['sigma_fit_LOSwind']
    sigWind = np.sqrt(sigWindCal**2 + sigWindFit**2)
    losTemp = FPI_Results['T']
    sigTemp = FPI_Results['sigma_T']

    laserTimes = FPI_Results['laser_times']

    n = int(len(losTimes)/2)
    sDate = losTimes[n].strftime('%b %d, %Y')
    outFile = abb + losTimes[n].strftime('_%Y%m%d_sum.png')

    fig = plt.figure(figsize=(10, 10))

    plt.rcParams.update({'font.size': 14})
    ax1 = fig.add_axes([0.09,0.09,0.85,0.42])
    ax2 = fig.add_axes([0.09,0.55,0.85,0.42])

    ax2.errorbar(losTimes, losWind, yerr = sigWind, capsize=3, fmt="r--o", ecolor = "black")
    ax2.axhline(0.0, linestyle = ':')
    ax2.set_ylabel('Vertical Wind (m/s)')
    ax2.set_title(siteName + ' on ' + sDate)

    sTime = losTimes[0].strftime('%b %d, %Y %H:%M UT') + ' - ' + \
        losTimes[-1].strftime('%b %d, %Y %H:%M UT')

    ax1.errorbar(losTimes, losTemp, yerr = sigTemp, capsize=3, fmt="r--o", ecolor = "black")
    ax1.set_xlabel(sTime)
    ax1.set_ylabel('Temperature (K)')

    for time in laserTimes:
        ax1.axvline(time, color = 'c')
        ax2.axvline(time, color = 'c')

    plotFile = outFile
    print('  --> Outputting plotfile : ', plotFile)
    fig.savefig(plotFile)
    plt.close(fig)

