#!/usr/bin/env python

import matplotlib
matplotlib.use('AGG')

import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import datetime
import h5py
import argparse

# ----------------------------------------------------------------------
# Function to parse input arguments
# ----------------------------------------------------------------------

def get_args_fpi():

    parser = argparse.ArgumentParser(description =
                                     'Display file image(s)')

    parser.add_argument('-files', nargs='+', \
                        help = 'list of files for displaying')

    parser.add_argument('-last',  default = False,
                        action = 'store_true',
                        help = 'Only use the last image')
    
    args = parser.parse_args()

    return args


args = get_args_fpi()

if (args.files == None):
    filelist = sorted(glob.glob('*_BG_*.hdf5'))
else:
    filelist = args.files

if (args.last):
    filelist = [filelist[-1]]

for file in filelist:
    print('Reading file : ', file)
    temp = h5py.File(file,'r')
    im = temp['image']
    image = np.asarray(im)

    fig = plt.figure(figsize=(6, 6))
    plt.rcParams.update({'font.size': 14})
    ax1 = fig.add_axes([0.05,0.05,0.9,0.9])

    maxi = np.percentile(image,98)
    mini = np.percentile(image,2)
    cax1 = ax1.pcolor(image, cmap = 'bwr', vmin = mini, vmax = maxi)
    cbar1 = fig.colorbar(cax1, ax = ax1, shrink = 0.5, pad=0.01)
    ax1.set_aspect(1.0)

    outfile = file[:-4] + 'png'
    print('  --> Outputting plotfile : ', outfile)
    fig.savefig(outfile)
    plt.close(fig)

