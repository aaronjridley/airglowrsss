#!/usr/bin/env python

# Functions to process FPI data on remote2

#import matplotlib
#matplotlib.use('AGG')

import FPI
import fpiinfo
import datetime
import glob
import os
import argparse
import h5py
import numpy as np

# ----------------------------------------------------------------------
# Function to parse input arguments
# ----------------------------------------------------------------------

def get_args_fpi():

    parser = argparse.ArgumentParser(description =
                                     'Process FPI data ring pattern image')

    parser.add_argument('filelist', nargs='+', \
                        help = 'list files to use for generating plots')

    parser.add_argument('-backdir',  default ='backgrounds', \
                        help = 'Directory to find background files')
    
    parser.add_argument('-savedir', default = 'preprocessed', \
                        help='Directory to put preprocessed files')
    
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
# read background files
# ----------------------------------------------------------------------

def read_backgrounds(dirIn):
    filelist = sorted(glob.glob(dirIn + '/' + '*.hdf5'))
    print(filelist)

    images = []
    times = []
    for file in filelist:
        temp = h5py.File(file,'r')
        image = np.array(temp['image'])
        time = datetime.datetime(temp['image'].attrs['year'],
                                 temp['image'].attrs['month'],
                                 temp['image'].attrs['day'],
                                 temp['image'].attrs['hour'],
                                 temp['image'].attrs['minute'],
                                 temp['image'].attrs['second'])
        images.append(image)
        times.append(time)
    backgrounds = {'images': images,
                   'times': times}
    return backgrounds

# ----------------------------------------------------------------------
# Main code!
# ----------------------------------------------------------------------

args = get_args_fpi()

bgDir = args.backdir
saveDir = args.savedir

backgrounds = read_backgrounds(bgDir)

filelist = args.filelist

for file in filelist:
    imgData = FPI.ReadIMG(file)
    img = np.asarray(imgData)
    time = imgData.info['LocalTime']
    exposure = imgData.attrs['ExposureTime']
    # search for the right backgrounds:
    diffs = []
    for bgTime in backgrounds['times']:
        diffs.append(np.abs((time - bgTime).total_seconds()))
    iBG_ = np.argmin(diffs)
    back = backgrounds['images'][iBG_] * exposure

    newImage = img - back
    newImage = newImage - np.percentile(newImage,1)
    
    if (not os.path.exists(saveDir)):
        command = 'mkdir ' + saveDir
        print('Running command : ', command)
        os.system(command)
    save_new_image(file, saveDir, imgData, newImage)
