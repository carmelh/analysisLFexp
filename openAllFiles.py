# -*- coding: utf-8 -*-
"""
Created on Wed May 27 08:46:20 2020

@author: chowe7
"""

#load in stack files
    
import numpy as np
import os    
import glob       
import re
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import plotting_functions as pf


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

keyword='refocused'
path = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190724\slice1\cell1\MLA_1x1_50ms_200pA_A-STIM_1'
pathDarkTrial = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190730\slice2\cell1\MLA_1x1_50ms_100pA_A_Stim_DARK_1'
path_stack = path + '\stack_refoc' + '\\' + keyword

files=[]
for infile in sorted(glob.glob(os.path.join(path_stack, '*.npy')), key=numericalSort):
    files.append(np.load(infile))
    print('File loaded {}'.format(infile))
    
refoc_mean_stack = np.array(files)
decon_mean_stack = np.array(files)
    
np.save(path_stack + '\\refoc_mean_stack_40um_2um_RL.npy',refoc_mean_stack)     


reFstackA =[]
for ii in range(len(files)):
    test = files[ii]
    reFstackA.append(test[40,:,:])

reFstackA=np.array(reFstackA)    
    
varImage = np.var(reFstackA/np.mean(reFstackA),axis=0)

decStackA =[]
for ii in range(len(files)):
    test = files[ii]
    decStackA.append(test[40,:,:])

decStackA=np.array(decStackA)
varImage_dec = np.var(reFstackA/np.mean(decStackA),axis=0)

  
varImage = np.var(decStackA/np.mean(decStackA),axis=0)
signalPixels = np.array(np.where(varImage > np.percentile(varImage,99.92)))
trialData = np.average(decStackA[:,signalPixels[0],signalPixels[1]], axis=1)    
plt.plot(trialData)  


    #background average
backgroundData=np.average(decStackA[:,10:30,10:30],axis=1)
backgroundData=np.average(backgroundData,axis=1)
    
darkTrialData = gf.loadPickles(pathDarkTrial,'\\deconvolvedDarkTrialData_RL_infocus_3')

stim = 'A Stim'
processedTrace, diffROI,processedBackgroundTrace,baselineIdx = ia.processRawTrace(trialData, darkTrialData, backgroundData, stim)
print('Finished Processing')
    
    # get stats
baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(processedTrace,trialData,darkTrialData,baselineIdx)










#pixelwise SNR
maxValue = max(trialData)
peakIdx = np.array(np.where(trialData == maxValue))





baselineIdx = 13
baselineFluorescence = np.mean(trialData[0:baselineIdx])
baselineFluorescence = np.mean(decStackA[0:baselineIdx],axis=0)

baselineBackgroundFluorescence = np.mean(backgroundData[0:baselineIdx])

# calculate the average number of counts for the dark trial. This is to get the f_dark value for the dF/F calculation
darkTrialAverage = np.mean(darkTrialData)

processedImages = (decStackA-baselineFluorescence)/(baselineFluorescence-darkTrialAverage)
varImageProcessed = np.var(processedImages/np.mean(processedImages),axis=0)


processedImages=processedImages+1
peak_dF_F=np.zeros((101,101))
for x in range(len(processedImages[1])):
    for y in range(len(processedImages[2])):
        peak_dF_F[x,y]=max(processedImages[:,x,y])

peak_dF_F_1 = np.where(peak_dF_F<0,0,peak_dF_F)

peak_dF_F_old = (processedImages[peakIdx[0,0]])*100
plt.imshow(peak_dF_F)
plt.colorbar()

df_noise = (np.sqrt(np.var(processedImages[0:baselineIdx],axis=0))) # in %
plt.imshow(df_noise)
plt.colorbar()


SNR = (peak_dF_F)/df_noise
plt.imshow(SNR)
plt.colorbar()











import os
import sys
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from PIL import Image

def make_interpolated_image(nsamples):
    """Make an interpolated image from a random selection of pixels.

    Take nsamples random pixels from im and reconstruct the image using
    scipy.interpolate.griddata.

    """

    ix = np.random.randint(im.shape[1], size=nsamples)
    iy = np.random.randint(im.shape[0], size=nsamples)
    samples = im[iy,ix]
    int_im = griddata((iy, ix), samples, (Y, X))
    return int_im

# Read in image and convert to greyscale array object
img_name = sys.argv[1]
im = Image.open(img_name)
im = np.array(im.convert('L'))
im=SNR

# A meshgrid of pixel coordinates
nx, ny = im.shape[1], im.shape[0]
X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))

# Create a figure of nrows x ncols subplots, and orient it appropriately
# for the aspect ratio of the image.
nrows, ncols = 1, 1
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4), dpi=100)
if nx < ny:
    w, h = fig.get_figwidth(), fig.get_figheight()
    fig.set_figwidth(h), fig.set_figheight(w)

# Convert an integer i to coordinates in the ax array
get_indices = lambda i: (i // nrows, i % ncols)


nsamples = 20000
plt.imshow(make_interpolated_image(nsamples))


for i in range(4):
    nsamples = 10**(i+2)
    axes = ax[get_indices(i)]
    axes.imshow(make_interpolated_image(nsamples),
                          cmap=plt.get_cmap('Greys_r'))
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title('nsamples = {0:d}'.format(nsamples))
filestem = os.path.splitext(os.path.basename(img_name))[0]
plt.savefig('{0:s}_interp.png'.format(filestem), dpi=100)