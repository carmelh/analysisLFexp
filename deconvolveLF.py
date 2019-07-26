# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 13:44:02 2019

@author: peq10 & Carmel
"""

import os
import sys
sys.path.insert(1, r'\\icnas4.cc.ic.ac.uk\chowe7\GitHub\lightfield_HPC_processing')
import deconvolve as de
import numpy as np
import pickle


def getDeconvolution(stack,stackDark,r,center,path):

    sum_ = np.sum(stack[0,:,:])
    
    Nnum = 19
    new_center = (1023,1023)
    locs = de.get_locs(new_center,Nnum)
    
    folder_to_data = 'H:/Python_Scripts/FireflyLightfield/PSF/correct_prop_550/'
    df_path = r'H:\Python_Scripts\FireflyLightfield\PSF\correct_prop_550\sim_df.xlsx'
    
    H = de.load_H_part(df_path,folder_to_data,zmax = 50.5*10**-6,zmin = -50.5*10**-6,zstep = 5)
    num_iterations=2
    
    #multiple images
    decStack = []
    for ii in range(len(stack)):
        lightfield_image = stack[ii,...]
        print('Stack loaded...', ii)    
        rectified = de.rectify_image(lightfield_image,r,center,new_center,Nnum)
        start_guess = de.backward_project3(rectified/sum_,H,locs)
        result_is = de.ISRA(start_guess,rectified/sum_,H,num_iterations,locs)
        print('Deconvolved.')
        decStack.append(result_is[10,:,:])
        
    decStackA=np.array(decStack)    
    with open(path + '\\deconvolvedStack_infocus', 'wb') as f:
        pickle.dump(decStackA, f)    
        
    varImage = np.var(decStackA,axis=-0)
    signalPixels = np.array(np.where(varImage > np.percentile(varImage,99.92)))
    trialData = np.average(decStackA[:,signalPixels[0],signalPixels[1]], axis=1)    

    with open(path + '\\deconvolvedTrialData_infocus', 'wb') as f:
        pickle.dump(trialData, f) 
        
    backgroundData=np.average(decStackA[:,10:30,10:30],axis=1)
    backgroundData=np.average(backgroundData,axis=1)
    
    with open(path + '\\refocussedBackgroundData_infocus', 'wb') as f:
        pickle.dump(backgroundData, f)     
        
        
    # Dark trial  
    decStackDark = []
    for ii in range(len(stackDark)):
        im = stackDark[ii,...]
        print('Stack loaded...', ii)    
        rectified = de.rectify_image(im,r,center,new_center,Nnum)
        start_guess = de.backward_project3(rectified/sum_,H,locs)
        result_is = de.ISRA(start_guess,rectified/sum_,H,num_iterations,locs)
        print('Deconvolved.')
        decStackDark.append(result_is[10,:,:])
        
    with open(path + '\\deconvolvedDarkData_infocus', 'wb') as f:
        pickle.dump(decStackDark, f) 
                   
    darkTrialData=[]   
    for jj in range(len(decStackDark)):
        x=decStackDark[jj]
        d=np.average(x[10:40,10:20])
        darkTrialData.append(d)    
        
    with open(path + '\\deconvolvedDarkTrialData_infocus', 'wb') as f:
        pickle.dump(darkTrialData, f)        
            
    return trialData, varImage, backgroundData, darkTrialData