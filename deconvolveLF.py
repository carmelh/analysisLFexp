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
import time
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import general_functions as gf

def getDeconvolution(stack,stackDark,r,center,path,signalPixels):

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
        start=time.time()
        lightfield_image = stack[ii,...]
        print('Stack loaded...', ii)    
        rectified = de.rectify_image(lightfield_image,r,center,new_center,Nnum)
        start_guess = de.backward_project3(rectified/sum_,H,locs)
        result_is = de.ISRA(start_guess,rectified/sum_,H,num_iterations,locs)
        print('Deconvolved.')
        decStack.append(result_is[10,:,:])
        end =time.time()
        elapsedTime = end-start
        print('elapsed time = {}'.format(elapsedTime))
        
        
    #find signal pixels average
    decStackA=np.array(decStack)    
    varImage = np.var(decStackA,axis=-0)
   # signalPixels = need to load from refocus for some reason....    
    trialData = np.average(decStackA[:,signalPixels[0],signalPixels[1]], axis=1)    
    
    #trialData = np.average(decStackA[:,52:55,46:53], axis=1)   
    #trialData = np.average(trialData,axis=1)    

    #background average
    backgroundData=np.average(decStackA[:,10:30,10:30],axis=1)
    backgroundData=np.average(backgroundData,axis=1)
    
        
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
        
                   
    darkTrialData=[]   
    for jj in range(len(decStackDark)):
        x=decStackDark[jj]
        d=np.average(x[10:40,10:20])
        darkTrialData.append(d)    
        
    #save
    gf.savePickes(path,'\\deconvolvedStack_infocus',decStackA)
    gf.savePickes(path,'\\deconvolvedTrialData_infocus',trialData)
    gf.savePickes(path,'\\deconvolvedBackgroundData_infocus',backgroundData)
    gf.savePickes(path,'\\deconvolvedDarkData_infocus',decStackDark)
    gf.savePickes(path,'\\deconvolvedDarkTrialData_infocus',darkTrialData)
            
    return trialData, varImage, backgroundData, darkTrialData