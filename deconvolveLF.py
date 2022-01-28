# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 13:44:02 2019

@author: peq10 & Carmel
"""

import os
import tifffile
import sys
sys.path.insert(1, r'\\icnas4.cc.ic.ac.uk\chowe7\GitHub\lightfield_HPC_processing')
import deconvolve as de
import numpy as np
import time
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import general_functions as gf

def getDeconvolution(stack,r,center,num_iterations,signalPixels,path,pathDarkTrial,fileNameDark):

    sum_ = np.sum(stack[0,:,:])
    
    Nnum = 19
    new_center = (1023,1023)
    locs = de.get_locs(new_center,Nnum)
    
    folder_to_data = 'H:/Python_Scripts/FireflyLightfield/PSF/correct_prop_550/'
    df_path = r'H:\Python_Scripts\FireflyLightfield\PSF\correct_prop_550\sim_df.xlsx'
    
    H = de.load_H_part(df_path,folder_to_data,zmax = 40.5*10**-6,zmin = -40.5*10**-6,zstep =1)
    
    #multiple images
    decStack = []
    depths = np.arange(-40.5,40.5,1)
    decon_mean_stack=np.zeros((len(stack),len(depths),101,101))
    for ii in range(len(stack)):
        start=time.time()
        lightfield_image = stack[ii,...]
        print('Stack loaded...', ii)    
        rectified = de.rectify_image(lightfield_image,r,center,new_center,Nnum)
        start_guess = de.backward_project3(rectified/sum_,H,locs)
        result_rl = de.RL_deconv(start_guess,rectified/sum_,H,num_iterations,locs)
        print('Deconvolved.')
        decon_mean_stack[ii] = result_rl
#        np.save(path + r'\\stack_refoc\\deconvolved\\' + 'decon_mean_stack_RL_it-{}-{}.npy'.format(num_iterations,ii),result_rl)
#        decStack.append(result_rl[round(len(result_rl)/2),:,:])
        end =time.time()
        elapsedTime = end-start
        print('elapsed time = {}'.format(elapsedTime))
        
        
    #find signal pixels average
    decStackA=np.array(decStack)    
    varImage = np.var(decStackA,axis=0)
   # signalPixels = load from refocus   
#    signalPixels = np.array(np.where(varImage > np.percentile(varImage,99.92)))
    trialData = np.average(decStackA[:,signalPixels[0],signalPixels[1]], axis=1)    
  
    #trialData = np.average(decStackA[:,52:54,46:48], axis=1)   
    #trialData = np.average(trialData,axis=1)    

    #background average
    backgroundData=np.average(decStackA[:,10:30,10:30],axis=1)
    backgroundData=np.average(backgroundData,axis=1)
    
    #save
    gf.savePickes(path,'\\deconvolvedStack_RL_infocus_{}'.format(num_iterations),decStackA)
    gf.savePickes(path,'\\deconvolvedTrialData_RL_infocus_{}'.format(num_iterations),trialData)
    gf.savePickes(path,'\\deconvolvedBackgroundData_RL_infocus_{}'.format(num_iterations),backgroundData)    
    
    try:
         darkTrialData = gf.loadPickles(pathDarkTrial,'\\deconvolvedDarkTrialData_RL_infocus_{}'.format(num_iterations))
         print('Loaded dark trial data')
    except:
        stackDark = tifffile.imread(pathDarkTrial + fileNameDark + '.tif')   
        print('Loaded dark stack')
        # Dark trial  
        decStackDark = []
        for ii in range(len(stackDark)):
            im = stackDark[ii,...]
            print('Stack loaded...', ii)    
            rectified = de.rectify_image(im,r,center,new_center,Nnum)
            start_guess = de.backward_project3(rectified/sum_,H,locs)
            result_is = de.RL_deconv(start_guess,rectified/sum_,H,num_iterations,locs)
            print('Deconvolved.')
            decStackDark.append(result_is[10,:,:])
            
        darkTrialData=[]   
        for jj in range(len(decStackDark)):
            x=decStackDark[jj]
            d=np.average(x[10:40,10:20])
            darkTrialData.append(d)    
        gf.savePickes(pathDarkTrial,'\\deconvolvedDarkData_RL_infocus_{}'.format(num_iterations),decStackDark)
        gf.savePickes(pathDarkTrial,'\\deconvolvedDarkTrialData_RL_infocus_{}'.format(num_iterations),darkTrialData)   


            
    return trialData, varImage, backgroundData, darkTrialData