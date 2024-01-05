# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 12:36:44 2021

@author: chowe7
"""

import sys
sys.path.insert(1, r'C:\Users\howeca\Documents\GitHub\lightfield_HPC_processing')
import deconvolve as de
import numpy as np
import time

num_iterations=3

def getDeconvolution(stack,r,center,num_iterations,depths,path):
    sum_ = np.sum(stack[0,:,:])
    
    Nnum = 19
    new_center = (1023,1023)
    locs = de.get_locs(new_center,Nnum)
    
    folder_to_data = 'C:/Users/howeca/Dropbox/Imperial/OneDrive/python_code/Imperial/FireflyLightfield/PSF/correct_prop_550/'
    df_path = 'C:/Users/howeca/Dropbox/Imperial/OneDrive/python_code/Imperial/FireflyLightfield/PSF/correct_prop_550/sim_df.xlsx'
    
    H = de.load_H_part(df_path,folder_to_data,zmax = 20.5*10**-6,zmin = -20.5*10**-6,zstep =2)
    
    decon_mean_stack=np.zeros((len(stack),21,101,101))
    for ii in range(len(stack)):
        start=time.time()
        lightfield_image = stack[ii,...]
        print('Stack loaded...', ii)    
        rectified = de.rectify_image(lightfield_image,r,center,new_center,Nnum)
        start_guess = de.backward_project3(rectified/sum_,H,locs)
        result_rl = de.RL_deconv(start_guess,rectified/sum_,H,num_iterations,locs)
        print('Deconvolved.')
        decon_mean_stack[ii] = result_rl
        end =time.time()
        elapsedTime = end-start
        print('elapsed time = {}'.format(elapsedTime))
        
    np.save(path + r'\\stack_refoc\\deconvolved\\' + 'decon_mean_stack_RL_it-{}.npy'.format(num_iterations),decon_mean_stack)     
    return decon_mean_stack