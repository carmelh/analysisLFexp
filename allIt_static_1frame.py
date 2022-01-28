# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 10:56:11 2022

@author: chowe7
"""
%reset -f

import numpy as np
import tifffile
import sys
import pandas as pd
sys.path.insert(1, r'H:\Python_Scripts\analysisLFexp')
import imagingAnalysis as ia
import idx_refocus as ref
import deconvolveLF as dlf
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import general_functions as gf
import deconvolve as de
import time



path=r'Y:\projects\thefarm2\live\Firefly\Lightfield\Phantoms\Non Scattering\10umBeads_best\lf_refl660nm_stack_1\z44'

depths = np.arange(-40,41,1)

r,center = (np.array([19.51,0.115]),np.array([1019.9,1024.4])) 

stack = tifffile.imread(path + '\\z44.tif')
    
    
refoc_mean_stack=np.zeros((len(depths),101,101))
refoc_mean_stack = get_refocussed(stack,r,center,depths,n_views = 19)


np.save(path + '\\refoc_mean_stack.npy',refoc_mean_stack)




num_it = [1,3,5,7,9,13,17,21]

sum_ = np.sum(stack)

Nnum = 19
new_center = (1023,1023)
locs = de.get_locs(new_center,Nnum)

folder_to_data = 'Y:/home/psf_calculation/prop_660_n-1_2201/'
df_path = r'Y:/home/psf_calculation/prop_660_n-1_2201/sim_df.xlsx'
 
H = de.load_H_part(df_path,folder_to_data,zmax = 41*10**-6,zmin = -40*10**-6,zstep =1)

decon_mean_stack=np.zeros((len(num_it),81,101,101))
for it in range(len(decon_mean_stack)):
    start=time.time()
    print('Iteration Number...', num_it[it])    
    rectified = de.rectify_image(stack,r,center,new_center,Nnum)
    start_guess = de.backward_project(rectified/sum_,H,locs)
    result_rl = de.RL(start_guess,rectified/sum_,H,num_it[it],locs)
    print('Deconvolved.')
    decon_mean_stack[it] = result_rl
    end =time.time()
    elapsedTime = end-start
    print('elapsed time = {}'.format(elapsedTime))
    
np.save(path + r'\\stack_refoc\\deconvolved\\' + 'decon_mean_stack_80um_1um_RL_newROI.npy',decon_mean_stack)     


