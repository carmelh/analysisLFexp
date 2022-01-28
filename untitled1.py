# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 10:56:11 2022

@author: chowe7
"""


sum_ = np.sum(stack)

Nnum = 19
new_center = (1023,1023)
locs = de.get_locs(new_center,Nnum)

folder_to_data = 'Y:/home/psf_calculation/prop_660_n-1_2201/'
df_path = r'Y:/home/psf_calculation/prop_660_n-1_2201/sim_df.xlsx'
 
H = de.load_H_part(df_path,folder_to_data,zmax = 41*10**-6,zmin = -40*10**-6,zstep =1)

num_it=
decon_mean_stack=np.zeros((8,81,101,101))
for ii in range(len(stack)):
    start=time.time()
    lightfield_image = stack[ii,...]
    print('Stack loaded...', ii)    
    rectified = de.rectify_image(lightfield_image,r,center,new_center,Nnum)
    start_guess = de.backward_project(rectified/sum_,H,locs)
    result_rl = de.RL(start_guess,rectified/sum_,H,num_iterations,locs)
    print('Deconvolved.')
    decon_mean_stack[ii] = result_rl
    end =time.time()
    elapsedTime = end-start
    print('elapsed time = {}'.format(elapsedTime))
    
np.save(path + r'\\stack_refoc\\deconvolved\\' + 'decon_mean_stack_80um_1um_RL_it-{}_Jan2022.npy'.format(num_iterations),decon_mean_stack)     


