# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 13:34:43 2021

@author: chowe7
"""





inFocus=11


plt.imshow(stack[inFocus])

stack=stack_full[inFocus-1:inFocus+2]
stack=stack_full[inFocus]

#  Rdy,Rdx,y,x,
r,center = (np.array([0.885,19.51]),np.array([1121.525,1094.85])) 


r,center = (np.array([0.885,19.51]),np.array([1025.3,1035])) 

num_iter=[1,3,5,7,9,13,17,21]
decon_mean_stack=np.zeros((len(num_iter),len(stack),41,101,101))
for it in range(len(num_iter)):
    print(num_iter[it])
    result = getDeconvolution(stack,r,center,num_iter[it],path)
    decon_mean_stack[it] = result


np.save(path + r'\\stack_refoc\\' + 'decon_mean_stack_correctpsf_80um_2um_RL.npy',decon_mean_stack)     
   