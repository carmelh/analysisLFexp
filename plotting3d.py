# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:27:57 2020

@author: chowe7
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import plotting_functions as pf
from mpl_toolkits import mplot3d
import pyqtgraph as pg

font = {'family': 'sans',
        'weight': 'normal',
        'size': 18,}

plt.rc('font',**font)    

# variance 3d map
figurePath = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Bulk\191105\slice2\cell1\LF_1x1_50ms-50pA__1\figures\660nm'
keyword = 'refocusedAM'

# get files from openAllFiles.py
stack =[]
for ii in range(len(files)):
    test = files[ii]
    for depth in range(len(test)):
        stack.append(test[depth,:,:])
   
testttt=np.reshape(stack, (len(files),81,len(files[0][0]),len(files[0][1])))


depths_ref = np.arange(-40,40,1)
depths_dec = np.arange(-40,42,1)
depths_dec550 = np.arange(40,-41,-1)


decon_mean_stack=decon_mean_stack_all[1]
# DECONVOLVED
darkTrialAverage_dec=0
df_dec=np.zeros((82,200,101,101))
var3d_dec=np.zeros((len(df_dec),len(df_dec[0][0]),len(df_dec[0][1])))

for ii in range(82):
    file=decon_mean_stack[ii,:,...]
    back=np.mean(file[:13,...],0)
    df_dec[ii]=100*(file-back[None,...]) / (back[None,...] - darkTrialAverage_dec)
    var3d_dec[ii] = np.var(df_dec[ii,...],axis=0)


#550 
back=np.mean(decon_mean_stack_RL_it3[:13,...],0)
df_dec550=100*(decon_mean_stack_RL_it3-back[None,...]) / (back[None,...] - darkTrialAverage_dec)
var3d_dec550= np.var(df_dec550,axis=0)


    


darkTrialAverage_ref=90

# REFOCUSED df & variance
df_ref=np.zeros((len(refoc_mean_stack[1]),200,101,101))
var3d_ref=np.zeros((len(df_ref),len(df_ref[0][0]),len(df_ref[0][1])))

for ii in range(len(refoc_mean_stack[1])):
    file=refoc_mean_stack[:,ii,...]
    back=np.mean(file[:13,...],0)
    df_ref[ii]=100*(file-back[None,...]) / (back[None,...] - darkTrialAverage_ref)
    var3d_ref[ii] = np.var(df_ref[ii,...],axis=0)

pg.image(var3d_ref)





minVal=np.min(var3d)
maxVal=np.max(var3d)

for ii in range(len(var3d)):
    fig = plt.gcf()      
    ax = plt.subplot(111)  
    plt.imshow(var3d[ii],cmap='gray')
    plt.clim(minVal,maxVal)
       
    #axis formatting
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(False)
    ax.spines['left'].set_linewidth(False)
    ax.axes.get_yaxis().set_ticks([])
    ax.axes.get_xaxis().set_ticks([]) 
    pf.forceAspect(ax,aspect=1)
    plt.tight_layout()  
    plt.savefig(figurePath + r'\\{}\\varImage_depth-{}.png'.format(keyword,ii), format='png', dpi=600, bbox_inches='tight')
    plt.close(fig)
    

    
# plot (scatter) 3d map
coords = []
for depth in range(len(var3d)):
    coords.append(np.array(np.where(var3d[depth] > np.percentile(var3d,99.98))))

start=29
stop=57
color_idx = np.linspace(0, 1, len(coords[start:stop]))

fig = plt.gcf()    
ax = plt.axes(projection='3d')
for i, z in zip(color_idx, range(len(coords[start:stop]))):
    ax.scatter3D(coords[z+start][0], coords[z+start][1], z-5, color=plt.cm.inferno(i))
ax.set_zlabel('Z (um)')    
ax.set_zlim((-20,20))
ax.set_xlim((40,80))
ax.set_ylim((20,70))
ax.set_xticklabels([])
ax.view_init(elev=0, azim=0) 
plt.tight_layout()  
fig.set_size_inches(10,5)
plt.savefig(figurePath + r'\\{}\\varImage3d.png'.format(keyword), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figurePath + r'\\{}\\varImage3d.eps'.format(keyword), format='eps', dpi=600, bbox_inches='tight')


fig = plt.gcf()    
ax = plt.axes(projection='3d')
for i, z in zip(color_idx, range(len(coords[start:stop]))):
    ax.scatter3D(coords[z+start][0], coords[z+start][1], z-5, color=plt.cm.inferno(i))
ax.set_zlabel('Z (um)')    
ax.set_zlim((-20,20))
ax.set_xlim((40,80))
ax.set_ylim((20,70))
ax.view_init(elev=20, azim=63)   
plt.tight_layout()  
fig.set_size_inches(10,5)
plt.savefig(figurePath + r'\\{}\\varImage3d_angled.png'.format(keyword), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figurePath + r'\\{}\\varImage3d_angled.eps'.format(keyword), format='eps', dpi=600, bbox_inches='tight')


fig = plt.gcf()    
ax = plt.axes(projection='3d')
for i, z in zip(color_idx, range(len(coords[start:stop]))):
    ax.scatter3D(coords[z+start][0], coords[z+start][1], z-5, color=plt.cm.inferno(i))
ax.set_zlim((-20,20))
ax.set_xlim((40,80))
ax.set_ylim((20,70))
ax.set_zticklabels([])
ax.view_init(elev=-90, azim=90)    # (0,0) or (20,63) or (-90,90)
plt.tight_layout()  
plt.savefig(figurePath + r'\\{}\\varImage3d_top.png'.format(keyword), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figurePath + r'\\{}\\varImage3d_top.eps'.format(keyword), format='eps', dpi=600, bbox_inches='tight')






lengthSB = 50
pixelSize = 5
xS = 44
y = 55

x1=35
x2=95
y1=25
y2=85
z1=10
z2=70


file = decon_mean_stack[0,...]
maxProj_z = np.max(file, axis=0)
maxProj_x = np.max(file, axis=1)
maxProj_y = np.max(file, axis=2)


var3d_dec=var3d

# VARIANCE FIGURES
# max projections
refMaxProj_z = np.max(var3d_ref[:,x1:x2,y1:y2], axis=0)
refMaxProj_x = np.max(var3d_ref[:,x1:x2,y1:y2], axis=1)
refMaxProj_y = np.max(var3d_ref[:,x1:x2,y1:y2], axis=2)

decMaxProj_z = np.max(var3d_dec[:,x1:x2,y1:y2], axis=0)
decMaxProj_x = np.max(var3d_dec[:,x1:x2,y1:y2], axis=1)
decMaxProj_y = np.max(var3d_dec[:,x1:x2,y1:y2], axis=2)

maxProj_z = np.array([refMaxProj_z,decMaxProj_z])
maxProj_x = np.concatenate([refMaxProj_x,decMaxProj_x[0:80]],axis=1)
maxProj_y = np.array([refMaxProj_y,decMaxProj_y[0:80]])

#Get the min and max of all your data
keyword = 'refoc'
keyword = 'decon_660nm_3it'


_min, _max = np.amin(maxProj_z), np.amax(maxProj_z)

fig = plt.gcf()      
ax = plt.subplot(111)  
im = plt.imshow(decMaxProj_z)
plt.plot([xS,xS+(lengthSB/pixelSize)],[y,y],linewidth=6.0,color='w')

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_xaxis().set_ticks([]) 
ax.axes.get_yaxis().set_ticks([]) 
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.3)
#plt.colorbar(im, cax=cax)
pf.forceAspect(ax,aspect=1)
plt.tight_layout()  

plt.savefig(figurePath + r'\\{}_maxProj_z.png'.format(keyword), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figurePath + r'\\{}_maxProj_z.eps'.format(keyword), format='eps', dpi=600, bbox_inches='tight')



_min, _max = np.amin(maxProj_x[0:80,...]), np.amax(maxProj_x[0:80,...])

fig = plt.gcf()      
ax = plt.subplot(111)  
im = plt.imshow(refMaxProj_x[z1:z2])

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_xaxis().set_ticks([]) 
plt.tight_layout()  
plt.savefig(figurePath + r'\\{}_maxProj_x.png'.format(keyword), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figurePath + r'\\{}_maxProj_x.eps'.format(keyword), format='eps', dpi=600, bbox_inches='tight')


_min, _max = np.amin(maxProj_y), np.amax(maxProj_y)

fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(refMaxProj_y[z1:z2])

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_xaxis().set_ticks([]) 
plt.tight_layout()  
plt.savefig(figurePath + r'\\{}_maxProj_y.png'.format(keyword), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figurePath + r'\\{}_maxProj_y.eps'.format(keyword), format='eps', dpi=600, bbox_inches='tight')



# line plot through x=12
fig = plt.gcf()      
ax = plt.subplot(111)  
plt.plot(refMaxProj_x[z1:z2,12],linewidth=3.0,color='k',label='Refocused')
plt.plot(decMaxProj_x[z1:z2,12],linewidth=3.0,color='r',label='Deconvolved')

#axis formatting
#plt.legend(loc='upper left',frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([]) 
ax.set_ylim((-5,30))
fig.set_size_inches(10,4)
plt.tight_layout()  
plt.savefig(figurePath + r'\\maxProj_x12.png'.format(keyword), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figurePath + r'\\maxProj_x12.eps'.format(keyword), format='eps', dpi=600, bbox_inches='tight')



var3d_ref_ROI=var3d_ref[:,x1:x2,y1:y2]
var3d_dec_ROI=var3d_dec[:,x1:x2,y1:y2]
var3d_dec_ROI550=var3d_dec550[:,x1:x2,y1:y2]


z1=10
z2=70

cell1_ref=var3d_ref_ROI[:,13,12]
cell2_ref=var3d_ref_ROI[:,12,47]
cell3_ref=var3d_ref_ROI[:,32,12]
cell4_ref=var3d_ref_ROI[:,30,43]
cell5_ref=var3d_ref_ROI[:,47,48]
cell6_ref=var3d_ref_ROI[:,42,32]

fig = plt.gcf()      
ax = plt.subplot(111)  
plt.plot(depths_ref[z1:z2],cell1_ref[z1:z2],linewidth=2.5,color='k')
plt.plot(depths_ref[z1:z2],cell2_ref[z1:z2],linewidth=2.5,color='#f6a21b')
plt.plot(depths_ref[z1:z2],cell3_ref[z1:z2],linewidth=2.5,color='#942367')
plt.plot(depths_ref[z1:z2],cell4_ref[z1:z2],linewidth=2.5,color='#006633')
plt.plot(depths_ref[z1:z2],cell5_ref[z1:z2],linewidth=2.5,color='#df513b')
#plt.plot(depths_ref,cell6_ref,linewidth=2.5,color='b')
x = [-30,0,30]
plt.xticks(x) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([]) 
fig.set_size_inches(5,4)
plt.tight_layout()  
plt.savefig(figurePath + r'\\cellDepths_refoc.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figurePath + r'\\cellDepths_refoc.eps', format='eps', dpi=600, bbox_inches='tight')




cell1_dec=var3d_dec_ROI[:,13,12]
cell2_dec=var3d_dec_ROI[:,12,47]
cell3_dec=var3d_dec_ROI[:,32,12]
cell4_dec=var3d_dec_ROI[:,30,43]
cell5_dec=var3d_dec_ROI[:,47,48]
cell6_dec=var3d_dec_ROI[:,42,32]

fig = plt.gcf()      
ax = plt.subplot(111)  
plt.plot(depths_dec[z1:z2],cell1_dec[z1:z2],linewidth=2.5,color='k')
plt.plot(depths_dec[z1:z2],cell2_dec[z1:z2],linewidth=2.5,color='#f6a21b')
plt.plot(depths_dec[z1:z2],cell3_dec[z1:z2],linewidth=2.5,color='#942367')
plt.plot(depths_dec[z1:z2],cell4_dec[z1:z2],linewidth=2.5,color='#006633')
plt.plot(depths_dec[z1:z2],cell6_dec[z1:z2],linewidth=2.5,color='#df513b')
x = [-30,0,30]
plt.xticks(x)         
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([]) 
fig.set_size_inches(5,4)
plt.tight_layout()  
plt.savefig(figurePath + r'\\cellDepths_decon-660.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figurePath + r'\\cellDepths_decon-660.eps', format='eps', dpi=600, bbox_inches='tight')




cell1_dec=var3d_dec_ROI550[:,13,12]
cell2_dec=var3d_dec_ROI550[:,12,47]
cell3_dec=var3d_dec_ROI550[:,32,12]
cell4_dec=var3d_dec_ROI550[:,30,43]
cell5_dec=var3d_dec_ROI550[:,47,48]
cell6_dec=var3d_dec_ROI550[:,42,32]

fig = plt.gcf()      
ax = plt.subplot(111)  
plt.plot(depths_dec550[1:82],cell1_dec[1:82],linewidth=2.5,color='k')
plt.plot(depths_dec550[1:82],cell2_dec[1:82],linewidth=2.5,color='#f6a21b')
plt.plot(depths_dec550[1:82],cell3_dec[1:82],linewidth=2.5,color='#942367')
plt.plot(depths_dec550[1:82],cell4_dec[1:82],linewidth=2.5,color='#006633')
plt.plot(depths_dec550[1:82],cell6_dec[1:82],linewidth=2.5,color='#df513b')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([]) 
fig.set_size_inches(5,4)
plt.tight_layout()  
plt.savefig(figurePath + r'\\cellDepths_decon-550.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figurePath + r'\\cellDepths_decon-550.eps', format='eps', dpi=600, bbox_inches='tight')







fig = plt.gcf()      
ax = plt.subplot(111)  
plt.plot(depths_ref,cell1_ref,linewidth=2.5,color='k')
plt.plot(depths_ref,cell2_ref,linewidth=2.5,color='#f6a21b')
plt.plot(depths_ref,cell3_ref,linewidth=2.5,color='#942367')
plt.plot(depths_ref,cell4_ref,linewidth=2.5,color='#006633')
plt.plot(depths_ref,cell5_ref,linewidth=2.5,color='#df513b')
plt.plot(depths_dec[2:82],cell1_dec[2:82],linewidth=2.5,color='k',linestyle='--')
plt.plot(depths_dec[2:82],cell2_dec[2:82],linewidth=2.5,color='#f6a21b',linestyle='--')
plt.plot(depths_dec[2:82],cell3_dec[2:82],linewidth=2.5,color='#942367',linestyle='--')
plt.plot(depths_dec[2:82],cell4_dec[2:82],linewidth=2.5,color='#006633',linestyle='--')
plt.plot(depths_dec[2:82],cell6_dec[2:82],linewidth=2.5,color='#df513b',linestyle='--')
plt.plot([0,0],[0,35])        
#plt.plot(depths_ref,cell6_ref,linewidth=2.5,color='b')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([]) 
fig.set_size_inches(5,4)
plt.tight_layout()  





