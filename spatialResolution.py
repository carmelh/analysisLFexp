# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 09:18:04 2020

@author: chowe7

In this file


at the end maximum intensity projections
but also see plotting3d.py
"""

import numpy as np
import sys
import pandas as pd
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import general_functions as gf
import matplotlib.pyplot as plt
import plotting_functions as pf
from scipy.interpolate import UnivariateSpline
import pyqtgraph as pg

font = {'family': 'sans',
        'weight': 'normal',
        'size': 20,}

plt.rc('font',**font)    


analysisFolder=r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Analysis\spatialRes'
figureFolder=r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190605\slice1\Cell2\ZSTCK\MLA_NOPIP_1x1_f_2-8_660nm_200mA_1\figures'
date=210927
pixelSize = 5

reconstruMeth = 'refoc'

z_opt = 40
z_spacing=1

depth = np.arange(-40,41,1)
depth_ref = np.arange(-40,41,1)
depth_old = np.arange(-40.5,40.5,1)

result_refoc=refoc_mean_stack[0,...]
result_decon_old=decon_mean_stack_old[1,:,43,...]
result_decon_80=decon_mean_stack[0,1,...]


#scale bar
xSB=50
lengthSB = 25
y = 55



################################
#     Synthetically Refocused       #
################################
#open files
#result=

result=refoc_mean_stack

xLat=np.linspace(0, result.shape[1]*pixelSize, result.shape[2])  #x in um
xAx=np.linspace(0, result.shape[0], result.shape[0])  #x in um
plt.imshow(result[40,...])

inFocus=40

#190730
xLoc=49
yLoc=56

#190605\slice1\Cell2
xLoc=43
yLoc=63

xLoc=39
yLoc=58

# synthetically refocused
xy_refoc=result_refoc[inFocus,:,xLoc:xLoc+1]
yx_refoc=result_refoc[inFocus,yLoc:yLoc+1,:]

#
#xAx_old=np.linspace(0.5, 80.5, result_decon_old.shape[0])  #x in um
#
#xy_decon_old=result_decon_old[41,:,xLoc:xLoc+1]
#yx_decon_old=result_decon_old[41,yLoc:yLoc+1,:]


#xAx_new=np.linspace(0, 81, result_decon_new.shape[0])  #x in um

xy_decon=decon_mean_stack[:,40,:,xLoc:xLoc+1]
yx_decon=decon_mean_stack[:,40,yLoc:yLoc+1,:]



#################
# line plots for xy and yx
#################
fig = plt.gcf()    
ax = fig.add_subplot(111)
plt.plot(xLat,xy_refoc,linewidth=2.5,color='k')
    
ax.set_xlabel('x (um)')    
ax.set_xlim((200,400))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
fig.set_size_inches(7.5,5)
plt.tight_layout()  
plt.savefig(figureFolder + r'\\xy_{}.png'.format(reconstruMeth), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\xy_{}.eps'.format(reconstruMeth), format='eps', dpi=600, bbox_inches='tight')


fig = plt.gcf()    
ax = fig.add_subplot(111)
plt.plot(xLat,yx_refoc[0,:],linewidth=2.5, color='k')
ax.set_xlabel('x (um)')  
  
ax.set_xlim((100,300))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
fig.set_size_inches(7.5,5)
plt.tight_layout()  
    
plt.savefig(figureFolder + r'\\yx_{}.png'.format(reconstruMeth), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\yx_{}.eps'.format(reconstruMeth), format='eps', dpi=600, bbox_inches='tight')



color_idx = np.linspace(0, 1, 9)

reconstruMeth='all_decon_it_660new'
#################
# all
#################
fig = plt.gcf()    
ax = fig.add_subplot(111)
#plt.plot(xLat,xy_refoc,linewidth=2.5,color='k',linestyle='--')
#plt.plot(xLat,xy_decon_old,linewidth=2.5,color='k')
for col,it in zip(color_idx, range(len(xy_decon))):
    plt.plot(xLat,xy_decon[it],linewidth=2.5,color=plt.cm.tab10(col))
plt.legend(['1', '3', '5', '7', '9', '13', '17', '21'],frameon=False,ncol=2)      
ax.set_xlabel('x (um)')    
ax.set_xlim((200,400))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
fig.set_size_inches(7.5,5)
plt.tight_layout()  
plt.savefig(figureFolder + r'\\xy_{}.png'.format(reconstruMeth), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\xy_{}.eps'.format(reconstruMeth), format='eps', dpi=600, bbox_inches='tight')





# FOV image with scale bar
fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(result[inFocus,10:70,20:80])
plt.plot([xSB,xSB+(lengthSB/pixelSize)],[y,y],linewidth=6.0,color='w')

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
pf.forceAspect(ax,aspect=1)
plt.tight_layout()  
plt.savefig(figureFolder + r'\\FOV_{}_25umSB.png'.format(reconstruMeth), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\FOV_{}_25umSB.eps'.format(reconstruMeth), format='eps', dpi=800, bbox_inches='tight')



# FOV images through z
for z in range(len(result)):
    print(z)
    fig = plt.gcf()      
    ax = plt.subplot(111)  
    plt.imshow(result[z,10:70,20:80],cmap='gray')
    plt.plot([xS,xS+(lengthSB/pixelSize)],[y,y],linewidth=6.0,color='w')
    
    #axis formatting
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(False)
    ax.spines['left'].set_linewidth(False)
    ax.axes.get_yaxis().set_ticks([])
    ax.axes.get_xaxis().set_ticks([]) 
    pf.forceAspect(ax,aspect=1)
    plt.tight_layout()  
    plt.savefig(figureFolder + r'\\z_images\\FOV_{}_gray_z-{}_25umSB.png'.format(reconstruMeth,z), format='png', dpi=600, bbox_inches='tight')
    plt.close(fig)




#########################
#     Lateral FWHM      #
#########################


spline = UnivariateSpline(xLat, xy_refoc-np.max(xy_refoc)/2, s=0)
r1, r2 = spline.roots() # find the roots
FWHM_xy=r2-r1


spline = UnivariateSpline(xLat, yx_refoc-np.max(yx_refoc)/2, s=0)
r1, r2 = spline.roots() # find the roots
FWHM_yx=r2-r1

fields=[date,reconstruMeth,FWHM_xy,FWHM_yx,xLoc,yLoc,figureFolder]
gf.appendCSV(analysisFolder ,r'\lateralResolution_FWHM',fields)




################################
#      Axial Resolution        #
################################

# xz and yz are 2d images through z. 


xz_refoc=result_refoc[:,:,xLoc]
yz_refoc=result_refoc[:,yLoc,:]


#########################
#      Axial FWHM       #
#########################

y=xz_refoc[:,yLoc]
spline = UnivariateSpline(xAx, y-np.max(y)/2, s=0)
r1, r2 = spline.roots() # find the roots
FWHM_xz=r2-r1

y=yz_refoc[:,xLoc]
spline = UnivariateSpline(xAx, y-np.max(y)/2, s=0)
r1, r2 = spline.roots() # find the roots
FWHM_yz=r2-r1


fields=[date,reconstruMeth,FWHM_xz*z_spacing,FWHM_yz*z_spacing,xLoc,yLoc,figureFolder]
gf.appendCSV(analysisFolder ,r'\axialResolution_FWHM',fields)



FOV = 20

fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(xz_refoc[:,round(yLoc-(FOV/2)):round(yLoc+(FOV/2))])

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
plt.tight_layout()  
plt.savefig(figureFolder + r'\\xz_refocused_80um.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\xz_refocused_80um.eps', format='eps', dpi=800, bbox_inches='tight')


fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(yz_refoc[:,round(xLoc-(FOV/2)):round(xLoc+(FOV/2))])

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
plt.tight_layout()  
plt.savefig(figureFolder + r'\\yz_refocused_80um.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\yz_refocused_80um.eps', format='eps', dpi=800, bbox_inches='tight')



########################
#     DECONVOLVED      #
########################
reconstruMeth = 'deconvolved'
num_it = [1,3,5,7,9,13,17,21]

#open deconvolved_result
dataPath = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190730\slice2\cell1\z_stack\MLA3_1x1_50ms_150pA_1\stack_refoc\deconvolved'
deconvolved_result = np.load(dataPath + '\\deconvolved_result.npy')

xLat=np.linspace(0, deconvolved_result.shape[4]*pixelSize, deconvolved_result.shape[4])  #x in um
xAx=np.linspace(0, deconvolved_result.shape[1], deconvolved_result.shape[1])  #x in um

################################
#     Lateral Resolution       #
################################
# xy and yx are the 2d images at the native focal plane
#got to manually set the x or y location of the cell
plt.imshow(deconvolved_result[0,41,0,:,:])

#xLoc=54
#yLoc=62
xLoc=51
yLoc=58

xy=[]
yx=[]
for ii in range(len(decon_mean_stack)):
    xy.append(decon_mean_stack[ii,40,:,xLoc:xLoc+1])
    yx.append(decon_mean_stack[ii,40,yLoc:yLoc+1,:])
    
xy=np.array(xy)
yx=np.array(yx)


#################
# line plots for xy and yx for different iteration number
#plasma and tab10
#################

color_idx = np.linspace(0, 1, 9)

fig = plt.gcf()    
ax = fig.add_subplot(111)
for col,ii in zip(color_idx, range(len(decon_mean_stack))):
    lines=plt.plot(xLat,xy[ii,:,0],linewidth=2.5, color=plt.cm.tab10(col))
    
plt.legend(['1', '3', '5', '7', '9', '13', '17', '21'],frameon=False)
ax.set_xlabel('x (um)')    
ax.set_xlim((274,314))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
fig.set_size_inches(7.5,5)
plt.tight_layout()  

plt.savefig(figureFolder + r'\\xy_{}_cm-tab10.png'.format(reconstruMeth), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\xy_{}_cm-tab10.eps'.format(reconstruMeth), format='eps', dpi=600, bbox_inches='tight')


fig = plt.gcf()    
ax = fig.add_subplot(111)
for col,ii in zip(color_idx, range(len(decon_mean_stack))):
    lines=plt.plot(xLat,yx[ii,0,:],linewidth=2.5, color=plt.cm.tab10(col))
    
plt.legend(['1', '3', '5', '7', '9', '13', '17', '21'],frameon=False)
ax.set_xlabel('x (um)')    
ax.set_xlim((235,285))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
fig.set_size_inches(7.5,5)
plt.tight_layout()  
    
plt.savefig(figureFolder + r'\\yx_{}_cm-tab10.png'.format(reconstruMeth), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\yx_{}_cm-tab10.eps'.format(reconstruMeth), format='eps', dpi=600, bbox_inches='tight')


# FOV image with scale bar
xS=12
lengthSB = 25
y = 17

fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(decon_mean_stack[1,41,55:75,40:60])
plt.plot([xS,xS+(lengthSB/pixelSize)],[y,y],linewidth=6.0,color='w')

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
pf.forceAspect(ax,aspect=1)
plt.tight_layout()  
plt.savefig(figureFolder + r'\\FOV_{}_3it_25umSB.png'.format(reconstruMeth), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\FOV_{}_3it_25umSB.eps'.format(reconstruMeth), format='eps', dpi=800, bbox_inches='tight')


#########################
#     Lateral FWHM      #
#########################
#yy=[]
#rr1=[]
#rr2=[]

for ii in range(len(deconvolved_result)): 
    print(num_it[ii])
    y=xy[ii,:,0]
    spline = UnivariateSpline(xLat, y-np.max(y)/2, s=0)
    r1, r2 = spline.roots() # find the roots
    FWHM_xy=r2-r1
    
    y=yx[ii,0,:]
    spline = UnivariateSpline(xLat, y-np.max(y)/2, s=0)
    r1, r2 = spline.roots() # find the roots
    FWHM_yx=r2-r1
    
    fields=[date,num_it[ii],FWHM_xy,FWHM_yx,xLoc,yLoc,figureFolder]
    gf.appendCSV(analysisFolder ,r'\lateralResolution_{}_FWHM'.format(reconstruMeth),fields)

#    yy.append(y)
#    rr1.append(r1)
#    rr2.append(r2)
#    


# plot them?  doesnt work properly yet  
fig, ax = plt.subplots(nrows=2, ncols=4)
for row in ax:
    for col in row:
        for it in range(len(yy)):
            col.plot(xLat, yy[it])
            col.axvspan(rr1[it], rr2[it], facecolor='g', alpha=0.5)
fig.set_size_inches(12,4)



################################
#      Axial Resolution        #
################################

# xz and yz are 2d images through z. 

xz=[]
yz=[]
for ii in range(len(deconvolved_result)):
    xz.append(deconvolved_result[ii,:,0,:,xLoc])
    yz.append(deconvolved_result[ii,:,0,yLoc,:])
    
xz=np.array(xz)
yz=np.array(yz)


# plot them?  havent finished
for it in range(len(yz)):
    plt.imshow(yz[it,...])
    
    plt.imshow(xz[it,...])
    


#########################
#      Axial FWHM       #
#########################

for ii in range(len(deconvolved_result)): 
    print(num_it[ii])
    y=xz[ii,:,yLoc]    
    try:
        spline = UnivariateSpline(xAx, y-np.max(y)/2, s=0)
        r1, r2 = spline.roots() # find the roots
    except: # fit a gaussian 
        gaussian = lambda x: 3*np.exp(-(30-x)**2/20.)
        xAx = np.arange(y.size)
        x = np.sum(xAx*y)/np.sum(y)
        width = np.sqrt(np.abs(np.sum((xAx-x)**2*y)/np.sum(y)))
        max = y.max()
        fit = lambda t : max*np.exp(-(t-x)**2/(2*width**2))
        yFit = fit(xAx)
        spline = UnivariateSpline(xAx, yFit-np.max(yFit)/2, s=0)
        r1, r2 = spline.roots() # find the roots
    FWHM_xz=r2-r1
    
    y=yz[ii,:,xLoc]  
    try:
        spline = UnivariateSpline(xAx, y-np.max(y)/2, s=0)
        r1, r2 = spline.roots() # find the roots
    except: 
    # fit a gaussian 
        gaussian = lambda x: 3*np.exp(-(30-x)**2/20.)
        xAx = np.arange(y.size)
        x = np.sum(xAx*y)/np.sum(y)
        width = np.sqrt(np.abs(np.sum((xAx-x)**2*y)/np.sum(y)))
        max = y.max()
        fit = lambda t : max*np.exp(-(t-x)**2/(1*width**2))
        yFit = fit(xAx)
        spline = UnivariateSpline(xAx, yFit-np.max(yFit)/2, s=0)
        r1, r2 = spline.roots() # find the roots
    FWHM_yz=r2-r1
    print(FWHM_yz)
    fields=[date,num_it[ii],FWHM_xz,FWHM_yz,xLoc,yLoc,figureFolder]
    gf.appendCSV(analysisFolder ,r'\axialResolution_660_{}_FWHM'.format(reconstruMeth),fields)

plt.plot(y)
plt.plot(yFit)


ii=7
y=xz[ii,:,yLoc]
plt.plot(y)
plt.plot(yFit)


fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(xz[1,:,round(yLoc-(FOV/2)):round(yLoc+(FOV/2))])

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
plt.tight_layout()  
plt.savefig(figureFolder + r'\\xz_decon_3it_80um.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\xz_decon_3it_80um.eps', format='eps', dpi=800, bbox_inches='tight')


fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(yz[1,:,round(xLoc-(FOV/2)):round(xLoc+(FOV/2))])

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
plt.tight_layout()  
plt.savefig(figureFolder + r'\\yz_decon_3it_80um.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\yz_decon_3it_80um.eps', format='eps', dpi=800, bbox_inches='tight')


##############################
#     Spatial Var Image      #
##############################
# this is from a different cell
#190730\slice2\cell1\MLA2_1x1_50ms_100pA_A_Stim_3

varImage_dec=np.zeros((79,101,101))
for z in range(79):
    varImage_dec[z] = np.var(decon_mean_stack[:,z,...],axis=-0)

xLoc=50
yLoc=57

fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(varImage_dec[40,48:68,40:60])
plt.plot([xS,xS+(lengthSB/pixelSize)],[y,y],linewidth=6.0,color='w')

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
pf.forceAspect(ax,aspect=1)
plt.tight_layout()  
plt.savefig(figureFolder + r'\\varImage_FOV_{}_3it_25umSB.png'.format(reconstruMeth), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\varImage_FOV_{}_3it_25umSB.eps'.format(reconstruMeth), format='eps', dpi=800, bbox_inches='tight')

xz=varImage_dec[:,:,xLoc]
yz=varImage_dec[:,yLoc,:]

fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(xz[:,round(yLoc-(FOV/2)):round(yLoc+(FOV/2))])

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
plt.tight_layout()  
plt.savefig(figureFolder + r'\\varImage_xz_deconvolved_3it_80um.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\varImage_xz_deconvolved_3it_80um.eps', format='eps', dpi=800, bbox_inches='tight')


fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(yz[:,round(xLoc-(FOV/2)):round(xLoc+(FOV/2))])

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
plt.tight_layout()  
plt.savefig(figureFolder + r'\\varImage_yz_deconvolved_3it_80um.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\varImage_yz_deconvolved_3it_80um.eps', format='eps', dpi=800, bbox_inches='tight')


##############################
#      Summary Figures       #
##############################

keyword = 'lateral_yx' # axial or lateral_xy or yx
df = pd.read_csv(analysisFolder + '\\indi_delete\\{}Resolution_deconvolved_FWHM.csv'.format(keyword))
df=df.sort_values(by=['num it'])
df_ref = pd.read_csv(analysisFolder + '\\indi_delete\\{}Resolution_refocused_FWHM.csv'.format(keyword))

df1= pd.read_csv(analysisFolder + '\\indi_delete\\{}Resolution_deconvolved_FWHM_1.csv'.format(keyword))
df2= pd.read_csv(analysisFolder + '\\indi_delete\\{}Resolution_deconvolved_FWHM_2.csv'.format(keyword))
df3= pd.read_csv(analysisFolder + '\\indi_delete\\{}Resolution_deconvolved_FWHM_3.csv'.format(keyword))




#import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
#
#def func(x, a, b, c):
#    return a * np.exp(-b * x) + c
#
#if keyword == 'axial':
#    popt, pcov = curve_fit(func, df['num it'], np.mean(df_ref['xz'])/df['xz'],p0=[8e10,0.4,1])
#elif keyword == 'lateral':
#    popt, pcov = curve_fit(func, df['num it'], np.mean(df_ref['xy'])/df['xy'])




fig = plt.gcf()      
ax = plt.subplot(111)
if keyword == 'axial':
    avg_xz = (np.mean(df_ref['xz'][0])/df1['xz'] + np.mean(df_ref['xz'][1])/df2['xz'] +np.mean(df_ref['xz'][2])/df3['xz'] )/3
    plt.plot(df1['num it'],np.mean(df_ref['xz'][0])/df1['xz'],color='darkgrey',linewidth='2',markersize='14',label='data')
    plt.plot(df2['num it'],np.mean(df_ref['xz'][1])/df2['xz'],color='darkgrey',linewidth='2',markersize='14',label='data')
    plt.plot(df3['num it'],np.mean(df_ref['xz'][2])/df3['xz'],color='darkgrey',linewidth='2', markersize='14',label='data')
    plt.plot(df1['num it'],avg_xz,color='r',linewidth='3',label='Fit')
elif keyword == 'lateral_xy':
    avg_xy = (np.mean(df_ref['xy'][0])/df1['xy'] + np.mean(df_ref['xy'][1])/df2['xy'] +np.mean(df_ref['xy'][2])/df3['xy'] )/3
    plt.plot(df1['num it'],np.mean(df_ref['xy'][0])/df1['xy'],color='darkgrey',linewidth='2',markersize='14',label='data')
    plt.plot(df2['num it'],np.mean(df_ref['xy'][1])/df2['xy'],color='darkgrey',linewidth='2',markersize='14',label='data')
    plt.plot(df3['num it'],np.mean(df_ref['xy'][2])/df3['xy'],color='darkgrey',linewidth='2', markersize='14',label='data')
    plt.plot(df1['num it'],avg_xy,color='r',linewidth='3',label='Fit')
elif keyword == 'lateral_yx':
    avg_yx = (np.mean(df_ref['yx'][0])/df1['yx'] + np.mean(df_ref['yx'][1])/df2['yx'] +np.mean(df_ref['yx'][2])/df3['yx'] )/3
    plt.plot(df1['num it'],np.mean(df_ref['yx'][0])/df1['yx'],color='darkgrey',linewidth='2',markersize='14',label='data')
    plt.plot(df2['num it'],np.mean(df_ref['yx'][1])/df2['yx'],color='darkgrey',linewidth='2',markersize='14',label='data')
    plt.plot(df3['num it'],np.mean(df_ref['yx'][2])/df3['yx'],color='darkgrey',linewidth='2', markersize='14',label='data')
    plt.plot(df1['num it'],avg_yx,color='r',linewidth='3',label='Fit')
    
#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
#plt.yticks(np.arange(1, 1.6, step=0.5)) 
plt.xlabel('Num. Iterations', fontdict = font)
plt.ylabel('$FWHM_r$ / $FWHM_d$', fontdict = font)
fig.set_size_inches(6,5.5)
plt.tight_layout()  
plt.savefig(r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Analysis\spatialRes\Figures' + r'\\{}Resolution_norm_summary.png'.format(keyword), format='png', dpi=600, bbox_inches='tight')
plt.savefig(r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Analysis\spatialRes\Figures' + r'\\{}Resolution_norm_summary.eps'.format(keyword), format='eps', dpi=800, bbox_inches='tight')








#########################
#     Projections       #
#########################

reconstruMeth = 'decon_3it'

result=result_decon

maxProj_z = np.max(result, axis=0)
maxProj_x = np.max(result, axis=1)
maxProj_y = np.max(result, axis=2)

xS=10
xE=70
yS=20
yE=80


xSB=82
lengthSB = 50
y = 93

# z projection
fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(maxProj_z)
#plt.imshow(maxProj_z[xS:xE,yS:yE])
plt.plot([xSB,xSB+(lengthSB/pixelSize)],[y,y],linewidth=6.0,color='w')

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
pf.forceAspect(ax,aspect=1)
plt.tight_layout()  
plt.savefig(figureFolder + r'\\z-proj_{}_{}umSB.png'.format(reconstruMeth,lengthSB), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\z-proj_{}_{}umSB.eps'.format(reconstruMeth,lengthSB), format='eps', dpi=1900, bbox_inches='tight')
plt.close(fig)


# xz, yz projection 
fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(maxProj_x)
#plt.imshow(maxProj_x[:,yS:yE],cmap='gray')
#,aspect='auto',cmap='gray'
#plt.plot([xS,xS+(lengthSB/pixelSize)],[y,y],linewidth=6.0,color='w')

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
#ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
plt.tight_layout()  
plt.savefig(figureFolder + r'\\xz-proj_{}.png'.format(reconstruMeth), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\xz-proj_{}.eps'.format(reconstruMeth), format='eps', dpi=1900, bbox_inches='tight')
plt.close(fig)

fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(maxProj_y)
#plt.imshow(maxProj_y[:,xS:xE],cmap='gray')
#plt.plot([xS,xS+(lengthSB/pixelSize)],[y,y],linewidth=6.0,color='w')

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
#ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
plt.tight_layout()  
plt.savefig(figureFolder + r'\\yz-proj_{}.png'.format(reconstruMeth), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\yz-proj_{}.eps'.format(reconstruMeth), format='eps', dpi=1900, bbox_inches='tight')
plt.close(fig)
    


# dowsample the widefield to the same pixel size as the light field
from scipy import ndimage, misc

downSampledResult = np.zeros((40,101,101))
for z in range(len(result)):
    downSampledResult[z]= ndimage.zoom(result[z,...], 101/len(result[1]))


result=downSampledResult
