# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 13:18:36 2021

@author: chowe7
"""









maxProj_z_660 = np.max(result, axis=0)
maxProj_x_600 = np.max(result, axis=1)
maxProj_y_660 = np.max(result, axis=2)


fig = plt.gcf()      
ax1 = plt.subplot(121)  
ax1=plt.imshow(maxProj_z[45:55,45:55])
ax2 = plt.subplot(122)  
ax2=plt.imshow(maxProj_z_660[45:55,45:55])




pixelSize=5

xLat=np.linspace(0, result.shape[1]*pixelSize, result.shape[2])  #x in um
xAx=np.linspace(0, result.shape[0], result.shape[0])  #x in um



xLoc=50
yLoc=50

# synthetically refocused
xy_550=decon_mean_stack[18,10,:,xLoc:xLoc+1]
yx_550=decon_mean_stack[18,10,yLoc:yLoc+1,:]


xy_660=decon_mean_stack_660[18,10,:,xLoc:xLoc+1]
yx_660=decon_mean_stack_660[18,10,yLoc:yLoc+1,:]


plt.plot(xy_550)
plt.plot(xy_660)



spline = UnivariateSpline(xLat, xy_660-np.max(xy_660)/2, s=0)
r1, r2 = spline.roots() # find the roots
FWHM_xy=r2-r1


spline = UnivariateSpline(xLat, xy_660-np.max(xy_660)/2, s=0)
r1, r2 = spline.roots() # find the roots
FWHM_yx=r2-r1




xz_550=decon_mean_stack[18,:,:,xLoc]
yz_550=decon_mean_stack[18,:,yLoc,:]

xz_660=decon_mean_stack_660[18,:,:,xLoc]
yz_660=decon_mean_stack_660[18,:,yLoc,:]

y2=xz_660[:,yLoc]
#########################
#      Axial FWHM       #
#########################

y=xz_550[:,yLoc]
spline = UnivariateSpline(xAx, y-np.max(y)/2, s=0)
r1, r2 = spline.roots() # find the roots
FWHM_xz=r2-r1

y=xz_550[:,xLoc]
spline = UnivariateSpline(xAx, y-np.max(y)/2, s=0)
r1, r2 = spline.roots() # find the roots
FWHM_yz=r2-r1

plt.plot(y)
plt.plot(y2)