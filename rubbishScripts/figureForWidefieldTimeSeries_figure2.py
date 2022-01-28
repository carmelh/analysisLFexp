# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 09:32:25 2022

@author: chowe7
"""



fd=90

trialData=north_of_somatxt[:,1]
f0 = np.mean(trialData[0:13])

processedTrace = []

# now calc the following: (f-f0)/(f0-fdark)    
for element in trialData:
    processedTrace.append((element-f0)/(f0-fd))


farDend=np.array(processedTrace)*100
leftSoma=np.array(processedTrace)*100
northSoma=np.array(processedTrace)*100

ts=1/20
time = np.arange(0, ts*len(trialData), ts)

fig = plt.gcf()      
ax = plt.subplot(111)  
plt.plot(time[0:122],northSoma[0:122]+14,linewidth=3.0,color='#e30613')
plt.plot(time[0:122],farDend[0:122]+6.5,linewidth=3.0,color='#f39200')
plt.plot(time[0:122],leftSoma[0:122],linewidth=3.0,color='#e6007e')

plt.plot([6,6],[9,12],linewidth=4.0,color='k')
plt.plot([6,8],[9,9],linewidth=4.0,color='k')

fig.set_size_inches(8,8)

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
pf.saveFigure(fig,path,'timeSeries_wf_3ROIs_SBs-3per_2sec')