# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 09:47:41 2021

@author: chowe7
"""

import os
import sys
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import general_functions as gf


# input
date = '210317'


# automatically creat results summary from files in folder
folder = r'Y:\projects\thefarm2\live\Firefly\NIR-GECO_imaging' + '\\' + date 



lowest_dirs=[]
for root,dirs,files in os.walk(folder):
    if not dirs:
        lowest_dirs.append(root)

first = 0
for trial in lowest_dirs:
    inFile = trial
    print(trial)
    
    (head, tail) = os.path.split( inFile )
    trialFolder = tail
    (head, tail) = os.path.split( head )
    (head, tail2) = os.path.split( head )
    subFolder = tail2 + '\\' + tail

    splitResult = trialFolder.split( "_" ) #split on underscores
    imaging = splitResult[0] 
    photons = splitResult[1] 
    binning = splitResult[2] 
    LEDpower = splitResult[3] 
    LEDpower = int(LEDpower[0:-2])
    frameRate = splitResult[4] 
    frameRate = int(frameRate[0:-5])
    frameRate = frameRate/1000
    funcStat = splitResult[5] 
    
    if funcStat == 'func':
        noFrames = splitResult[6] 
        noFrames = int(noFrames[0:-6])
        totalTime = frameRate * noFrames
        noPlanes = '-'
        stepSize = '-'
        totalDepth = '-'
        
    elif funcStat == 'stack':    
        totalTime = '-'
        noFrames='-'
        noPlanes = 0
        stepSize = splitResult[6] 
        stepSize = int(stepSize[0:-2])
        totalDepth = noPlanes * stepSize
    
    if first == 0:
        fields=['main folder', 'subFolder', 'trialFolder', 'WF/LF', '1/2P', 'LED Power (mA)', 'Binning','static/func', 'timePeriod (s)', 'frames', 'totalTime (s)', 'numPlanes', 'stepSize (um)', 'totalDepth (um)', 'Usable?', 'x','y','Rdx','Rdy','Ldx','Ldy']
        gf.appendCSV(folder ,r'\results_summary_{}'.format(date),fields)
        first = 1
    else:
        fields=[folder, subFolder, trialFolder, imaging, photons,LEDpower,binning,funcStat,frameRate,noFrames,totalTime,noPlanes,stepSize,totalDepth]
        gf.appendCSV(folder ,r'\results_summary_{}'.format(date),fields)

import pandas as pd





