# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 16:06:21 2021

@author: chowe7
"""

import os
import sys
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import general_functions as gf
import json



# input
date = '211201'

# automatically creat results summary from files in folder
folder = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\GCaMP8' + '\\' + date 

lowest_dirs=[]
for root,dirs,files in os.walk(folder):
    if not dirs:
        lowest_dirs.append(root)

#file = r'Y:\projects\thefarm2\live\Firefly\NIR-GECO_imaging\210317\slice1\area1\LF_1P_1x1_800mA_100msExp_stack_1um_1\LF_1P_1x1_800mA_100msExp_stack_1um_1_MMStack_Default_metadata.txt'

count = 1
for trial in lowest_dirs:
    inFile = trial
    print(trial)
    
    (head, tail) = os.path.split( inFile )
    mainFolder = head
    trialFolder = tail
    metaFile = inFile + '\\' + tail + '_MMStack_Default_metadata.txt'
    
    splitResult = trialFolder.split( "_" ) #split on underscores
    identifier = splitResult[0] 
    imaging = splitResult[1] 
    photons = splitResult[2] 
    LEDpower = splitResult[4] 
    LEDpower = int(LEDpower[0:-2])
    chemStim = splitResult[8] 
    
    with open(metaFile, 'r') as f:
        jsonDict = json.loads(f.read())
        
        binning = jsonDict['FrameKey-0-0-0']['Binning']
        frameRate = jsonDict['FrameKey-0-0-0']['Exposure-ms']/1000
        noFrames = jsonDict['Summary']['Frames']
        if noFrames == 1:
            funcStat = 'stack'
        else:
            funcStat = 'func'
        totalTime = frameRate * noFrames
        noPlanes = jsonDict['Summary']['Slices']
        stepSize = jsonDict['Summary']['z-step_um']
        totalDepth = noPlanes * stepSize
        
        if chemStim == '4AP':
            stimYN = 'Y'
        else:
            stimYN = 'N'
    
    if count == 1:
        fields=['main folder', 'trialFolder', 'Identifier','WF/LF', '1/2P', 'LED Power (mA)', 'Binning','static/func', 'Stim', 'timePeriod (s)', 'frames', 'totalTime (s)', 'numPlanes', 'stepSize (um)', 'totalDepth (um)', 'Depth (um)' ,'Usable?', 'x','y','Rdx','Rdy','Ldx','Ldy']
        gf.appendCSV(folder ,r'\results_summary_{}'.format(date),fields)
#        first = 1
    else:
        fields=[mainFolder, trialFolder,identifier, imaging, photons,LEDpower,binning,funcStat,stimYN,frameRate,noFrames,totalTime,noPlanes,stepSize,totalDepth]
        gf.appendCSV(folder ,r'\results_summary_{}'.format(date),fields)
    
    count=count+1


     