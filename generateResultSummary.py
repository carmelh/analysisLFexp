# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:02:30 2019

@author: chowe7

get result summary
output = result_summary_{}.format(date)

"""

import numpy as np
import sys
import pandas as pd
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import general_functions as gf



cwd=r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190730'

# create file and save header
fields=['','date','slice','cell','Imaging','Order','Exp Time','Frames','Total Time','LED power','Stim Prot','Usable','Dark folder','Dark file','Depth','x','y','Rdx','Rdy','Ldx','Ldy','Refoc','Decon','WF','Notes']
gf.appendCSV(cwd ,r'\result_summary_{}'.format(date),fields)







fields=[currentFile,df.at[currentFile, 'slice'], df.at[currentFile, 'cell'],stim,df.at[currentFile, 'LED power'],'',SNR,baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, bleach, fileNameDark, baselineDarkNoise]
gf.appendCSV(cwd ,r'\result_summary_{}'.format(date),fields)




def getFilenamesByExtension(path,fileExtension = '.tif',recursive_bool = True):
    if recursive_bool:
        return [file for file in glob.glob(path + '/**/*'+fileExtension, recursive=recursive_bool)]
    else:
        return [file for file in glob.glob(path + '/**'+fileExtension, recursive=recursive_bool)]


def get_tif_metadata(tif_filepath):
    with tifffile.TiffFile(tif_filepath) as tif:
        meta_dict = tif.micromanager_metadata
    return meta_dict


tif_filepath = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Bulk\191024\slice1\cell1\WF_1x1_100mA_50ms_A-Stim_2\WF_1x1_100mA_50ms_A-Stim_2_MMStack_Pos0.ome.tif'


def get_int_from_string(string,loc,direction = 1):
    count = 0
    while True:
        try:
            if direction == 1:
                int(string[loc:loc + count +1])
                if loc +count > len(string):
                    break
            elif direction == -1:
                int(string[loc-count:loc+1])
            else:
                raise ValueError('Direction argument must be 1 or -1')
            count += 1
        except Exception:
            break
        
    if direction == 1:
        return int(string[loc:loc + count])
    elif direction == -1:
        return int(string[loc-count+1:loc+1])


import pandas as pd
import os
import f.general_functions as gf



#this script goes through an experimental day and extracts parameters from the filenames
repeats = [x for x in os.walk(cwd) if np.logical_and(len(x[1])%5 == 0,len(x[1])!=0)]
folder = [x[0] for x in repeats]


for root, dirs, files in os.walk(cwd):
  # for name in files:
   #   print(os.path.join(root, name))
   for name in dirs:
      print(os.path.join(root, name))
      dirname = os.path.join(root, name)
      slicenum.append(get_int_from_string(dirname,dirname.find('slice')+5))
      cellnum.append(get_int_from_string(dirname,dirname.find('cell')))


folder = []
day = []
slicenum = []
cellnum = []
n_repeats = []
f = []
pos = []
MLA = []
file_key = []
ephys_file = []

for repeat in repeats:
    dirname = repeat[0]
    folder.append(dirname)
    day.append(get_int_from_string(dirname,dirname.find('19')))
    slicenum.append(get_int_from_string(dirname,dirname.find('slice')+5))
    cellnum.append(get_int_from_string(dirname,dirname.find('cell')+4))
    n_repeats.append(len(repeat[1]))
    
    celldir = os.path.realpath(os.path.join(dirname,os.pardir,os.pardir))
    ephys_files = [a for a in os.listdir(celldir) if a.find('.smr')!=-1]
    if len(ephys_files) ==1:
        ephys_file.append(os.path.join(celldir,ephys_files[0]))
    else:
        ephys_file.append(-1)
    
    if any('A-Stim' in string for string in repeat[1]):
        file_key.append('A-Stim')
    else:
        file_key.append(0)
    
    if dirname.find('WF') == -1 and dirname.find('LF') == -1:
        MLA.append(True)
    else:
        MLA.append(False)
        
df = pd.DataFrame({'day':day,'cell':cellnum,'slice':slicenum,'MLA':MLA,'folder':folder, 'file_key':file_key})
df.to_excel(cwd+'/auto_data_summary_'+str(day[0])+'.xlsx')



import os

path = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Bulk\191024'


dirname = []

# r=root, d=directories, f = files
for root, d, f in os.walk(path):
    for folder in d:
        dirname.append(os.path.join(root, folder))
        

for f in dirname:
    print(f)
    
    
    if dirname.find('WF') == -1 and dirname.find('LF') == -1:
        MLA.append('Y')
    else:
        MLA.append('N')
    
    df = pd.DataFrame({'':f,'date':date,'slice':slicenum,'cell':cellnum,'Imaging':MLA})
    df.to_csv(cwd+'/auto_data_summary_'+str(day[0])+'.csv')
