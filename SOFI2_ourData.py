# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:12:53 2020

@author: Monik
"""

import os, tifffile
import numpy as np
import matplotlib.pyplot as plt
import sofi
import SOFI2_0_fromMatlab as sofi2

#%% read data and show mean

fname='QD_series_1'
data_dir='pliki/2020_02_21-QDs/'
data_path =data_dir+fname

data=np.array(tifffile.imread(os.path.join(data_dir, fname+'.tif')))
data_mean=data.mean(axis=0)
plt.imshow(data_mean)
plt.colorbar()

#%% calculate cumulants
ac2=sofi.autocumulant_2(data, 1)
plt.imshow(ac2)
plt.colorbar()

#%%
ac4=sofi.autocumulant_4(data, 1, 2, 3)
plt.imshow(ac4)
plt.colorbar()

#%% m4 with ldrc
m4_ldrc=sofi2.ldrc(sofi2.M4(data), data_mean, 15)
plt.imshow(m4_ldrc)
plt.colorbar()

#%% calculate m6 with ldrc
m6_ldrc=sofi2.ldrc(sofi2.M6(data), data_mean, 15)
plt.imshow(m6_ldrc)
plt.colorbar()

#%% save tifs
def normTo16bit(a):
    return np.int16(32500*(a/a.max()))

tifffile.imsave(data_dir+fname+'_mean'+'.tif', normTo16bit(data_mean))
tifffile.imsave(data_dir+fname+'_ac2'+'.tif', normTo16bit(ac2))
tifffile.imsave(data_dir+fname+'_ac4'+'.tif', normTo16bit(ac4))
tifffile.imsave(data_dir+fname+'_m4'+'.tif', normTo16bit(m4_ldrc))
tifffile.imsave(data_dir+fname+'_m6'+'.tif', normTo16bit(m6_ldrc))
