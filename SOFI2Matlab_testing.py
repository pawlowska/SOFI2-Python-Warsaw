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
"""
fname='2020_02_21_QD_series'
#fname='Bio_QD_test3'
data_dir='pliki/2020_02_21/'
data_path =data_dir+fname+'.asc'

data, setup = sofi.load_asc_movie(data_path)
data=np.array(data)
plt.imshow(data.mean(axis=0))
plt.colorbar()
"""
#%% read data and show mean
data_dir='pliki/SOFI2/'
T=20

data_timelapse=[np.array(tifffile.imread(os.path.join(data_dir, 'Block'+str(k)+'.tif'))) for k in range(1, T+1)]
data_mean_series=[np.mean(data_timelapse[k], axis=0) for k in range(T)]

plt.imshow(data_mean_series[-1])
plt.colorbar()

#%%
m6_series=[sofi2.M6(data_timelapse[k], verbose=True) for k in range(T)]
plt.imshow(m6_series[-1])
plt.colorbar()

#%%
m6_f=sofi2.filter_image(np.moveaxis(np.array(m6_series), 0, -1))
m6_dcnv=[sofi2.deconvolution(m6_f[k]) for k in range(T)]
m6_dcnv_f=sofi2.filter_image(np.moveaxis(np.array(m6_dcnv), 0, -1))
plt.imshow(m6_dcnv_f[-1])
plt.colorbar()

#%%
m6_ldrc_series=[sofi2.ldrc(m6_dcnv_f[k], data_mean_series[k], 25) for k in range(T)]
plt.imshow(m6_ldrc_series[-1])
plt.colorbar()
#%%
temp=m6_ldrc*32500/m6_ldrc_series[-1].max()
tifffile.imsave(fname+'_M6_Deconv_ldrc'+'.tif', np.int16(temp))
