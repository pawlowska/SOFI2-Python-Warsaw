# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:12:53 2020

@author: Monik
"""

import os, tifffile
import numpy as np
import matplotlib.pyplot as plt
import SOFI2_0_fromMatlab as sofi2

#%% helper functions
def where_max(a):
    print(a.shape)
    return np.unravel_index(np.argmax(a, axis=None), a.shape)

#%% read data and show mean
data_dir='SOFI2-demo-data/'
T=20

data_timelapse=[np.array(tifffile.imread(os.path.join(data_dir, 'Block'+str(k)+'.tif')), dtype=np.float32) for k in range(1, T+1)]
data_mean_series=np.array([np.mean(data_timelapse[k], axis=0) for k in range(T)])

plt.imshow(data_mean_series[-1])
plt.colorbar()

#%% calculate m6 for all data

m6_series=np.array([sofi2.M6(data_timelapse[k], verbose=True, comment=str(k)) for k in range(T)])

plt.imshow(m6_series[-1])

#%% here I need a better deconvolution!
m6_f=sofi2.filter_timelapse(sofi2.kill_outliers(m6_series))
m6_dcnv=np.array([sofi2.deconvolution(m6_f[k], verbose=True, comment=str(k)) for k in range(T)], dtype=np.float32)
m6_dcnv_f=sofi2.filter_timelapse(m6_dcnv)

#plt.imshow(m6_dcnv_f[-1])
#plt.colorbar()

plt.imshow(m6_dcnv_f[-1])

#%% do ldrc
m6_ldrc_series=np.array([sofi2.ldrc(m6_dcnv_f[k], data_mean_series[k], 25) for k in range(T)])
plt.imshow(m6_ldrc_series[-1])
plt.colorbar()

#%% alternative: ldrc without deconv
m6_ldrc_nodeconv=np.array([sofi2.ldrc(m6_f[k], data_mean_series[k], 25) for k in range(T)])
plt.imshow(m6_ldrc_series[-1])
plt.colorbar()

#%%
tifffile.imsave('demo_means'+'.tif', np.uint16(65500*data_mean_series/data_mean_series.max()))
tifffile.imsave('demo_M6_Deconv_ldrc'+'.tif', np.uint16(65500*m6_ldrc_series/m6_ldrc_series.max()))
tifffile.imsave('demo_M6_noDeconv_ldrc'+'.tif', np.uint16(65500*m6_ldrc_nodeconv/m6_ldrc_nodeconv.max()))

