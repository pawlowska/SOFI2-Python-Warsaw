# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:10:00 2020

@author: Monik
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import restoration
  
def M4(movie, verbose=False, comment=""):
    """    
    calcualte M4 for time series.
    movie : input data as 3D numpy array with time as axis 0
    """
    if (verbose):
        print("Calculating M4"+" "+comment)

    N, xdim, ydim = movie.shape

    #   calculate average image
    ImMean = movie.mean(axis=0);
    
    #   calculate M6 
    M4 = np.zeros((xdim, ydim), dtype=np.float32)
    for i0 in range(N):
        M4=M4+np.power(movie[i0, :, :] - ImMean, 4)
    return M4/N    

def M6(movie, verbose=False, comment=""):
    """    
    calcualte M6 for time series.
    movie : input data as 3D numpy array with time as axis 0
    """
    if (verbose):
        print("Calculating M6"+" "+comment)

    N, xdim, ydim = movie.shape

    #   calculate average image
    ImMean = movie.mean(axis=0);
    
    #   calculate M6 
    M6 = np.zeros((xdim, ydim), dtype=np.float32)
    for i0 in range(N):
        M6=M6+np.power(movie[i0, :, :] - ImMean, 6)
    return M6/N

def kill_outliers(data, threshold=1000):
    m=np.median(data)
    print(m)
    data[data/m>threshold]=m
    return data    
    
    
    

def filter_timelapse(data_series, size=2):
    """
    perform noise filter along time axis using scipy.ndimage
    data : input data as 3D numpy array with time as axis 0
    """
    N, xdim, ydim = data_series.shape
    output=np.array([[gaussian_filter(data_series[:,i,j], sigma=size) for j in range(ydim)] for i in range(xdim)])
    return np.moveaxis(output, -1, 0)

def deconvolution(data, mirror_images=True, verbose=False, comment="", size=5, sigma=2):
    """
    Parameters
    ----------
    data : 2D np array
    mirror_images: Boolean; extend image to reduce FT artifacts on edges
    verbose : Boolean

    Returns
    -------
    Deconvolved array using skimage.restoration.richardson_lucy

    """
    if (verbose):
        print("Calculating Deconvolution "+comment)
    
    #psf = np.ones((size, size), dtype=np.float32) / size**2
    x, y = np.meshgrid(np.linspace(-1,1,size), np.linspace(-1,1,size))
    psf=np.exp(-(x*x+y*y) / (2.0 * sigma**2) ) 
    mx=data.max()
    if (mirror_images):
        data=np.hstack((data, np.flip(data,1)))
        data=np.vstack((data, np.flip(data,0)))
    deconvolved_RL = restoration.richardson_lucy(np.array(data/mx, dtype=np.float16), psf)
    if (mirror_images):
        x, y= deconvolved_RL.shape
        deconvolved_RL=deconvolved_RL[:int(x/2), :int(y/2)]

    return mx*deconvolved_RL

def shrinking_kernel_deconvolution(data, I=20, verbose=False, comment=""):
    """
    

    Parameters
    ----------
    data : 2D np array
    I : Number of iterations. The default is 20.
    verbose : Whether to print to console. The default is False.
    comment : String. The default is "".

    Returns
    -------
    None.

    """
    if (verbose):
        print("Calculating Deconvolution "+comment)

    mx=data.max()
        
    l = 1.5 #DeconvSK parameter
    size=10
    sigma=2
    x, y = np.meshgrid(np.linspace(-1,1,size), np.linspace(-1,1,size))
    psf0=np.exp(-(x*x+y*y) / (2.0 * sigma**2) ) 
    print(np.max(psf0))

    deconvolved_SK=np.array(data/mx, dtype= np.float16)

    for i in range(I):
        alpha = np.power(l, i)/(l-1)
        deconvolved_SK = restoration.richardson_lucy(deconvolved_SK, np.power(psf0, alpha), 1)

    return mx*deconvolved_SK

    

def ldrc(data, mask, w):
    """
    data : 2D image input
    mask: the reference mask of brightness (choose either AC2 or Mean)
    w: windowsize.
    """
    xdim, ydim = data.shape
    output = np.zeros((xdim, ydim))
    Fla = np.zeros((xdim, ydim))
    FlaP = np.ones((w, w))
    for i0 in range(xdim-w+1):
        for i1 in range(ydim-w+1):
            p = data[i0:i0+w,i1:i1+w]
            p = p/p.max()
            output[i0:i0+w,i1:i1+w] = output[i0:i0+w,i1:i1+w] + p* mask[i0:i0+w,i1:i1+w].max();
            Fla[i0:i0+w,i1:i1+w] = Fla[i0:i0+w,i1:i1+w] + FlaP;
            
    return output/Fla
