#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:45:38 2024

@author: sydneydybing
"""
import numpy as np
import matplotlib.pyplot as plt
from obspy.core import read, Stream

path = '/Users/sydneydybing/SW4/strain/rc_fake-results/'
stas = ['B916', 'B917', 'B918', 'B921']
test_stas = ['B917']

for sta in test_stas:
    
    xx = read(path + str(sta) + '.xx')
    xy = read(path + str(sta) + '.xy')
    xz = read(path + str(sta) + '.xz')
    yy = read(path + str(sta) + '.yy')
    yz = read(path + str(sta) + '.yz')
    zz = read(path + str(sta) + '.zz')
    
    rms = np.sqrt((xx[0].data**2 + xy[0].data**2 + xz[0].data**2 + yy[0].data**2 + yz[0].data**2 + zz[0].data**2)/6)
    
    i = np.where(rms > 0)[0]
    p_arr_idx = i[0]
    
    rms_st = xx.copy()
    rms_st[0].data = rms
    
    rms_data = rms_st[0].data
    rms_times = rms_st[0].times()
    
    mod_rms = rms_st.copy()
    mod_rms_data = mod_rms[0].data
    mod_rms_times = mod_rms[0].times()
    
    p_arr_strain = rms_data[p_arr_idx]
    
    mod_rms_data[:p_arr_idx] = p_arr_strain
    
    pst = mod_rms.copy()
    
    # Loop over samples
    for k in range(0,len(mod_rms[0].data)): #avoid starting at zero

        if k == 0:
            strain = mod_rms[0].data[0]
            max_strain = strain

        else:
            # Grab progressively longer windows and save the biggest strain
            strain = mod_rms[0].data[:k+1] # Has to be k+1 because slicing doesn't include last one
            max_strain = max(strain)

        # Put peak strain back into the output stream
        pst[0].data[k] = max_strain 
    
    plt.figure(dpi = 100)
    plt.title(sta)
    # plt.plot(rms_times, rms_data, color = 'green', label = 'RMS strain')
    plt.plot(mod_rms_times, mod_rms_data, color = 'blue', label = 'RMS strain')
    plt.plot(pst[0].times(), pst[0].data, color = 'red', label = 'Peak strain')
    plt.semilogy()
    plt.axvline(rms_st[0].times()[i[0]], color = 'gray', linestyle = '--', label = 'P-wave arrival')
    plt.xlabel('Time (s)')
    plt.ylabel('Strain')
    plt.xlim(0,20)
    plt.legend()
    plt.show()
    
    
    
    
    
    