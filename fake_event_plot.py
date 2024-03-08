#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:45:38 2024

@author: sydneydybing
"""
import numpy as np
import matplotlib.pyplot as plt

path = ''
data = np.genfromtxt('./VCI_example/ALBH4.txt')

plt.plot(data[:,0],data[:,1], label = 'E-W')
plt.plot(data[:,0],data[:,2], label = 'N-S')
plt.plot(data[:,0],data[:,3], label = 'Vertical')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()