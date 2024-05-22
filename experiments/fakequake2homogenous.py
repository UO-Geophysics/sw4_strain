#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:35:56 2024

@author: dmelgarm
"""

import numpy as np
import matplotlib.pyplot as plt
from pyproj import Geod




rupture = np.genfromtxt('/Users/dmelgarm/FakeQuakes/Alasak_PTHA/output/ruptures/production.002598.rupt')
out_file = '/Users/dmelgarm/FakeQuakes/Alasak_PTHA/output/homogenous/production_homog_MID.002598.rupt'

hypo_depth =  20
depth_tol = 1.5
v_rupt = 2.8
rise_time = 20

#Where is slip non zero?
i = np.where(rupture[:,9]>0)[0]

#get magnitude
slip = (rupture[:,8]**2 + rupture[:,9]**2)**0.5
area = rupture[:,10]*rupture[:,11]
rigidity = rupture[:,13]
M0 = np.sum(slip*area*rigidity)
Mw = (2/3)*(np.log10(M0)-9.1)

#equivalent slip
equivalent_slip = M0 / (np.sum(area*rigidity)) 
#sanity check
equivalent_Mw = (2/3)*(np.log10(np.sum(area*rigidity*equivalent_slip))-9.1)

#replace slip and rise time
rupture[:,7] = 0
rupture[i,7] = rise_time
rupture[i,8] = 0
rupture[i,9] = equivalent_slip

# #Find new hypocenter for WEST to EAST
# j = np.where((rupture[i,3]>hypo_depth-depth_tol) & (rupture[i,3]<hypo_depth+depth_tol))[0]
# hypo_fault = np.argmin(rupture[i[j],1])
# hypo_fault = i[j[hypo_fault]]
# hypocenter = rupture[hypo_fault,1:4]

# #Find new hypocenter for EAST to WEST
# j = np.where((rupture[i,3]>hypo_depth-depth_tol) & (rupture[i,3]<hypo_depth+depth_tol))[0]
# hypo_fault = np.argmax(rupture[i[j],1])
# hypo_fault = i[j[hypo_fault]]
# hypocenter = rupture[hypo_fault,1:4]

#Find new hypocenter for BILATERAL
hypocenter = np.array([-152.65,57.05,0])
dist_to_hypo = (rupture[i,1]-hypocenter[0])**2 + (rupture[i,2]-hypocenter[1])**2
j = np.argmin(dist_to_hypo)
hypocenter = rupture[i[j],1:4]



#get inter-fault_distances

# Define the WGS84 ellipsoid
geod = Geod(ellps="WGS84")
distances = [geod.inv(hypocenter[0], hypocenter[1], point[0], point[1])[2] for point in rupture[i,1:3]]
distances = np.array(distances)/1000
# distances = (distances**2 + (hypocenter[2] - rupture[i,3])**2)**0.5
onset_time = distances / v_rupt
rupture[i,12] = onset_time


formatter = "%d\t%.6f\t%.6f\t%.4f\t%.2f\t%.2f\t%.1f\t%.9e\t%.4e\t%.4e\t%.2f\t%.2f\t%.9e\t%.6e\t%.6f"
np.savetxt(out_file,rupture,fmt=formatter)

#make figures to check
plt.subplot(1,3,1)
plt.scatter(rupture[:,1],rupture[:,2],c=slip,cmap='magma')
plt.colorbar(label='Slip (m)')

plt.subplot(1,3,2)
plt.scatter(rupture[:,1],rupture[:,2],c=rupture[:,9],cmap='magma',vmax=slip.max())
plt.colorbar(label='Slip (m)')
plt.scatter(hypocenter[0],hypocenter[1],marker='*',s=90,facecolor='w')

plt.subplot(1,3,3)
plt.scatter(rupture[:,1],rupture[:,2],c=rupture[:,12],cmap='inferno')
plt.colorbar(label='onset time (s)')
plt.scatter(hypocenter[0],hypocenter[1],marker='*',s=90,facecolor='w')

