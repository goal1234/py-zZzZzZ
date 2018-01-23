# -*- coding: utf-8 -*-
"""
Created on Fri Feb 03 18:03:49 2017

@author: arch618
"""

import numpy as np
import matplotlib.pyplot as plt

#%% 
def is_outlier(points,threshold = 3.5):
      """
      return a boolean array with True if points are outliers and False
      otherwise
      
      Data points with a modified z-score greater than this
      #value will be classified as outliers
      """
      
      #transform into vector
      if len(points.shape)==1:
          points = points[:,None]
          
      #compute median value
      median = np.median(points,axis=0)
      
      #compute diff sums along the axis
      diff = np.sum((points-median)**2,axis=-1)
      diff = np.sqrt(diff)
      #compute MAD
      med_abs_deviation = np.median(diff)
      
      #compute modeifed Z-score
      #http://www.itl.nist.gov/div888/handbook/eda/
      #iglewicz
      modified_z_score = 0.6745*diff/med_abs_deviation
      
      #return a mask for each outlier
      return modified_z_score > threshold
  
      #Random data
      x = np.random.random(100)
      
      #histogram buckers
      buckers = 50
      
      #Add in a few outliers
      x = np.r_[x,-49,95,100,-100]
      
      #keep valid data points
      #note here that
      #"~" is logical not on boolearn numpy arrays
      filtered =x[~is_outlier(x)]
      
      #plot histograms
      plt.figure()
      
      plt.subplot(211)
      plt.hist(x,buckers)
      plt.xlabel('Raw')
      
      plt.subplot(212)
      plt.hist(filtered,buckers)
      plt.xlabel('Cleaned')
      
      plt.show()
      
      