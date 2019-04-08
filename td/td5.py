#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:43:04 2019

@author: jijie liu
"""

# Classification des images
# La présentation des images qui sont proches de l'image d'origine
# Il faut, d'abord, calculer descr_hist0, descr_hist qui ne sont pas présentés dans ce fichier.

import scipy as sp

d = sp.spatial.distance_matrix(descr_hist0, descr_hist)
d_i = np.argsort(d[0,:])

plt.subplot(3,3,1) # subplot(nrows, ncols, index, **kwargs)
plt.imshow(io.imread('cougar.jpg'))
for i in range(2, 9+1):
    plt.subplot(3,3,i)
    plt.imshow(plt.imread('images/' + '%03d'%(d_i[i-2]) + '.jpg'))