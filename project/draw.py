#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 18:14:02 2019

@author: Jijie LIU
"""
import numpy as np

label = np.loadtxt('project_train/label.txt')

from skimage import io, util, color

I = color.rgb2gray(io.imread("project_train/train/0001.jpg"))
I = util.img_as_float(I)

Irgb = color.gray2rgb(I)
coor = label[0, 1:5].astype(int)

# [1,0,0] == red
# [0,1,0] == green
Irgb[coor[0]:coor[0]+coor[2], coor[1],:] = [0,1,0]
Irgb[coor[0]:coor[0]+coor[2], coor[1]+coor[3],:] = [0,1,0]
# the 4 corners have been drawn
Irgb[coor[0]+1, coor[1]:coor[1]+coor[3],:] = [0,1,0]
Irgb[coor[0]+coor[2], coor[1]:coor[1]+coor[3],:] = [0,1,0]

io.imshow(Irgb)