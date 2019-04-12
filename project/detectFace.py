#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:49:30 2019

Le but de ce fichier: 
1. la détection de visages par fenêtre glissante

@author: jijie liu
"""
import numpy as np

# 0. obtenir la classifier
from sklearn.externals import joblib
clf = joblib.load('clf_hog_v1.pkl')

# 1. obtenir les images originales
from skimage import io, util, color, transform
img_raw = []
n_total = 500
for i in range(n_total):
    # lire les image, et convertir RGB images aux images gray
    im = color.rgb2gray(io.imread("test/" + "%04d"%(i+1) + ".jpg"))
    # Convertir les images en valeurs
    im = util.img_as_float(im)
    img_raw.append(im)

# 2. créer l'algo de "fênetre glissant"
h_fixed = 120
l_fixed = 80
n_dim = 8424
# On utilise une image pour créer l'algo
test_1 = np.array(img_raw[0])
# créer 15 fenetres pour chaque ligne et colonne
from skimage.feature import hog
nbFenetre = 30
step_h = int(np.floor((test_1.shape[0] - h_fixed)/nbFenetre))
step_l = int(np.floor((test_1.shape[1] - l_fixed)/nbFenetre))
n_test = nbFenetre * nbFenetre
x_hog = np.zeros((n_test, n_dim))
idx = 0
faces_predicts = []
for i in range(nbFenetre):
    for j in range(nbFenetre):
        # coordonnées (ligne, colonne) du coin supérieure gauche de la boîte
        # Ils sont (o_h, o_l)
        o_h = step_h * i
        o_l = step_l * j
        img_fenetre = test_1[o_h: o_h + h_fixed, o_l: o_l + l_fixed]
        x_hog[idx,:] = hog(img_fenetre, block_norm='L2-Hys')
        x_predict = clf.predict(x_hog)
        if (x_predict[idx] == 1):
            # stocker les informations de la img_fenetre
            face = np.array([1, o_h, o_l, h_fixed, l_fixed])
            faces_predicts.append(face)
        idx += 1