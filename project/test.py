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
clf_proba = joblib.load('clf_proba_v1.pkl')

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
# Dans cette partie, on commence à utiliser une image.
h_fixed = 120
l_fixed = 80
n_dim = 8424
# On utilise une image pour créer l'algo
test_1 = np.array(img_raw[9])
# créer 15 fenetres pour chaque ligne et colonne
from skimage.feature import hog
nbFenetre = 30
step_h = int(np.floor((test_1.shape[0] - h_fixed)/nbFenetre))
step_l = int(np.floor((test_1.shape[1] - l_fixed)/nbFenetre))
n_test = nbFenetre * nbFenetre
x_hog = np.zeros((n_test, n_dim))
idx = 0
faces_predicts = []
# 2.1 un itération pour trouver tous les fênetres qui pouvent obtenir les visgae.
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
            # obtenir le score de détection
            x_predict_proba = clf_proba.predict_proba(x_hog)
            score = x_predict_proba[idx][1]
            # stocker les informations de la img_fenetre
            # il faut faire attention sur la format de la première élément !
            face = np.array([10, int(o_h), int(o_l), int(h_fixed), int(l_fixed), score])
            faces_predicts.append(face)
        idx += 1
faces_predicts = np.array(faces_predicts)
# 2.2 trouver le visage avec le score max
# Pour l'instance, le score est la probabilité d'être un visage
def calcul_aire_recouvrement(line1, line2):
    H = int(max(line1[0]+line1[2], line2[0]+line2[2]))
    L = int(max(line1[1]+line1[3], line2[1]+line2[3]))
    bg1 = np.zeros((H, L))
    bg2 = np.zeros((H,L))
    for i in np.arange(int(line1[0]),int(line1[0]+line1[2])):
        for j in np.arange(int(line1[1]), int(line1[1]+line1[3])):
            bg1[i,j] = 1
    for i in np.arange(int(line2[0]),int(line2[0]+line2[2])):
        for j in np.arange(int(line2[1]), int(line2[1]+line2[3])):
            bg2[i,j] = 1        
    uni = np.logical_or(bg1,bg2).sum()
    inter = np.logical_and(bg1, bg2).sum()
    return inter/uni

# sort par le score asc
faces_sort = np.array(faces_predicts[np.argsort(faces_predicts[:,5])])
faces = []
faces.append(faces_sort[len(faces_sort)-1])
for i in range(len(faces_sort)-2, -1, -1):
    new_face = True
    for img_face in faces:
        f1 = np.copy(img_face[1:5])
        f2 = np.copy(faces_sort[i][1:5])
        aire = calcul_aire_recouvrement(f1,f2)
        if aire > 0.5:
            new_face = False
    if new_face == True:
        faces.append(faces_sort[i])
faces = np.array(faces)