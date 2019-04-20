#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:49:30 2019
Le but de ce fichier: 
1. la détection de visages par fenêtre glissante
@author: jijie liu, Zhiqi KANG
"""
import numpy as np

# 0. obtenir la classifier
from sklearn.externals import joblib
clf_LinearSVC = joblib.load('clf_LinearSVC_v1.pkl')
clf_rf = joblib.load('clf_rf_v1.pkl')
# clf_svc = joblib.load('clf_svc_v1.pkl')
# 1. obtenir les images originales
from skimage import io, util, color
img_raw = []
n_total = 500
for i in range(n_total):
    # lire les image, et convertir RGB images aux images gray
    im = color.rgb2gray(io.imread("test/" + "%04d"%(i+1) + ".jpg"))
    # Convertir les images en valeurs
    im = util.img_as_float(im)
    img_raw.append(im)

# 2. créer l'algo de "fênetre glissant"
# 2.1 créer la fonction pour calculer l'aire de recouvrement
def calcul_aire_recouvrement(fenetre_a, fenetre_b):
    IoU = 0
    position_h_a = fenetre_a[0]
    position_l_a = fenetre_a[1]
    h_a = fenetre_a[2]
    l_a = fenetre_a[3]
    
    position_h_b = fenetre_b[0]
    position_l_b = fenetre_b[1]
    h_b = fenetre_b[2]
    l_b = fenetre_b[3]
    
    #Intersection de hauteur
    start_h = min(position_h_a, position_h_b)
    end_h = max(position_h_a+h_a, position_h_b+h_b)
    intersec_h = h_a + h_b - (end_h-start_h)
    
    #Intersection de largeur
    start_l = min(position_l_a, position_l_b)
    end_l = max(position_l_a+l_a, position_l_b+l_b)
    intersec_l = l_a+l_b-(end_l-start_l)
        
    #Si Intersection >0 on calcule IoU
    if intersec_h > 0 and intersec_l > 0:
        intersec_area = intersec_h * intersec_l
        aire_a = h_a*l_a
        aire_b = h_b*l_b
        IoU = intersec_area/(aire_a+aire_b-intersec_area)
    return IoU
# 2.2 créer la fonction pour les fênetres glissantes
from skimage.feature import hog
def execute_fenetre_glissante(clf_name, img, nb_img):  
    if clf_name == 'LinearSVC':
        clf = clf_LinearSVC
    if clf_name == 'rf':
        clf = clf_rf
    #if clf_name == 'svc':
    #    clf = clf_svc
    h_fixed = 120
    l_fixed = 80
    # On utilise une image pour créer l'algo
    img = np.array(img)  
    idx = 0
    faces_predicts = np.empty((0, 6))
    # 2.1 un itération pour trouver tous les fênetres qui pouvent obtenir les visgae.
    for h in range(0, img.shape[0]-h_fixed, 30):
        for l in range(0, img.shape[1]-l_fixed, 20):
            # coordonnées (ligne, colonne) du coin supérieure gauche de la boîte
            # Ils sont (o_h, o_l)
            img_fenetre = img[h: h + h_fixed, l: l + l_fixed]
            feature_hog = hog(img_fenetre, block_norm='L2-Hys', transform_sqrt=True)
            x_predict = clf.predict(feature_hog.reshape(1, -1))
            if (x_predict == 1):
                # obtenir le score de détection
                if clf_name == 'LinearSVC' or clf_name == 'svc':
                    x_predict_proba = clf.decision_function(feature_hog.reshape(1, -1))
                    score = x_predict_proba[0]
                if clf_name == 'rf':
                    x_predict_proba = clf.predict_proba(feature_hog.reshape(1, -1))
                    score = x_predict_proba[0][1]
                # stocker les informations de la img_fenetre
                face = np.array([nb_img, int(h), int(l), int(h_fixed), int(l_fixed), score])
                faces_predicts = np.concatenate((faces_predicts, face.reshape(1,6)))
                idx += 1    
    return faces_predicts

# 2.3 supprimier les doublons, et choisir les meilleurs visages 
def select_meilleurs(faces_candidat):
    faces_sort = np.array(faces_candidat[np.argsort(faces_candidat[:,5])])
    faces = []
    faces.append(faces_sort[len(faces_sort)-1])
    for i in range(len(faces_sort)-2, -1, -1):
        new_face = True
        for img_face in faces:
            f1 = np.copy(img_face[1:5])
            f2 = np.copy(faces_sort[i][1:5])
            aire = calcul_aire_recouvrement(f1,f2)
            # si l'aire de recouvrement > 0.1, cette visage est déjà stockée.
            if aire > 0.1:
                new_face = False
                break
        if new_face == True:
            faces.append(faces_sort[i])
    faces = np.array(faces)
    return faces

# 2.4 utiliser la fonction "execute_fenetre_glissante" et la fonction "select_meilleurs"
# pour predicter les visages    
def predict_img(clf, img, nb_img):
    from skimage.transform import  rescale
    # on fixe la taile de la  boîte englobante de visages.
    # on change la taile d'image par ratios
    ratios = [0.1, 0.3, 0.6, 0.8, 1, 1.2, 1.5, 2]    
    faces_candidat = np.empty((0,6))
    for ratio in ratios:
        img_temp = rescale(img, ratio, multichannel=False,
                           mode='reflect', anti_aliasing=True)
        faces = execute_fenetre_glissante(clf, img_temp, nb_img)
        if faces is not None:
            faces[:, 1:5] = (faces[:, 1:5]/ratio).astype(int)
            faces_candidat = np.concatenate((faces_candidat, faces))
    faces_best = np.empty((0, 6))
    if len(faces_candidat) > 0:
        faces_best = select_meilleurs(faces_candidat)
        print(faces_best)
    return faces_best

# 3. predicter les visages et les stocker dans le fichier
faces = []
for i in np.arange(0,500):
        print("-------" + str(i) + "-------")
        i_faces = predict_img('LinearSVC',img_raw[i], i+1)
        if i_faces is not None:    
            for face_dec in i_faces:
                faces.append(face_dec)

faces = np.array(faces)            
output = np.savetxt('detection.txt', faces)
