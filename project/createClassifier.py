# -*- coding: utf-8 -*-
"""
Spyder Editor

Le but de ce fichier
1. générer les exemples postifs et negetifs des visages
2. générer la classifier des visages 

@author: jijie liu
"""            
import numpy as np

label = np.loadtxt('project_train/label.txt')

from skimage import io, util, color, transform
# 1. obtenir les images originales
img_raw = []
n_total = 1000
for i in range(n_total):
    # lire les image, et convertir RGB images aux images gray
    im = color.rgb2gray(io.imread("project_train/train/" + "%04d"%(i+1) + ".jpg"))
    # Convertir les images en valeurs
    im = util.img_as_float(im)
    img_raw.append(im)
    
# 2. obtenir les visages
img_target = []
for img in label:
    idx = int(img[0])
    o_h = int(img[1])
    o_l = int(img[2])
    h = int(img[3])
    l = int(img[4])
    sub_img = img_raw[idx-1][o_h :o_h + h, o_l : o_l + l]
    img_target.append(sub_img)

# 3. Calculer le ratio moyen entre la hauteur et la largeur d'une boîte englobante de visages.
mean_hl = np.mean(label[:,3]/label[:,4])
ratio_HL = 1.5

# 4. Définir une taille fixe de boîte de visage
# puis extraire tous les visages des images en les redimensionnant à la bonne taille.
from statistics import mode
l_fixed = int(mode(label[:,4]))+1
h_fixed = int(ratio_HL * l_fixed)

data_train_pos = []
for img in img_target:
    im = transform.resize(img,(h_fixed,l_fixed))
    data_train_pos.append(im)

# 5. générer les exemples négatives
# 5.1. calculer l'aire recouvrement entre l'images positif et l'image généré
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

# 5.2  générer les exemples négatives
# Si l'aire recouvrement > 0.5, alors cette image est positive 
data_train_neg = []
for i in range(len(img_raw)):
    H = img_raw[i].shape[0]
    L = img_raw[i].shape[1]
    image = img_raw[i]
    exemples_pos = []
    for face in label:
        if face[0] == i+1:
            exemples_pos.append(face)
    nb_img_neg = 0
    while nb_img_neg < 10:
        # obtenir le coordonnées (ligne, colonne) du coin supérieure gauche de la boîte
        # position_h, position_l et la taille de hauteur h aléatoirement
        position_h = int(np.random.uniform(0,H))
        position_l = int(np.random.uniform(0,L))
        h = int(np.random.uniform(int(H/5),int(H/2))) # C'est difficile de choisir la bonne taille
        l = int(h/ratio_HL)
        img_atr = np.array([position_h, position_l, h, l])
        is_pos = False
        for face in exemples_pos:
            aire_re = calcul_aire_recouvrement(face[1:5], img_atr)
            if aire_re > 0.1:
                nb_cover = True
        if is_pos == False:
            img_neg = image[position_h: position_h + h, position_l: position_l + l]
            img_neg = transform.resize(img_neg, (h_fixed, l_fixed))
            data_train_neg.append(img_neg)
            nb_img_neg += 1

# 6. Apprendre un classifieur de visage 
data_train = data_train_pos + data_train_neg
train = np.array(data_train)

n_pos = len(data_train_pos)
n_neg = len(data_train_neg)
n_train = n_pos + n_neg

# 6.1.  On utilise l'algo HOG pour améliorer les données de training   
from skimage.feature import hog
# nb of the output of the function hog
n_dim = 8424

x_train_hog = np.zeros((n_train, n_dim))
for i in range(n_train):
    x_train_hog[i,:] = hog(train[i,:,:], block_norm='L2-Hys')     
y_train = np.concatenate((np.ones(n_pos) , -np.ones(n_neg)))

# 6.2 training
from sklearn.svm import LinearSVC
clf_hog = LinearSVC()
clf_hog.fit(x_train_hog,y_train)

# 6.3 sauvegarder la classifieur
from sklearn.externals import joblib
joblib.dump(clf_hog, 'clf_hog_v1.pkl')
