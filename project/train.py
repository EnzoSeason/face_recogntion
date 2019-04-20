# -*- coding: utf-8 -*-
"""
Spyder Editor

Le but de ce fichier
1. générer les exemples postifs et negetifs des visages
2. générer la classifier des visages 

@author: jijie liu, Zhiqi KANG
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
# On double les nombre des visages par stoker les visages inversés.    
img_target = []
for img in label:
    idx = int(img[0])
    o_h = int(img[1])
    o_l = int(img[2])
    h = int(img[3])
    l = int(img[4])
    sub_img = img_raw[idx-1][o_h :o_h + h, o_l : o_l + l]
    im_inverse = np.zeros(sub_img.shape)
    for i in range(sub_img.shape[1]):
        # On met la première colonne à la dernier colonne, etc
        im_inverse[:,sub_img.shape[1]-i-1] = sub_img[:,i]
    img_target.append(sub_img)
    img_target.append(im_inverse)

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
    im = transform.resize(img,(h_fixed,l_fixed), mode='reflect', anti_aliasing=True)
    data_train_pos.append(im)

# 5. générer les exemples négatives
# 5.1. calculer l'aire recouvrement entre l'images positif et l'image généré
def calcul_aire_recouvrement2(line1, line2):
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
    intersec_h =  h_a + h_b - (end_h-start_h)
    
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
        isValide = False
        while not isValide:
            position_h = int(np.random.uniform(0,H))
            h = int(np.random.uniform(int(H/5),int(H/2)))# C'est difficile de choisir la bonne taille 
            if position_h+h > H:
                continue
            position_l = int(np.random.uniform(0,L))
            l = int(h/ratio_HL)
            if position_l+l > L:
                continue
            isValide = True
        img_atr = np.array([position_h, position_l, h, l])
        is_pos = False
        for face in exemples_pos:
            aire_re = calcul_aire_recouvrement(face[1:5], img_atr)
            if aire_re > 0.5:
                is_pos = True
                break
        if is_pos == False:
            img_neg = image[position_h: position_h + h, position_l: position_l + l]
            # Vérification pour les images peu contrastées
            if np.var(img_neg) > 0.001:
                img_neg = transform.resize(img_neg, (h_fixed, l_fixed))
                data_train_neg.append(img_neg)
                nb_img_neg += 1
                #io.imsave("./face_recogntion/neg/%d_%d.jpg"%(i,nb_img_neg), img_neg)
            
# 6. Evaluaiton de la classifieur
data_train = data_train_pos + data_train_neg
train = np.array(data_train)

n_pos = len(data_train_pos)
n_neg = len(data_train_neg)
n_train = n_pos + n_neg
# On utilise l'algo HOG pour améliorer les données de training   
from skimage.feature import hog
# nb of the output of the function hog
n_dim = 8424

x_train_hog = np.zeros((n_train, n_dim))
for i in range(n_train):
    x_train_hog[i,:] = hog(train[i,:,:], block_norm='L2-Hys', transform_sqrt=True)     
y = np.concatenate((np.ones(n_pos), -np.ones(n_neg)))

# Dans cette partie, on évalue la classifieur du "LinearSVC"
# 6.1 séparer "training data" à "training set" et "validation set"
x_train = np.copy(x_train_hog)
y_train = np.copy(y)
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train, 
                                                    test_size=0.33)
# 6.2 training dataon the training set
from sklearn.svm import LinearSVC, SVC
clf_valid = LinearSVC()
clf_valid.fit(X_train, y_train)
y_pred = clf_valid.predict(X_valid)

# 6.3 average_precision
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_pred, y_valid)

# 6.4 plot precision/recall curve
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
precision, recall, _ = precision_recall_curve(y_pred, y_valid)
# 'post': The y value is continued constantly to the right from every x position, 
# i.e. the interval [x[i], x[i+1]) has the value y[i]
plt.step(recall, precision, color='b', where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

# 7. Apprendre un classifieur de visage 
# 7.1 training
# 7.1.1 LinearSVC
x_train = np.copy(x_train_hog)
y_train = np.copy(y)
X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train, 
                                                    test_size=0.33)
clf_valid = LinearSVC()
# cette iteration est pour mettre les faux positifs dans "training set"
# Elle peut améliorer la performence de la classifieur
for turn in range(3):
    clf_valid = LinearSVC()
    clf_valid.fit(X_train, y_train)
    y_pred = clf_valid.predict(X_valid)
    # trouver les faux positifs
    faux_positifs_index = np.logical_and(y_pred[:] == 1, y_valid[:] == -1)
    print(np.sum(faux_positifs_index))
    faux_positifs = X_valid[faux_positifs_index] 
    # Prendre tous les faux positifs comme exemples négatifs supplémentaires.
    X_train = np.vstack((X_train,faux_positifs))
    y_train = np.concatenate((y_train, -np.ones([np.sum(faux_positifs_index)])))
    
# 7.1.2 RandomForest
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier()
clf_rf.fit(x_train_hog,y)
# 7.1.3 SVC
# Ça prend beaucoup du temps pour le générer
# clf_svc = SVC(gamma='auto')
# clf_svc.fit(x_train_hog,y)

# 7.2 sauvegarder la classifieur
from sklearn.externals import joblib
joblib.dump(clf_valid, 'clf_LinearSVC_v1.pkl')
joblib.dump(clf_rf, 'clf_rf_v1.pkl')
# joblib.dump(clf_svc, 'clf_svc_v1.pkl')
