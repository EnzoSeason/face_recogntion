# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: jijie liu
"""
import numpy as np

label = np.loadtxt('project_train/label.txt')

from skimage import io, util, color, transform
# obtenir les images originales
img_raw = []
n_total = 1000
for i in range(n_total):
    # lire les image, et convertir RGB images aux images gray
    im = color.rgb2gray(io.imread("project_train/train/" + "%04d"%(i+1) + ".jpg"))
    # Convertir les images en valeurs
    im = util.img_as_float(im)
    img_raw.append(im)
    
# obtenir les visages
img_target = []
for img in label:
    idx = int(img[0])
    o_h = int(img[1])
    o_l = int(img[2])
    h = int(img[3])
    l = int(img[4])
    sub_img = img_raw[idx-1][o_h :o_h + h, o_l : o_l + l]
    img_target.append(sub_img)

# Calculer le ratio moyen entre la hauteur et la largeur d'une boîte englobante de visages.
mean_hl = np.mean(label[:,3]/label[:,4])
ratio_HL = 1.5

# Définir une taille fixe de boîte de visage
# puis extraire tous les visages des images en les redimensionnant à la bonne taille.
from statistics import mode
l_fixed = int(mode(label[:,4]))+1
h_fixed = int(ratio_HL * l_fixed)

img_resize = []
for img in img_target:
    im = transform.resize(img,(h_fixed,l_fixed))
    img_resize.append(im)
io.imshow(img_resize[9])


