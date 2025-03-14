# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:46:15 2024

@author: achal
"""

import cv2
import numpy as np
from scipy.ndimage import maximum_filter

start_time = cv2.getTickCount()

# Charger l'image
image = cv2.imread('fourn.png', cv2.IMREAD_GRAYSCALE)

# Réduction du bruit
blurred = cv2.GaussianBlur(image, (5, 5), 0)
blurred = cv2.medianBlur(blurred, 5)

# Sobel pour la magnitude et la direction des gradients
grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
magnitude = cv2.magnitude(grad_x, grad_y)
direction = np.arctan2(grad_y, grad_x)

# Détection des contours
threshold = 0.3 * magnitude.max()  # Seuil réduit
contours = (magnitude > threshold).astype(int)

# Nettoyage des contours
kernel = np.ones((3, 3), np.uint8)
contours = cv2.morphologyEx(contours.astype(np.uint8), cv2.MORPH_OPEN, kernel)

# Paramètres de discrétisation
rmin, rmax, dr = 1, 100, 1
cmin, cmax, dc = 1, 100, 1
rad_min, rad_max, drad = 5, 40, 1

# Initialisation de l'accumulateur
acc = np.zeros(((rmax - rmin) // dr + 1,
                (cmax - cmin) // dc + 1,
                (rad_max - rad_min) // drad + 1))

# Étape 4 : Votes dans l'accumulateur
for x in range(image.shape[0]):
    for y in range(image.shape[1]):
        if contours[x, y]:  # Si c'est un pixel de contour
            for r in range(rmin, rmax + 1, dr):
                for c in range(cmin, cmax + 1, dc):
                    rad = int(np.sqrt((x - r)**2 + (y - c)**2))
                    if rad_min <= rad <= rad_max:
                        angle = np.arctan2(x - r, y - c)
                        if np.abs(angle - direction[x, y]) <= np.pi / 18:
                            i = (r - rmin) // dr
                            j = (c - cmin) // dc
                            k = (rad - rad_min) // drad
                            acc[i, j, k] += magnitude[x, y] / ((rad + 1e-5))

# Détection des maxima locaux
local_max = maximum_filter(acc, size=5)
maxima = (acc == local_max) & (acc > 0.25 * acc.max())

indices = np.argwhere(maxima)

# Visualisation des cercles détectés
output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for idx in indices:
    i, j, k = idx
    r = rmin + i * dr
    c = cmin + j * dc
    rad = rad_min + k * drad
    cv2.circle(output, (c, r), rad, (0, 255, 0), 2)
    
end_time = cv2.getTickCount()
time_taken = (end_time - start_time) / cv2.getTickFrequency()
print(f"Temps de calcul : {time_taken:.2f} secondes")

cv2.imshow('Detected Circles', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
