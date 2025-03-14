# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 19:54:10 2024

@author: achal
"""

import cv2
import numpy as np
from scipy.ndimage import maximum_filter

start_time = cv2.getTickCount()

# Charger l'image en niveaux de gris
image = cv2.imread('four.png', cv2.IMREAD_GRAYSCALE)

# Calculer les gradients en x et y
grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Calculer la magnitude du gradient
gradient_magnitude = cv2.magnitude(grad_x, grad_y)
gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)

# Binariser la magnitude pour extraire les contours
_, contours = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)
contours = (contours > 0).astype(int)

#contours_visualization = (contours * 255).astype(np.uint8) 
#cv2.imshow('Contours Detected', contours_visualization)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Paramètres de discrétisation
rmin, rmax, dr = 1, 100, 1
cmin, cmax, dc = 1, 100, 1
rad_min, rad_max, drad = 5, 100, 1

# Initialisation de l'accumulateur
acc = np.zeros(((rmax - rmin) // dr + 1,
                (cmax - cmin) // dc + 1,
                (rad_max - rad_min) // drad + 1))

# Votes dans l'accumulateur
for x in range(image.shape[0]):
    for y in range(image.shape[1]):
        if contours[x, y]:  # Si c'est un pixel de contour
            for r in range(rmin, rmax + 1, dr):
                for c in range(cmin, cmax + 1, dc):
                    rad = int(np.sqrt((x - r)**2 + (y - c)**2))
                    if rad_min <= rad <= rad_max:
                        i = (r - rmin) // dr
                        j = (c - cmin) // dc
                        k = (rad - rad_min) // drad
                        acc[i, j, k] += 1 / (rad + 1e-5)  # Votes pondérés avec normalisation

# Détection des maxima locaux
local_max = maximum_filter(acc, size=3)
maxima = (acc == local_max) & (acc > 0.55 * acc.max()) 
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



