# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:12:25 2022
@author: arnab
"""

import cv2
import numpy as np
import scipy.signal as sig

#Preserve Edges
src = cv2.imread("D:/tulsi-leaves.jpg")
edge_preserve = cv2.edgePreservingFilter(src, flags=1, sigma_s=40, sigma_r=0.3)
cv2.imshow("edge preserve",edge_preserve)

#Stylize Leaf Boundary
sty = cv2.stylization(edge_preserve, sigma_s=60, sigma_r=0.07)
cv2.imshow("Stylize",sty)

#Multi Channel to Single Channel Conversion
B,G,R = sty[:,:,0], sty[:,:,1], sty[:,:,2]
imgGray = 0.114 * B + 0.5870 * G + 0.299 * R
cv2.imshow("3 channel to 1 channel",imgGray)

#Blur Background Area
gb = cv2.GaussianBlur(imgGray,(9,9), 1.5, 1.5,cv2.BORDER_DEFAULT )
cv2.imshow('Gaussian Filter',gb)

#Expanding Boundary Areas by Sobel Filter
gb=np.asarray(gb)
kernel1 = np.asarray([[-1,0,1],[-2,0,2],[-1,0,1]])
kernel2 = np.asarray([[-1,-2,-1],[0,0,0],[1,2,1]])
sobel_output = sig.convolve2d(imgGray,kernel1,mode="same")
sobel_output = sig.convolve2d(imgGray,kernel2,mode="same")
cv2.imshow('Sobel Convolution',sobel_output)

#Spearting backgound from foreground
blur = cv2.blur(sobel_output,(5,5))
cv2.imshow("blu", blur)

#Preprocessing for detecting contours
blur = np.absolute(blur)
blur = blur / np.max(blur) # normalize the data to 0 - 1
blur = 255 * blur # Now scale by 255
blur = blur.astype(np.uint8)
cv2.imshow("test",blur)

#Detecting Contours
ret, thresh = cv2.threshold(blur,255,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#Drawing the countors
color = cv2.cvtColor(blur,cv2.COLOR_GRAY2BGR)
img = cv2.drawContours(color, contours, -1, (0,0,255), 1)
cv2.imshow("Contours",img)