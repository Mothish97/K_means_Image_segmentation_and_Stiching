#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 12:34:06 2022

@author: mothish
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt



    








if __name__ == "__main__":

    img = cv2.imread("Q1image.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel= np.ones((11,11),np.uint8) 
    erosion= cv2.erode(img,kernel,iterations=3)
    dilate= cv2.dilate(erosion,kernel)
    
    contour, heirarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print(len(contour))
    plt.imshow(dilate,cmap='gray')
    print("Code")
