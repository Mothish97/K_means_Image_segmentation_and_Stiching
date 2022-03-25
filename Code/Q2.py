# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:31:06 2022

@author: mothi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 21:09:27 2022

@author: mothi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def getHomography(matches,kp1,kp2):

    ratio = []
    for m in matches:
        if m[0].distance < 0.5*m[1].distance:
            ratio.append(m)
            matches = np.asarray(ratio)
        
    if len(matches[:,0]) >= 4:
        points_1 = np.float32([ keyPoint1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        points_2 = np.float32([ keyPoint2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        H, masked = cv2.findHomography(points_1, points_2, cv2.RANSAC, 5)
        
        
    return H

       
        
    
    




if __name__ == "__main__":


    img_1 = cv2.imread("Q2imageB.png")
    img1_gray = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    
    img_2 = cv2.imread("Q2imageA.png")
    img2_gray = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
    
    
    sift = cv2.xfeatures2d.SIFT_create()
    

    keyPoint1, feature1 = sift.detectAndCompute(img1_gray,None)
    keyPoint2, feature2 = sift.detectAndCompute(img2_gray,None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(feature1,feature2, 2)
    
    H=getHomography(matches, keyPoint1, keyPoint2)
        
    dst = cv2.warpPerspective(img_1,H,(img_2.shape[1] + img_1.shape[1], img_2.shape[0]))
    

    dst[0:img_2.shape[0], 0:img_2.shape[1]] = img_2
    plt.imshow(dst)


