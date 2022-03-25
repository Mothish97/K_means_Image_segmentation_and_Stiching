# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 23:04:00 2022

@author: mothi
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt


def getCentroids():
    black= [0,0,0]
    blue=[0,0,255]
    green= [0,255,0]
    red = [255,0,0]
    centroid= []
    centroid.append(red)
    centroid.append(green)
    centroid.append(blue)
    centroid.append(black)
    return np.asarray(centroid)

def getClusteredData(img_data, centroid):
    red_cluster=[]
    blue_cluster=[]
    green_cluster=[]
    black_cluster=[]
    
    
    for i in range(len(img_data)):
        
        distance_red= getEucledianDistance(img_data[i], centroid[0])
        distance_green= getEucledianDistance(img_data[i], centroid[1])
        distance_blue= getEucledianDistance(img_data[i], centroid[2])
        distance_black= getEucledianDistance(img_data[i], centroid[3])
        #print(distance_red)
        distances=[distance_red,distance_green,distance_blue,distance_black]
        
        short_distance= np.argmin(distances)
        
        if(short_distance==0):
            red_cluster.append(img_data[i])
        
        elif(short_distance==1):
            blue_cluster.append(img_data[i])
            
        elif(short_distance==2):
            green_cluster.append(img_data[i])   
            
        elif(short_distance==3):
            black_cluster.append(img_data[i])   
    
    
    red_cluster=np.asarray(red_cluster)
    blue_cluster=np.asarray(blue_cluster)
    green_cluster=np.asarray(green_cluster)
    black_cluster=np.asarray(black_cluster)
    
    
                   
    return red_cluster,blue_cluster,green_cluster,black_cluster
 


def getImage(img_data, centroid):

    
    
    for i in range(len(img_data)):
        
        distance_red= getEucledianDistance(img_data[i], centroid[0])
        distance_green= getEucledianDistance(img_data[i], centroid[1])
        distance_blue= getEucledianDistance(img_data[i], centroid[2])
        distance_black= getEucledianDistance(img_data[i], centroid[3])
        #print(distance_red)
        distances=[distance_red,distance_green,distance_blue,distance_black]
        
        short_distance= np.argmin(distances)
        
        if(short_distance==0):
            img_data[i]=centroid[0]
        
        elif(short_distance==1):
            img_data[i]=centroid[1]
            
        elif(short_distance==2):
            img_data[i]=centroid[2]  
            
        elif(short_distance==3):
            img_data[i]=centroid[3]   
                    
    return img_data   



def getEucledianDistance(point,centroid):
    distance = ((point[0] - centroid[0])**2 + (point[1] - centroid[1])**2 + (point[2] - centroid[2])**2)**0.5
    return distance

if __name__ == "__main__":

    img = cv2.imread("Q4image.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_data= img.reshape((-1,3))
    centroid=getCentroids()
    
    i=0
    while True:
        old_centroid= centroid
        red_cluster,blue_cluster,green_cluster,black_cluster = getClusteredData(img_data,centroid)
        
        red_cluster_mean= np.mean(red_cluster, axis=0)
        
        blue_cluster_mean= np.mean(blue_cluster, axis=0)  
        
        green_cluster_mean= np.mean(green_cluster, axis=0) 
        
        black_cluster_mean= np.mean(black_cluster, axis=0)   
        
        centroid=[]
        
        centroid.append(red_cluster_mean)
        centroid.append(blue_cluster_mean)
        centroid.append(green_cluster_mean)
        centroid.append(black_cluster_mean)
        
        centroid=np.asanyarray(centroid) 
        print(i)
        i=i+1

        if(getEucledianDistance(centroid[0], old_centroid[0])<0.7  and getEucledianDistance(centroid[1], old_centroid[1])<0.7 and getEucledianDistance(centroid[2], old_centroid[2])<0.7 and getEucledianDistance(centroid[3], old_centroid[3])<0.7 ):
            break;
    
    
    
    
    
    
    
    img_new_data= getImage(img_data, centroid)
    
    img_segmented= img_new_data.reshape((img.shape))

    
    
    
    
    
    plt.imshow(img_segmented)

    
    