# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 14:46:03 2022

@author: mothi
"""

import numpy as np


def getAmatrix():
    param  = np.array([[0,0,0,757,213], [0,3,0,758,415], [0,7,0,758,686], [0,11,0,759,966], [7,1,0,1190,172], [0,11,7,329,1041], [7,9,0,1204,850], [0,1,7,340,159]])
    A = np.array([[param[0][0], param[0][1], param[0][2], 1, 0, 0, 0, 0, -param[0][3]*param[0][0], -param[0][3]*param[0][1], -param[0][3]*param[0][2], -param[0][3]], 
          [0, 0, 0, 0, param[0][0], param[0][1], param[0][2], 1, -param[0][4]*param[0][0], -param[0][4]*param[0][1], -param[0][4]*param[0][2], -param[0][4]],
          [param[1][0], param[1][1], param[1][2], 1, 0, 0, 0, 0, -param[1][3]*param[1][0], -param[1][3]*param[1][1], -param[1][3]*param[1][2], -param[1][3]], 
          [0, 0, 0, 0, param[1][0], param[1][1], param[1][2], 1, -param[1][4]*param[1][0], -param[1][4]*param[1][1], -param[1][4]*param[1][2], -param[1][4]],
          [param[2][0], param[2][1], param[2][2], 1, 0, 0, 0, 0, -param[2][3]*param[2][0], -param[2][3]*param[2][1], -param[2][3]*param[2][2], -param[2][3]], 
          [0, 0, 0, 0, param[2][0], param[2][1], param[2][2], 1, -param[2][4]*param[2][0], -param[2][4]*param[2][1], -param[2][4]*param[2][2], -param[2][4]],
          [param[3][0], param[3][1], param[3][2], 1, 0, 0, 0, 0, -param[3][3]*param[3][0], -param[3][3]*param[3][1], -param[3][3]*param[3][2], -param[3][3]], 
          [0, 0, 0, 0, param[3][0], param[3][1], param[3][2], 1, -param[3][4]*param[3][0], -param[3][4]*param[3][1], -param[3][4]*param[3][2], -param[3][4]],

          [param[4][0], param[4][1], param[4][2], 1, 0, 0, 0, 0, -param[4][3]*param[4][0], -param[4][3]*param[4][1], -param[4][3]*param[4][2], -param[4][3]], 
          [0, 0, 0, 0, param[4][0], param[4][1], param[4][2], 1, -param[4][4]*param[4][0], -param[4][4]*param[4][1], -param[4][4]*param[4][2], -param[4][4]],
          [param[5][0], param[5][1], param[5][2], 1, 0, 0, 0, 0, -param[5][3]*param[5][0], -param[5][3]*param[5][1], -param[5][3]*param[5][2], -param[5][3]], 
          [0, 0, 0, 0, param[5][0], param[5][1], param[5][2], 1, -param[5][4]*param[5][0], -param[5][4]*param[5][1], -param[5][4]*param[5][2], -param[5][4]],
          [param[6][0], param[6][1], param[6][2], 1, 0, 0, 0, 0, -param[6][3]*param[6][0], -param[6][3]*param[6][1], -param[6][3]*param[6][2], -param[6][3]], 
          [0, 0, 0, 0, param[6][0], param[6][1], param[6][2], 1, -param[6][4]*param[6][0], -param[6][4]*param[6][1], -param[6][4]*param[6][2], -param[6][4]],
          [param[7][0], param[7][1], param[7][2], 1, 0, 0, 0, 0, -param[7][3]*param[7][0], -param[7][3]*param[7][1], -param[7][3]*param[7][2], -param[7][3]], 
          [0, 0, 0, 0, param[7][0], param[7][1], param[7][2], 1, -param[7][4]*param[7][0], -param[7][4]*param[7][1], -param[7][4]*param[7][2], -param[7][4]]])
    return A


if __name__ == "__main__":
    A= getAmatrix()
    A_t = A.transpose()
  
    

    (S,U,V)= np.linalg.svd(np.matmul(A_t,A))
    p = np.reshape(V[-1][:], (3, 4))
    p_normalized = p/p[-1][-1]
    projection_matrix = p_normalized[0:3, 0:3]

    
    
    rot, intrinsic = np.linalg.qr(projection_matrix)
    intrinsic_normal = intrinsic / intrinsic[-1][-1]
    intrinsic_inv = np.linalg.inv(intrinsic)

    
    
    p_last = p_normalized[:, -1]
    trans = np.matmul(intrinsic_inv, p_last)
    trans = np.matrix(trans)
    trans= trans.transpose()
    
    
    
    rot_trans = np.hstack((rot, trans))

    extrinsic = np.vstack((rot_trans, [0,0,0,1]))

    


