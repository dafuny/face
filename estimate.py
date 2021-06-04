"""
    Estimate transform matrix from correspondence.
"""
import numpy as np
import math
from transform import *

########################### estimate transform  matrix from 3d to 3d ###########################
def estimate_affine_mat_3d23d(X,Y):
    """
    Using least-squares solution
    :param X:vertices [n,3] 3d points(fixed)
    :param Y:vertices [n,3]  corresponding 3d points(moving). Y = PX
    :return: affine matrix [3,4] (the third row is [0,0,0,1])
    """
    X_homo = np.hstack((X, np.ones([X.shape[1], 1])))  # n x 4
    P = np.linalg.lstsq(X_homo, Y)[0].T  # Affine matrix. 3 x 4 最小二乘求解矩阵
    return P


########################### estimate transform  matrix from 3d to 2d ###########################
# def estimate_affine_matrix_3d22d(X, x):
    """
    Using Golden Standard Algorithm for estimating an affine camera
    matrix P from world to image correspondences.
    Ref:https://github.com/patrikhuber/eos/blob/master/include/eos/fitting/detail/nonlinear_camera_estimation_detail.hpp

    x_homo = X_homo.dot(P_Affine)
    人脸模型的三维点以及对应照片中的二维点存在映射关系
    :param X: corresponding 3d points(fixed)
    :param x:n>=4. 2d points(moving). x = PX
    :return: [3, 4]. Affine camera matrix
    """


########################### estimate transform  matrix from 3d to 2d--Gussian-Neton Method ###########################
class Gauss_Newton:
    """
    Using Gauss_Newton Algorithm for estimating a transformation camera
    :param vertice:[n, 3].  corresponding 3d points(fixed)
    :param pixel: n>=4. 2d points(moving). x = PX
    """
    def __init__(self,vertice,pixel,param):
        self.vertice = vertice
        self.pixel = pixel
        self.param = param


    def jacobian(self):
        """
        Compute the jacobian matrix of params
        :param param:[rx,ry,rz,tx,ty,s]
        :param J:jacobian matrix [2*n,6]
        :return:J
        """
        n=np.array(self.vertice).shape[0]
        J=np.zeros((n*2,6))
        R=angle2matrix(self.param[:3],r=False)  # rotate matrix dont need 2radian

        sx = sin(self.param[0])
        cx = cos(self.param[0])
        sy = sin(self.param[1])
        cy = cos(self.param[1])
        sz = sin(self.param[2])
        cz = cos(self.param[2])

        for i in range(n):
            x=self.vertice[i,0]
            y=self.vertice[i,1]
            z=self.vertice[i,2]
############################# Z-Y-X顺规 jacobian matrix #############################
            # J[i * 2, 0] = self.param[5]*(sz*sx*y+cz*cy*cx*y+sz*cx*z-cz*sy*sx*z)
            # J[i * 2, 1] = self.param[5]*(-cz*sy*x+cz*cy*sx*y+cz*cy*cx*z)
            # J[i * 2, 2] = self.param[5]*(-sz*cy*x-cz*cx*y-sz*sy*sx*y+cz*sx*z-sz*sy*cx*z)
            # J[i * 2, 3] = 1
            # J[i * 2, 4] = 0
            # J[i * 2, 5] = (cz*cy*x+cz*sy*sx*y-sz*cx*y+sz*sx*z+cz*sy*cx*z)
            # J[i * 2 + 1, 0] = self.param[5]*(-cz*sx*y+sz*sy*cx*y-cz*cx*z-sz*sy*sx*z)
            # J[i * 2 + 1, 1] = self.param[5]*(-sz*sy*x+sz*cy*sx*y+sz*cy*cx*z)
            # J[i * 2 + 1, 2] = self.param[5]*(cz*cy*x-sz*cx*y+cz*sy*sx*y+sz*sx*z+cz*sy*cx*z)
            # J[i * 2 + 1, 3] = 0
            # J[i * 2 + 1, 4] = 1
            # J[i * 2 + 1, 5] = (cz*cy*x+cz*sy*sx*y-sz*cx*y+sz*sx*z+cz*sy*cx*z)
############################# Y-Z-X顺规 jacobian matrix #############################
            J[i * 2, 5] = R[0,0] * x + R[0,1] * y + R[0,2] * z
            J[i * 2 + 1, 5] = R[0,0] * x + R[0,1] * y + R[0,2] * z

            J[i * 2, 3] = 1
            J[i * 2, 4] = 0
            J[i * 2 + 1, 3] = 0
            J[i * 2 + 1, 4] = 1

            J[i * 2, 0] = self.param[5] * (cx * sy * y + sx * cy * sz * y - sx * sy * z + cx * cy * sz * z)
            J[i * 2, 1] = self.param[5] * (-sy * cz * x + sx * cy * y + sy * cx * sz * y + cx * cy * z - sx * sy * sz * z)
            J[i * 2, 2] = self.param[5] * (-cx * sz * x - cx * cy * cz * y + cy * sx * cz * z)

            J[i * 2 + 1, 0] = self.param[5] * (-sx * cz * y - cz * cx * z)
            J[i * 2 + 1, 1] = 0
            J[i * 2 + 1, 2] = self.param[5] * (cz * x - cx * sz * y + sx * sz * z)
        return J


    def residul(self):
       """
       Update the residual error of \prej_vert-pixel\
       :param param: [rx,ry,rz,tx,ty,s]
       :param E: residual error
       :return: E
       """
       n = np.array(self.vertice).shape[0]
       E = np.zeros((n*2,1))
       R = angle2matrix(self.param[:3], r=False)
       s=self.param[5]
       t2d = self.param[3:5]
       t3d = np.append(t2d,1)
       transformed_vertice = similarity_transform(self.vertice,R,s,t3d)
       projected_vertice = transformed_vertice  #[n,3]
       for i in range(n):
           E[i * 2, 0] = projected_vertice[i, 0] - self.pixel[i, 0]
           E[i * 2+1, 0] = projected_vertice[i, 1] - self.pixel[i, 1]

       return E

    def estimate_transform_matrix(self):
        for _ in range(5):
            J = self.jacobian()
            E = self.residul()
            a=np.linalg.inv(((J.T).dot(J)))
            self.param=self.param.reshape(6,1)
            b=a.dot(J.T).dot(E)
            self.param = self.param - b
        return self.param



