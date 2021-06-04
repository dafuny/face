"""
Examples of estimation.
"""
from utils.util import TriMesh
import numpy as np
import cv2
from estimate import *
from transform import *

# --------- load mesh data ----------------------
mesh = TriMesh()
mesh.load("/home/SENSETIME/yuefurong/GitProjects/face/face_model/mean.obj")
# mesh.load("/home/SENSETIME/yuefurong/GitProjects/face/test_data/3d_model/Backlight~11~IMG_20180608_132319.obj")
vertices = mesh.vertices
triangles = mesh.faces

# -------------load 3d landmark index --------------
ld_3d_index = np.loadtxt('/home/SENSETIME/yuefurong/GitProjects/face/face_model/face_lmk_idx_106.txt')
# ld_3d_index = np.loadtxt('/home/SENSETIME/yuefurong/GitProjects/face/test_data/3d_model/index244_2106.txt')
ld_3d = mesh.vertices[ld_3d_index.astype(int)]

#------------- read image and get landmarks ---------------------
img = cv2.imread("/home/SENSETIME/yuefurong/GitProjects/face/test_data/img/5.jpg")
ld_2d = np.loadtxt("/home/SENSETIME/yuefurong/GitProjects/face/test_data/lmk/5.txt")
# img = cv2.imread("/home/SENSETIME/yuefurong/3d_face/dataset/test/image/Backlight~11~IMG_20180608_132319.jpg")
# ld_2d = np.loadtxt("/home/SENSETIME/yuefurong/3d_face/dataset/test/Backlight~11~IMG_20180608_132319.txt")
ld_2d=ld_2d.reshape(-1,2)
ld_2d = np.delete(ld_2d,[74,77,104,105],axis=0)

param =np.array([0,0,0,0,0,0])
param[5] = img.shape[0]/2

Gauss_Newton = Gauss_Newton(ld_3d,ld_2d,param)
param = Gauss_Newton.estimate_transform_matrix()
print(param)