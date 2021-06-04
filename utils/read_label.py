import numpy as np
import os
from util import *

label_path="/home/SENSETIME/yuefurong/3d_face/dataset/test/label"
filenames = os.listdir(label_path)
filenames.sort()

# load 2106 vertices index
kpts_file = f'/home/SENSETIME/yuefurong/3d_face/dataset/test/index_2106pts_bs.txt'
kpts_index = np.loadtxt(kpts_file).astype(np.int32)

# load face
file_path = f'/home/SENSETIME/yuefurong/3d_face/dataset/test/face_2106pts.txt'
triangles = np.loadtxt(file_path).astype(np.int32) - 1

mesh_paths = "/home/SENSETIME/yuefurong/3d_face/dataset/test/mesh"
euler_pose_path = '/home/SENSETIME/yuefurong/3d_face/dataset/test/euler_pose.txt'
euler_pose_path = open(euler_pose_path,'w')
pose_path = '/home/SENSETIME/yuefurong/3d_face/dataset/test/pose.txt'
pose_path = open(pose_path,'w')
# landmark_path = '/home/SENSETIME/yuefurong/3d_face/dataset/test/landmark.txt'
# landmark_path = open(landmark_path,'w')
#filenames = []
mesh_dict = dict()
euler_pose_dict = dict()

def save_landmarks(finalname, landmark):
    for i in range(len(landmark)):
        landmark_path.write(str(np.round(landmark[i], 3)) + " ")
    landmark_path.write(finalname + '\n')
    return True

def save_euler_pose(filename,euler_pose):
    for i in range(len(euler_pose)):
        euler_pose_path.write(str(euler_pose[i])+' ')
    euler_pose_path.write(filename[:-4]+'\n')
    return True

def save_pose(filename,pose):
    for i in range(len(pose)):
        pose_path.write(str(pose[i])+' ')
    pose_path.write(filename[:-4]+'\n')
    return True

i=0
for filename in filenames:
    path = label_path+'/'+filename
    label = np.load(path,encoding='bytes',allow_pickle=True)
    euler_pose = label['pose_euler']
    pose = label['pose']
    if save_pose(filename,pose):
        print("pose save success", i)
    if save_euler_pose(filename,euler_pose):
        print("euler_pose save success", i)
    mesh = label['mesh']
    landmark = label['landmark']
    landmark_path = f'/home/SENSETIME/yuefurong/3d_face/dataset/test/{filename[:-4]}.txt'
    landmark_path = open(landmark_path, 'w')
    for i in range(len(landmark)):
        landmark_path.write(str(np.round(landmark[i][0], 3)) + " "+str(np.round(landmark[i][1], 3))+"\n")
    # if save_landmarks(filename,landmark):
    #     print("landmark save success", i)
    i+=1
    vertices =mesh[kpts_index]
    mesh = TriMesh(vertices=vertices,faces=triangles)
    # mesh.save(f'/home/SENSETIME/yuefurong/GitProjects/face/test_data/3d_model/{filename[:-4]}.obj')


