"""
load mesh
"""
import numpy as np

# 三角网格的读写处理
class TriMesh:
    def __init__(self, vertices=None, faces=None):
        self.vertices = vertices
        self.faces = faces

    def load(self, path):
        vertices = []
        faces = []

        lines = open(path, 'r+').readlines()
        for line in lines:
            line = line.strip('\n')
            ss = line.split(' ')
            if ss[0] == 'v':
                for i in range(1, 4):
                    vertices.append(float(ss[i]))

            if ss[0] == 'f':
                for i in range(1, 4):
                    sv = ss[i].split('/')
                    faces.append(int(sv[0]) - 1)

        self.vertices = np.reshape(vertices, (-1, 3)).astype(np.float32)
        self.faces = np.reshape(faces, (-1, 3)).astype(np.int)

    def save(self, path, with_color=True, with_vt=True):
        out_file = open(path, 'w+')

        for i in range(len(self.vertices)):
            v = self.vertices[i]
            line = 'v ' + "{:.6f}".format(v[0]) + ' ' + '{:.6f}'.format(v[1]) + ' ' + '{:.6f}'.format(v[2])
            out_file.write(line + '\n')

        for i in range(len(self.faces)):
            line = 'f'
            for j in range(3):
                line += ' ' + str(self.faces[i, j] + 1)
            out_file.write(line + '\n')