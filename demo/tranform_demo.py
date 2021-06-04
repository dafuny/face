'''
 Examples of transformation & camera model.
'''
import os, sys
from skimage import io
import transform
from utils.util import TriMesh
import cv2

sys.path.append('..')

def transform_test(vertices, obj, camera, h=256, w=256):
    '''
    Args:
        obj: dict contains obj transform paras
        camera: dict contains camera paras
    '''
    R = transform.angle2matrix(obj['angles'],r=False)
    # R = np.array([0.9895531156982811,-0.05223090807800712,-0.13437471284869995,-0.04220924492221025,-0.9961850630260141,0.07637866093977615,-0.13785140860695122,-0.06990888673983575,-0.9879826601210552]).reshape(3,3)
    transformed_vertices = transform.similarity_transform(vertices,  R, obj['s'],obj['t'])

    if camera['proj_type'] == 'orthographic':
        projected_vertices = transformed_vertices
        image_vertices = transform.to_image(projected_vertices, h, w)
    else:
        ## world space to camera space. (Look at camera.)
        camera_vertices = transform.view_transform(transformed_vertices, camera['eye'], camera['at'], camera['up'])
        ## camera space to image space. (Projection) if orth project, omit
        projected_vertices = transform.perspective_project(camera_vertices, camera['fovy'], near=camera['near'],
                                                                far=camera['far'])
        ## to image coords(position in image)
        image_vertices = transform.to_image(projected_vertices, h, w, True)

    return image_vertices,projected_vertices


# --------- load mesh data ----------------------
mesh = TriMesh()
# mesh.load("/home/SENSETIME/yuefurong/GitProjects/face/test_data/3d_model/Backlight~11~IMG_20180608_132346.obj")
mesh.load("/home/SENSETIME/yuefurong/GitProjects/face/face_model/mean.obj")
# mesh.load("/home/SENSETIME/yuefurong/GitProjects/face/test_data/3d_model/Backlight~11~IMG_20180608_132319.obj")
vertices = mesh.vertices
triangles = mesh.faces

# move center to [0,0,0]
# vertices = vertices - np.mean(vertices, 0)[np.newaxis, :]

# ------------- load image -------------------------
img = cv2.imread("/home/SENSETIME/yuefurong/GitProjects/face/test_data/img/5.jpg")
# img = cv2.imread("/home/SENSETIME/yuefurong/3d_face/dataset/test/image/Backlight~11~IMG_20180608_132319.jpg")

h=img.shape[0]
w=img.shape[1]
# save folder
save_folder = '/home/SENSETIME/yuefurong/GitProjects/face/results/transform/'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

# ---- start
obj = {}
camera = {}
camera['proj_type'] = 'orthographic'
# scale
# obj['s'] = 139.81374664197887
# obj['angles'] = [147.23205621759652,5.106463958543056,-4.503194666694818]
# obj['t'] = [199.37268468979613,245.97889341583436,5.205738664283617e-05]
# obj['s'] =  1.37350154e+02
# # obj['angles'] = [3.15664948e+00, 1.35591130e-01,-3.54319350e-02]
# obj['angles'] = [-3.12653583e+00, 1.35591130e-01,-3.54319350e-02]
# # obj['angles'] = [9.43983299e+00, 1.35590885e-01,-3.17702444e+00]  #30次的结果
# obj['t'] = [1.99756804e+02,1.83682810e+02,1]
obj['s'] = 6.47793004e+01
obj['angles'] = [-2.22009726e-01, -6.06524383e-01,3.70425288e-02]
obj['t'] = [ 2.34456858e+02,1.87549714e+02,1]
### 大概需要60次才收敛
image_vertices,projected_vertices = transform_test(vertices, obj, camera)
for i in range(projected_vertices.shape[0]):
    cv2.circle(img, (int(projected_vertices[i, 0]), int(projected_vertices[i, 1])), 1, (255, 0, 0), -1)
io.imsave('{}/5_new.jpg'.format(save_folder), img)
