"""
transform:
    1.3d transform(homogeneous)--scale rotation translation
    2.viewing transform(homogeneous)--view/camera transformation
                                    --projection transformation(orthographic perspective)
"""
import numpy as np
import math
from math import sin, cos

########################## angle2matrix ###########################
# right-hand axis,note y axis
def angle2matrix(angles,r=True):
    """
    Get rotation matrix from angle.right-hand
    DirectXMath use ZXY rule.
    :param angles: [3,]. x, y, z angles
                  x: pitch. positive for looking down.
                  y: yaw. positive for looking left.
                  z: roll. positive for tilting head right.
    :return: rotation matrix:[3,3]
    """
    # angle2radian
    if r:
        x,y,z = np.deg2rad(angles[0]),np.deg2rad(angles[1]),np.deg2rad(angles[2])
    else:
        x,y,z = angles[0],angles[1],angles[2]
    # # x
    Rx = np.array([[1, 0, 0],
                   [0, cos(x), -sin(x)],
                   [0, sin(x), cos(x)]])
    # y
    Ry = np.array([[cos(y), 0, sin(y)],
                   [0, 1, 0],
                   [-sin(y), 0, cos(y)]])
    # z
    Rz = np.array([[cos(z), -sin(z), 0],
                   [sin(z), cos(z), 0],
                   [0, 0, 1]])
    # R = Rz.dot(Ry).dot(Rx)   #Z-Y-X顺规
    R = Ry.dot(Rz).dot(Rx)
    return R.astype(np.float32)

######################### 3d transformation ################################
def rotate(vertices,angles):
    """
    Rotate vertices.
    X_new=R.dot(X) X:vertice [3,]
    :param vertices:[n,3]
    :param angles:[3,]
    :return: rotate_vertices:[n,3]
    """
    R=angle2matrix(angles)
    rotate_vertices=vertices.dot(R.T)
    return rotate_vertices

def similarity_transform(vertices,R,s,t3d):
    """
    After similarity transformed,the shape is not changed.
    3D: s*R.dot(X) + t
    Homo: M = [[sR, t],[0^T, 1]].  M.dot(X)
    :param vertices:[n,3]
    :param R:[3,3] rotate matrix
    :param s:[1,]  scale factor
    :param t3d:[3,] 3d translation vector.
    :return:transformed_vertices [n,3]
    """
    M=np.concatenate((s*R,np.array(t3d).reshape(3,1)),axis=1)
    M=np.concatenate((M,np.array([[0,0,0,1]])),axis=0)
    vertices=np.hstack((vertices,np.ones([vertices.shape[0],1])))
    transformed_vertices=M.dot(vertices.T)
    return transformed_vertices.T[:,:3]

######################### Camera. from world space to camera space #####################
def normalize(x):
    epsilon = 1e-12
    norm = np.sqrt(np.sum(x**2, axis = 0))
    norm = np.maximum(norm, epsilon)
    return x/norm

def view_transform(vertices,eye,at=None,up=None):
    """
    Transform the objects along with the camera.
    Define the camera: --position:e  --look-at/gaze:g --up direction:t
    Transform the camera by M_view: located at the origin,up at Y,look at -Z
    M_view=R_view.dot(T_view)
    Xcam = R(X - C)
    Homo: [[R, -RC], [0, 1]]
    :param vertices:[n,3]
    :param eye:[3,] the XYZ world space position of the camera.
    :param at:[3,] a position along the center of the camera's gaze.
    :param up:[3,] up direction
    :return:transformed_vertices [n,3]
    """
    if at is None:
        at=np.array([0,0,0],np.float32)
    if up is None:
        up=np.array([0,1,0],np.float32)

    eye = np.array(eye).astype(np.float32)
    at = np.array(at).astype(np.float32)
    z_aixs = -normalize(at-eye) # look forward
    x_aixs = normalize(np.cross(up, z_aixs))  # look right
    y_aixs = np.cross(z_aixs,x_aixs)

    R=np.stack((x_aixs,y_aixs,z_aixs),aixs=0)  #3*3
    M=np.concatenate((R,-R.dot(eye.T).reshape(3,1)),axis=1)
    M = np.concatenate((M, np.array([[0, 0, 0, 1]])), axis=0)

    vertices=np.hstack((vertices,np.ones([vertices.shape[0],1])))
    # vertices_homo = np.vstack((vertices.T, np.ones((1, vertices.shape[0])))) #[4,n]
    transformed_vertices = M.dot(vertices.T)
    return transformed_vertices.T[:,:3]

###################### 3d-2d tranformation. from camera space to img space #######################
def orthographic_project(vertices):
    """
    delete z
    Homo: P = [[1,0,0,0], [0,1,0,0], [0,0,1,0]]
    :param vertices: [n,3]
    :return: project_vertices [n,3]
    """
    return vertices[:,:3]

def perspective_project(vertices, fovy, aspect_ratio = 1., near = 0.1, far = 1000.):
    ''' perspective projection.
    Args:
        vertices: [nver, 3]
        fovy: vertical angular field of view. degree.
        aspect_ratio : width / height of field of view
        near : depth of near clipping plane
        far : depth of far clipping plane
    Returns:
        projected_vertices: [nver, 3]
    '''
    fovy = np.deg2rad(fovy)
    top = near*np.tan(fovy)
    bottom = -top
    right = top*aspect_ratio
    left = -right

    #-- homo
    P = np.array([[near/right, 0, 0, 0],
                 [0, near/top, 0, 0],
                 [0, 0, -(far+near)/(far-near), -2*far*near/(far-near)],
                 [0, 0, -1, 0]])
    vertices_homo = np.hstack((vertices, np.ones((vertices.shape[0], 1)))) # [nver, 4]
    projected_vertices = vertices_homo.dot(P.T)
    projected_vertices = projected_vertices/projected_vertices[:,3:]
    projected_vertices = projected_vertices[:,:3]
    projected_vertices[:,2] = -projected_vertices[:,2]

    #-- non homo. only fovy
    # projected_vertices = vertices.copy()
    # projected_vertices[:,0] = -(near/right)*vertices[:,0]/vertices[:,2]
    # projected_vertices[:,1] = -(near/top)*vertices[:,1]/vertices[:,2]
    return projected_vertices[:,:3]

############################## image. from image space to pixel space ##############################
def to_image(vertices, h, w, is_perspective = False):
    ''' change vertices to image coord system
    3d system: XYZ, center(0, 0, 0)
    2d image: x(u), y(v). center(w/2, h/2), flip y-axis.
    Args:
        vertices: [nver, 3]
        h: height of the rendering
        w : width of the rendering
    Returns:
        image_vertices: [nver, 3]
    '''
    image_vertices = vertices.copy()
    if is_perspective:
        # if perspective, the projected vertices are normalized to [-1, 1]. so change it to image size first.
        image_vertices[:,0] = image_vertices[:,0]*w/2
        image_vertices[:,1] = image_vertices[:,1]*h/2
    # move to center of image
    image_vertices[:,0] = image_vertices[:,0] + w/2
    image_vertices[:,1] = image_vertices[:,1] + h/2
    # flip vertices along y-axis.
    image_vertices[:,1] = h - image_vertices[:,1] - 1
    return image_vertices