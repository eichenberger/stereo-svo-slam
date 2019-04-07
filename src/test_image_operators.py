import image_operators as imo
import numpy as np
import math
from math import cos, sin

def _rot_mat_x(angle):
    return np.mat([[1, 0, 0],
                    [0, cos(angle), -sin(angle)],
                    [0, sin(angle), cos(angle)]])

def _rot_mat_y(angle):
    return np.mat([[cos(angle), 0, sin(angle)],
                    [0, 1, 0],
                    [-sin(angle), 0, cos(angle)]])

def _rot_mat_z(angle):
    return np.mat([[cos(angle), -sin(angle), 0],
                    [sin(angle), cos(angle), 0],
                    [0, 0, 1]])

def rotation_matrix(angle):
    rot_x = _rot_mat_x(angle[0])
    rot_y = _rot_mat_y(angle[1])
    rot_z = _rot_mat_z(angle[2])

    rot = np.matmul(rot_x, rot_y)
    rot = np.matmul(rot, rot_z)

    return rot

def transform_keypoints(pose, keypoints3d, fx, fy, cx, cy):
    extrinsic = np.zeros((3,4))
    extrinsic[:,0:3] = rotation_matrix(pose[3:])
    extrinsic[:,3] = np.transpose(pose[0:3])

    kps3d = np.ones((4, keypoints3d.shape[1]))
    kps3d[0:3, :] = keypoints3d

    kps2d = np.matmul(extrinsic, kps3d)
    kps2d = (kps2d*[[fx],[fy],[1]])/kps2d[2,:] + [[cx],[cy],[0]]
    return kps2d

print(imo.rotation_matrix(np.array([0.0, 0.0, 0.0])))
print(imo.rotation_matrix(np.array([math.pi/4, 0.0, 0.0])))
print(imo.rotation_matrix(np.array([0.0, math.pi/4, 0.0])))
print(imo.rotation_matrix(np.array([0.0, 0.0, math.pi])))

pose = np.array([1.0,2,3.0,0.0,0.0,0.0])
kps = np.transpose(np.array([[1.0,2.0,3.0],[3.0,4.0,5.0]]))
fx = 1.0
fy = 1.0
cx = 0.0
cy = 0.0

kps2d = imo.transform_keypoints(pose, kps, fx, fy, cx, cy)
kps2d_verify = transform_keypoints(pose, kps, fx, fy, cx, cy)
print(f"kps2d: {kps2d}")
print(f"kps2d_verify: {kps2d_verify}")

fx = 1.0
fy = 2.0
cx = 3.0
cy = 4.0
kps2d = imo.transform_keypoints(pose, kps, fx, fy, cx, cy)
kps2d_verify = transform_keypoints(pose, kps, fx, fy, cx, cy)
print(f"kps2d: {kps2d}")
print(f"kps2d_verify: {kps2d_verify}")
