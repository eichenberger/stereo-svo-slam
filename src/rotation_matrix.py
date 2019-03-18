import numpy as np
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

