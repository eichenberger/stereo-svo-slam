import numpy as np
import cv2
from scipy.linalg import expm
from math import cos,sin

def get_skew(twist):
    v = twist[0:3]
    w = twist[3:6]
    skew_mat = np.mat([[0, -w[2], w[1], v[0]],
                        [w[2], 0, -w[0], v[1]],
                        [-w[1], w[0], 0, v[2]],
                        [0,0,0,1]])

    return skew_mat

def exp_to_hom(twist):
    skew_mat = get_skew(twist)
    homogenous = expm(skew_mat)
    angles = cv2.Rodrigues(homogenous[0:3,0:3])[0].reshape(3)
    return np.array([homogenous[0,3], homogenous[1,3], homogenous[2,3], angles[0], angles[1], angles[2] ])

def exponential_map(twist):
    v = twist[0:3]
    w = twist[3:6]

    w_skew = np.array([[0,    -w[2],  w[1]],
                       [w[2],  0,    -w[0]],
                       [-w[1], w[0],  0  ]]);
    #_norm = np.linalg.norm(w);
    _norm = 1.0
    _ones = np.eye(3,3);

    p1 = _ones*_norm
    p2 = (1-cos(_norm))*w_skew
    p3 = (_norm-sin(_norm))*np.matmul(w_skew,w_skew)
    translation = (p1 +p2 + p3)*np.mat(v).transpose();

    return [translation[0,0], translation[1,0], translation[2,0], w[0], w[1], w[2]]

test = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
#test = np.array([0.4, 0.5, 0.6, 0.1, 0.2, 0.3])
grad1 = exp_to_hom(test)
print(grad1)

ver_grad1 = exponential_map(test)
print(ver_grad1)
print(ver_grad1/grad1)
angles = np.asarray(ver_grad1[3:6])
print(cv2.Rodrigues(angles)[0])


test = np.random.rand(6)
print(test)
grad1 = exp_to_hom(test)
print(grad1)

ver_grad1 = exponential_map(test)
print(ver_grad1)
print(ver_grad1/grad1)
