import numpy as np

from rotation_matrix import rotation_matrix

def transform_keypoints(pose, keypoints3d, fx, fy, cx, cy):
    extrinsic = np.zeros((3,4))
    extrinsic[:,0:3] = rotation_matrix(pose[3:])
    extrinsic[:,3] = np.transpose(pose[0:3])

    kps3d = np.ones((4, keypoints3d.shape[1]))
    kps3d[0:3, :] = keypoints3d

    kps2d = np.matmul(extrinsic, kps3d)
    kps2d = (kps2d*[[fx],[fy],[1]])/kps2d[2,:] + [[cx],[cy],[0]]
    return kps2d


