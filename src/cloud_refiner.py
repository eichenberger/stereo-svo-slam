import numpy as np

import scipy.optimize as opt
from slam_accelerator import refine_cloud

class CloudRefiner:
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy


    def refine_cloud(self, pose, keypoints3d, keypoints2d):
        kps3d = refine_cloud(self.fx, self.fy, self.cx, self.cy, pose, keypoints3d.copy(), keypoints2d)
        return kps3d
        #for i in range(0, keypoints3d.shape[1]):
        #    kp2d = keypoints2d[:,i]
        #    kp3d = keypoints3d[:,i]

        #    def _update_point(tmp_kp3d):
        #        temp_kp3d = np.reshape(tmp_kp3d, (3,1))
        #        temp_kp2d = transform_keypoints(pose, temp_kp3d,
        #                                        self.fx, self.fy, self.cx, self.cy)
        #        return np.linalg.norm(kp2d - temp_kp2d.flatten())

        #    res = opt.minimize(_update_point, kp3d, method = 'CG')
        #    keypoints3d[:,i] = res.x

        #return keypoints3d
