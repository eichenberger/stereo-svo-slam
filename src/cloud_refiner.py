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
