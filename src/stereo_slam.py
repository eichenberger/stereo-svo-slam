import cv2
import numpy as np
import math

from keyframe import KeyFrame
from pose_estimator import PoseEstimator
from depth_calculator import DepthCalculator
from draw_kps import draw_kps

from image_operators import DepthAdjuster

class StereoSLAM:
    def __init__(self, baseline, fx, fy, cx, cy):
        self.keypoints = None
        self.left = None
        self.right = None
        self.baseline = baseline
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        # vx, vy, vz, gx, gy, gz
        self.motion_model = np.array([0, 0, 0, 0, 0, 0])
        self.pose = np.array([0, 0, 0, 0, 0, 0])
        self.keyframes = []

        self.depth_calculator = DepthCalculator(self.baseline, self.fx, self.fy, self.cx, self.cy)
        # self.depth_adjustment = DepthAdjustment()

    def new_image(self, left, right):
        self.left = left
        self.right = right

        if len(self.keyframes) == 0:
            self._calculate_depth()
        else:
            new_pose = self._estimate_pose()
            self.motion_model =  new_pose - self.pose
            self.pose = new_pose

            kf = self.keyframes[-1]
            ad = DepthAdjuster()
            ad.set_image(kf.image)
            ad.set_new_image(self.left)
            ad.set_keypoints3d(kf.keypoints3d)
            ad.set_keypoints2d(kf.keypoints2d)
            ad.set_pose(self.pose)
            ad.set_fx(self.fx)
            ad.set_fy(self.fy)
            ad.set_cx(self.cx)
            ad.set_cy(self.cy)

            new_z, cost = ad.adjust_depth()

            MAX_PROP = 0.5
            prop = MAX_PROP*np.exp(-(cost))
            kf.keypoints3d[2,:] = (prop * new_z) + (1-prop)*kf.keypoints3d[2,:]

#            draw_kps(new_pose, self.left,
#                     self.keyframes[-1].image,
#                     self.keyframes[-1].keypoints2d,
#                     self.keyframes[-1].keypoints3d,
#                     self.fx, self.fy,
#                     self.cx, self.cy)


    def _estimate_pose(self):
        kf = self.keyframes[-1]
        estimator = PoseEstimator(self.left, kf.image, kf.keypoints2d, kf.keypoints3d,
                                  self.fx, self.fy, self.cx, self.cy)
        return estimator.estimate_pose(self.pose, self.motion_model)

    def _calculate_depth(self):
        keypoints2d, keypoints3d = self.depth_calculator.calculate_depth(self.left, self.right)
        self.keyframes.append(KeyFrame(self.left, keypoints2d, keypoints3d, self.pose))
