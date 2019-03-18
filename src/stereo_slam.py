import cv2
import numpy as np
import math

from keyframe import KeyFrame
from pose_estimator import PoseEstimator
from rotation_matrix import rotation_matrix
from depth_calculator import DepthCalculator
from depth_adjustment import DepthAdjustment

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
        self.depth_adjustment = DepthAdjustment()

    def new_image(self, left, right):
        self.left = left
        self.right = right

        if len(self.keyframes) == 0:
            self._calculate_depth()
        else:
            new_pose = self._estimate_pose()
            self.motion_model =  new_pose - self.pose

            self.pose = new_pose

            self.depth_adjustment(self.keyframes[-1], new_pose)

            extrinsic = np.zeros((3,4))
            extrinsic[:,0:3] = rotation_matrix(self.pose[3:])
            extrinsic[:,3] = np.transpose(self.pose[0:3])

            keypoints3d = np.ones((4, len(self.keyframes[-1].keypoints3d)))
            keypoints3d[0:3, :] = np.transpose(self.keyframes[-1].keypoints3d)
            estimated_keypoints2d = np.matmul(extrinsic, keypoints3d)
            estimated_keypoints2d = (estimated_keypoints2d * [[self.fx],[self.fy], [1]])/ estimated_keypoints2d[2,:] + [[self.cx],[self.cy], [0]]

            keypoints_new_ocv = [None]*estimated_keypoints2d.shape[1]
            keypoints_old_ocv = [None]*estimated_keypoints2d.shape[1]
            matches = [None]*estimated_keypoints2d.shape[1]

            for i in range(estimated_keypoints2d.shape[1]):
                keypoints_new_ocv [i] = cv2.KeyPoint(estimated_keypoints2d[0,i], estimated_keypoints2d[1, i], 1)
                keypoints_old_ocv [i] = cv2.KeyPoint(self.keyframes[-1].keypoints2d[i, 0], self.keyframes[-1].keypoints2d[i, 1], 1)
                matches[i] = cv2.DMatch(i ,i, 1)

            result = self.keyframes[-1].image.copy()
            result = cv2.drawMatches(self.keyframes[-1].image,
                                     keypoints_old_ocv,
                                     self.left, keypoints_new_ocv, matches, result)

            cv2.imshow("Matches", result)
            cv2.waitKey(1)


    def _estimate_pose(self):
        kf = self.keyframes[-1]
        estimator = PoseEstimator(self.left, kf.image, kf.keypoints2d, kf.keypoints3d,
                                  self.fx, self.fy, self.cx, self.cy)
        return estimator.estimate_pose(self.pose, self.motion_model)

    def _calculate_depth(self):
        keypoints2d, keypoints3d = self.depth_calculator.calculate_depth(self.left, self.right)
        self.keyframes.append(KeyFrame(self.left, keypoints2d, keypoints3d, self.pose)
