import cv2
import numpy as np
import math

from keyframe import KeyFrame
from pose_estimator import PoseEstimator
from depth_calculator import DepthCalculator
from draw_kps import draw_kps

from image_operators import DepthAdjuster, transform_keypoints
from pose_refiner import PoseRefiner
from cloud_refiner import CloudRefiner

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
        self.motion_model = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.keyframes = []

        self.depth_calculator = DepthCalculator(self.baseline, self.fx, self.fy, self.cx, self.cy)
        self.pose_refiner = PoseRefiner(self.fx, self.fy, self.cx, self.cy)
        self.cloud_refiner = CloudRefiner(self.fx, self.fy, self.cx, self.cy)
        # self.depth_adjustment = DepthAdjustment()

    def new_image(self, left, right):
        self.prev_left = self.left
        self.prev_right = self.right

        self.left = left
        self.right = right

        if len(self.keyframes) == 0:
            self._calculate_depth()
            self.keypoints2d = self.keyframes[-1].keypoints2d
        else:
            new_pose = self._estimate_pose()
            self.motion_model =  new_pose - self.pose
            self.pose =  new_pose
            # We get the pose between the last and the current image.
            # We need to update the global pose
            self.keypoints2d = transform_keypoints(self.pose,
                                                   self.keyframes[-1].keypoints3d,
                                                   self.fx, self.fy,
                                                   self.cx, self.cy)


            kf = self.keyframes[-1]
            kps2d_prev = np.array(kf.keypoints2d.transpose(), dtype=np.float32)
            kps2d_next = np.array(self.keypoints2d.transpose(), dtype=np.float32)
            keypoints2d, status, err = cv2.calcOpticalFlowPyrLK(kf.image,
                                                                self.left,
                                                                kps2d_prev,
                                                                kps2d_next,
                                                                maxLevel=0,
                                                                winSize=(21,21),
                                                                flags=cv2.OPTFLOW_USE_INITIAL_FLOW)

            keypoints2d = cv2.UMat.get(keypoints2d).transpose()
            status = cv2.UMat.get(status)
            err = cv2.UMat.get(err)

            valid = (status*(err<1.0)).transpose()
            keypoints2d = valid*keypoints2d + (1-valid)*self.keypoints2d
            self.keypoints2d = keypoints2d

            # Not verified yet!
            self.pose = self.pose_refiner.refine_pose(self.pose,
                                                      kf.keypoints3d,
                                                      keypoints2d)

            # Not verified yet, super slow!
            kf.keypoints3d = self.cloud_refiner.refine_cloud(self.pose,
                                                             kf.keypoints3d,
                                                             keypoints2d)

            draw_kps(self.pose, self.left,
                     self.keyframes[-1].image,
                     self.keyframes[-1].keypoints2d,
                     self.keyframes[-1].keypoints3d,
                     self.fx, self.fy,
                     self.cx, self.cy)


    def _estimate_pose(self):
        kf = self.keyframes[-1]
        estimator = PoseEstimator(self.left, self.prev_left, self.keypoints2d, kf.keypoints3d,
                                  self.fx, self.fy, self.cx, self.cy)
        return estimator.estimate_pose(self.pose + self.motion_model)

    def _calculate_depth(self):
        keypoints2d, keypoints3d = self.depth_calculator.calculate_depth(self.left, self.right)
        self.keyframes.append(KeyFrame(self.left, keypoints2d, keypoints3d, self.pose))
