import cv2
import numpy as np
import math

from pose_estimator import PoseEstimator
from depth_calculator import DepthCalculator
from draw_kps import draw_kps

from slam_accelerator import transform_keypoints
from pose_refiner import PoseRefiner
from cloud_refiner import CloudRefiner

from mapping import Mapping

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

        # self.depth_calculator = DepthCalculator(self.baseline, self.fx, self.fy, self.cx, self.cy)
        self.pose_refiner = PoseRefiner(self.fx, self.fy, self.cx, self.cy)
        self.cloud_refiner = CloudRefiner(self.fx, self.fy, self.cx, self.cy)
        # self.depth_adjustment = DepthAdjustment()

        self.mapping = Mapping(baseline, fx, fy, cx, cy)

    def new_image(self, left, right):
        self.prev_left = self.left
        self.prev_right = self.right

        self.left = left
        self.right = right

        if self.mapping.number_of_keyframes() == 0:
            self.mapping.new_image(left, right, self.pose, 0, 0)
            self.mapping.process_image()
            kf = self.mapping.get_last_keyframe()
            self.previous_keypoints2d = kf.keypoints2d
        else:
            kf = self.mapping.get_last_keyframe()

            new_pose, cost = self._estimate_pose(kf)
            self.motion_model =  new_pose - self.pose
            self.pose =  new_pose
            # We get the pose between the last and the current image.
            # We need to update the global pose
            keypoints2d = transform_keypoints(self.pose,
                                              kf.keypoints3d,
                                              self.fx, self.fy,
                                              self.cx, self.cy)


            kps2d_prev = np.array(kf.keypoints2d.transpose(), dtype=np.float32)
            kps2d_next = np.array(keypoints2d.transpose(), dtype=np.float32)
            ref_keypoints2d, status, err = cv2.calcOpticalFlowPyrLK(kf.left,
                                                                self.left,
                                                                kps2d_prev,
                                                                kps2d_next,
                                                                maxLevel=0,
                                                                winSize=(21,21),
                                                                flags=cv2.OPTFLOW_USE_INITIAL_FLOW)

            ref_keypoints2d = cv2.UMat.get(ref_keypoints2d).transpose()
            status = cv2.UMat.get(status)
            err = cv2.UMat.get(err)

            valid = (status*(err<1.0)).transpose()
            keypoints2d = valid*ref_keypoints2d + (1-valid)*keypoints2d

            # Needs more testing!
            self.pose = self.pose_refiner.refine_pose(self.pose,
                                                      kf.keypoints3d,
                                                      keypoints2d)

            # Needs more testing!
            keypoints3d_refined = self.cloud_refiner.refine_cloud(self.pose,
                                                             kf.keypoints3d,
                                                             keypoints2d)
            # Only take refined keypoints that are valid
            kf.keypoints3d = valid*keypoints3d_refined + (1-valid)*kf.keypoints3d

            MARGIN = 10
            valid_indexes = (keypoints2d[0,:]>MARGIN) &\
                            (keypoints2d[1,:]>MARGIN) &\
                            (keypoints2d[0,:]<(self.left.shape[1]-MARGIN))&\
                            (keypoints2d[1,:]<(self.left.shape[0]-MARGIN))

            matches = np.count_nonzero(valid_indexes==True)

            print(f"Found matches: {matches}")

            self.mapping.new_image(left, right, self.pose, matches, 0)
            self.mapping.process_image()

            draw_kps(self.pose, self.left,
                     kf.left,
                     kf.keypoints2d,
                     kf.keypoints3d,
                     self.fx, self.fy,
                     self.cx, self.cy)
            self.previous_keypoints2d = keypoints2d


    def _estimate_pose(self, kf):
        estimator = PoseEstimator(self.left, self.prev_left, self.previous_keypoints2d,
                                  kf.keypoints3d, self.fx, self.fy, self.cx, self.cy)
        return estimator.estimate_pose(self.pose + self.motion_model)

#    def _calculate_depth(self):
#        keypoints2d, keypoints3d = self.depth_calculator.calculate_depth(self.left, self.right)
#        self.keyframes.append(KeyFrame(self.left, keypoints2d, keypoints3d, self.pose))
