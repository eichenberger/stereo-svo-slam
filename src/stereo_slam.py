import cv2
import numpy as np
import math

from pose_estimator import PoseEstimator
#from depth_calculator import DepthCalculator
from draw_kps import draw_kps
from point_aligner import PointAligner

from slam_accelerator import transform_keypoints, get_intensity_diff
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
        self.pyramid_levels = 5
        self.pyramid_step = 1.2
        # vx, vy, vz, gx, gy, gz
        self.motion_model = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # self.depth_calculator = DepthCalculator(self.baseline, self.fx, self.fy, self.cx, self.cy)
        self.pose_refiner = PoseRefiner(self.fx, self.fy, self.cx, self.cy)
        self.cloud_refiner = CloudRefiner(self.fx, self.fy, self.cx, self.cy)
        # self.depth_adjustment = DepthAdjustment()

        self.mapping = Mapping(baseline, fx, fy, cx, cy)


    def detect_outliers(self, current, previous,
                        current_keypoints2d, previous_keypoints, kf):
        for i in range(0, len(current)):
            for j in range(0, current_keypoints2d[i].shape[1]):
                diff = get_intensity_diff(current[i], previous[i],
                                    current_keypoints2d[i][:, j],
                                    previous_keypoints[i][:,j], 0)
                if diff > 100:
                    kf.blacklist[i][j] = True

    def get_kps3d(self, kf):
        kps3d = [None]*len(kf.blacklist)
        for i in range(0, len(kf.blacklist)):
            kps3d[i] = kf.keypoints3d[i][:, np.array(kf.blacklist[i]) == False]

        return kps3d



    def new_image(self, left, right):
        self.prev_left = self.left
        self.prev_right = self.right

        self.left = [None]*self.pyramid_levels
        self.right = [None]*self.pyramid_levels
        self.left[0] = left
        self.right[0] = right
        for i in range(1, self.pyramid_levels):
            self.left[i] = cv2.resize(self.left[i-1],None,
                                      fx=1.0/self.pyramid_step,
                                      fy=1.0/self.pyramid_step)
            self.right[i] = cv2.resize(self.right[i-1],None,
                                       fx=1.0/self.pyramid_step,
                                       fy=1.0/self.pyramid_step)


        if self.mapping.number_of_keyframes() == 0:
            self.mapping.new_image(self.left, self.right, self.pose, 0, 0)
            self.mapping.process_image()
            kf = self.mapping.get_last_keyframe()
            self.previous_keypoints2d = kf.keypoints2d
        else:
            kf = self.mapping.get_last_keyframe()

            kps3d = self.get_kps3d(kf)
            new_pose, cost = self._estimate_pose(kf, kps3d)

            # We get the pose between the last and the current image.
            # We need to update the global pose
            keypoints2d = transform_keypoints(new_pose,
                                              kf.keypoints3d[0],
                                              self.fx, self.fy,
                                              self.cx, self.cy)

            keypoints2d_list = [None]*len(kf.keypoints2d)
            for i in range(0, len(keypoints2d_list)):
                keypoints2d_list[i] = transform_keypoints(new_pose,
                                                kf.keypoints3d[i],
                                                self.fx, self.fy,
                                                self.cx, self.cy)

            self.detect_outliers(self.left, self.prev_left, keypoints2d_list,
                                 self.previous_keypoints2d, kf)

            point_aligner = PointAligner(kf.keypoints2d[0], keypoints2d,
                                         kf.left[0], self.left[0])

            warp, cost = point_aligner.align_points()

            keypoints2d_ext = np.ones((3, keypoints2d.shape[1]))
            keypoints2d_ext[0:2,:] = keypoints2d

            warped_points = np.matmul(warp, keypoints2d_ext)

            # Needs more testing!
            new_pose = self.pose_refiner.refine_pose(new_pose,
                                                      kf.keypoints3d[0],
                                                      warped_points)

            # Needs more testing!
            keypoints3d_refined = self.cloud_refiner.refine_cloud(new_pose,
                                                             kf.keypoints3d[0],
                                                             warped_points)
            # Only take refined keypoints that are valid
            kf.keypoints3d[0] = keypoints3d_refined

            MARGIN = 10
            valid_indexes = (keypoints2d[0,:]>MARGIN) &\
                            (keypoints2d[1,:]>MARGIN) &\
                            (keypoints2d[0,:]<(self.left[0].shape[1]-MARGIN))&\
                            (keypoints2d[1,:]<(self.left[0].shape[0]-MARGIN))

            matches = np.count_nonzero(valid_indexes==True)

            print(f"Found matches: {matches}")

            self.mapping.new_image(self.left, self.right, new_pose, matches, 0)
            self.mapping.process_image()


            self.motion_model =  new_pose - self.pose
            self.pose =  new_pose

            draw_kps(self.pose, self.left[0],
                     kf.left[0],
                     kf.keypoints2d[0],
                     kf.keypoints3d[0],
                     kf.colors,
                     self.fx, self.fy,
                     self.cx, self.cy)

            current_kf = self.mapping.get_last_keyframe()

            # Be careful when new kf is added
            if current_kf == kf:
                self.previous_keypoints2d = [None]*len(kf.keypoints2d)
                for i in range(0, len(self.previous_keypoints2d)):
                    self.previous_keypoints2d[i] = transform_keypoints(new_pose,
                                                    kf.keypoints3d[i],
                                                    self.fx, self.fy,
                                                    self.cx, self.cy)
            else:
                self.previous_keypoints2d = current_kf.keypoints2d

    def _estimate_pose(self, kf, kps3d):
        estimator = PoseEstimator(self.left, self.prev_left, self.previous_keypoints2d,
                                  kf.keypoints3d, self.fx, self.fy, self.cx, self.cy)
        return estimator.estimate_pose(self.pose + self.motion_model)
