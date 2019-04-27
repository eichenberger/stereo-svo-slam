import math
import numpy as np
import scipy.optimize as opt

from slam_accelerator import get_total_intensity_diff, transform_keypoints

class PoseEstimator:
    def __init__(self, current_image, previous_image, previous_keypoints,
                 keypoints3d, fx, fy, cx, cy):
        self.current_image = current_image
        self.previous_image = previous_image
        self.previous_keypoints = previous_keypoints
        self.keypoints3d = keypoints3d
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def _optimize_pose(self, pose):
        kps2d = transform_keypoints(pose, self.keypoints3d,
                                    self.fx, self.fy,
                                    self.cx, self.cy)


        diff = get_total_intensity_diff(self.previous_image, self.current_image,
                                        self.previous_keypoints, kps2d)

        return diff


    def estimate_pose(self, pose_guess):

        res = opt.least_squares(self._optimize_pose, pose_guess,
                                method = 'lm')

        print(f"Guess: {pose_guess}")
        print(f"New pose: {res.x}")
        print(f"Cost: {res.cost}")

        return res.x, res.cost

