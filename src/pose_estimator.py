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
        kps2d = transform_keypoints(pose, self.keypoints3d[self.round],
                                    self.fx, self.fy,
                                    self.cx, self.cy)


        diff = get_total_intensity_diff(self.previous_image[self.round],
                                        self.current_image[self.round],
                                        self.previous_keypoints[self.round],
                                        kps2d)

        return diff


    def estimate_pose(self, pose_guess):
        cost = 0
        current_guess = pose_guess
        res = None
        # Only estimate pose based on last 3 pyramid levels
        for i in range(0, min(3, len(self.current_image))):
            self.round = len(self.current_image) - 1 - i
            res = opt.least_squares(self._optimize_pose, current_guess,
                                    method = 'lm')
            # Take the estimated pose as new guess
            current_guess = res.x
            cost = res.cost

        print(f"Guess: {pose_guess}")
        print(f"New pose: {current_guess}")
        print(f"Cost: {cost}")

        return current_guess, cost

