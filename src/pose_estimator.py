import math
import numpy as np
import scipy.optimize as opt
import cv2

from slam_accelerator import get_total_intensity_diff, project_keypoints

class PoseEstimator:
    def __init__(self, current_image, previous_image, previous_kf,
                 keypoints3d, camera_settings):
        self.current_image = current_image
        self.previous_image = previous_image
        self.previous_kps = previous_kps
        self.camera_settings = camera_settings

    def _optimize_pose(self, pose):
        _kps3d = self.previous_kps[self.round].kps3d
        _kps2d = self.previous_kps[self.round].kps2d
        kps2d = project_keypoints(pose, _kps3d, self.camera_settings)


        diff = get_total_intensity_diff(self.previous_image[self.round].left,
                                        self.current_image[self.round].left,
                                        _kps2d, kps2d, 4)

        return diff

    def estimate_pose(self, pose_guess):
        cost = 0
        current_guess = pose_guess
        res = None
        # Only estimate pose based on last 3 pyramid levels
        for i in range(0, min(3, len(self.current_image))):
            self.hessian = None
            self.jacobian = []

            self.round = len(self.current_image) - 1 - i


            self.gradient_x = cv2.Sobel(self.previous_image[self.round], cv2.CV_16S, 1, 0)
            self.gradient_y = cv2.Sobel(self.previous_image[self.round], cv2.CV_16S, 0, 1)
            res = opt.least_squares(self._optimize_pose, current_guess,
                                    method = 'lm')
            # Take the estimated pose as new guess
            current_guess = res.x
            cost = res.cost

        print(f"Guess: {pose_guess}")
        print(f"New pose: {current_guess}")
        print(f"Cost: {cost}")

        return current_guess, cost

