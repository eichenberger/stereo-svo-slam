import math
import numpy as np
import scipy.optimize as opt

from rotation_matrix import rotation_matrix

class PoseEstimator:
    def __init__(self, current_image, previous_image, previous_keypoints, keypoints3d, fx, fy, cx, cy):
        self.current_image = current_image
        self.previous_image = previous_image
        self.previous_keypoints = previous_keypoints
        self.keypoints3d = keypoints3d
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy


    def _get_sub_pixel(self, image, pt):
        pt = np.array(pt)
        pt_floor = np.floor(pt)
        pt_ceil = np.ceil(pt + [2e-10, 2e-10])

        prob_floor = [1,1] - (pt - pt_floor)
        prob_ceil = [1,1] - (pt_ceil - pt)

        sub_pixel_val = prob_floor[0]*prob_floor[1]*image[int(pt_floor[1]), int(pt_floor[0])]
        sub_pixel_val += prob_floor[0]*prob_ceil[1]*image[int(pt_ceil[1]), int(pt_floor[0])]
        sub_pixel_val += prob_ceil[0]*prob_floor[1]*image[int(pt_floor[1]), int(pt_ceil[0])]
        sub_pixel_val += prob_ceil[0]*prob_ceil[1]*image[int(pt_ceil[1]), int(pt_ceil[0])]

        return sub_pixel_val


    def _get_intensity_diff(self, pose):
        extrinsic = np.zeros((3,4))
        extrinsic[:,0:3] = rotation_matrix(pose[3:])
        extrinsic[:,3] = np.transpose(pose[0:3])

        keypoints3d = np.ones((4, len(self.keypoints3d)))
        keypoints3d[0:3, :] = np.transpose(self.keypoints3d)
        estimated_keypoints2d = np.matmul(extrinsic, keypoints3d)
        estimated_keypoints2d = (estimated_keypoints2d * [[self.fx],[self.fy], [1]])/ estimated_keypoints2d[2,:] + [[self.cx],[self.cy], [0]]

        diff = np.zeros((estimated_keypoints2d.shape[1]))
        for i in range(estimated_keypoints2d.shape[1]):
            prev_x = int(self.previous_keypoints[i, 0])
            prev_y = int(self.previous_keypoints[i, 1])

            cur_x = estimated_keypoints2d[0][i]
            cur_y = estimated_keypoints2d[1][i]

            if cur_x - 2 < 0 or cur_x + 2 > self.current_image.shape[1] or \
                    cur_y - 2 < 0 or cur_y + 2 > self.current_image.shape[0]:
                continue

            diff[i] = self.previous_image[prev_y, prev_x] - self._get_sub_pixel(self.current_image, [cur_x, cur_y])
            diff[i] += self.previous_image[prev_y, prev_x-1] - self._get_sub_pixel(self.current_image, [cur_x-1, cur_y])
            diff[i] += self.previous_image[prev_y-1, prev_x] - self._get_sub_pixel(self.current_image, [cur_x, cur_y-1])
            diff[i] += self.previous_image[prev_y, prev_x+1] - self._get_sub_pixel(self.current_image, [cur_x+1, cur_y])
            diff[i] += self.previous_image[prev_y+1, prev_x] - self._get_sub_pixel(self.current_image, [cur_x, cur_y+1])

        return diff

    def _optimize_pose(self, pose):
        return self._get_intensity_diff(pose)

    def estimate_pose(self, initial_pose, motion_model):
        pose_guess = initial_pose + motion_model

        res = opt.least_squares(self._optimize_pose, pose_guess,
                                 method = 'lm')


        print(f"guess: {pose_guess}, optimized: {res.x}")
        print(f"New pose: {res.x}")
        print(f"Cost: {res.cost}")

        return res.x

