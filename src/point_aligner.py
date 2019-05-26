import numpy as np
import scipy.optimize as opt

from slam_accelerator import get_total_intensity_diff

class PointAligner:
    def __init__(self, points_kf, points_current, image_kf, image_current):
        self.points_kf = points_kf
        self.points_current = np.ones((3, points_current.shape[1]))
        self.points_current[0:2,:] = points_current
        self.image_kf = image_kf
        self.image_current = image_current

    def _optimize_warp(self, warp):
        warp_matrix = np.reshape(warp, (2,3)) + np.array([[1,0,0],[0,1,0]])

        warped_points = np.matmul(warp_matrix, self.points_current)

        intensity_diff = get_total_intensity_diff(self.image_kf, self.image_current,
                                                  self.points_kf, warped_points)

        return intensity_diff

    def align_points(self):
        warp_matrix_guess = [0,0,0,0,0,0]
        res = opt.least_squares(self._optimize_warp, warp_matrix_guess,
                                method = 'lm')

        print(f"Warp matrix: {res.x}")
        print(f"Cost: {res.cost}")

        return np.reshape(res.x, (2,3)) + np.array([[1,0,0],[0,1,0]]), res.cost


