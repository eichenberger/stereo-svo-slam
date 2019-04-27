import scipy.optimize as opt
from slam_accelerator import transform_keypoints

class PoseRefiner:
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def _update_pose(self, pose):
        kps2d = transform_keypoints(pose, self.keypoints3d,
                            self.fx, self.fy,
                            self.cx, self.cy)

        return (kps2d - self.keypoints2d).flatten()

    def refine_pose(self, pose, keypoints3d, keypoints2d):
        self.keypoints3d = keypoints3d
        self.keypoints2d = keypoints2d
        res = opt.least_squares(self._update_pose, pose,
                                method = 'lm')
        print(f'Refined pose: {res.x}')

        return res.x
