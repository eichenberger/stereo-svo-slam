import numpy as np
import scipy.optimize as opt

from transform_keypoints import transform_keypoints
from image_operators import get_intensity_diff

class DepthAdjustment:
    def adjust_depth(self, keyframe, new_image, pose, fx, fy, cx, cy):
        kps3d = keyframe.keypoints3d.copy()
        costs = np.zeros((1, kps3d.shape[1]))
        for i in range(0, kps3d.shape[1]):
            # We need the keypoint still in vector format
            keypoint = kps3d[:, i:i+1]
            def _optimize_depth(z):
                kp3d = keypoint.copy()
                kp3d[2] = z
                kp2d = transform_keypoints(pose, kp3d, fx, fy, cx, cy)

                diff = get_intensity_diff(keyframe.image,
                                          new_image,
                                          keyframe.keypoints2d[:, i],
                                          kp2d[0:2,0],
                                          100000000000.0)
                return diff

            res = opt.minimize(_optimize_depth, keypoint[2], options={'maxiter':100})
            kps3d[2, i] = res.x[0]
            costs[0, i] = res.fun
        prop = 0.4*(1.0-(costs/10.0))
        for i in range(0, prop.shape[1]):
            prop[0, i] = max(prop[0, i],0)

        return kps3d, prop
