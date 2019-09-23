import math
import numpy as np
import scipy.optimize as opt
import cv2
from scipy.linalg import expm

from slam_accelerator import get_total_intensity_diff, project_keypoints
from slam_accelerator import CameraSettings

class PoseEstimator:
    def __init__(self, current_image, previous_image, previous_kps,
                 camera_settings):
        self.current_image = current_image
        self.previous_image = previous_image
        self.previous_kps = previous_kps
        self.camera_settings = camera_settings


    def _optimize_pose(self, pose):
        divider = 2**self.round
        _kps3d = self.previous_kps.kps3d
        _kps2d = self.previous_kps.kps2d
        for i in range(0, len(_kps2d)):
            _kps2d[i]['x'] = _kps2d[i]['x']/divider
            _kps2d[i]['y'] = _kps2d[i]['y']/divider

        kps2d = project_keypoints(pose, _kps3d,
                                  self._camera_settings[self.round])


        diff = get_total_intensity_diff(self.current_image[self.round].left,
                                        self.previous_image[self.round].left,
                                        kps2d, _kps2d, 4)

        return diff

    def _optimize_pose_newton(self, pose):
        diff = self._optimize_pose(pose)

        return np.sum(diff), diff

    def _get_derivate2(self, pose):
        DIFF = 0.0001
        _pose = pose.copy()
        _pose['x'] -= DIFF
        t41, _none_ = self._optimize_pose_newton(_pose)
        _pose['x'] += 2*DIFF
        t42, _none_ = self._optimize_pose_newton(_pose)

        _pose = pose.copy()
        _pose['y'] -= DIFF
        t51, _none_ = self._optimize_pose_newton(_pose)
        _pose['y'] += 2*DIFF
        t52, _none_ = self._optimize_pose_newton(_pose)

        _pose = pose.copy()
        _pose['z'] -= DIFF
        t61, _none_ = self._optimize_pose_newton(_pose)
        _pose['z'] += 2*DIFF
        t62, _none_ = self._optimize_pose_newton(_pose)

        _pose = pose.copy()
        _pose['roll'] -= DIFF
        t11, _none_ = self._optimize_pose_newton(_pose)
        _pose['roll'] += 2*DIFF
        t12, _none_ = self._optimize_pose_newton(_pose)

        _pose = pose.copy()
        _pose['pitch'] -= DIFF
        t21, _none_ = self._optimize_pose_newton(_pose)
        _pose['pitch'] += 2*DIFF
        t22, _none_ = self._optimize_pose_newton(_pose)

        _pose = pose.copy()
        _pose['yaw'] -= DIFF
        t31, _none_ = self._optimize_pose_newton(_pose)
        _pose['yaw'] += 2*DIFF
        t32, _none_ = self._optimize_pose_newton(_pose)

        grad = np.array([t11-t12, t21-t22, t31-t32, t41-t42, t51-t52, t61-t62])
        return grad

    def _get_derivate(self, pose, diff):
        # This is slightly shaky because the gradient is only taken at the
        # exact point position instead an averiged one

        kps3d = self.previous_kps.kps3d
        kps2d = self.previous_kps.kps2d


        divider = 2**self.round
        _image = self.previous_image[self.round].left
        if self.hessian is None:
            hessian = np.zeros((6,6))
            for i in range(0, len(kps3d)):
                kp2d = kps2d[i]
                x = kps3d[i]['x']
                y = kps3d[i]['y']
                z = kps3d[i]['z']

                # minus becuse we want to minimize
                jacobian = -np.mat([[1/z, 0, -x/z**2, -x*y/z**2,1+x**2/z**2, -y/z],
                                   [0, 1/z, -y/z**2, -(1+y**2/z**2), x*y/z**2, x/z]])

                x2d = kp2d['x']/divider
                y2d = kp2d['y']/divider

                int1 = cv2.getRectSubPix(_image, (1,1), (x2d, y2d), patchType=cv2.CV_32F)
                int2 = cv2.getRectSubPix(_image, (1,1), (x2d+1.0, y2d), patchType=cv2.CV_32F)
                int3 = cv2.getRectSubPix(_image, (1,1), (x2d, y2d+1.0), patchType=cv2.CV_32F)
                # We have to norm the gradient
                grad = np.mat([int1[0]-int2[0], int1[0]-int3[0]])
                #grad = grad/np.sum(grad)

                jac = grad.transpose()*jacobian
                self.jacobian.append(jac)
                hessian = hessian + jac.transpose()*jac

            self.hessian = np.linalg.pinv(hessian)

        _kps2d = project_keypoints(pose, kps3d,
                                  self._camera_settings[self.round])
        __image = self.current_image[self.round].left
        dp = np.zeros((6,1))
        for i in range(0, len(kps3d)):
            kp2d = kps2d[i]
            _kp2d = _kps2d[i]

            x2d = kp2d['x']/divider
            y2d = kp2d['y']/divider

            int1 = cv2.getRectSubPix(_image, (1,1), (x2d, y2d), patchType=cv2.CV_32F)
            int2 = cv2.getRectSubPix(__image, (1,1), (_kp2d['x'], _kp2d['y']), patchType=cv2.CV_32F)

            jacobian = self.jacobian[i]
            dp = dp + np.transpose(jacobian)*(int2[0]-int1[0])

        dp = np.asarray(np.matmul(self.hessian, dp)).reshape(6)
        skew_mat = np.mat([[0, -dp[5], dp[4], dp[0]],
                           [dp[5], 0, -dp[3], dp[1]],
                           [-dp[4], dp[3], 0, dp[2]],
                           [0,0,0,0]])
        homogenous = expm(skew_mat)
        angles = cv2.Rodrigues(homogenous[0:3,0:3])[0].reshape(3)
        dp = np.array([homogenous[0,3], homogenous[1,3], homogenous[2,3],
                       angles[0], angles[1], angles[2]])

        return dp/np.linalg.norm(dp)

    def _optimize(self, x0):
        MAX_ITER = 50
        x = x0
        k = 0.001
        total_diff, diff = self._optimize_pose_newton(x)
        new_x = x.copy()
        for i in range(0, MAX_ITER):
            # Check 3d cloud. Is it really okay?
            dp = self._get_derivate(x, diff)
            #dp2 = np.array([0.0,0.0,0.0,-1.0,0.0,0.0])
            #dp3 = self._get_derivate2(x)
            while i < MAX_ITER:
                new_x['x'] =  new_x['x'] + k*dp[0]
                new_x['y'] =  new_x['y'] + k*dp[1]
                new_x['z'] =  new_x['z'] + k*dp[2]
                new_x['roll'] =  new_x['roll'] + k*dp[3]
                new_x['pitch'] =  new_x['pitch'] + k*dp[4]
                new_x['yaw'] =  new_x['yaw'] + k*dp[5]
                new_total_diff, new_diff = self._optimize_pose_newton(new_x)

                if new_total_diff < total_diff:
                    x = new_x
                    total_diff = new_total_diff
                    diff = new_diff
                    break
                else:
                    k = 0.5*k
                    i += 1

        return x, total_diff

    def estimate_pose(self, pose_guess):
        cost = 0
        current_guess = pose_guess
        res = None

        self._camera_settings = []
        self._camera_settings.append(self.camera_settings)

        for i in range(1, len(self.current_image)):
            _cs = CameraSettings(self._camera_settings[i-1])
            _cs.baseline /= 2
            _cs.fx /= 2
            _cs.fy /= 2
            _cs.cx /= 2
            _cs.cy /= 2
            self._camera_settings.append(_cs)

        total_cost = 0
        # Only estimate pose based on last 3 pyramid levels
        #for i in range(0, min(3, len(self.current_image))):
        for i in range(0, len(self.current_image)):
            self.hessian = None
            self.jacobian = []

            self.round = len(self.current_image) - 1 - i

            current_guess, cost = self._optimize(current_guess)
            total_cost = total_cost+cost
            print(current_guess)

        #self.hessian = None
        #self.jacobian = []
        #self.round = 0
        #current_guess, cost = self._optimize(current_guess)
        #total_cost = total_cost+cost

        print(f"Guess: {pose_guess}")
        print(f"New pose: {current_guess}")
        print(f"Cost: {cost}")
        print(f"Total cost: {total_cost}")

        return current_guess, cost

