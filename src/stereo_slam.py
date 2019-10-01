import cv2
import numpy as np

#from depth_calculator import DepthCalculator
from draw_kps import draw_kps

from slam_accelerator import CameraSettings, StereoImage, KeyPoints
from slam_accelerator import project_keypoints
from slam_accelerator import PoseRefiner, OpticalFlow
#from pose_refiner import PoseRefiner
#from cloud_refiner import CloudRefiner

#from pose_estimator import PoseEstimator
#from pose_estimator_grad import PoseEstimator
from slam_accelerator import PoseEstimator

from mapping import Mapping

class StereoSLAM:
    def __init__(self, camera_settings):
        self.keypoints = None
        self.stereo_image = None

        self.camera_settings = camera_settings
        self.pyramid_levels = 4
        # vx, vy, vz, gx, gy, gz
        self.motion_model = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.pose = {'x': 0.0, 'y': 0.0, 'z': 0.0,
                     'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}

        # self.depth_calculator = DepthCalculator(self.baseline, self.fx, self.fy, self.cx, self.cy)
        # self.pose_refiner = PoseRefiner(self.camera_settings)
        # self.cloud_refiner = CloudRefiner(self.camera_settings)
        # self.depth_adjustment = DepthAdjustment()

        self.mapping = Mapping(self.camera_settings)

    def new_image(self, left, right):
        self.prev_stereo_image = self.stereo_image


        stereo_image = StereoImage()
        # We need to copy it to align the data properly
        stereo_image.left = left.copy()
        stereo_image.right = right.copy()

        self.stereo_image = [None]*self.pyramid_levels
        self.stereo_image[0] = stereo_image
        for i in range(1, self.pyramid_levels):
            rows, cols = map(lambda x: int(x/2), self.stereo_image[i-1].left.shape)
            stereo_image = StereoImage()
            stereo_image.left = cv2.pyrDown(self.stereo_image[i-1].left, dstsize=(cols, rows))
            stereo_image.right = cv2.pyrDown(self.stereo_image[i-1].right, dstsize=(cols, rows))
            self.stereo_image[i] = stereo_image

        if self.mapping.number_of_keyframes() == 0:
            self.mapping.new_image(self.stereo_image, self.pose, 0, 0)
            self.mapping.process_image()
            kf = self.mapping.get_last_keyframe()
            self.previous_kps = kf.kps.copy()
        else:
            kf = self.mapping.get_last_keyframe()

            #kps3d = self.get_kps3d(kf)
            timer = cv2.TickMeter()
            timer.start()
            new_pose, cost = self._estimate_pose()
            timer.stop()
            print(f"Time estimate pose: {timer.getTimeSec()}")
            timer.reset()

            _cs = CameraSettings(self.camera_settings)

            estimated_keypoints = project_keypoints(new_pose,
                                                    self.previous_kps.kps3d,
                                                    self.camera_settings)

            #draw_kps(self.stereo_image, estimated_keypoints, kf)

            refined_keypoints = self.previous_kps.copy()

            timer.start()
            optical_flow = OpticalFlow()
            refined_keypoints.kps2d, err = optical_flow.calculate_optical_flow(
                kf.stereo_images, kf.kps.kps2d,
                self.stereo_image, estimated_keypoints)

            timer.stop()
            print(f"Time optical flow: {timer.getTimeSec()}")
            timer.reset()


            for i in range(0, len(refined_keypoints.kps2d)):
                if err[i] > 100:
                    refined_keypoints.kps2d[i] = estimated_keypoints[i]

            draw_kps(self.stereo_image, refined_keypoints.kps2d, kf)

            timer.start()
            pose_refiner = PoseRefiner(self.camera_settings)
            refined_pose = pose_refiner.refine_pose(refined_keypoints, new_pose)
            self.previous_kps.kps2d = project_keypoints(
                refined_pose, kf.kps.kps3d, _cs)

            timer.stop()
            print(f"Time refine keypoints: {timer.getTimeSec()}")
            timer.reset()

            draw_kps(self.stereo_image, self.previous_kps.kps2d, kf)
            self.pose = refined_pose


    def _estimate_pose(self):
        #estimator = PoseEstimator(self.stereo_image, self.prev_stereo_image,
        #                          self.previous_kps, self.camera_settings)
        #return estimator.estimate_pose(self.pose)

        pose = self.pose
        for i in range(len(self.stereo_image)):
            level = len(self.stereo_image) - 1 - i
            divider = 2**level
            current = self.stereo_image[level]
            previous = self.prev_stereo_image[level]

            previous_kps = KeyPoints()
            kps2d = self.previous_kps.kps2d.copy()
            for j in range(len(kps2d)):
                kps2d[j]['x'] /= divider
                kps2d[j]['y'] /= divider

            previous_kps.kps2d = kps2d
            previous_kps.kps3d = self.previous_kps.kps3d

            camera_settings = CameraSettings(self.camera_settings)
            camera_settings.fx /= divider
            camera_settings.fy /= divider
            camera_settings.cx /= divider
            camera_settings.cy /= divider

            estimator = PoseEstimator(current, previous,
                                      previous_kps, camera_settings)
            pose, cost = estimator.estimate_pose(pose)
            print (f'current pose: {pose}, cost: {cost}')

        print (f'final pose: {pose}, cost: {cost}')
        return pose, cost

