import cv2
import numpy as np

#from depth_calculator import DepthCalculator
from draw_kps import draw_kps

from slam_accelerator import CameraSettings, StereoImage, KeyPoints
from slam_accelerator import project_keypoints
#from pose_refiner import PoseRefiner
#from cloud_refiner import CloudRefiner

#from pose_estimator import PoseEstimator
#from pose_estimator_grad import PoseEstimator
from slam_accelerator import PoseEstimator

from mapping import Mapping

class StereoSLAM:
    def __init__(self, baseline, fx, fy, cx, cy):
        self.keypoints = None
        self.stereo_image = None

        self.camera_settings = CameraSettings()
        self.camera_settings.baseline = baseline
        self.camera_settings.fx = fx
        self.camera_settings.fy = fy
        self.camera_settings.cx = cx
        self.camera_settings.cy = cy
        self.camera_settings.grid_width = 40
        self.camera_settings.grid_height = 30
        self.camera_settings.search_x = 30
        self.camera_settings.search_y = 6
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


#    def detect_outliers(self, current, previous,
#                        current_keypoints2d, previous_keypoints, kf):
#        for i in range(0, len(current)):
#            for j in range(0, current_keypoints2d[i].shape[1]):
#                diff = get_intensity_diff(current[i], previous[i],
#                                    current_keypoints2d[i][:, j],
#                                    previous_keypoints[i][:,j], 0)
#                if diff > 100:
#                    kf.blacklist[i][j] = True
#
#    def get_kps3d(self, kf):
#        kps3d = [None]*len(kf.blacklist)
#        for i in range(0, len(kf.blacklist)):
#            kps3d[i] = kf.keypoints3d[i][:, np.array(kf.blacklist[i]) == False]
#
#        return kps3d



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
            self.previous_kps = kf.kps
        else:
            kf = self.mapping.get_last_keyframe()

            #kps3d = self.get_kps3d(kf)
            new_pose, cost = self._estimate_pose()
            _cs = CameraSettings(self.camera_settings)
            self.previous_kps.kps2d = project_keypoints(new_pose,
                                                        self.previous_kps.kps3d,
                                                        _cs)

            draw_kps(self.stereo_image, self.previous_kps, kf)
            self.pose = new_pose

            # # We get the pose between the last and the current image.
            # # We need to update the global pose
            # keypoints2d = transform_keypoints(new_pose,
            #                                   kf.keypoints3d[0],
            #                                   self.fx, self.fy,
            #                                   self.cx, self.cy)

            # keypoints2d_list = [None]*len(kf.keypoints2d)
            # for i in range(0, len(keypoints2d_list)):
            #     keypoints2d_list[i] = transform_keypoints(new_pose,
            #                                     kf.keypoints3d[i],
            #                                     self.fx, self.fy,
            #                                     self.cx, self.cy)

            # self.detect_outliers(self.left, self.prev_left, keypoints2d_list,
            #                      self.previous_keypoints2d, kf)

            # point_aligner = PointAligner(kf.keypoints2d[0], keypoints2d,
            #                              kf.left[0], self.left[0])

            # warp, cost = point_aligner.align_points()

            # keypoints2d_ext = np.ones((3, keypoints2d.shape[1]))
            # keypoints2d_ext[0:2,:] = keypoints2d

            # warped_points = np.matmul(warp, keypoints2d_ext)

            # # Needs more testing!
            # #new_pose = self.pose_refiner.refine_pose(new_pose,
            # #                                          kf.keypoints3d[0],
            # #                                          warped_points)

            # # Needs more testing!
            # keypoints3d_refined = self.cloud_refiner.refine_cloud(new_pose,
            #                                                  kf.keypoints3d[0],
            #                                                  warped_points)
            # # Only take refined keypoints that are valid
            # kf.keypoints3d[0] = keypoints3d_refined

            # MARGIN = 10
            # valid_indexes = (keypoints2d[0,:]>MARGIN) &\
            #                 (keypoints2d[1,:]>MARGIN) &\
            #                 (keypoints2d[0,:]<(self.left[0].shape[1]-MARGIN))&\
            #                 (keypoints2d[1,:]<(self.left[0].shape[0]-MARGIN))

            # matches = np.count_nonzero(valid_indexes==True)

            # print(f"Found matches: {matches}")

#            self.mapping.new_image(self.stereo_image, new_pose, matches, 0)
#            self.mapping.process_image()
#
#
#            self.motion_model =  new_pose - self.pose
#            self.pose =  new_pose

#            draw_kps(self.pose, self.stereo_image[0].left,
#                     kf.stereo_image[0].left,
#                     kf.kps[0].kps2d[0],
#                     kf.kps[0].kps3d[0],
#                     kf.colors,
#                     self.fx, self.fy,
#                     self.cx, self.cy)
#
#            current_kf = self.mapping.get_last_keyframe()
#
#            # Be careful when new kf is added
#            if current_kf == kf:
#                self.previous_kps2d = project_keypoints(new_pose,
#                                                        kf.kps3d[i],
#                                                        self.camera_settings)
#            else:
#                self.previous_keypoints2d = current_kf.keypoints2d

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

