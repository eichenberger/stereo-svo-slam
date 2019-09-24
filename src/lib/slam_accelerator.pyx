cimport numpy as np
import numpy as np
import cv2
from libcpp.vector cimport vector

from cython.parallel import parallel, prange
from cython import boundscheck, wraparound
from cpython.ref cimport PyObject
from cython cimport view
from libc.math cimport fabs
from libc.math cimport sin, cos
from libc.string cimport memcpy

ctypedef double (*opt_fun)(const double*)

cimport depth_calculator
cimport transform_keypoints
cimport image_comparison
cimport stereo_slam_types
cimport pose_estimator
cimport optical_flow
cimport pose_refinement

cdef _c2np(stereo_slam_types.Mat image):
    rows = image.rows
    cols = image.cols
    return np.asarray(<np.uint8_t[:rows,:cols]> image.data)

cdef _np2c(pyimage, stereo_slam_types.Mat &cimage):
    r = pyimage.shape[0]
    c = pyimage.shape[1]
    cimage.create(r, c, cv2.CV_8UC1)
    cdef unsigned char[:,:] image_buffer = pyimage
    memcpy(cimage.data, &image_buffer[0,0], r*c)

def project_keypoints(pose, input, CameraSettings camera_settings):
    cdef vector[stereo_slam_types.KeyPoint2d] keypoints2d

    transform_keypoints.project_keypoints(pose, input, camera_settings._camera_settings, keypoints2d)

    return keypoints2d

def get_total_intensity_diff(image1, image2, keypoints1, keypoints2, patchSize):
    cdef vector[float] diff
    cdef stereo_slam_types.Mat _image1
    cdef stereo_slam_types.Mat _image2

    _np2c(image1, _image1)
    _np2c(image2, _image2)

    image_comparison.get_total_intensity_diff(_image1, _image2, keypoints1,
                                              keypoints2, patchSize, diff)

    return np.asarray(diff)

def transform_keypoints_inverse(pose, input):
    cdef vector[stereo_slam_types.KeyPoint3d] output

    transform_keypoints.transform_keypoints_inverse(pose, input, output)

    return output

cdef class DepthCalculator:
    cdef depth_calculator.DepthCalculator _depth_calculator

    def calculate_depth(self, stereo_images, CameraSettings camera_settings):
        keypoints = KeyPoints()

        cdef vector[stereo_slam_types.StereoImage] _stereo_images
        for stereo_image in stereo_images:
            _stereo_images.push_back((<StereoImage>stereo_image)._stereo_image)

        self._depth_calculator.calculate_depth(_stereo_images,
                                               camera_settings._camera_settings,
                                               keypoints._keypoints)

        return keypoints

cdef class StereoImage:
    cdef stereo_slam_types.StereoImage _stereo_image

    @property
    def left(self):
        return _c2np(self._stereo_image.left)
    @left.setter
    def left(self, left):
        _np2c(left, self._stereo_image.left)
    @property
    def right(self):
        return _c2np(self._stereo_image.right)
    @right.setter
    def right(self, right):
        _np2c(right, self._stereo_image.right)

cdef class CameraSettings:
    cdef stereo_slam_types.CameraSettings _camera_settings

    def __init__(self, CameraSettings camera_settings = None):
        if camera_settings:
            memcpy(&self._camera_settings,
                   &camera_settings._camera_settings, sizeof(stereo_slam_types.CameraSettings))

    @property
    def baseline(self):
        return self._camera_settings.baseline
    @baseline.setter
    def baseline(self, baseline):
        self._camera_settings.baseline = baseline
    @property
    def fx(self):
        return self._camera_settings.fx
    @fx.setter
    def fx(self, fx):
        self._camera_settings.fx = fx
    @property
    def fy(self):
        return self._camera_settings.fy
    @fy.setter
    def fy(self, fy):
        self._camera_settings.fy = fy
    @property
    def cx(self):
        return self._camera_settings.cx
    @cx.setter
    def cx(self, cx):
        self._camera_settings.cx = cx
    @property
    def cy(self):
        return self._camera_settings.cy
    @cy.setter
    def cy(self, cy):
        self._camera_settings.cy = cy
    @property
    def grid_height(self):
        return self._camera_settings.grid_height
    @grid_height.setter
    def grid_height(self, grid_height):
        self._camera_settings.grid_height = grid_height
    @property
    def grid_width(self):
        return self._camera_settings.grid_width
    @grid_width.setter
    def grid_width(self, grid_width):
        self._camera_settings.grid_width = grid_width
    @property
    def search_x(self):
        return self._camera_settings.search_x
    @search_x.setter
    def search_x(self, search_x):
        self._camera_settings.search_x = search_x
    @property
    def search_y(self):
        return self._camera_settings.search_y
    @search_y.setter
    def search_y(self, search_y):
        self._camera_settings.search_y = search_y
    @property
    def window_size(self):
        return self._camera_settings.window_size
    @window_size.setter
    def window_size(self, window_size):
        self._camera_settings.window_size = window_size


cdef class KeyPoints:
    cdef stereo_slam_types.KeyPoints _keypoints

    @property
    def kps2d(self):
        return self._keypoints.kps2d
    @kps2d.setter
    def kps2d(self, kps2d):
        self._keypoints.kps2d = kps2d
    @property
    def kps3d(self):
        return self._keypoints.kps3d
    @kps3d.setter
    def kps3d(self, kps3d):
        self._keypoints.kps3d = kps3d

cdef class KeyFrame:
    cdef stereo_slam_types.KeyFrame _keyframe

    @property
    def pose(self):
        return self._keyframe.pose
    @pose.setter
    def pose(self, pose):
        self._keyframe.pose = pose

    @property
    def stereo_images(self):
        # Create a list of stereo images from C vector
        stereo_images = []
        for i in range(0, self._keyframe.stereo_images.size()):
            stereo_image = StereoImage()
            rows = self._keyframe.stereo_images[i].left.rows
            cols = self._keyframe.stereo_images[i].left.cols

            # Because of a strange reason if the matrix is not created yet
            # it will fail to assign another matrix to it. Therfore, we create one
            stereo_image._stereo_image.left.create(rows, cols, cv2.CV_8UC1)
            stereo_image._stereo_image.left = self._keyframe.stereo_images[i].left

            stereo_image._stereo_image.right.create(rows, cols, cv2.CV_8UC1)
            stereo_image._stereo_image.right = self._keyframe.stereo_images[i].right
            stereo_images.append(stereo_image)
        return stereo_images
    @stereo_images.setter
    def stereo_images(self, stereo_images):
        # Assign list of StereoImages to C type
        self._keyframe.stereo_images.resize(len(stereo_images))
        for i, stereo_image in enumerate(stereo_images):
            self._keyframe.stereo_images[i] = (<StereoImage>stereo_image)._stereo_image

    @property
    def kps(self):
        kps = KeyPoints()
        kps._keypoints = self._keyframe.kps
        return kps
    @kps.setter
    def kps(self, kps):
        self._keyframe.kps = (<KeyPoints>kps)._keypoints

    @property
    def colors(self):
        # Let autoconvert do the magic (Color will be a dict)
        return self._keyframe.colors
    @colors.setter
    def colors(self, colors):
        # Let autoconvert do the magic
        self._keyframe.colors = colors

cdef class PoseEstimator:
    cdef pose_estimator.PoseEstimator *_pose_estimator

    def __cinit__(self, StereoImage current_stereo_image,
                 StereoImage previous_stereo_image,
                 KeyPoints previous_keypoints,
                 CameraSettings camera_settings):
        self._pose_estimator = new pose_estimator.PoseEstimator(
            current_stereo_image._stereo_image,
            previous_stereo_image._stereo_image,
            previous_keypoints._keypoints,
            camera_settings._camera_settings)

    def __dealloc__(self):
        del self._pose_estimator

    def estimate_pose(self, pose_guess):
        cdef stereo_slam_types.Pose estimated_pose
        cost = self._pose_estimator.estimate_pose(pose_guess, estimated_pose)
        return estimated_pose, cost


cdef class OpticalFlow:
    cdef optical_flow.OpticalFlow *_optical_flow

    def __cinit__(self):
        self._optical_flow= new optical_flow.OpticalFlow()

    def __dealloc__(self):
        del self._optical_flow

    def calculate_optical_flow(self, previous_stereo_image_pyr,
                               previous_keypoints2d,
                               current_stereo_image_pyr,
                               current_keypoints2d):
        cdef vector[stereo_slam_types.StereoImage] _previous_stereo_image_pyr
        for stereo_image in previous_stereo_image_pyr:
            _previous_stereo_image_pyr.push_back((<StereoImage>stereo_image)._stereo_image)

        cdef vector[stereo_slam_types.StereoImage] _current_stereo_image_pyr
        for stereo_image in current_stereo_image_pyr:
            _current_stereo_image_pyr.push_back((<StereoImage>stereo_image)._stereo_image)

        cdef vector[stereo_slam_types.KeyPoint2d] refined_keypoints = current_keypoints2d.copy()
        cdef vector[float] err

        self._optical_flow.calculate_optical_flow(_previous_stereo_image_pyr,
                                                  previous_keypoints2d,
                                                  _current_stereo_image_pyr,
                                                  refined_keypoints,
                                                  err)

        return refined_keypoints, err

cdef class PoseRefiner:
    cdef pose_refinement.PoseRefiner *_pose_refiner

    def __cinit__(self, CameraSettings camera_settings):
        self._pose_refiner = new pose_refinement.PoseRefiner(camera_settings._camera_settings)

    def __dealloc__(self):
        del self._pose_refiner

    def refine_pose(self, keypoints2d, keypoints3d, estimated_pose):
        cdef stereo_slam_types.Pose refined_pose

        self._pose_refiner.refine_pose(keypoints2d, keypoints3d, estimated_pose,
                                       refined_pose)

        return refined_pose
