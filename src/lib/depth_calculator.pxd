from libcpp.vector cimport vector
cimport stereo_slam_types

cdef extern from "depth_calculator.hpp":
    cdef cppclass DepthCalculator:
        DepthCalculator()
        void calculate_depth(const vector[stereo_slam_types.StereoImage] &stereo_image,
            const stereo_slam_types.CameraSettings camera_settings,
            stereo_slam_types.KeyPoints &keypoints) nogil

