from libcpp.vector cimport vector
from stereo_slam_types cimport StereoImage, KeyPoint2d

cdef extern from "optical_flow.hpp":
    cdef cppclass OpticalFlow:
        OpticalFlow()

        void calculate_optical_flow(const vector[StereoImage] &previous_stereo_image_pyr,
            const vector[KeyPoint2d] &previous_keypoints2d,
            const vector[StereoImage] &current_stereo_image_pyr,
            vector[KeyPoint2d] &current_keypoints2d,
            vector[float] &err)

