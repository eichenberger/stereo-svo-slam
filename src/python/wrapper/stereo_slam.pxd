from stereo_slam_types cimport Mat, CameraSettings, KeyFrame, Frame
from libcpp.vector cimport vector

cdef extern from "stereo_slam.hpp":
    cdef cppclass StereoSlam:
        StereoSlam(const CameraSettings &camera_settings)
        void new_image(const Mat &left, const Mat &right) nogil
        void get_keyframe(KeyFrame &keyframe) nogil
        void get_frame(Frame &frame) nogil
        void get_keyframes(vector[KeyFrame] &keyframes) nogil

