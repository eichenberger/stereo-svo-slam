from libcpp.vector cimport vector
from libc.stdint cimport uint64_t
from libcpp cimport bool

from pose_manager cimport PoseManager as _PoseManager

# Declares OpenCV's cv::Mat class
cdef extern from "opencv2/core/core.hpp" namespace "cv":
    cdef cppclass Mat:
        Mat()
        Mat (int rows, int cols, int type, void *data)
        void create(int rows, int cols, int type)
        void copyTo(Mat m) const
        unsigned char* data
        int rows
        int cols
        pass

# Declares OpenCV's cv::Mat class
cdef extern from "opencv2/video/tracking.hpp" namespace "cv":
    cdef cppclass KalmanFilter:
        KalmanFilter()
#        KalmanFilter(KalmanFilter rhs)
#
#        Mat statePre
#        Mat statePost
#        Mat transitionMatrix
#        Mat controlMatrix
#        Mat measurementMatrix
#        Mat processNoiseCov
#        Mat measurementNoiseCov
#        Mat errorCovPre
#        Mat gain
#        Mat errorCovPost
#
#        Mat temp1
#        Mat temp2
#        Mat temp3
#        Mat temp4
#        Mat temp5
        pass


cdef extern from "stereo_slam_types.hpp":
    cdef enum KeyPointType:
        KP_FAST,
        KP_EDGELET

    cdef struct KeyPoint2d:
        float x
        float y

    cdef struct KeyPoint3d:
        float x
        float y
        float z

    cdef struct Color:
        unsigned char r
        unsigned char g
        unsigned char b

    cdef struct KeyPointInformation:
        float score
        int level
        KeyPointType type
        float confidence
        uint64_t keyframe_id
        size_t keypoint_index;
        Color color
        bool ignore_during_refinement
        bool ignore_completely
        int outlier_count
        int inlier_count
        KalmanFilter kf

    cdef struct StereoImage:
        vector[Mat] left
        vector[Mat] right

    cdef struct CameraSettings:
        float baseline
        float fx
        float fy
        float cx
        float cy
        float k1
        float k2
        float k3
        float p1
        float p2
        int grid_height
        int grid_width
        int search_x
        int search_y
        int window_size
        int window_size_opt_flow
        int window_size_depth_calculator
        int max_pyramid_levels
        int min_pyramid_level_pose_estimation

        int image_width
        int image_height

        float dist_window_k0
        float dist_window_k1
        float dist_window_k2
        float dist_window_k3

        float cost_k0
        float cost_k1

    cdef struct KeyPoints:
        vector[KeyPoint2d] kps2d
        vector[KeyPoint3d] kps3d
        vector[KeyPointInformation] info

    cdef struct Pose:
        float x
        float y
        float z
        float rx # around x
        float ry   # around y
        float rz  # around z

    cdef struct Frame:
        uint64_t id
        _PoseManager pose
        StereoImage stereo_image
        KeyPoints kps

    cdef struct KeyFrame:
        uint64_t id
        _PoseManager pose
        StereoImage stereo_image
        KeyPoints kps



