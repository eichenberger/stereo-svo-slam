from libcpp.vector cimport vector

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

#cdef extern from "<array>" namespace "std" nogil:
#    cdef cppclass array3 "std::array<float, 3>":
#        array3() except+
#        float& operator[](size_t)
#    cdef cppclass array2 "std::array<float, 2>":
#        array2() except+
#        float& operator[](size_t)

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

    cdef struct KeyPointInformation:
        float score
        int level
        KeyPointType type
        float confidence

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
        int max_pyramid_levels

    cdef struct KeyPoints:
        vector[KeyPoint2d] kps2d
        vector[KeyPoint3d] kps3d
        vector[KeyPointInformation] info

    cdef struct Pose:
        float x
        float y
        float z
        float pitch # around x
        float yaw   # around y
        float roll  # around z

    cdef struct Color:
        unsigned char r
        unsigned char g
        unsigned char b

    cdef struct Frame:
        Pose pose
        StereoImage stereo_image
        KeyPoints kps

    cdef struct KeyFrame:
        Pose pose
        StereoImage stereo_image
        KeyPoints kps
        vector[Color] colors



