from libcpp.vector cimport vector

# Declares OpenCV's cv::Mat class
cdef extern from "opencv2/core/core.hpp" namespace "cv":
    cdef cppclass Mat:
        Mat()
        Mat (int rows, int cols, int type, void *data)
        void create(int rows, int cols, int type)
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
    cdef enum KeyPointType
        KP_FAST,
        KP_EDGELET

    cdef struct KeyPoint2d:
        float x
        float y
        int level
        KeyPointType type

    cdef struct KeyPoint3d:
        float x
        float y
        float z

    cdef struct StereoImage:
        Mat left
        Mat right

    cdef struct CameraSettings:
        float baseline
        float fx
        float fy
        float cx
        float cy
        int grid_height
        int grid_width
        int search_x
        int search_y
        int window_size

    cdef struct KeyPoints:
        vector[KeyPoint2d] kps2d
        vector[KeyPoint3d] kps3d
        vector[float] confidence

    cdef struct Pose:
        float x
        float y
        float z
        float roll # around x
        float pitch #around y
        float yaw #around z

    cdef struct Color:
        unsigned char r
        unsigned char g
        unsigned char b

    cdef struct KeyFrame:
        Pose pose
        vector[StereoImage] stereo_images
        KeyPoints kps
        vector[Color] colors
