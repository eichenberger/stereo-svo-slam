cimport numpy as np
import numpy as np

from cython.parallel import parallel, prange
from cython import boundscheck, wraparound
from cpython.ref cimport PyObject
from cython cimport view
from libc.math cimport fabs
from libcpp.vector cimport vector
from libc.math cimport sin, cos

ctypedef double (*opt_fun)(const double*)

cdef extern from "slam_accelerator_helper.hpp":
    cdef void c_rotation_matrix(double* angle, double* rotation_matrix)
    cdef void c_transform_keypoints(double *pose, double *keypoints3d,
                               int number_of_keypoints, double fx, double fy,
                               double cx, double cy, double *keypoints2d)
    cdef double c_get_intensity_diff(unsigned char *image1,
        unsigned char *image2,
        unsigned int image_width,
        unsigned int image_height,
        double *keypoint1,
        double *keypoint2,
        double errorval) nogil
    cdef double c_get_total_intensity_diff(unsigned char *image1,
        unsigned char *image2,
        unsigned int image_width,
        unsigned int image_height,
        double *keypoint1,
        double *keypoint2,
        unsigned int n_keypoints,
        double *diff) nogil

    cdef void c_refine_cloud(double fx,
        double fy,
        double cx,
        double cy,
        double *pose,
        double *keypoints3d,
        double *keypoints2d,
        int number_of_keypoints) nogil

cdef extern from "<array>" namespace "std" nogil:
  cdef cppclass array3 "std::array<float, 3>":
    array3() except+
    float& operator[](size_t)
  cdef cppclass array2 "std::array<float, 2>":
    array2() except+
    float& operator[](size_t)


cdef extern from "depth_calculator.hpp":
    cdef cppclass CDepthCalculator "DepthCalculator":
        CDepthCalculator(float baseline,
                float fx, float fy, float cx, float cy,
                int window_size, int search_x, int search_y, int margin)
        void calculate_depth(Mat &left, Mat &right, int split_count,
                             vector[array2] &keypoints2d,
                             vector[array3] &keypoints3d,
                             vector[unsigned int] &err) nogil

# Declares OpenCV's cv::Mat class
cdef extern from "opencv2/core/core.hpp":
    cdef cppclass Mat:
        Mat (int rows, int cols, int type, void *data)
        pass



def rotation_matrix(double[:] angle):
    cdef double[:,:] rot_mat = np.zeros((3,3))

    c_rotation_matrix(&angle[0], &rot_mat[0,0])
    return np.asarray(rot_mat)

def transform_keypoints(double[:] pose, double[:,:] keypoints3d,
                        double fx, double fy, double cx, double cy):
    cdef double[:,:] keypoints2d = np.zeros((3, keypoints3d.shape[1]))
    c_transform_keypoints(&pose[0], &keypoints3d[0,0], keypoints3d.shape[1],
                          fx, fy, cx, cy, &keypoints2d[0,0])
    return np.asarray(keypoints2d)[0:2,:]

def get_intensity_diff(unsigned char[:, :] image1, unsigned char[:, :] image2, double[:] keypoint1, double[:] keypoint2, double errorval):
    return c_get_intensity_diff(&image1[0,0],
                                &image2[0,0],
                                image1.shape[1],
                                image1.shape[0],
                                &keypoint1[0],
                                &keypoint2[0],
                                errorval)

def get_total_intensity_diff(unsigned char[:,:] image1, unsigned char[:,:] image2, double[:,:] keypoints1, double[:,:] keypoints2):
    cdef double[:] diff = np.empty((keypoints2.shape[1]), dtype=np.float64)

    c_get_total_intensity_diff(&image1[0,0],
                               &image2[0,0],
                               image1.shape[1],
                               image1.shape[0],
                               &keypoints1[0,0],
                               &keypoints2[0,0],
                               keypoints1.shape[1],
                               &diff[0])

    return np.asarray(diff)

def refine_cloud(double fx, double fy, double cx, double cy, double[:] pose, double[:,:] keypoints3d, double[:,:] keypoints2d):
    c_refine_cloud(fx, fy, cx, cy, &pose[0], &keypoints3d[0,0], &keypoints2d[0,0], keypoints3d.shape[1])

    return np.asarray(keypoints3d)

cdef class DepthCalculator:
    cdef CDepthCalculator *depth_calculator

    def __cinit__(self, baseline, fx, fy, cx, cy, window_size, search_x, search_y, margin):
        self.depth_calculator = new CDepthCalculator(baseline, fx, fy, cx, cy,
                                                window_size, search_x, search_y,
                                                margin)

    def __dealloc__(self):
        del self.depth_calculator

    def calculate_depth(self, unsigned char[:,:] left, unsigned char[:,:] right,
                        unsigned int split_count):
        cdef vector[array2] keypoints2d
        cdef vector[array3] keypoints3d
        cdef vector[unsigned int] err

        cdef Mat *_left = new Mat(left.shape[0], left.shape[1], 0, &left[0,0])
        cdef Mat *_right = new Mat(right.shape[0], right.shape[1], 0, &right[0,0])
        self.depth_calculator.calculate_depth(_left[0], _right[0], split_count,
                                              keypoints2d, keypoints3d, err)
        del _left
        del _right

        _keypoints2d = np.empty((2, keypoints2d.size()))
        for i in range(0, keypoints2d.size()):
            _keypoints2d[0,i] = keypoints2d[i][0]
            _keypoints2d[1,i] = keypoints2d[i][1]

        _keypoints3d = np.empty((3, keypoints3d.size()))
        for i in range(0, keypoints3d.size()):
            _keypoints3d[0,i] = keypoints3d[i][0]
            _keypoints3d[1,i] = keypoints3d[i][1]
            _keypoints3d[2,i] = keypoints3d[i][2]
        

        _err = np.empty((1, err.size()))
        for i in range(0, err.size()):
            _err[0,i] = err[i]

        return _keypoints2d, _keypoints3d, err
