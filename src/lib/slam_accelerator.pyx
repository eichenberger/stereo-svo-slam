import numpy as np

from cython.parallel import parallel, prange
from cython import boundscheck, wraparound
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
