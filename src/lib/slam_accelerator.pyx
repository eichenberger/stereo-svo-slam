import numpy as np

from cython.parallel import parallel, prange
from cython import boundscheck, wraparound
from libc.math cimport fabs
from libcpp.vector cimport vector
from libc.math cimport sin, cos

ctypedef double (*opt_fun)(const double*)

cdef extern from "slam_accelerator_helper.hpp":
    cdef cppclass c_DepthAdjuster:
        c_DepthAdjuster()
        void adjust_depth(double *new_kps, double* cost)
        void set_image(unsigned char *image, int rows, int cols)
        void set_new_image(unsigned char* new_image, int rows, int cols)
        void set_keypoints3d(double *keypoint3d, int size)
        void set_keypoints2d(double *keypoints2d, int size)
        void set_pose(double *pose)
        void set_fx(double fx)
        void set_fy(double fy)
        void set_cx(double cx)
        void set_cy(double cy)


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

cdef class DepthAdjuster:

    cdef c_DepthAdjuster *_impl
    cdef unsigned int count

    def __init__(self):
        self._impl = new c_DepthAdjuster()
        self.count = 0

    def __dealloc__(self):
        del self._impl

    def adjust_depth(self):
        cdef double[:] new_z
        cdef double[:] cost
        new_z = np.empty((self.count))
        cost = np.empty((self.count))
        self._impl.adjust_depth(&new_z[0], &cost[0])
        return np.asarray(new_z), np.asarray(cost)

    def set_image(self,unsigned char[:,:] image):
        self._impl.set_image(&image[0,0], image.shape[0], image.shape[1])

    def set_new_image(self, unsigned char[:,:] image):
        self._impl.set_new_image(&image[0,0], image.shape[0], image.shape[1])

    def set_keypoints3d(self, double[:,:] kps):
        self._impl.set_keypoints3d(&kps[0,0], kps.shape[1])
        self.count = kps.shape[1]

    def set_keypoints2d(self, double[:,:] kps):
        self._impl.set_keypoints2d(&kps[0,0], kps.shape[1])

    def set_pose(self, double[:] pose):
        self._impl.set_pose(&pose[0])

    def set_fx(self, double fx):
        self._impl.set_fx(fx)

    def set_fy(self, double fy):
        self._impl.set_fy(fy)

    def set_cx(self, double cx):
        self._impl.set_cx(cx)

    def set_cy(self, double cy):
        self._impl.set_cy(cy)


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
