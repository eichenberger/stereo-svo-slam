import numpy as np

from cython.parallel import parallel, prange
from cython import boundscheck, wraparound
from libc.math cimport fabs
from libcpp.vector cimport vector
from libc.math cimport sin, cos

ctypedef double (*opt_fun)(const double*)

cdef extern from "depth_adjustment_helper.hpp":
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
    return np.asarray(keypoints2d)

#@boundscheck(False)
#@wraparound(False)
#cdef double c_get_sub_pixel(unsigned char[:,:] image, double x, double y) nogil:
#    cdef int x_floor = int(x)
#    cdef int y_floor = int(y)
#
#    cdef int x_ceil = int(x+1)
#    cdef int y_ceil = int(y+1)
#
#    cdef double x_floor_prob =  1.0 - (x - x_floor)
#    cdef double y_floor_prob =  1.0 - (y - y_floor)
#
#    cdef double x_ceil_prob =  1.0 - (x_ceil - x)
#    cdef double y_ceil_prob =  1.0 - (y_ceil - y)
#
#    cdef double sub_pixel_val = 0.0
#
#    sub_pixel_val = x_floor_prob*y_floor_prob*image[y_floor, x_floor]
#    sub_pixel_val += x_floor_prob*y_ceil_prob*image[y_ceil, x_floor]
#    sub_pixel_val += x_ceil_prob*y_floor_prob*image[y_floor, x_ceil]
#    sub_pixel_val += x_ceil_prob*y_ceil_prob*image[y_ceil, x_ceil]
#
#    return sub_pixel_val
#
#@boundscheck(False)
#@wraparound(False)
#cdef double c_get_intensity_diff(unsigned char[:, :] image1, unsigned char[:, :] image2, double[:] keypoint1, double[:] keypoint2, double errorval) nogil:
#    cdef int x1 = int(keypoint1[0])
#    cdef int y1 = int(keypoint1[1])
#    cdef double x2 = keypoint2[0]
#    cdef double y2 = keypoint2[1]
#
#    # If keypoint is outside of second image we ignore it
#    if x2 - 2 < 0 or x2 + 2 > image2.shape[1] or \
#            y2 - 2 < 0 or y2 + 2 > image2.shape[0]:
#        return errorval
#
#    cdef double diff = 0
#
#    diff =  fabs(image1[y1, x1] - c_get_sub_pixel(image2, x2, y2))
#    diff += fabs(image1[y1, x1-1] - c_get_sub_pixel(image2, x2-1, y2))
#    diff += fabs(image1[y1-1, x1] - c_get_sub_pixel(image2, x2, y2-1))
#    diff += fabs(image1[y1, x1+1] - c_get_sub_pixel(image2, x2+1, y2))
#    diff += fabs(image1[y1+1, x1] - c_get_sub_pixel(image2, x2, y2+1))
#
#    return diff

def get_intensity_diff(unsigned char[:, :] image1, unsigned char[:, :] image2, double[:] keypoint1, double[:] keypoint2, double errorval):
    return c_get_intensity_diff(&image1[0,0], 
                                &image2[0,0], 
                                image1.shape[1], 
                                image1.shape[0], 
                                &keypoint1[0], 
                                &keypoint2[0], 
                                errorval)

@boundscheck(False)
@wraparound(False)
def get_total_intensity_diff(unsigned char[:,:] image1, unsigned char[:,:] image2, double[:,:] keypoints1, double[:,:] keypoints2):
    cdef double[:] diff = np.zeros((keypoints2.shape[1]), dtype=np.float64)
    cdef int i
    with nogil, parallel():
        for i in prange(keypoints2.shape[1]):
            diff[i] = c_get_intensity_diff(&image1[0,0], 
                                           &image2[0,0], 
                                           image1.shape[1], 
                                           image1.shape[0], 
                                           [keypoints1[:,i][0], keypoints1[:,i][1]], 
                                           [keypoints2[:,i][0], keypoints2[:,i][1]], 
                                           0)

    return np.asarray(diff)

#cdef double[:,:] transform_keypoints(double[:] pose, double[:,:] keypoints3d, double fx, double fy, double cx, double cy):
#    cdef double[:,:] extrinsic = np.zeros((3,4))
#
#    extrinsic[:,0:3] = rotation_matrix(pose[3:])
#    extrinsic[0,3] = pose[0]
#    extrinsic[1,3] = pose[1]
#    extrinsic[2,3] = pose[2]
#
#    cdef double[:,:] kps3d = np.ones((4, keypoints3d.shape[1]))
#    kps3d[0:3, :] = keypoints3d
#
#    cdef double[:,:] kps2d = np.matmul(extrinsic, kps3d)
#    kps2d = np.add(np.divide(np.multiply(kps2d,[[fx],[fy],[1]]),kps2d[2,:]),[[cx],[cy],[0]])
#    return kps2d
#
#cdef double[:,:] m_kps3d
#cdef unsigned char[:,:] m_image
#cdef unsigned char[:,:] m_new_image
#cdef double[:,:] m_keypoints2d
#cdef double[:] m_pose
#cdef double m_fx
#cdef double m_fy
#cdef double m_cx
#cdef double m_cy
#cdef int i
#
#cdef double optimize_depth(const double* x):
#    global m_image
#    global m_new_image
#    global m_keypoints2d
#    global m_pose
#    global m_fx
#    global m_fy
#    global m_cx
#    global m_cy
#    global i
#
#    cdef double[:,:] kps3d = np.zeros((1,3))
#    kps3d[0,0] = x[0]
#    kps3d[0,1] = x[1]
#    kps3d[0,2] = x[2]
#    cdef double[:,:] kp2d = transform_keypoints(m_pose, kps3d, m_fx, m_fy, m_cx, m_cy)
#
#    cdef double diff = c_get_intensity_diff(m_image,
#                              m_new_image,
#                              m_keypoints2d[:, i],
#                              kp2d[0:2,0],
#                              100000000000.0)
#
#    print(f"Diff {i}: {diff}")
#
#    return diff
#
#def adjust_depth(unsigned char[:,:] image, unsigned char[:,:] new_image,
#                 double[:,:] keypoints3d, double[:,:] keypoints2d,
#                 double[:] pose, double fx, double fy, double cx, double cy):
#    global m_image
#    global m_new_image
#    global m_keypoints2d
#    global m_pose
#    global m_fx
#    global m_fy
#    global m_cx
#    global m_cy
#    global i
#    kps3d = keypoints3d.copy()
#    m_image = image
#    m_new_image = new_image
#    m_keypoints2d = keypoints2d
#    m_pose = pose
#    m_fx = fx
#    m_fy = fy
#    m_cx = cx
#    m_cy = cy
#
#    print("adjust_depth: " + str(kps3d.shape[0]))
#
#    cdef GradientDescent gd
#    gd.setFunction(optimize_depth, 3)
#    cdef vector[double] steps = [0, 0, 0.001]
#    gd.setInitStep(steps)
#
#    cdef vector[double] x0
#    costs = np.zeros((1, kps3d.shape[1]))
#    for i in range(0, kps3d.shape[1]):
#        keypoint = kps3d[:, i:i+1]
#        x0 = [keypoint[0,0], keypoint[1,0], keypoint[2,0]]
#        cost = gd.minimize(x0)
#
#        kps3d[2, i] = x0[2]
#        costs[0, i] = cost
#
#    prop = 1.0-(costs/10.0)
#    for i in range(0, prop.shape[1]):
#        prop[0, i] = max(prop[0, i],0)
#
#    return kps3d, prop
#
