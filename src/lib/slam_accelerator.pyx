import numpy as np

from cython.parallel import parallel, prange
from cython import boundscheck, wraparound
from libc.math cimport fabs
from libcpp.vector cimport vector
from libc.math cimport sin, cos

ctypedef double (*opt_fun)(const double*)

cdef extern from "depth_adjustment_helper.hpp":
    cdef cppclass GradientDescent:
        GradientDescent()
        void setFunction(opt_fun fun, int dims)
        void minimize(vector[double] x0)
        void setInitStep(vector[double] step)

@boundscheck(False)
@wraparound(False)
cdef double c_get_sub_pixel(unsigned char[:,:] image, double x, double y) nogil:
    cdef int x_floor = int(x)
    cdef int y_floor = int(y)

    cdef int x_ceil = int(x+1)
    cdef int y_ceil = int(y+1)

    cdef double x_floor_prob =  1.0 - (x - x_floor)
    cdef double y_floor_prob =  1.0 - (y - y_floor)

    cdef double x_ceil_prob =  1.0 - (x_ceil - x)
    cdef double y_ceil_prob =  1.0 - (y_ceil - y)

    cdef double sub_pixel_val = 0.0

    sub_pixel_val = x_floor_prob*y_floor_prob*image[y_floor, x_floor]
    sub_pixel_val += x_floor_prob*y_ceil_prob*image[y_ceil, x_floor]
    sub_pixel_val += x_ceil_prob*y_floor_prob*image[y_floor, x_ceil]
    sub_pixel_val += x_ceil_prob*y_ceil_prob*image[y_ceil, x_ceil]

    return sub_pixel_val

@boundscheck(False)
@wraparound(False)
cdef double c_get_intensity_diff(unsigned char[:, :] image1, unsigned char[:, :] image2, double[:] keypoint1, double[:] keypoint2, double errorval) nogil:
    cdef int x1 = int(keypoint1[0])
    cdef int y1 = int(keypoint1[1])
    cdef double x2 = keypoint2[0]
    cdef double y2 = keypoint2[1]

    # If keypoint is outside of second image we ignore it
    if x2 - 2 < 0 or x2 + 2 > image2.shape[1] or \
            y2 - 2 < 0 or y2 + 2 > image2.shape[0]:
        return errorval

    cdef double diff = 0

    diff =  fabs(image1[y1, x1] - c_get_sub_pixel(image2, x2, y2))
    diff += fabs(image1[y1, x1-1] - c_get_sub_pixel(image2, x2-1, y2))
    diff += fabs(image1[y1-1, x1] - c_get_sub_pixel(image2, x2, y2-1))
    diff += fabs(image1[y1, x1+1] - c_get_sub_pixel(image2, x2+1, y2))
    diff += fabs(image1[y1+1, x1] - c_get_sub_pixel(image2, x2, y2+1))

    return diff

def get_intensity_diff(unsigned char[:, :] image1, unsigned char[:, :] image2, double[:] keypoint1, double[:] keypoint2, double errorval):
    return c_get_intensity_diff(image1, image2, keypoint1, keypoint2, errorval)

@boundscheck(False)
@wraparound(False)
def get_total_intensity_diff(unsigned char[:,:] image1, unsigned char[:,:] image2, double[:,:] keypoints1, double[:,:] keypoints2):
    cdef double[:] diff = np.zeros((keypoints2.shape[1]), dtype=np.float64)
    cdef int i
    with nogil, parallel():
        for i in prange(keypoints2.shape[1]):
            diff[i] = c_get_intensity_diff(image1, image2, keypoints1[:,i], keypoints2[:,i], 0)

    return np.asarray(diff)

cdef double[:,:] _rot_mat_x(double angle):
    return np.array([[1, 0, 0],
                     [0, cos(angle), -sin(angle)],
                     [0, sin(angle), cos(angle)]])

cdef double[:,:]_rot_mat_y(double angle):
    return np.array([[cos(angle), 0, sin(angle)],
                     [0, 1, 0],
                     [-sin(angle), 0, cos(angle)]])

cdef double[:,:] _rot_mat_z(double angle):
    return np.array([[cos(angle), -sin(angle), 0],
                     [sin(angle), cos(angle), 0],
                     [0, 0, 1]])

cdef double[:,:] matmul(double[:,:] a, double[:,:] b):
    cdef int i, j, k
    cdef double s
    cdef double[:,:] out = np.array((a.shape[0], a.shape[1]))

    # Take each row of A
    for i in range (0 , a.shape [0]):
        # And multiply by every column of B
        for j in range ( b.shape [1]):
            s = 0
            for k in range ( a.shape [1]):
                s += a[i , k] * b[k, j]
                out [i, j] = s

    return out

cdef double[:,:] rotation_matrix(double[:] angle):
    cdef double[:,:] rot_x = _rot_mat_x(angle[0])
    cdef double[:,:] rot_y = _rot_mat_y(angle[1])
    cdef double[:,:] rot_z = _rot_mat_z(angle[2])

    cdef double[:,:] rot = matmul(rot_x, rot_y)
    rot = matmul(rot, rot_z)

    return rot

cdef double[:,:] transform_keypoints(double[:] pose, double[:,:] keypoints3d, double fx, double fy, double cx, double cy):
    cdef double[:,:] extrinsic = np.array((3,4))

    extrinsic[:,0:3] = rotation_matrix(pose[3:])
    extrinsic[:,3] = np.transpose(pose[0:3])

    cdef double[:,:] kps3d = np.ones((4, keypoints3d.shape[1]))
    kps3d[0:3, :] = keypoints3d

    cdef double[:,:] kps2d = np.matmul(extrinsic, kps3d)
    kps2d = (kps2d*[[fx],[fy],[1]])/kps2d[2,:] + [[cx],[cy],[0]]
    return kps2d

cdef double[:,:] m_kps3d
cdef unsigned char[:,:] m_image
cdef unsigned char[:,:] m_new_image
cdef double[:,:] m_keypoints2d
cdef double[:] m_pose
cdef double m_fx
cdef double m_fy
cdef double m_cx
cdef double m_cy
cdef int i

cdef double optimize_depth(const double* x):
    m_kps3d[2] = x[0]
    print("optimize_depth")
    cdef double[:,:] kp2d = transform_keypoints(m_pose, m_kps3d, m_fx, m_fy, m_cx, m_cy)

    cdef double diff = c_get_intensity_diff(m_image,
                              m_new_image,
                              m_keypoints2d[:, i],
                              kp2d[0:2,0],
                              100000000000.0)

    return diff

def adjust_depth(unsigned char[:,:] image, unsigned char[:,:] new_image,
                 double[:,:] keypoints3d, double[:,:] keypoints2d,
                 double[:] pose, double fx, double fy, double cx, double cy):
    m_kps3d = keypoints3d.copy()
    m_image = image
    m_new_image = new_image
    m_keypoints2d = keypoints2d
    m_pose = pose
    m_fx = fx
    m_fy = fy
    m_cx = cx
    m_cy = cy

    print("adjust_depth")

    cdef GradientDescent gd
    gd.setFunction(optimize_depth, 1)
    cdef vector[double] steps = [1]
    gd.setInitStep(steps)
    cdef vector[double] x0 = [0.0]

    costs = np.zeros((1, m_kps3d.shape[1]))
    for i in range(0, m_kps3d.shape[1]):
        keypoint = m_kps3d[:, i:i+1]

        cost = gd.minimize(x0)
        m_kps3d[2, i] = x0[0]
        costs[0, i] = cost

    prop = 1.0-(costs/10.0)
    for i in range(0, prop.shape[1]):
        prop[0, i] = max(prop[0, i],0)

    return m_kps3d, prop

