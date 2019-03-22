import numpy as np

from libc.math cimport fabs
from cython.parallel import parallel, prange
from cython import boundscheck, wraparound

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
cdef double c_get_intensity_diff(unsigned char[:, :] image1, unsigned char[:, :] image2, double[:] keypoint1, double[:] keypoint2, double errorval=0) nogil:
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

def get_intensity_diff(unsigned char[:, :] image1, unsigned char[:, :] image2, double[:] keypoint1, double[:] keypoint2, double errorval=0):
    return c_get_intensity_diff(image1, image2, keypoint1, keypoint2, errorval)

@boundscheck(False)
@wraparound(False)
def get_total_intensity_diff(unsigned char[:,:] image1, unsigned char[:,:] image2, double[:,:] keypoints1, double[:,:] keypoints2):
    cdef double[:] diff = np.zeros((keypoints2.shape[1]), dtype=np.float64)
    cdef int i
    with nogil, parallel():
        for i in prange(keypoints2.shape[1]):
            diff[i] = c_get_intensity_diff(image1, image2, keypoints1[:,i], keypoints2[:,i])

    return np.asarray(diff)

