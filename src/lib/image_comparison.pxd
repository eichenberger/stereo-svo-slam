from libcpp.vector cimport vector
from stereo_slam_types cimport Mat, KeyPoint2d

cdef extern from "image_comparison.hpp":
    cdef void get_total_intensity_diff(const Mat &image1, const Mat &image2,
                                       const vector[KeyPoint2d] &keypoints1,
                                       const vector[KeyPoint2d] &keypoints2,
                                       unsigned int patchSize,
                                       vector[float] &diff) nogil

