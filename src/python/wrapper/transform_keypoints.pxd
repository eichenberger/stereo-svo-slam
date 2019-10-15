from libcpp.vector cimport vector

cimport stereo_slam_types

cdef extern from "transform_keypoints.hpp":
    cdef void project_keypoints(const stereo_slam_types.Pose &pose,
        const vector[stereo_slam_types.KeyPoint3d] &input,
        const stereo_slam_types.CameraSettings &camera_settings,
        vector[stereo_slam_types.KeyPoint2d] &out)

#    cdef void transform_keypoints_inverse(
#        const stereo_slam_types.Pose &pose,
#        const vector[stereo_slam_types.KeyPoint3d] &input,
#        vector[stereo_slam_types.KeyPoint3d] &output)
#
