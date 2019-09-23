from libcpp.vector cimport vector
from stereo_slam_types cimport StereoImage, KeyPoints, CameraSettings, Pose

cdef extern from "pose_estimator.hpp":
    cdef cppclass PoseEstimator:
        PoseEstimator(const StereoImage current_stereo_image,
                    const StereoImage previous_stereo_image,
                    const KeyPoints previous_keypoints,
                    const CameraSettings camera_settings)

        float estimate_pose(const Pose &pose_guess, Pose &estimaged_pose)
