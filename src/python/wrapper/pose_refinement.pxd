from libcpp.vector cimport vector
from stereo_slam_types cimport KeyPoints, CameraSettings, Pose

cdef extern from "pose_refinement.hpp":
    cdef cppclass PoseRefiner:
        PoseRefiner(const CameraSettings &camera_settings)

        float refine_pose(KeyPoints keypoints,
                const Pose &estimated_pose,
                Pose &refined_pose);

