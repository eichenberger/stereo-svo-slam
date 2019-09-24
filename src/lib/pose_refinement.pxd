from libcpp.vector cimport vector
from stereo_slam_types cimport KeyPoint2d, KeyPoint3d, CameraSettings, Pose

cdef extern from "pose_refinement.hpp":
    cdef cppclass PoseRefiner:
        PoseRefiner(const CameraSettings &camera_settings)

        float refine_pose(const vector[KeyPoint2d] keypoints2d,
                const vector[KeyPoint3d] keypoints3d,
                const Pose &estimated_pose,
                Pose &refined_pose);

