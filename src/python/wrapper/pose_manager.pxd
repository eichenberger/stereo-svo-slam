from stereo_slam_types cimport Pose
from libcpp.vector cimport vector

cdef extern from "pose_manager.hpp":
    cdef cppclass PoseManager:
        PoseManager()
        Pose get_pose()
        void set_pose(Pose pose)
