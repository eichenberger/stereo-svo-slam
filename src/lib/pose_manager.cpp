#include "pose_manager.hpp"

using namespace cv;

PoseManager::PoseManager()
{
}

void PoseManager::set_pose(Pose &pose)
{
    this->pose = pose;
    angles = Vec3f(&this->pose.pitch);
    translation = Vec3f(&this->pose.x);

    Rodrigues(angles, rot_mat);
    Rodrigues(-angles, inv_rot_mat);
}


Matx33f PoseManager::get_inv_rotation_matrix() const
{
    return inv_rot_mat;
}

Matx33f PoseManager::get_rotation_matrix() const
{
    return rot_mat;
}

Vec3f PoseManager::get_translation() const
{
    return translation;
}

Pose PoseManager::get_pose() const
{
    return pose;
}

Vec3f PoseManager::get_angles() const
{
    return angles;
}


Vec6f PoseManager::get_vector() const
{
    return Vec6f(&pose.x);
}


void PoseManager::set_vector(cv::Vec6f &pose)
{
    Pose *_pose = (Pose*)&pose[0];
    set_pose(*_pose);
}


std::ostream& operator<<(std::ostream& os, const PoseManager& pose)
{
    const Pose &_pose = pose.pose;
    os << _pose.x << "," << _pose.y << "," << _pose.z << "," << _pose.pitch <<
        "," << _pose.yaw << "," << _pose.roll;
    return os;
}
