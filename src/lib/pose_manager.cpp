#include "pose_manager.hpp"

using namespace cv;

PoseManager::PoseManager()
{
}

void PoseManager::set_pose(Pose &pose)
{
    this->pose = pose;
    angles = Vec3f(&this->pose.rx);
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

Vec3f PoseManager::get_robot_angles() const
{
    Matx33f rot_mat_x , rot_mat_y, rot_mat_z;
    Rodrigues(Vec3f(angles[0],0,0), rot_mat_x);
    Rodrigues(Vec3f(0,angles[1],0), rot_mat_y);
    Rodrigues(Vec3f(0,0,angles[2]), rot_mat_z);

    Vec3f robot_angles;
    // The normal order would be rot_mat_z*rot_mat_y*rot_mat_x.
    // However, for e.g. opengl or blender the order is vice versa.
    // This is a hacky convertion.
    Rodrigues(rot_mat_z*(rot_mat_x*rot_mat_y), robot_angles);

    return robot_angles;
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
    os << _pose.x << "," << _pose.y << "," << _pose.z << "," << _pose.rx <<
        "," << _pose.ry << "," << _pose.rz;
    return os;
}
