#ifndef POSE_MANAGER_HPP
#define POSE_MANAGER_HPP

#include <iostream>
#include <opencv2/opencv.hpp>

struct Pose {
    float x;
    float y;
    float z;
    float pitch;    // around x
    float yaw;      // around y
    float roll;     // around z
};

class PoseManager
{
public:
    PoseManager();

    void set_pose(Pose &pose);
    void set_vector(cv::Vec6f &pose);
    cv::Matx33f get_rotation_matrix() const;
    cv::Matx33f get_inv_rotation_matrix() const;
    cv::Vec3f get_translation() const;
    cv::Vec3f get_angles() const;
    Pose get_pose() const;
    cv::Vec6f get_vector() const;
    friend std::ostream& operator<<(std::ostream& os, const PoseManager& pose);

private:
    Pose pose;
    cv::Matx33f rot_mat;
    cv::Matx33f inv_rot_mat;
    cv::Vec3f angles;
    cv::Vec3f translation;

};

std::ostream& operator<<(std::ostream& os, const PoseManager& pose);

#endif
