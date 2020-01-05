#ifndef POSE_MANAGER_HPP
#define POSE_MANAGER_HPP

#include <iostream>
#include <opencv2/opencv.hpp>

/*!
 * \brief Structure representing a global Pose in 3D
 *
 * We use the same right-handed-coordinate system as OpenCV.\n
 * positive x: move right\n
 * positive y: move down\n
 * positive z: move forward\n
 * This is not the same as an extrinsic camera matrix. We first rotate and then
 * move. Also it is inverse to the extrinsic camera matrix. We need to do
 * -position and then rotate with -rotation. Also note that first we rotate
 * around z axis, then around y and finally around x. This may be different
 * for describtion in Qt3D or robotoics. There it's often rotate around x, y,
 * z. PoseManager can help in this case by providing a wrapper.
 */
struct Pose {
    float x;    //!< X position in room
    float y;    //!< Y position in room
    float z;    //!< Z position in room
    float rx;   //!< Rotation around x axis
    float ry;   //!< Rotation around y axis
    float rz;   //!< Rotation around z axis
};

/*!
 * \brief Class for managing the pose
 *
 * Also see Pose. The pose manager makes sure that we calculate several
 * matrices (e.g. rotation) only once. This should help to speed up the
 * calculations. It also provides other interfaces to set or get poses and
 * allows to receive a robot angels instead of OpenCV angles (x,y,z instead of
 * z,y,x).
 */

class PoseManager
{
public:
    PoseManager();

    void set_pose(Pose &pose); //!< Set the pose via Pose structure
    void set_vector(cv::Vec6f &pose);   //!< Set the pose via OpenCV Vector
    cv::Matx33f get_rotation_matrix() const;    //!< Receive the rotation matrix
    cv::Matx33f get_inv_rotation_matrix() const;    //!< Receive inverse rotation matrix
    cv::Vec3f get_translation() const;  //!< Receive translation (position)
    cv::Vec3f get_angles() const;   //!< Receive rotation/angles as OpenCV vector
    cv::Vec3f get_robot_angles() const; //!< Receive rotation/angles applied in inverse order (rx*ry*rz instead rz*ry*rx)
    Pose get_pose() const;  //!< Receive the pose as Pose
    cv::Vec6f get_vector() const;   //!< Receive the Pose as vector
    friend std::ostream& operator<<(std::ostream& os, const PoseManager& pose); //!< allow printing a Pose

private:
    Pose pose;
    cv::Matx33f rot_mat;
    cv::Matx33f inv_rot_mat;
    cv::Vec3f angles;
    cv::Vec3f translation;

};

std::ostream& operator<<(std::ostream& os, const PoseManager& pose);

#endif
