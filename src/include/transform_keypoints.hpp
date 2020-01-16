#ifndef _TRANSFORM_KEYPOINTS_H
#define _TRANSFORM_KEYPOINTS_H

#include <vector>
#include <opencv2/opencv.hpp>

#include "stereo_slam_types.hpp"

/*!
 * \brief Project keypoints
 *
 * @param[in] pose The camera pose
 * @param[in] in The global 3D points
 * @param[in] camera_settings The camera settings to use
 * @param[out] out The projected 2D points
 */
void project_keypoints(const PoseManager &pose,
        const std::vector<KeyPoint3d> &in, const CameraSettings &camera_settings,
        std::vector<KeyPoint2d> &out);

#endif
