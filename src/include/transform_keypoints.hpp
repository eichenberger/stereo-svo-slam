#ifndef _TRANSFORM_KEYPOINTS_H
#define _TRANSFORM_KEYPOINTS_H

#include <vector>
#include <opencv2/opencv.hpp>

#include "stereo_slam_types.hpp"

void project_keypoints(const PoseManager &pose,
        const std::vector<KeyPoint3d> &in, const CameraSettings &camera_settings,
        std::vector<KeyPoint2d> &out);

//void transform_keypoints_inverse(const struct Pose &pose,
//        const std::vector<KeyPoint3d> &in, std::vector<KeyPoint3d> &out);
#endif
