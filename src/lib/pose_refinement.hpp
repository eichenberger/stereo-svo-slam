#ifndef POSE_REFINEMENT_H
#define POSE_REFINEMENT_H

#include <vector>

#include <opencv2/opencv.hpp>

#include "stereo_slam_types.hpp"

class PoseRefinerCallback;

class PoseRefiner
{
public:
    PoseRefiner(const CameraSettings &camera_settings);

    float refine_pose(const std::vector<KeyPoint2d> &keypoints2d,
            const std::vector<KeyPoint3d> &keypoints3d,
            const Pose &estimated_pose,
            Pose &refined_pose);

private:
    const CameraSettings &camera_settings;
    cv::Ptr<PoseRefinerCallback> solver_callback;
};

#endif
