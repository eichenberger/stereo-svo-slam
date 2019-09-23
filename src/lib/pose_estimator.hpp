#ifndef POSE_ESTIMATOR_H
#define POSE_ESTIMATOR_H

#include <vector>

#include <opencv2/opencv.hpp>

#include "stereo_slam_types.hpp"

class PoseEstimatorCallback;

class PoseEstimator
{
public:
    PoseEstimator(const StereoImage &current_stereo_image,
                const StereoImage &previous_stereo_image,
                const KeyPoints &previous_keypoints,
                const CameraSettings &camera_settings);



    float estimate_pose(const Pose &pose_guess, Pose &estimaged_pose);

private:
    cv::Ptr<PoseEstimatorCallback> solver_callback;
};

#endif
