#ifndef STEREO_SLAM_HPP
#define STEREO_SLAM_HPP

#include <vector>

#include <opencv2/opencv.hpp>

#include "stereo_slam_types.hpp"
#include "keyframe_manager.hpp"

class StereoSlam
{
public:
    StereoSlam(const CameraSettings &camera_settings);

    void new_image(const cv::Mat &left, const cv::Mat &right, const float dt);

    void get_keyframe(KeyFrame &keyframe);
    void get_keyframes(std::vector<KeyFrame> &keyframes);
    bool get_frame(Frame &frame);
    void get_trajectory(std::vector<Pose> &trajectory);
    Pose update_pose(const Pose &pose, const cv::Vec6f &speed,
        const cv::Vec6f &pose_variance, const cv::Vec6f &speed_variance,
        double current_time);
    double get_current_time();

private:
    void new_keyframe();

    void estimate_pose(Frame *previous_frame);
    void remove_outliers(Frame *frame);

    const CameraSettings &camera_settings;
    KeyFrameManager keyframe_manager;
    KeyFrame* keyframe;
    cv::Ptr<Frame> frame;
    std::vector<Pose> trajectory;
    cv::Vec6f motion;

    cv::KalmanFilter kf;
    cv::TickMeter time_measure;
};

#endif
