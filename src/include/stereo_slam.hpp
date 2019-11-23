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

    void new_image(const cv::Mat &left, const cv::Mat &right);

    void get_keyframe(KeyFrame &keyframe);
    void get_keyframes(std::vector<KeyFrame> &keyframes);
    void get_frame(Frame &frame);
    void get_trajectory(std::vector<Pose> &trajectory);

private:
    void new_keyframe();

    void estimate_pose(Frame *previous_frame);

    const CameraSettings &camera_settings;
    KeyFrameManager keyframe_manager;
    KeyFrame* keyframe;
    cv::Ptr<Frame> frame;
    std::vector<Pose> trajectory;
};

#endif
