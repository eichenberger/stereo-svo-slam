#ifndef STEREO_SLAM_TYPES_H
#define STEREO_SLAM_TYPES_H

#include <vector>
#include <opencv2/opencv.hpp>

struct CameraSettings {
    float baseline;
    float fx;
    float fy;
    float cx;
    float cy;
    float k1;
    float k2;
    float k3;
    float p1;
    float p2;
    int grid_height;
    int grid_width;
    int search_x;
    int search_y;
    int window_size;
    int window_size_opt_flow;
    int window_size_depth_calculator;
    int max_pyramid_levels;
    int min_pyramid_level_pose_estimation;

    int image_width;
    int image_height;

    // Windows for confidence calculation
    float dist_window_k0;   // min distance _/-\_
    float dist_window_k1;   // min distance confidence 1.0
    float dist_window_k2;   // max distance confidence 1.0
    float dist_window_k3;   // max distance

    float cost_k0;  // max cost confidence 1.0 -\_
    float cost_k1;  // max cost
};

struct StereoImage {
    std::vector<cv::Mat> left;
    std::vector<cv::Mat> right;
};

enum KeyPointType {
    KP_FAST,
    KP_EDGELET
};

struct KeyPoint2d {
    float x;
    float y;
};

struct KeyPoint3d {
    float x;
    float y;
    float z;
};

struct Color {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

struct KeyPointInformation {
    float score;
    int level;
    enum KeyPointType type;
    float confidence;
    uint64_t keyframe_id;
    size_t keypoint_index;
    Color color;
    bool ignore_during_refinement;
    bool ignore_completely;
    int outlier_count;
    int inlier_count;
    cv::KalmanFilter kf;
};

// KeyPoints, each entry has the same index. We try to avaoid mixing
// information to speed up calculation (e.g. kp2d<->kp3d mixing)
struct KeyPoints {
    std::vector<KeyPoint2d> kps2d;
    std::vector<KeyPoint3d> kps3d;
    std::vector<KeyPointInformation> info;
};

struct Pose {
    float x;
    float y;
    float z;
    float pitch;    // around x
    float yaw;      // around y
    float roll;     // around z
};

struct Frame {
    uint64_t id;
    struct Pose pose;
    struct StereoImage stereo_image;
    struct KeyPoints kps;
};

struct KeyFrame : Frame{
};


#endif
