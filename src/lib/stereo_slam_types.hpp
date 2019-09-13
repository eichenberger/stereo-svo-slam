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
    int grid_height;
    int grid_width;
    int search_x;
    int search_y;
    int window_size;
};

struct StereoImage {
    cv::Mat left;
    cv::Mat right;
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

struct KeyPointInformation {
    float score;
    int level;
    enum KeyPointType type;
    float confidence;
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
    float roll; // around x
    float pitch; //around y
    float yaw; //around z
};

struct Color {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

struct KeyFrame {
    struct Pose pose;
    std::vector<struct StereoImage> stereo_images;
    struct KeyPoints kps;
    std::vector<struct Color> colors;
};

#endif
