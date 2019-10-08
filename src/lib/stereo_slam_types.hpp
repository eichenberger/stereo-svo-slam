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
    int max_pyramid_levels;
    int image_width;
    int image_height;

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
    float pitch;    // around x
    float yaw;      // around y
    float roll;     // around z
};

struct Color {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

struct Frame {
    struct Pose pose;
    struct StereoImage stereo_image;
    struct KeyPoints kps;
};

struct KeyFrame : Frame{
    std::vector<struct Color> colors;
};


#endif
