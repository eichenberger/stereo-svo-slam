#include <vector>

#include <opencv2/opencv.hpp>

#include "stereo_slam_types.hpp"

class Match
{
public:
    uint32_t x;
    uint32_t y;
    uint32_t err;
};

class DepthCalculator
{
public:
    DepthCalculator(){}
    void calculate_depth(Frame &frame,
            const struct CameraSettings &camera_settings);

};

