#ifndef DEPTH_FILTER_HPP
#define DEPTH_FILTER_HPP

#include "keyframe_manager.hpp"

#include "stereo_slam_types.hpp"

class DepthFilter
{
public:
    DepthFilter(KeyFrameManager &keyframe_manager,
            const CameraSettings &camera_settings);

    void update_depth(Frame &frame);

private:
    void calculate_disparities(Frame &frame, std::vector<float> &disparity);

    KeyFrameManager &keyframe_manager;
    const CameraSettings &camera_settings;

};



#endif
