#ifndef KEYFRAME_INSERTER_HPP
#define KEYFRAME_INSERTER_HPP

#include "stereo_slam_types.hpp"

class KeyframeInserter
{
public:
    KeyframeInserter(const CameraSettings &camera_settings) :
        camera_settings(camera_settings)
    {}

    bool keyframe_needed(const Frame &frame);
    void new_keyframe(Frame &frame, KeyFrame &keyframe);

private:
    const CameraSettings &camera_settings;
};

#endif
