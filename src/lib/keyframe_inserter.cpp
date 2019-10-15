#include <vector>
#include <cstdlib>

#include "depth_calculator.hpp"
#include "transform_keypoints.hpp"

#include "keyframe_inserter.hpp"

using namespace std;

bool KeyframeInserter::keyframe_needed(const Frame &frame)
{
    auto image_width = frame.stereo_image.left[0].cols;
    auto image_height = frame.stereo_image.left[0].rows;
    int inside_frame = 0;

    for (auto kp: frame.kps.kps2d) {
        if ((kp.x > 0) && (kp.y > 0) &&
                (kp.x < image_width) &&
                (kp.y < image_height)) {
            inside_frame++;
        }
    }

    if ((inside_frame < (frame.kps.kps3d.size()*0.7)) &&
            (inside_frame < 200)) {
        return true;
    }

    return false;
}

void KeyframeInserter::new_keyframe(Frame &frame, KeyFrame &keyframe)
{
    DepthCalculator depth_calculator;

    depth_calculator.calculate_depth(frame, camera_settings);

    keyframe.kps = frame.kps;
    keyframe.pose = frame.pose;
    keyframe.stereo_image = frame.stereo_image;
    keyframe.id = frame.id;
}
