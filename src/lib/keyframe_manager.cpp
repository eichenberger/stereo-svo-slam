#include "keyframe_manager.hpp"

#include "depth_calculator.hpp"
#include "transform_keypoints.hpp"

using namespace std;

uint32_t KeyFrameManager::keyframe_counter = 0;

KeyFrameManager::KeyFrameManager(const CameraSettings &camera_settings) :
	camera_settings(camera_settings)
{
}

KeyFrame* KeyFrameManager::create_keyframe(Frame &frame)
{
	keyframes.push_back(KeyFrame());

	KeyFrame &keyframe = keyframes.back();

	keyframe.id = keyframe_counter; keyframe_counter++;

    DepthCalculator depth_calculator;

    depth_calculator.calculate_depth(frame, camera_settings);

    keyframe.kps = frame.kps;
    keyframe.pose = frame.pose;
    keyframe.stereo_image = frame.stereo_image;

	return &keyframe;
}

KeyFrame* KeyFrameManager::get_keyframe(uint32_t id)
{
	if (id < keyframes.size())
		return &keyframes[id];

	return NULL;
}

void KeyFrameManager::get_keyframes(std::vector<KeyFrame> &keyframes)
{
	keyframes = this->keyframes;
}

bool KeyFrameManager::keyframe_needed(const Frame &frame)
{
    auto image_width = frame.stereo_image.left[0].cols;
    auto image_height = frame.stereo_image.left[0].rows;
    int inside_frame = 0;

    for (size_t i = 0; i < frame.kps.kps2d.size(); i++) {
        KeyPoint2d kp = frame.kps.kps2d[i];
        KeyPointInformation info = frame.kps.info[i];
        if ((kp.x > 0) && (kp.y > 0) &&
                (kp.x < image_width) &&
                (kp.y < image_height) &&
                !info.ignore_completely) {
            inside_frame++;
        }
    }

    int cols = frame.stereo_image.left[0].cols;
    int rows= frame.stereo_image.left[0].rows;

    int max_keypoints = (cols/camera_settings.grid_width)*(rows/camera_settings.grid_height);

    if (inside_frame < 3*(max_keypoints/4)) {
        return true;
    }

    return false;
}
