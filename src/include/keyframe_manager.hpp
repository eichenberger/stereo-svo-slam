#ifndef KEYFRAME_MANAGER_HPP
#define KEYFRAME_MANAGER_HPP

#include <vector>

#include "stereo_slam_types.hpp"

/*!
 * \brief Manages key frames (internal use only)
 *
 * Decides when to insert new keyframes, allows to create new keyfames, etc.
 */
class KeyFrameManager {
public:
    KeyFrameManager(const CameraSettings &camera_settings);

	KeyFrame* create_keyframe(Frame &frame);
	KeyFrame* get_keyframe(uint32_t id);
	void get_keyframes(std::vector<KeyFrame> &keyframes);

    bool keyframe_needed(const Frame &frame);
private:
    const CameraSettings &camera_settings;
	static uint32_t keyframe_counter;

    std::vector<KeyFrame> keyframes;

};

#endif
