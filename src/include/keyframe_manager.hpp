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

    /*!
     * \brief Create a new keyframe
     *
     * @param[in] frame The frame that should be used to create the keyframe
     * @return Pointer to the generated keyframe
     */
    KeyFrame* create_keyframe(Frame &frame);
    /*!
     * \brief Get the keyframe with id
     *
     * @param[in] id The keyframe id
     * @return A pointer to the keyframe
     */
    KeyFrame* get_keyframe(uint32_t id);
    /*!
     * \brief Get the all keyframes
     *
     * @param[out] keyframes vector with keyframes
     * @return A pointer to the keyframe
     */
    void get_keyframes(std::vector<KeyFrame> &keyframes);

    /*!
     * \brief Is a new keyframe needed?
     *
     * @param[in] Current frame
     */
    bool keyframe_needed(const Frame &frame);
private:
    const CameraSettings &camera_settings;
	static uint32_t keyframe_counter;

    std::vector<KeyFrame> keyframes;

};

#endif
