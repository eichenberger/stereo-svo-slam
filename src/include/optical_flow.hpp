#ifndef OPTICAL_FLOW_H
#define OPTICAL_FLOW_H

#include <vector>

#include "stereo_slam_types.hpp"


/*!
 * \brief Wrapper for opencv optical flow (internal use)
 */
class OpticalFlow {

public:
    OpticalFlow(const CameraSettings &camera_settings);

    /*!
     * \brief Calculate the optical flow
     *
     * @param[in] previous_stereo_image_pyr Reference stereo image pyramid
     * @param[in] previous_keypoints2d Keypoints in the previous image
     * @param[in] current_stereo_image_pyr Stereo image pyramid to compare
     * @param[in] current_keypoints2d Keypoints in the image to compare
     * @param[out] err Intensity difference for each point
     */
    void calculate_optical_flow(const StereoImage &previous_stereo_image_pyr,
        const std::vector<KeyPoint2d> &previous_keypoints2d,
        const StereoImage &current_stereo_image_pyr,
        std::vector<KeyPoint2d> &current_keypoints2d,
        std::vector<float> &err);

private:
    const CameraSettings &camera_settings;

};

#endif
