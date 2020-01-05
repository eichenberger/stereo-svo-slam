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

    void calculate_optical_flow(const StereoImage &previous_stereo_image_pyr,
        const std::vector<KeyPoint2d> &previous_keypoints2d,
        const StereoImage &current_stereo_image_pyr,
        std::vector<KeyPoint2d> &current_keypoints2d,
        std::vector<float> &err);

private:
    const CameraSettings &camera_settings;

};

#endif
