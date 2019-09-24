#ifndef OPTICAL_FLOW_H
#define OPTICAL_FLOW_H

#include <vector>

#include "stereo_slam_types.hpp"

class OpticalFlow {

public:
    OpticalFlow();

    void calculate_optical_flow(const std::vector<StereoImage> &previous_stereo_image_pyr,
        const std::vector<KeyPoint2d> &previous_keypoints2d,
        const std::vector<StereoImage> &current_stereo_image_pyr,
        std::vector<KeyPoint2d> &current_keypoints2d,
        std::vector<float> &err);

};

#endif
