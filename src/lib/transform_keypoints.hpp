#ifndef _TRANSFORM_KEYPOINTS_H
#define _TRANSFORM_KEYPOINTS_H

#include <vector>
#include <opencv2/opencv.hpp>

void transform_keypoints(const std::vector<float> &pose,
        const std::vector<std::array<float, 3>>& keypoints3d,
        double fx, double fy, double cx, double cy,
        std::vector<std::array<float, 2>> keypoints2d);

#endif
