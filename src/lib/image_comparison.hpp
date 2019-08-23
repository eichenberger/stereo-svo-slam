#ifndef IMAGE_COMPARISON_H
#define IMAGE_COMPARISON_H

#include <vector>
#include <opencv2/opencv.hpp>

#include "stereo_slam_types.hpp"

void get_total_intensity_diff(const cv::Mat &image1, const cv::Mat &image2,
        const std::vector<struct KeyPoint2d> &keypoints1,
        const std::vector<struct KeyPoint2d> &keypoints2,
        size_t patchSize,
        std::vector<float> &diff);

#endif
