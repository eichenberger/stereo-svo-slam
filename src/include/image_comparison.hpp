#ifndef IMAGE_COMPARISON_H
#define IMAGE_COMPARISON_H

#include <vector>
#include <opencv2/opencv.hpp>

#include "stereo_slam_types.hpp"

/*!
 * \brief Calculate the intensity difference at some points
 *
 * @param[in] image1 Reference image
 * @param[in] image2 Image to compare
 * @param[in] keypoint1 Point in the reference image
 * @param[in] keypoint2 Point in the image to compare
 * @param[in] patch_size The patch size to compare (8->8x8 pixel)
 * @return Intensity difference
 */
float get_intensity_diff(const cv::Mat &image1, const cv::Mat &image2,
        const KeyPoint2d &keypoint1, const KeyPoint2d &keypoint2,
        size_t patch_size);

/*!
 * \brief Calculate the intensity difference over several points
 *
 * @param image1 Reference image
 * @param image2 Image to compare
 * @param keypoints1 Points in the reference image
 * @param keypoints2 Points in the image to compare
 * @param patch_size The patch size to compare (8->8x8 pixel)
 * @return Intensity difference
 */
float get_total_intensity_diff(const cv::Mat &image1, const cv::Mat &image2,
        const std::vector<struct KeyPoint2d> &keypoints1,
        const std::vector<struct KeyPoint2d> &keypoints2,
        size_t patch_size);

#endif
