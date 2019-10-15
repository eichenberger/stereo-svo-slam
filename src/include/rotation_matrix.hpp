#ifndef _ROATAION_MATRIX_H
#define _ROATAION_MATRIX_H

#include <vector>
#include <opencv2/opencv.hpp>

void rotation_matrix(const std::vector<float> &angle,
        cv::Matx33f &rotation_matrix);

#endif
