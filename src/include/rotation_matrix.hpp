#ifndef _ROATAION_MATRIX_H
#define _ROATAION_MATRIX_H

#include <vector>
#include <opencv2/opencv.hpp>

/*!
 * \brief Calculate the rotation matrix from angles
 *
 * @param[in] angle rx,ry,rz
 * @param[out] rotation_matrix The rotation matrix from rx,ry,rz
 */
void rotation_matrix(const std::vector<float> &angle,
        cv::Matx33f &rotation_matrix);

#endif
