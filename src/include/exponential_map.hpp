#ifndef EXPONENTIAL_MAP_HPP
#define EXPONENTIAL_MAP_HPP

#include <opencv2/opencv.hpp>

/*!
 * \brief Caclulate the exponential map (internal use)
 */
static inline void exponential_map(const cv::Mat &twist, cv::Mat &pose)
{
    cv::Vec3f v(twist.ptr<float>(0));
    cv::Vec3f w(twist.ptr<float>(3));

    twist.copyTo(pose);
    // The angles don't change. See robotics vision and control
    // page 53 for more details
    cv::Matx33f w_skew(0, -w(2), w(1),
            w(2), 0, -w(0),
            -w(1), w(0), 0);
    float _norm = 1.0;
    cv::Matx33f _eye = cv::Matx33f::eye();
    cv::Vec3f translation;

    // Take closed form solution from robotics vision and control page 53
    // Note: This exponential map doesn't give the exact same value as expm
    // from Matlab or Numpy. It is different up to a scaling. It seems that
    // expm uses norm set to 1.0. However, this differs from the closed form
    // solution written in robotics vision and control. We set norm = 1.0 to
    // make sure we get the same as in Numpy/Matlab.
    translation = (_eye*_norm + (1-cos(_norm))*w_skew+(_norm-sin(_norm))*(w_skew*w_skew))*v;

    memcpy(pose.ptr<float>(), &translation[0], 3*sizeof(float));

}

#endif
