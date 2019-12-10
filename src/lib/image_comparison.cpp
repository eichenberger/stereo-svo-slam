#include <vector>
#include <opencv2/opencv.hpp>

#include "stereo_slam_types.hpp"

using namespace cv;
using namespace std;

float get_intensity_diff(const cv::Mat &image1, const cv::Mat &image2,
        const KeyPoint2d &keypoint1, const KeyPoint2d &keypoint2,
        size_t patchSize)
{
    Size _patchSize(patchSize, patchSize);
    Point2f _center;
    _center.x = keypoint1.x;
    _center.y = keypoint1.y;
    Mat patch1, patch2;
    getRectSubPix(image1, _patchSize, _center, patch1, CV_32F);
    _center.x = keypoint2.x;
    _center.y = keypoint2.y;
    getRectSubPix(image2, _patchSize, _center, patch2, CV_32F);

    Mat diff;
    absdiff(patch1, patch2, diff);
    return sum(diff)[0];
}


void get_total_intensity_diff(const cv::Mat &image1, const cv::Mat &image2,
        const vector<struct KeyPoint2d> &keypoints1,
        const vector<struct KeyPoint2d> &keypoints2,
        size_t patchSize,
        vector<float> &diff)
{
    diff.resize(keypoints1.size());
#pragma omp parallel for default(none) shared(keypoints1, keypoints2, image1, image2, patchSize, diff)
    for (unsigned i = 0; i < keypoints1.size(); i++) {
        KeyPoint2d kp1 = keypoints1[i];
        KeyPoint2d kp2 = keypoints2[i];
        diff[i] = get_intensity_diff(image1, image2, kp1, kp2, patchSize);
    }
}

