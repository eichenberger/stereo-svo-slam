#include <vector>
#include <opencv2/opencv.hpp>

#include "stereo_slam_types.hpp"

using namespace cv;
using namespace std;

static inline float _get_intensity_diff(const Mat &image1, Mat image2,
        const KeyPoint2d &center1, const KeyPoint2d center2,
        size_t patch_size)
{
    KeyPoint2d start1 = center1;
    KeyPoint2d start2 = center2;
    Point ip1, ip2;

    // e.g. if we have patch_size = 1 we just take the pixel position
    // without additions. However if it's e.g. 2 we want also to consider
    // the previous pixel
    float half_size = ((float)patch_size-1.0f)/2.0f;
    start1.x -= half_size;
    start1.y -= half_size;

    start2.x -= half_size;
    start2.y -= half_size;

    ip1.x = floor(start1.x);
    ip1.y = floor(start1.y);

    ip2.x = floor(start2.x);
    ip2.y = floor(start2.y);

    // how much do we count the last pixel
    float x12 = start1.x - ip1.x;
    float y12 = start1.y - ip1.y;

    float x22 = start2.x - ip2.x;
    float y22 = start2.y - ip2.y;

    // how much do we count the first pixel
    float x11 = 1.0 - x12;
    float y11 = 1.0 - y12;

    float x21 = 1.0 - x22;
    float y21 = 1.0 - y22;

    float m11 = x11*y11;
    float m12 = x12*y11;
    float m13 = x11*y12;
    float m14 = x12*y12;

    float m21 = x21*y21;
    float m22 = x22*y21;
    float m23 = x21*y22;
    float m24 = x22*y22;

    Vec4f m1(m11, m12, m13, m14);
    Vec4f m2(m21, m22, m23, m24);

    float intensity = 0;

    if (ip1.y >= 0 && (int)(ip1.y+ patch_size) < image1.rows &&
        ip2.y >= 0 && (int)(ip2.y + patch_size) < image2.rows &&
        ip1.x >= 0 && (int)(ip1.x + patch_size) < image1.cols &&
        ip2.x >= 0 && (int)(ip2.x + patch_size) < image2.cols )
    {
        for (size_t i = 0; i < patch_size; i++) {
            const uint8_t *src11 = image1.ptr<uint8_t>(i+ip1.y, ip1.x);
            const uint8_t *src12 = image1.ptr<uint8_t>(i+ip1.y+1, ip1.x);
            const uint8_t *src21 = image2.ptr<uint8_t>(i+ip2.y, ip2.x);
            const uint8_t *src22 = image2.ptr<uint8_t>(i+ip2.y+1, ip2.x);
            for (size_t j = 0; j < patch_size; j++) {
                Vec4f px1 (*(src11+0),
                           *(src11+1),
                           *(src12+0),
                           *(src12+1));
                Vec4f px2 (*(src21+0),
                           *(src21+1),
                           *(src22+0),
                           *(src22+1));

                float i1 = (m1.t()*px1)[0];
                float i2 = (m2.t()*px2)[0];

                intensity += fabs(i1-i2);
                src11++; src12++; src21++; src22++;
            }
        }
    }
    return intensity;
}


float get_intensity_diff(const cv::Mat &image1, const cv::Mat &image2,
        const KeyPoint2d &keypoint1, const KeyPoint2d &keypoint2,
        size_t patch_size)
{
#if 0
    Size _patchSize(patch_size, patch_size);
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
#else
    return _get_intensity_diff(image1, image2, keypoint1, keypoint2, patch_size);
#endif

}


void get_total_intensity_diff(const cv::Mat &image1, const cv::Mat &image2,
        const vector<struct KeyPoint2d> &keypoints1,
        const vector<struct KeyPoint2d> &keypoints2,
        size_t patchSize,
        vector<float> &diff)
{
    diff.resize(keypoints1.size());
//#pragma omp parallel for default(none) shared(keypoints1, keypoints2, diff) firstprivate(image1, image2, patchSize)
    for (unsigned i = 0; i < keypoints1.size(); i++) {
        KeyPoint2d kp1 = keypoints1[i];
        KeyPoint2d kp2 = keypoints2[i];
        float _diff = _get_intensity_diff(image1, image2, kp1, kp2, patchSize);
        diff[i] = _diff;
    }
}

