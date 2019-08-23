#include <cstdlib>
#include <algorithm>

#include "depth_calculator.hpp"
#include "corner_detector.hpp"

Match DepthCalculator::match(Mat &roi, Mat &templ)
{
    Match _match;
    _match.err = UINT32_MAX;

    for (int x = 0; x < roi.cols - templ.cols; x++) {
        for (int y = 0; y < roi.rows - templ.rows; y++) {
            Mat diff;
            absdiff(templ, roi(Range(y, y+templ.rows),Range(x, x+templ.cols)), diff);
            double err = sum(diff)[0];

            if (err < _match.err) {
                _match.err = err;
                _match.x = x;
                _match.y = y;
            }
        }
    }
    return _match;
}

void DepthCalculator::calculate_depth(const struct StereoImage &stereo_image,
            const struct CameraSettings &camera_settings,
            struct KeyPoints &keypoints)
{
    auto detector = CornerDetector();
    const Mat &left = stereo_image.left;
    const Mat &right = stereo_image.right;
    auto grid_width = camera_settings.grid_width;
    auto grid_height = camera_settings.grid_width;
    auto &keypoints2d = keypoints.kps2d;
    auto &keypoints3d = keypoints.kps3d;
    auto &err = keypoints.err;
    float fx = camera_settings.fx;
    float fy = camera_settings.fy;
    float cx = camera_settings.cx;
    float cy = camera_settings.cy;
    float baseline = camera_settings.baseline;
    int search_x = camera_settings.search_x;
    int search_y = camera_settings.search_y;
    int window_size = camera_settings.window_size;
    int window_before = (window_size-1)/2;
    int window_after = window_size/2;

    vector<Point> _keypoints;
    detector.detect_keypoints(left, grid_width, grid_height, _keypoints);

    keypoints2d.clear();
    keypoints2d.resize(_keypoints.size());
    keypoints3d.clear();
    keypoints3d.resize(_keypoints.size());
    err.clear();
    err.resize(_keypoints.size());

//#pragma omp parallel for
    for (uint32_t i = 0; i < _keypoints.size(); i++) {
        auto keypoint = _keypoints[i];
        uint32_t x = static_cast<int>(keypoint.x);
        uint32_t y = static_cast<int>(keypoint.y);

        uint32_t x1 = std::max<int>(0, x - window_before);
        uint32_t x2 = std::min<int>(left.cols - search_x, x + window_after + search_x);
        uint32_t y1 = std::max<int>(0, y - window_before - search_y);
        uint32_t y2 = std::min<int>(left.rows, y + window_after + search_y);

//        if (x1 < 0 || (x2 + search_x) > (uint32_t)left.cols || y1 < (uint32_t)search_y ||
//                (y2 + search_y) >= (uint32_t)(left.rows-search_y))
//            continue;

        auto templ = left(Range(y1,y2), Range(x1, x2));
        auto roi = right(Range(y1,y2), Range(x1, x2));

        auto _match = match(roi, templ);
        float disparity = _match.x + window_before;
        keypoints2d[i].x = x;
        keypoints2d[i].y = y;

        keypoints3d[i].z = baseline/disparity;
        keypoints3d[i].x = (keypoints2d[i].x - cx)/fx*keypoints3d[i].z;
        keypoints3d[i].y = (keypoints2d[i].y - cy)/fy*keypoints3d[i].z;
    }
}
