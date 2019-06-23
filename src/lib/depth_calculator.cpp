#include <cstdlib>

#include "depth_calculator.hpp"

#include "corner_detector.hpp"

DepthCalculator::DepthCalculator(float baseline, float fx, float fy, float cx,
        float cy, int window_size, int search_x, int search_y, int margin):
    baseline(baseline), fx(fx), fy(fy), cx(cx), cy(cy),
    half_window_size(window_size/2), search_x(search_x), search_y(search_y),
    margin(margin)
{
}

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

void DepthCalculator::calculate_depth(Mat &left, Mat &right, int split_count,
        vector<array<float, 2>> &keypoints2d,
        vector<array<float, 3>> &keypoints3d,
        vector<uint32_t> &err)
{
    auto detector = CornerDetector(margin);

    vector<Point> keypoints;
    detector.detect_keypoints(left, split_count, keypoints);

    keypoints2d.clear();
    keypoints2d.resize(keypoints.size());
    keypoints3d.clear();
    keypoints3d.resize(keypoints.size());
    err.clear();
    err.resize(keypoints.size());

#pragma omp parallel for
    for (uint32_t i = 0; i < keypoints.size(); i++) {
        auto keypoint = keypoints[i];
        uint32_t x = static_cast<int>(keypoint.x);
        uint32_t y = static_cast<int>(keypoint.y);

        uint32_t x1 = x - half_window_size;
        uint32_t x2 = x + half_window_size;
        uint32_t y1 = y - half_window_size;
        uint32_t y2 = y + half_window_size;

        if (x1 < 0 || (x2 + search_x) > (uint32_t)left.cols || y1 < (uint32_t)search_y ||
                (y2 + search_y) >= (uint32_t)(left.rows-search_y))
            continue;

        auto templ = left(Range(y1,y2), Range(x1, x2));
        auto roi = right(Range(y1-search_y,y2+search_y), Range(x1, x2 + search_x));

        auto _match = match(roi, templ);
        float disparity = _match.x + half_window_size;
        keypoints2d[i][0] = x + _match.x + half_window_size;
        keypoints2d[i][1] = y + _match.y + half_window_size;

        keypoints3d[i][2] = baseline/disparity;
        keypoints3d[i][0] = (keypoints2d[i][0] - cx)/fx*keypoints3d[i][2];
        keypoints3d[i][1] = (keypoints2d[i][1] - cy)/fy*keypoints3d[i][2];

    }
}
