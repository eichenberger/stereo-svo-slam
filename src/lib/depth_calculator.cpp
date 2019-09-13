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

static void detect_keypoints_on_each_level(
        const vector<struct StereoImage> &stereo_images,
        const struct CameraSettings &camera_settings,
        vector<vector<KeyPoint2d>> &keypoints_pyr,
        vector<vector<KeyPointInformation>> &kp_info_pyr)
{
    auto detector = CornerDetector();

    auto grid_width = camera_settings.grid_width;
    auto grid_height = camera_settings.grid_width;

    keypoints_pyr.resize(stereo_images.size());
    kp_info_pyr.resize(stereo_images.size());
//#pragma omp parallel for
    for (unsigned int i = 0; i < stereo_images.size(); i++) {
        vector<KeyPoint2d> _keypoints;
        vector<KeyPointInformation> _kp_info;
        const Mat &left = stereo_images[i].left;
        detector.detect_keypoints(left, grid_width, grid_height, _keypoints,
                _kp_info, i);
        keypoints_pyr[i] = _keypoints;
        kp_info_pyr[i] = _kp_info;
        grid_width /= 2;
        grid_height /= 2;
    }
}

static void select_best_keypoints(
        const vector<vector<KeyPoint2d>> &keypoints_pyr,
        const vector<vector<KeyPointInformation>> &kp_info_pyr,
        vector<KeyPoint2d> &keypoints, vector<KeyPointInformation> &kp_info) {

    keypoints = keypoints_pyr[0];
    kp_info = kp_info_pyr[0];

    for (unsigned int i = 1; i < keypoints_pyr.size(); i++) {
        vector<KeyPoint2d> _keypoints = keypoints_pyr[i];
        vector<KeyPointInformation> _info = kp_info_pyr[i];
        for (unsigned int j = 0; j < keypoints.size(); j++) {
            // Fast is better than edgledt -> skip
            if (kp_info[j].type == KP_FAST && _info[j].type == KP_EDGELET)
                continue;
            // The types are equal but the score of the upper pyramid kp is higher -> skip
            if ((kp_info[j].type == _info[j].type) &&
                (kp_info[j].score > _info[j].score))
                continue;
            // The lower layer kp is better
            keypoints[j] = _keypoints[j];
            // Upscale the position of the kp to layer 0
            keypoints[j].x *= (1<<i);
            keypoints[j].y *= (1<<i);
            kp_info[j] = _info[j];
        }
    }
}

void DepthCalculator::calculate_depth(
        const vector<struct StereoImage> &stereo_images,
        const struct CameraSettings &camera_settings,
        struct KeyPoints &keypoints)
{
    auto &keypoints2d = keypoints.kps2d;
    auto &keypoints3d = keypoints.kps3d;
    auto &kp_info = keypoints.info;
    float fx = camera_settings.fx;
    float fy = camera_settings.fy;
    float cx = camera_settings.cx;
    float cy = camera_settings.cy;
    float baseline = camera_settings.baseline;
    int search_x = camera_settings.search_x;
    int search_y = camera_settings.search_y;
    int window_size = camera_settings.window_size;
    int window_before = window_size/2;
    int window_after = (window_size+1)/2;

    vector<vector<KeyPoint2d>> keypoints_pyr;
    vector<vector<KeyPointInformation>> kp_info_pyr;

    detect_keypoints_on_each_level(stereo_images,
            camera_settings, keypoints_pyr, kp_info_pyr);

    keypoints2d.clear();
    select_best_keypoints(keypoints_pyr, kp_info_pyr, keypoints2d, kp_info);

    keypoints3d.clear();
    keypoints3d.resize(keypoints2d.size());

    const Mat &left = stereo_images[0].left;
    const Mat &right= stereo_images[0].right;
    // Estimate the depth now
//#pragma omp parallel for
    for (uint32_t i = 0; i < keypoints2d.size(); i++) {
        auto keypoint = keypoints2d[i];
        int x = static_cast<int>(keypoint.x);
        int y = static_cast<int>(keypoint.y);

        uint32_t x11 = std::max<int>(0, x - window_before);
        uint32_t x12 = std::min<int>(left.cols - 1, x + window_after);
        uint32_t y11 = std::max<int>(0, y - window_before);
        uint32_t y12 = std::min<int>(left.rows, y + window_after);
        auto templ = left(Range(y11,y12), Range(x11, x12));

        uint32_t x21 = std::max<int>(0, x - window_before);
        uint32_t x22 = std::min<int>(left.cols - 1, x + window_after + search_x);
        uint32_t y21 = std::max<int>(0, y - window_before - search_y);
        uint32_t y22 = std::min<int>(left.rows - 1, y + window_after + search_y);
        auto roi = right(Range(y21,y22), Range(x21, x22));

        //auto _match = match(roi, templ);
        //float disparity = _match.x + window_before;
        Mat match;
        matchTemplate(roi, templ, match, TM_SQDIFF);
        double minVal, maxVal;
        Point minLoc, maxLoc;
        minMaxLoc(match, &minVal, &maxVal, &minLoc, &maxLoc);

        float minPos = 0;
        int matches = 0;
        for (int j = minLoc.x; j < match.cols; j++) {
            for (int k = minLoc.y; k < match.rows; k++) {
                if (match.at<float>(k,j) <= minVal) {
                    minPos += j;
                    matches++;
                }
            }
        }
        minPos = minPos/matches;

        float disparity = std::max<float>(0.5, minPos);

        keypoints3d[i].z = baseline/disparity;
        keypoints3d[i].x = (keypoint.x - cx)/fx*keypoints3d[i].z;
        keypoints3d[i].y = (keypoint.y - cy)/fy*keypoints3d[i].z;
        // Reduce confidence if intensitiy difference is high
        // float _confidence = 100-_match.err;
        float _confidence = 100-minVal;

        // Give penalty for edglets
        _confidence -= kp_info[i].type == KP_FAST ? 0 : 10;
        _confidence /= 100;
        kp_info[i].confidence = std::max<float>(0, _confidence);
    }
}
