#include <cstdlib>
#include <algorithm>

#include "depth_calculator.hpp"
#include "corner_detector.hpp"

using namespace std;

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
        const StereoImage &stereo_images,
        const struct CameraSettings &camera_settings,
        vector<vector<KeyPoint2d>> &keypoints_pyr,
        vector<vector<KeyPointInformation>> &kp_info_pyr)
{
    auto detector = CornerDetector();

    auto grid_width = camera_settings.grid_width;
    auto grid_height = camera_settings.grid_width;

    keypoints_pyr.resize(stereo_images.left.size()/2);
    kp_info_pyr.resize(stereo_images.left.size()/2);
//#pragma omp parallel for
    for (unsigned int i = 0; i < stereo_images.left.size()/2; i++) {
        vector<KeyPoint2d> _keypoints;
        vector<KeyPointInformation> _kp_info;
        const Mat &left = stereo_images.left[2*i];
        // TODO: We can use the gradient image from the pyramid
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

//#pragma omp parallel for
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

static void find_bad_keypoints(Frame &frame) {
    auto &width = frame.stereo_image.left[0].cols;
    auto &height = frame.stereo_image.left[0].rows;
    auto &kps2d = frame.kps.kps2d;
    auto &kps3d = frame.kps.kps3d;
    auto &info = frame.kps.info;

    for (size_t i = 0; i < kps2d.size(); i++) {
        auto &kp2d = kps2d[i];
        if ((kp2d.x < 0) || (kp2d.y < 0) ||
                (kp2d.x > width) || (kp2d.y > height) ||
                (info[i].confidence < 0.2)) {
            kps2d.erase(kps2d.begin() + i);
            kps3d.erase(kps3d.begin() + i);
            info.erase(info.begin() +i);
            i--;
        }
    }
}

static void merge_keypoints(Frame &frame,
        vector<KeyPoint2d> new_keypoints, vector<KeyPointInformation> new_info,
        int grid_height, int grid_width) {
    auto &kps2d = frame.kps.kps2d;
    auto &info = frame.kps.info;

    int image_width = frame.stereo_image.left[0].cols;
    int image_height = frame.stereo_image.left[0].rows;

//#pragma omp parallel for
    for (int x = 0; x < image_width; x += grid_width) {
        int left = x;
        int right = left + grid_width;
        for (int y = 0; y < image_height; y += grid_height) {
            int top = y;
            int bottom = y + grid_height;
            bool match = false;

            // Check if we already have a keypoint in the current grid box
            for (size_t i = 0; i < kps2d.size(); i++) {
                auto &kp2d = kps2d[i];

                if (kp2d.x  > left && kp2d.x < right&&
                        kp2d.y > top && kp2d.y < bottom) {
                    match = true;
                    break;
                }
            }

            if (!match) {
                // Add one of the new points if no old one can be found
                for (size_t i = 0; i < new_keypoints.size(); i++) {
                    auto &kp2d = new_keypoints[i];

                    if (kp2d.x  > left && kp2d.x < right&&
                            kp2d.y > top && kp2d.y < bottom) {
                        kps2d.push_back(kp2d);
                        info.push_back(new_info[i]);
                    }
                }
            }
       }
    }
}

void DepthCalculator::calculate_depth(Frame &frame,
        const struct CameraSettings &camera_settings)
{
    auto &keypoints2d = frame.kps.kps2d;
    auto &keypoints3d = frame.kps.kps3d;
    auto &kp_info = frame.kps.info;
    float fx = camera_settings.fx;
    float fy = camera_settings.fy;
    float cx = camera_settings.cx;
    float cy = camera_settings.cy;
    float dist_window_k0 = camera_settings.dist_window_k0;
    float dist_window_k1 = camera_settings.dist_window_k1;
    float dist_window_k2 = camera_settings.dist_window_k2;
    float dist_window_k3 = camera_settings.dist_window_k3;
    float dist_fac0 = 1/(dist_window_k1 - dist_window_k0);
    float dist_fac1 = 1/(dist_window_k3 - dist_window_k2);

    float cost_k0 = camera_settings.cost_k0;
    float cost_k1 = camera_settings.cost_k1;
    float cost_fac0 = 1/(cost_k1-cost_k0);

    float baseline = camera_settings.baseline;
    int search_x = camera_settings.search_x;
    int search_y = camera_settings.search_y;
    int window_size = camera_settings.window_size_depth_calculator;
    int window_before = window_size/2;
    int window_after = (window_size+1)/2;

    vector<vector<KeyPoint2d>> keypoints_pyr;
    vector<vector<KeyPointInformation>> kp_info_pyr;


    detect_keypoints_on_each_level(frame.stereo_image,
            camera_settings, keypoints_pyr, kp_info_pyr);

    // Remove keypoints that are not within the image
    find_bad_keypoints(frame);

    vector<KeyPoint2d> _keypoints2d;
    vector<KeyPointInformation> _kp_info;
    select_best_keypoints(keypoints_pyr, kp_info_pyr, _keypoints2d, _kp_info);


    size_t old_keypoint_count = keypoints2d.size();

    merge_keypoints(frame, _keypoints2d, _kp_info,
            camera_settings.grid_width, camera_settings.grid_height);

    keypoints3d.resize(keypoints2d.size());

    const Mat &left = frame.stereo_image.left[0];
    const Mat &right= frame.stereo_image.right[0];

    // Prepare rotation matrix for inverse transformation
    Mat angles(1, 3, CV_32F, (void*)&frame.pose.pitch);
    Mat rot_mat(3, 3, CV_32F);
    Rodrigues(angles, rot_mat);

    Mat translation(3, 1, CV_32F, &frame.pose.x);
    // Estimate the depth now
#pragma omp parallel for
    for (uint32_t i = old_keypoint_count; i < keypoints2d.size(); i++) {
        auto keypoint = keypoints2d[i];
        int x = static_cast<int>(keypoint.x);
        int y = static_cast<int>(keypoint.y);

        uint32_t x11 = max<int>(0, x - window_before);
        uint32_t x12 = min<int>(left.cols - 1, x + window_after);
        uint32_t y11 = max<int>(0, y - window_before);
        uint32_t y12 = min<int>(left.rows, y + window_after);
        auto templ = left(Range(y11,y12), Range(x11, x12));

        uint32_t x21 = max<int>(0, x - window_before);
        uint32_t x22 = min<int>(left.cols - 1, x + window_after + search_x);
        uint32_t y21 = max<int>(0, y - window_before - search_y);
        uint32_t y22 = min<int>(left.rows - 1, y + window_after + search_y);
        auto roi = right(Range(y21,y22), Range(x21, x22));

        //auto _match = match(roi, templ);
        //float disparity = _match.x + window_before;
        Mat match;
        matchTemplate(roi, templ, match, TM_SQDIFF);
        double minVal, maxVal;
        Point minLoc, maxLoc;
        minMaxLoc(match, &minVal, &maxVal, &minLoc, &maxLoc);

        // If we don't have a lot of contrast we take the average of x
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

        float disparity = max<float>(0.5, minPos);

        // Calculate depth and transform kp3d into global coordinates
        float _z = baseline/disparity;
        float _x = (keypoint.x - cx)/fx*_z;
        float _y = (keypoint.y - cy)/fy*_z;

        Mat kp3d = (Mat_<float>(3, 1) <<
                _x, _y, _z);

        // The position is fucked up therefore rot_mat and tranlation
        // are already the inverse and we can use it directly
        kp3d = rot_mat*kp3d;
        kp3d += translation;

        keypoints3d[i].x = kp3d.at<float>(0);
        keypoints3d[i].y = kp3d.at<float>(1);
        keypoints3d[i].z = kp3d.at<float>(2);

        // Reduce confidence if intensitiy difference is high
        // float _confidence = 100-_match.err;
        float _confidence;

        // Give penalty for edglets
        _confidence = (kp_info[i].type == KP_FAST) ? 1.0 : 0.4;

        // Create a convidence window _/-\_ between k0, k1, k2 and k3
        _confidence *= (0.2 + (max<float>(0, _z - dist_window_k0)*dist_fac0 -
                max<float>(0, _z - dist_window_k1)*dist_fac0 -
                max<float>(0, _z -dist_window_k2)*dist_fac1 +
                max<float>(0, _z-dist_window_k3)*dist_fac1)*0.8);

        cout << "z value: " << _z << " cost: " << minVal << " matches: " << matches << endl;
        // More convidence if cost is low -\_. Convidence going down > k0 until k1
        _confidence *= (1.0 - 0.4*(max<float>(0, minVal - cost_k0)*cost_fac0 -
                    max<float>(0, minVal - cost_k1)*cost_fac0));

        _confidence *= (0.6*(min<float>(1.0, kp_info[i].score/150.0)) + 0.4);
         // The more matches we had the less confident we are...
         // Normally we only have one match
        _confidence *= 1.0/matches;

        uint32_t color = rand();
        kp_info[i].color.r = (color >> 0) & 0xFF;
        kp_info[i].color.g = (color >> 8) & 0xFF;
        kp_info[i].color.b = (color >> 16) & 0xFF;

        kp_info[i].confidence = _confidence;
        kp_info[i].keyframe_id = frame.id;
    }
}
