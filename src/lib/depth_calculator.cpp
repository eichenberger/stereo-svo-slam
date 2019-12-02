#include <cstdlib>
#include <algorithm>

#include "depth_calculator.hpp"
#include "corner_detector.hpp"
#include "image_comparison.hpp"

using namespace std;

static void detect_keypoints_on_each_level(
        const StereoImage &stereo_images,
        const struct CameraSettings &camera_settings,
        vector<vector<KeyPoint2d>> &keypoints_pyr,
        vector<vector<KeyPointInformation>> &kp_info_pyr)
{
    auto detector = CornerDetector();

    auto grid_width = camera_settings.grid_width;
    auto grid_height = camera_settings.grid_height;

    keypoints_pyr.resize(stereo_images.left.size()/2);
    kp_info_pyr.resize(stereo_images.left.size()/2);
    for (unsigned int i = 0; i < stereo_images.left.size()/2; i++) {
        vector<KeyPoint2d> _keypoints;
        vector<KeyPointInformation> _kp_info;
        const Mat &left = stereo_images.left[i];
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

#pragma omp parallel for
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
                info[i].ignore_completely ||
                info[i].ignore_during_refinement) {
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
    static uint64_t keyframe_count = 0;
    auto &keypoints2d = frame.kps.kps2d;
    auto &keypoints3d = frame.kps.kps3d;
    auto &kp_info = frame.kps.info;
    float fx = camera_settings.fx;
    float fy = camera_settings.fy;
    float cx = camera_settings.cx;
    float cy = camera_settings.cy;
#if 0
    float dist_window_k0 = camera_settings.dist_window_k0;
    float dist_window_k1 = camera_settings.dist_window_k1;
    float dist_window_k2 = camera_settings.dist_window_k2;
    float dist_window_k3 = camera_settings.dist_window_k3;
    float dist_fac0 = 1/(dist_window_k1 - dist_window_k0);
    float dist_fac1 = 1/(dist_window_k3 - dist_window_k2);

    float cost_k0 = camera_settings.cost_k0;
    float cost_k1 = camera_settings.cost_k1;
    float cost_fac0 = 1/(cost_k1-cost_k0);
#endif

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
    Matx33f rot_mat(frame.pose.get_rotation_matrix());
    Vec3f translation(frame.pose.get_translation());

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

        float disparity = minPos;

        // Calculate depth and transform kp3d into global coordinates
        float _z = baseline/max<float>(0.5, disparity);
        float _x = (keypoint.x - cx)/fx*_z;
        float _y = (keypoint.y - cy)/fy*_z;

        Vec3f kp3d(_x, _y, _z);

        // The position is fucked up therefore rot_mat and tranlation
        // are already the inverse and we can use it directly
        kp3d = rot_mat*kp3d;
        kp3d += translation;

        keypoints3d[i].x = kp3d(0);
        keypoints3d[i].y = kp3d(1);
        keypoints3d[i].z = kp3d(2);

        uint32_t color = rand();
        kp_info[i].color.r = (color >> 0) & 0xFF;
        kp_info[i].color.g = (color >> 8) & 0xFF;
        kp_info[i].color.b = (color >> 16) & 0xFF;

        kp_info[i].confidence = 1.0;
        kp_info[i].keyframe_id = keyframe_count;
        kp_info[i].keypoint_index = i;
        kp_info[i].ignore_completely = false;
        kp_info[i].ignore_during_refinement = false;
        kp_info[i].inlier_count = 1;
        kp_info[i].outlier_count = 0;

        KalmanFilter &kf = kp_info[i].kf;
        kf.init(1,1);
        setIdentity(kf.transitionMatrix);
        setIdentity(kf.measurementMatrix);
        setIdentity(kf.processNoiseCov, Scalar::all(0.001));
        setIdentity(kf.errorCovPost, Scalar::all(1.0));

        // Variance can be +- 0.5 pixel
        float deviation = abs((baseline/(disparity+0.5)-baseline/(disparity-0.5)));

        // disparity can't be negaritve
        if (disparity < 0.5)
            deviation = abs((1/(disparity+0.5)));

        // This should probably be dependent on where it is in 2d
        float x_deviation = abs((deviation*keypoint.x-cx)/camera_settings.fx);
        float y_deviation = abs((deviation*keypoint.y-cy)/camera_settings.fy);
        float z_deviation = deviation;

        Vec3f deviation_vec(x_deviation, y_deviation, z_deviation);
        deviation_vec = rot_mat * deviation_vec;

        Matx<float, 1, 1> directed_deviation= deviation_vec.t() * deviation_vec;

        kf.errorCovPost.at<float>(0,0) = directed_deviation(0);

        kf.statePost = (Mat_<float>(1,1) << 1.0);
    }
    keyframe_count++;
}
