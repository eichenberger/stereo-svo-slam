#include <math.h>
#include <algorithm>

#include<opencv2/opencv.hpp>

#include "transform_keypoints.hpp"
#include "image_comparison.hpp"
#include "convert_pose.hpp"
#include "depth_filter.hpp"

using namespace cv;
using namespace std;

#define PRINT_TIME_TRACE

#ifdef PRINT_TIME_TRACE
static TickMeter tick_meter;
#define START_MEASUREMENT() tick_meter.reset(); tick_meter.start()

#define END_MEASUREMENT(_name) tick_meter.stop();\
    cout << _name << " took: " << tick_meter.getTimeMilli() << "ms" << endl

#else
#define START_MEASUREMENT()
#define END_MEASUREMENT(_name)
#endif



DepthFilter::DepthFilter(const vector<KeyFrame> &keyframes, const CameraSettings &camera_settings) :
    keyframes(keyframes), camera_settings(camera_settings)
{
}

void DepthFilter::update_depth(Frame &frame)
{
    float fx = camera_settings.fx;
    float fy = camera_settings.fy;
    float cx = camera_settings.cx;
    float cy = camera_settings.cy;
    float baseline = camera_settings.baseline;


    // Calculate depth
    vector<float> disparities;
    calculate_disparities(frame, disparities);

    vector<KeyPoint2d> &kps2d = frame.kps.kps2d;
    vector<KeyPoint3d> kps3d;
    kps3d.resize(kps2d.size());

    Mat angles(1, 3, CV_32F, (void*)&frame.pose.pitch);
    Mat rot_mat(3, 3, CV_32F);
    Rodrigues(angles, rot_mat);

    Mat translation(3, 1, CV_32F, &frame.pose.x);
    for (size_t i = 0; i < disparities.size(); i++) {
        float disparity = disparities[i];
        if (disparity < 0)
            continue;
        // Calculate depth and transform kp3d into global coordinates
        float _z = baseline/std::max<float>(disparity, 0.5);
        float _x = (kps2d[i].x - cx)/fx*_z;
        float _y = (kps2d[i].y - cy)/fy*_z;

        Mat kp3d = (Mat_<float>(3, 1) <<
                _x, _y, _z);

        kp3d = rot_mat*kp3d;
        kp3d += translation;

        kps3d[i].x = kp3d.at<float>(0);
        kps3d[i].y = kp3d.at<float>(1);
        kps3d[i].z = kp3d.at<float>(2);

    }

    vector<KeyPointInformation> &info = frame.kps.info;
    vector<KeyPoint3d> &kps3d_orig = frame.kps.kps3d;
    const Pose &pose = frame.pose;
    for (size_t i = 0; i < kps3d.size(); i++) {
        float &disparity = disparities[i];
        if (disparity < 0)
            continue;
        KalmanFilter &kf = info[i].seed.kf;

        float dist = abs(kps3d[i].x-kps3d_orig[i].x) +
                    abs(kps3d[i].y-kps3d_orig[i].y) +
                    abs(kps3d[i].z-kps3d_orig[i].z);

        float var = dist*((1/(disparity+0.5)-1/(disparity-0.5)));
        float var2 = var*var;
        float x_var = 10*var2/(fx*fx);
        float y_var = 10*var2/(fy*fy);
        float z_var = 10*var2;

        Vec3f var_vect(x_var, y_var, z_var);
        Vec3f angles(pose.pitch, pose.yaw, pose.roll);
        Matx33f rotation;

        Rodrigues(angles, rotation);

        var_vect = rotation * var_vect;

        kf.measurementNoiseCov.at<float>(0,0) = abs(var_vect(0));
        kf.measurementNoiseCov.at<float>(1,1) = abs(var_vect(1));
        kf.measurementNoiseCov.at<float>(2,2) = abs(var_vect(2));

        kf.predict();
        kf.correct((Mat_<float>(3,1) << kps3d[i].x, kps3d[i].y, kps3d[i].z));
        cout << "New measurement: " << kps3d[i].x << ","
            << kps3d[i].y << ","  << kps3d[i].z;
        cout << " New estimate: " << kf.statePost.at<float>(0) << "," <<
            kf.statePost.at<float>(1) << "," <<
            kf.statePost.at<float>(2);
        cout << " Old measurement: " << kps3d_orig[i].x << ","
            << kps3d_orig[i].y << "," << kps3d_orig[i].z << endl;

        cout << "Error: " << endl <<
            kf.errorCovPost.at<float>(0,0) << ", " <<
            kf.errorCovPost.at<float>(0,1) << ", " <<
            kf.errorCovPost.at<float>(0,2) << ", " << endl <<
            kf.errorCovPost.at<float>(1,0) << ", " <<
            kf.errorCovPost.at<float>(1,1) << ", " <<
            kf.errorCovPost.at<float>(1,2) << ", " << endl <<
            kf.errorCovPost.at<float>(2,0) << ", " <<
            kf.errorCovPost.at<float>(2,1) << ", " <<
            kf.errorCovPost.at<float>(2,2) << endl;
    }

}

void DepthFilter::calculate_disparities(Frame &frame, std::vector<float> &disparity)
{
    int window_size = camera_settings.window_size_depth_calculator;
    int window_before = window_size/2;
    int window_after = (window_size+1)/2;

    int search_x = camera_settings.search_x;
    int search_y = camera_settings.search_y;

    const Mat &left = frame.stereo_image.left[0];
    const Mat &right= frame.stereo_image.right[0];

    disparity.resize(frame.kps.kps2d.size());

    for (size_t i = 0; i < frame.kps.kps2d.size(); i++) {
        disparity[i] = -1;
        auto keypoint = frame.kps.kps2d[i];
        int x = static_cast<int>(keypoint.x);
        int y = static_cast<int>(keypoint.y);

        int32_t x11 = max<int>(0, x - window_before);
        int32_t x12 = min<int>(left.cols - 1, x + window_after);
        int32_t y11 = max<int>(0, y - window_before);
        int32_t y12 = min<int>(left.rows, y + window_after);

        // If we can't see the point at all
        if (x12 <= 0 || y12 <= 0 || x11 >= (left.cols-1) ||
                y11 >= (left.rows-1))
            continue;

        auto templ = left(Range(y11,y12), Range(x11, x12));

        int32_t x21 = max<int>(0, x - window_before);
        int32_t x22 = min<int>(left.cols - 1, x + window_after + search_x);
        int32_t y21 = max<int>(0, y - window_before - search_y);
        int32_t y22 = min<int>(left.rows - 1, y + window_after + search_y);

        if (x22 <= 0 || y22 <= 0 || x21 >= (left.cols-1) ||
                y21 >= (left.rows-1))
            continue;
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

        disparity[i] =  minPos;
    }
}
