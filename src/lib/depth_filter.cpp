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



DepthFilter::DepthFilter(KeyFrameManager &keyframe_manager, const CameraSettings &camera_settings) :
    keyframe_manager(keyframe_manager), camera_settings(camera_settings)
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

    Vec3f angles(&frame.pose.pitch);
    Matx33f rot_mat;
    Rodrigues(angles, rot_mat);

    Vec3f translation(&frame.pose.x);
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
        KalmanFilter &kf = info[i].kf;

        float dist_x = kps3d[i].x-kps3d_orig[i].x;
        float dist_y = kps3d[i].y-kps3d_orig[i].y;
        float dist_z = kps3d[i].z-kps3d_orig[i].z;

        // Variance can be +- 0.5 pixel
        float deviation = abs((baseline/(disparity+0.5)-baseline/(disparity-0.5)));

        // disparity can't be negaritve
        if (disparity < 0.5)
            deviation = abs((1/(disparity+0.5)));
        // This should probably be dependent on where it is in 2d
        float x_deviation = abs((deviation*kps2d[i].x-camera_settings.cx)/camera_settings.fx);
        float y_deviation = abs((deviation*kps2d[i].y-camera_settings.cy)/camera_settings.fy);
        float z_deviation = deviation;

        Vec3f deviation_vec(x_deviation, y_deviation, z_deviation);
        Vec3f angles(&pose.pitch);
        Matx33f rotation;

        Rodrigues(angles, rotation);

        deviation_vec = rotation * deviation_vec;

        cout << "Deviation: " << deviation_vec(0) << ", " <<
            deviation_vec(1) << ", " << deviation_vec(2) << endl;
        cout << "Dist: " << dist_x << ", " << dist_y << ", " << dist_z << endl;

        // Let's define a point as inliner if it is within 2*deviation which should
        // match for 95% of all points
        // rotation can introduce a negative sign for deviation again therfore abs
        if (abs(dist_x) > 2*abs(deviation_vec(0)) ||
                abs(dist_y) > 2*abs(deviation_vec(1)) ||
                abs(dist_z) > 2*abs(deviation_vec(2))) {
            info[i].outlier_count++;
            continue;
        }
        else
            info[i].inlier_count++;

        kf.measurementNoiseCov.at<float>(0,0) = deviation_vec(0)*deviation_vec(0);
        kf.measurementNoiseCov.at<float>(1,1) = deviation_vec(1)*deviation_vec(1);
        kf.measurementNoiseCov.at<float>(2,2) = deviation_vec(2)*deviation_vec(2);

        kf.predict();
        kf.correct((Mat_<float>(3,1) << kps3d[i].x, kps3d[i].y, kps3d[i].z));
        cout << "New measurement: " << kps3d[i].x << ","
            << kps3d[i].y << ","  << kps3d[i].z;
        cout << " New estimate: " << kf.statePost.at<float>(0) << "," <<
            kf.statePost.at<float>(1) << "," <<
            kf.statePost.at<float>(2);
        cout << " Old measurement: " << kps3d_orig[i].x << ","
            << kps3d_orig[i].y << "," << kps3d_orig[i].z << endl;

//        cout << "Error: " << endl <<
//            kf.errorCovPost.at<float>(0,0) << ", " <<
//            kf.errorCovPost.at<float>(0,1) << ", " <<
//            kf.errorCovPost.at<float>(0,2) << ", " << endl <<
//            kf.errorCovPost.at<float>(1,0) << ", " <<
//            kf.errorCovPost.at<float>(1,1) << ", " <<
//            kf.errorCovPost.at<float>(1,2) << ", " << endl <<
//            kf.errorCovPost.at<float>(2,0) << ", " <<
//            kf.errorCovPost.at<float>(2,1) << ", " <<
//            kf.errorCovPost.at<float>(2,2) << endl;
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
