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

std::ostream& operator<<(std::ostream& os, const Vec3f& vector)
{
    os << vector(0) << "," << vector(1) << "," << vector(2);
    return os;
}

void DepthFilter::update_depth(Frame &frame, vector<KeyPoint3d> &updated_kps3d)
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

    Matx33f rot_mat(frame.pose.get_rotation_matrix());

    Vec3f translation(frame.pose.get_translation());
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
    const PoseManager &pose = frame.pose;
    Matx33f rotation(pose.get_rotation_matrix());
    for (size_t i = 0; i < kps3d.size(); i++) {
        float &disparity = disparities[i];
        if (disparity < 0)
            continue;

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

        // We rotation the standard deviation to the global coordinate system
        Vec3f deviation_vec(x_deviation, y_deviation, z_deviation);
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

        KeyFrame *keyframe = keyframe_manager.get_keyframe(info[i].keyframe_id);

        Vec3f p1(&kps3d_orig[i].x);
        Vec3f p2(&kps3d[i].x);
        Vec3f c1(keyframe->pose.get_translation());
        Vec3f c2(pose.get_translation());

        Vec3f line = p1-c1;
        Vec3f rot;
        rot[0] = -atan(line(0)/line(2));
        rot[1] = -atan(line(1)/line(2));
        rot[2] = 0;


        cout << "u/v: " << keyframe->kps.kps2d[info[i].keypoint_index].x <<
            ", " << keyframe->kps.kps2d[info[i].keypoint_index].y << endl;
        cout << "rot: " << rot(0) <<", " << rot(1) << "," << rot(2) << endl;

        // Now we rotate the standard deviation so that the z value will be
        // our deviation along the line from Cref to P
        Rodrigues(rot, rotation);
        deviation_vec = rotation*deviation_vec;

        Mat res = (Mat_<float>(3, 2) << p1(0)-c1(0), -(p2(0)-c2(0)),
                    p1(1)-c1(1), -(p2(1)-c2(1)),
                        p1(2)-c1(2), -(p2(2)-c2(2)));
        Vec3f y=c2-c1;
        Vec2f l;

        solve(res, y, l, DECOMP_SVD);

        Vec3f p1_stroke = c1+(l(0)*(p1-c1));

        cout << "p1: " << p1 << " p1': " << p1_stroke << " l0: " << l(0) << " l1: " << l(1) <<endl;
        cout << "deviation: " << deviation_vec(2) << endl;

        KalmanFilter &kf = info[i].kf;

        // We take the deviation along the line which is now the z part
        // and calculate the variance (deviation squared)
        kf.measurementNoiseCov.at<float>(0,0) = deviation_vec(2)*deviation_vec(2);

        cout << "Pre Error: " << endl <<
            kf.errorCovPre.at<float>(0,0) << ", " << endl;

        cout << "Post Error: " << endl <<
            kf.errorCovPost.at<float>(0,0) << ", " << endl;

        Mat old_l;
        old_l = kf.statePost;

        kf.predict();

        cout << "Pre Error after prediction: " << endl <<
            kf.errorCovPre.at<float>(0,0) << ", " << endl;

        cout << "Post Error after prediction: " << endl <<
            kf.errorCovPost.at<float>(0,0) << ", " << endl;

        Vec3f measurement(&kps3d[i].x);
        Mat new_l(1, 1, CV_32F, l(0));
        new_l = kf.correct(new_l);

        // Correct point. The old_l is the reference.
        Vec3f p1_corrected = c1+((new_l.at<float>(0)/old_l.at<float>(0))*(p1-c1));

        cout << "New measurement: " << kps3d[i].x << ","
            << kps3d[i].y << ","  << kps3d[i].z;

        kps3d[i].x = p1_corrected(0);
        kps3d[i].y = p1_corrected(1);
        kps3d[i].z = p1_corrected(2);

        cout << "New estimate: " << kps3d[i].x << ","
            << kps3d[i].y << ","  << kps3d[i].z;
        cout << " Old measurement: " << kps3d_orig[i].x << ","
            << kps3d_orig[i].y << "," << kps3d_orig[i].z << endl;

        cout << "Pre Error after correction: " << endl <<
            kf.errorCovPre.at<float>(0,0) << ", " << endl;

        cout << "Error after correction: " << endl <<
            kf.errorCovPost.at<float>(0,0) << ", " << endl;
        cout << endl;
    }

    updated_kps3d = kps3d;

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
