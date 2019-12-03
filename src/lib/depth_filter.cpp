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

    updated_kps3d = frame.kps.kps3d;

    vector<KeyPointInformation> &info = frame.kps.info;
    for (size_t i = 0; i < kps3d.size(); i++) {
        float &disparity = disparities[i];
        if (disparity < 0)
            continue;

        KeyFrame *keyframe = keyframe_manager.get_keyframe(info[i].keyframe_id);
        KeyPoint3d &kp3d_ref = keyframe->kps.kps3d[info[i].keypoint_index];
        Matx33f rotation(keyframe->pose.get_inv_rotation_matrix());
        Vec3f translation(keyframe->pose.get_translation());

        // We rotate the distances so that we get a z value with reference
        // current frame
        Vec3f kp3d(&kps3d[i].x);
        kp3d = kp3d - translation;
        kp3d = rotation*kp3d;

        Vec3f _kp3d_ref(&kp3d_ref.x);
        _kp3d_ref = _kp3d_ref - translation;
        _kp3d_ref = rotation*_kp3d_ref;

        float disp_ref = baseline/_kp3d_ref(2);
        float disp = baseline/kp3d(2);

        float pixel_distance = disp - disp_ref;

        cout << "Disparity reference: " << disp_ref<< endl;
        cout << "Disparity new: " << disparity << endl;
        cout << "Disparity transformed: " << disp << endl;
        cout << "Pixel distance: " << pixel_distance << endl;

        // Standard deviation can be +- 3 pixel
        float deviation = 1;
        // Let's define a point as inliner if it is within 5*deviation which should
        // match for 99% of all points
        // rotation can introduce a negative sign for deviation again therfore abs
        if (abs(pixel_distance) > 5*deviation) {
            info[i].outlier_count++;
            continue;
        }
        else
            info[i].inlier_count++;

        Matx33f rotation_frame(frame.pose.get_inv_rotation_matrix());
        Vec3f translation_frame(frame.pose.get_translation());
        Vec3f _kp3d_ref_cur(&kp3d_ref.x);
        _kp3d_ref_cur = _kp3d_ref_cur - translation_frame;
        _kp3d_ref_cur = rotation_frame*_kp3d_ref_cur;

        // Take the z distance of the current estimated 3d point form the
        // frame and from the reference frame
        float scaled_deviation = _kp3d_ref_cur(2)/_kp3d_ref(2) * deviation;

        cout << "Scaled deviation: " << scaled_deviation << endl;


#if 1
        // TODO: Why is z distance of points increasing and not decreasing?
        // They should be at 7.5 meters but increase up to 9.2m???
        Vec3f c1(keyframe->pose.get_translation());
        Vec3f c2(frame.pose.get_translation());

        cout << "Frame diff: " << abs((c1(0) -c2(0))) << endl;
        // TODO: We need to have at least a small diff in x direction
        if (abs((c1(0) -c2(0))) < 0.2)
            continue;

        if (info[i].ignore_completely || info[i].ignore_during_refinement)
            continue;

        KeyPoint2d &kp2d = frame.kps.kps2d[i];
        Vec3f p1(&kp3d_ref.x);
        Vec3f p2;
        p2[0] = kp2d.x-cx;
        p2[1] = kp2d.y-cy;
        p2[2] = fx;


        // This is the a matrix we need to solve to get the new l
        // where the two lines from reference and new frame intersect
        Mat a = (Mat_<float>(3, 2) << p1(0)-c1(0), -(p2(0)-c2(0)),
                    p1(1)-c1(1), -(p2(1)-c2(1)),
                        p1(2)-c1(2), -(p2(2)-c2(2)));
        Vec3f y=c2-c1;
        Vec2f l;

        solve(a, y, l, DECOMP_SVD);

        Vec3f point1 = c1 + l(0)*(p1-c1);
        Vec3f point2 = c2 + l(1)*(p2-c2);

        cout << "P1: " << p1(0) << "," << p1(1) << "," << p1(2) << endl;
        cout << "Point1: " << point1(0) << "," << point1(1) << "," << point1(2) << endl;
        cout << "P2: " << p2(0) << "," << p2(1) << "," << p2(2) << endl;
        cout << "Point2: " << point2(0) << "," << point2(1) << "," << point2(2) << endl;

        // Vec3f p1_stroke = c1+(l(0)*(p1-c1));

        KalmanFilter &kf = info[i].kf;

        // We take the deviation along the line which is now the z part
        // and calculate the variance (deviation squared)
        kf.measurementNoiseCov.at<float>(0,0) = 10.0;

        //cout << "Pre Error: " << endl <<
        //    kf.errorCovPre.at<float>(0,0) << ", " << endl;

        //cout << "Post Error: " << endl <<
        //    kf.errorCovPost.at<float>(0,0) << ", " << endl;


        kf.predict();

        //cout << "Pre Error after prediction: " << endl <<
        //    kf.errorCovPre.at<float>(0,0) << ", " << endl;

        //cout << "Post Error after prediction: " << endl <<
        //    kf.errorCovPost.at<float>(0,0) << ", " << endl;

        Vec3f measurement(&kps3d[i].x);
        Mat new_l(1, 1, CV_32F, l(0));
        new_l = kf.correct(new_l);

        // Correct point. The old_l is the reference.
        Vec3f p1_corrected = c1+(new_l.at<float>(0)*(p1-c1));

        cout << " l0: " << l(0) << " l1: " << l(1) <<endl;
        cout << " updated l: " << new_l.at<float>(0) << endl;

        updated_kps3d[i].x = p1_corrected(0);
        updated_kps3d[i].y = p1_corrected(1);
        updated_kps3d[i].z = p1_corrected(2);

        cout << "New estimate: " << updated_kps3d[i].x << ","
            << updated_kps3d[i].y << ","  << updated_kps3d[i].z;
        cout << " Old measurement: " << kp3d_ref.x << ","
            << kp3d_ref.y << "," << kp3d_ref.z << endl;

        kf.statePost.at<float>(0) = 1.0;
        //cout << "Pre Error after correction: " << endl <<
        //    kf.errorCovPre.at<float>(0,0) << ", " << endl;

        //cout << "Error after correction: " << endl <<
        //    kf.errorCovPost.at<float>(0,0) << ", " << endl;
        //cout << endl;
#endif
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
