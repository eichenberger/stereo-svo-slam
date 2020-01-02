#include <vector>

#include "transform_keypoints.hpp"
#include "depth_calculator.hpp"
#include "pose_estimator.hpp"
#include "pose_refinement.hpp"
#include "depth_filter.hpp"
#include "image_comparison.hpp"

#include "stereo_slam.hpp"

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

StereoSlam::StereoSlam(const CameraSettings &camera_settings) :
    camera_settings(camera_settings), keyframe_manager(this->camera_settings),
    motion(0, 0, 0, 0, 0, 0)
{
    kf.init(12,12);
    setIdentity(kf.transitionMatrix);
    setIdentity(kf.measurementMatrix);
    // We don't trust the prediction becasue the measurement is much more accurate
    setIdentity(kf.processNoiseCov, Scalar::all(100.0));
    setIdentity(kf.errorCovPost, Scalar::all(1.0));
    kf.statePost = Mat::zeros(12, 1, CV_32F);
    time_measure.start();
}

void StereoSlam::remove_outliers(Frame *frame)
{
    KeyPoints updated;
    // Set to maximum size
    for (size_t i = 0; i < frame->kps.kps2d.size(); i++) {
        if (frame->kps.info[i].ignore_completely)
            continue;
        updated.kps2d.push_back(frame->kps.kps2d[i]);
        updated.kps3d.push_back(frame->kps.kps3d[i]);
        updated.info.push_back(frame->kps.info[i]);
    }

    frame->kps = updated;
}

void StereoSlam::estimate_pose(Frame *previous_frame)
{
    PoseEstimator estimator(frame->stereo_image, previous_frame->stereo_image,
            previous_frame->kps, camera_settings);

    PoseManager estimated_pose;
    START_MEASUREMENT();
    float cost = estimator.estimate_pose(frame->pose, estimated_pose);
    END_MEASUREMENT("estimator");

    cout << "Estimation cost (intensity diff): " << cost << endl;
    cout << "Pose after estimation: " << estimated_pose << endl;

    vector<KeyPoint2d> estimated_kps;
    START_MEASUREMENT();
    project_keypoints(estimated_pose, previous_frame->kps.kps3d, camera_settings,
            estimated_kps);
    END_MEASUREMENT("project keypoints");

    frame->pose = estimated_pose;
    frame->kps.info = previous_frame->kps.info;
    frame->kps.kps3d = previous_frame->kps.kps3d;
    frame->kps.kps2d = estimated_kps;

    START_MEASUREMENT();
    PoseRefiner refiner(camera_settings);
    cost = refiner.refine_pose(keyframe_manager, *frame);

    END_MEASUREMENT("pose refinement");

    cout << "Refinement cost (reprojection error): " << cost << endl;
    cout << "Pose after refinement: " << frame->pose << endl;
}


static void halfSample(const cv::Mat& in, cv::Mat& out)
{
    assert( in.rows/2==out.rows && in.cols/2==out.cols);
    assert( in.type()==CV_8U && out.type()==CV_8U);

// OMP variant is slower...
// #pragma omp parallel for default(none) shared(out, in)
    for (int j = 0; j < out.rows; j++) {
        int y = j<<1;
        const uint8_t* upper_in= in.ptr<uint8_t>(y);
        const uint8_t* lower_in= in.ptr<uint8_t>(y+1);
        uint8_t* current_out = out.ptr<uint8_t>(j);
        for (int i = 0, x = 0; i < out.cols; i++, x+=2) {
            current_out[i] = (upper_in[x] + upper_in[x+1] + lower_in[x] + lower_in[x+1])/4;
        }
    }
}


static void createImgPyramid(const cv::Mat& img_level_0, int n_levels, vector<Mat>& pyr)
{
    pyr.resize(n_levels);
    pyr[0] = img_level_0;
    for(int i=1; i<n_levels; i++)
    {
        pyr[i] = Mat(pyr[i-1].rows/2, pyr[i-1].cols/2, CV_8U);
        halfSample(pyr[i-1], pyr[i]);
    }
}

void StereoSlam::new_image(const Mat &left, const Mat &right, const float time_stamp) {
    // Size window_size = Size(3, 3);
    Ptr<Frame> previous_frame = frame;
    frame = new Frame;

    // Add a pseudo timestamp
    frame->time_stamp = time_stamp;

    START_MEASUREMENT();
    // Unfortunately we can't use buildOpticalFlowPyramid because
    // it does some heavy gauss filtering which doesen't work well
    // with our pose estimator
    createImgPyramid(left, camera_settings.max_pyramid_levels, frame->stereo_image.left);
    createImgPyramid(right, 1, frame->stereo_image.right);
    Size patch_size(camera_settings.window_size_opt_flow,
            camera_settings.window_size_opt_flow);
    buildOpticalFlowPyramid(left, frame->stereo_image.opt_flow, patch_size, 2);
    END_MEASUREMENT("Create pyramid");

    // Check if this is the first frame
    if (previous_frame.empty()) {
        Pose pose;

        frame->id = 0;
        pose.x = 0;
        pose.y = 0;
        pose.z = 0;
        pose.rx = 0;
        pose.ry = 0; // M_PI; //0;
        pose.rz = 0;
        frame->pose.set_pose(pose);

        keyframe = keyframe_manager.create_keyframe(*frame);

        // Never ignore the keypoints for the first frame
        for (auto &info:frame->kps.info)
            info.ignore_temporary = false;
#if 0
        cout << "void createVector(std::vector<Vector2d> &pts2d, std::vector<Vector3d> &pts3d) { " << endl;
        for (size_t i = 0; i < frame->kps.kps2d.size(); i++) {
            cout << "    pts2d.push_back(Vector2d(" << frame->kps.kps2d[i].x << "," <<
                frame->kps.kps2d[i].y << "));" << endl;
            cout << "    pts3d.push_back(Vector3d(" << frame->kps.kps3d[i].x << "," <<
                frame->kps.kps3d[i].y << "," << frame->kps.kps3d[i].z << "));" << endl;
        }
        cout << "}" << endl;
        exit(0);
#endif
    }
    else {
        frame->id = previous_frame->id + 1;

        for (size_t i = 0; i < frame->kps.info.size(); i++) {
            KeyPointInformation &info = frame->kps.info[i];
            if (info.keyframe_id == 0 && info.keypoint_index == 143) {
                cout << "Fucking inlier count: " << info.inlier_count << "," <<
                    info.outlier_count << endl;
            }
        }

        Pose predicted_pose;
        predicted_pose.x = kf.statePre.at<float>(0);
        predicted_pose.y = kf.statePre.at<float>(1);
        predicted_pose.z = kf.statePre.at<float>(2);
        predicted_pose.rx = kf.statePre.at<float>(3);
        predicted_pose.ry = kf.statePre.at<float>(4);
        predicted_pose.rz = kf.statePre.at<float>(5);

        cout << "Previous pose: " << previous_frame->pose;
        frame->pose.set_pose(predicted_pose);
        cout << " Predicted pose: " << frame->pose << endl;

        remove_outliers(previous_frame);
        estimate_pose(previous_frame);

        DepthFilter filter(keyframe_manager, camera_settings);

        vector<KeyPoint3d> updated_kps3d;

        START_MEASUREMENT();
        filter.update_depth(*frame, updated_kps3d);

        for (size_t i = 0; i < frame->kps.kps3d.size(); i++) {
            KeyPointInformation &info = frame->kps.info[i];
            KeyFrame *keyframe = keyframe_manager.get_keyframe(info.keyframe_id);
            KeyPoint3d &kp3d = keyframe->kps.kps3d[info.keypoint_index];

            // If we count more inlier than outlier it's probably
            // a complete outlier then...
            if (info.outlier_count > info.inlier_count)
                info.ignore_completely = true;

            if (info.inlier_count > info.outlier_count)
                info.ignore_temporary = false;

            kp3d = updated_kps3d[i];
            frame->kps.kps3d[i] = kp3d;
            keyframe->kps.info[i].inlier_count = info.inlier_count;
            keyframe->kps.info[i].outlier_count = info.outlier_count;

        }
        END_MEASUREMENT("Filter update");
        project_keypoints(frame->pose, frame->kps.kps3d, camera_settings,
                frame->kps.kps2d);

        if (keyframe_manager.keyframe_needed(*frame)) {
            cout << "New keyframe is needed" << endl;
            START_MEASUREMENT();
            keyframe = keyframe_manager.create_keyframe(*frame);
            END_MEASUREMENT("Create new keyframe");

            size_t i = 0;
            for (auto &info:frame->kps.info)
                if (!info.ignore_temporary)
                    i++;
            // We have a problem now we need a backup
            if (i < frame->kps.info.size()/4) {
                for (auto &info:frame->kps.info)
                    info.ignore_temporary = false;
            }
        }
    }


    if (!previous_frame.empty()) {
        double dt = frame->time_stamp - previous_frame->time_stamp;
        const Vec6f current_pose(frame->pose.get_vector());
        const Vec6f previous_pose(previous_frame->pose.get_vector());
        motion = (current_pose - previous_pose)/dt;
        Vec6f pose_variance(0.1,0.1,0.1,0.1,0.1,0.1);
        Vec6f motion_variance(1.0,1.0,1.0,1.0,1.0,1.0);
        Pose filtered_pose = update_pose(frame->pose.get_pose(), motion, pose_variance, motion_variance, 0.0);

        frame->pose.set_pose(filtered_pose);

        previous_frame->kps.kps2d.clear();
        previous_frame->kps.kps3d.clear();
        previous_frame->kps.info.clear();
        previous_frame->stereo_image.left.clear();
        previous_frame->stereo_image.right.clear();
        previous_frame->stereo_image.opt_flow.clear();
        previous_frame.release();
    }

    trajectory.push_back(frame->pose.get_pose());
}

void StereoSlam::get_keyframe(KeyFrame &keyframe)
{
    keyframe = *this->keyframe;
}

bool StereoSlam::get_frame(Frame &frame)
{
    if (this->frame == nullptr)
        return false;
    frame = *this->frame;
    return true;
}

void StereoSlam::get_keyframes(std::vector<KeyFrame> &keyframes)
{
    this->keyframe_manager.get_keyframes(keyframes);
}

void StereoSlam::get_trajectory(std::vector<Pose> &trajectory)
{
    trajectory = this->trajectory;
}

Pose StereoSlam::update_pose(const Pose &pose, const Vec6f &speed,
        const Vec6f &pose_variance, const Vec6f &speed_variance, double dt)
{
    //cout << "Previous state: ";
    //for (size_t i = 0; i < 12; i++) {
    //    cout << kf.statePost.at<float>(i, 0) << ",";
    //}
    //cout << endl;

    //cout << current_time << "-" << last_update << ": " << dt << endl;
    kf.transitionMatrix.at<float>(0, 6) = dt;
    kf.transitionMatrix.at<float>(1, 7) = dt;
    kf.transitionMatrix.at<float>(2, 8) = dt;
    kf.transitionMatrix.at<float>(3, 9) = dt;
    kf.transitionMatrix.at<float>(4, 10) = dt;
    kf.transitionMatrix.at<float>(5, 11) = dt;

    kf.predict();

//    cout << "State Pre: " << endl << kf.statePre << endl;

    kf.measurementNoiseCov.at<float>(0, 0) = pose_variance(0);
    kf.measurementNoiseCov.at<float>(1, 1) = pose_variance(1);
    kf.measurementNoiseCov.at<float>(2, 2) = pose_variance(2);
    kf.measurementNoiseCov.at<float>(3, 3) = pose_variance(3);
    kf.measurementNoiseCov.at<float>(4, 4) = pose_variance(4);
    kf.measurementNoiseCov.at<float>(5, 5) = pose_variance(5);
    kf.measurementNoiseCov.at<float>(6, 6) = speed_variance(0);
    kf.measurementNoiseCov.at<float>(7, 7) = speed_variance(1);
    kf.measurementNoiseCov.at<float>(8, 8) = speed_variance(2);
    kf.measurementNoiseCov.at<float>(9, 9) = speed_variance(3);
    kf.measurementNoiseCov.at<float>(10, 10) = speed_variance(4);
    kf.measurementNoiseCov.at<float>(11, 11) = speed_variance(5);

    Matx<float, 12, 1> measurement(
            pose.x,
            pose.y,
            pose.z,
            pose.rx,
            pose.ry,
            pose.rz,
            speed(0),
            speed(1),
            speed(2),
            speed(3),
            speed(4),
            speed(5));

    kf.correct(Mat(measurement));

    Pose filtered_pose;
    filtered_pose.x = kf.statePost.at<float>(0);
    filtered_pose.y = kf.statePost.at<float>(1);
    filtered_pose.z = kf.statePost.at<float>(2);
    filtered_pose.rx = kf.statePost.at<float>(3);
    filtered_pose.ry = kf.statePost.at<float>(4);
    filtered_pose.rz = kf.statePost.at<float>(5);

    //cout << "Post error cov: " << endl << kf.errorCovPost << endl;
    //cout << "dt: " << dt << endl;
    //cout << "After correction: " << endl << kf.statePost << endl;

    return filtered_pose;
}

double StereoSlam::get_current_time() {
    // TODO: This is a hack because we can't call getTimeSec without stop...
    time_measure.stop();
    double current_time = time_measure.getTimeSec();
    time_measure.start();
    return current_time;
}
