#include <vector>

#include "transform_keypoints.hpp"
#include "depth_calculator.hpp"
#include "pose_estimator.hpp"
#include "optical_flow.hpp"
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
    camera_settings(camera_settings), keyframe_inserter(camera_settings)
{
}

void StereoSlam::new_image(const Mat &left, const Mat &right) {
    Size window_size = Size(camera_settings.window_size_opt_flow,
            camera_settings.window_size_opt_flow);

    Ptr<Frame> previous_frame = frame;
    frame = new Frame;

    buildOpticalFlowPyramid(left, frame->stereo_image.left,
            window_size, camera_settings.max_pyramid_levels);

    buildOpticalFlowPyramid(right, frame->stereo_image.right,
            window_size, camera_settings.max_pyramid_levels);

    // Check if this is the first frame
    if (previous_frame.empty()) {
        keyframes.push_back(KeyFrame());
        keyframe = &keyframes.back();

        frame->id = 0;
        frame->pose.x = 0;
        frame->pose.y = 0;
        frame->pose.z = 0;
        frame->pose.pitch = 0;
        frame->pose.yaw = 0;
        frame->pose.roll = 0;

        keyframe_inserter.new_keyframe(*frame, *keyframe);
    }
    else {
        START_MEASUREMENT();
        frame->id = previous_frame->id + 1;
        PoseEstimator estimator(frame->stereo_image, previous_frame->stereo_image,
                previous_frame->kps, camera_settings);

        Pose estimated_pose;
        float cost = estimator.estimate_pose(previous_frame->pose, estimated_pose);
        END_MEASUREMENT("estimator");

        cout << "Cost after estimation: " << cost << endl;
        cout << "Pose after estimation: " <<
            estimated_pose.x << ", " << estimated_pose.y << ", " << estimated_pose.z << ", " <<
            estimated_pose.pitch << ", " << estimated_pose.yaw << ", " << estimated_pose.roll << endl;

        vector<KeyPoint2d> estimated_kps;
        project_keypoints(estimated_pose, previous_frame->kps.kps3d, camera_settings,
                estimated_kps);

        frame->kps.info = previous_frame->kps.info;

//        vector<float> diffs;
//        get_total_intensity_diff(keyframe->stereo_image.left[0],
//                frame->stereo_image.left[0], keyframe->kps.kps2d,
//                estimated_kps, camera_settings.window_size_opt_flow, diffs);
//        for (size_t i = 0; i < diffs.size(); i++) {
//            if (diffs[i] > 1000 ||
//                    (frame->kps.info[i].seed.a+5) < frame->kps.info[i].seed.b) {
//                frame->kps.info[i].seed.accepted = false;
//            }
//            else {
//                frame->kps.info[i].seed.accepted = true;
//            }
//        }

        OpticalFlow optical_flow;
        vector<float> err;

        START_MEASUREMENT();
        KeyPoints refined_kps = previous_frame->kps;
        optical_flow.calculate_optical_flow(keyframe->stereo_image,
                keyframe->kps.kps2d, frame->stereo_image, refined_kps.kps2d, err);

        END_MEASUREMENT("optical flow");

        for (size_t i = 0; i < refined_kps.kps2d.size(); i++) {
            if (err[i] > 1000) {
                refined_kps.kps2d[i].x = 10000000;
            }
        }

        START_MEASUREMENT();
        Pose refined_pose;
        PoseRefiner refiner(camera_settings);
        refiner.refine_pose(refined_kps, estimated_pose, refined_pose);

        END_MEASUREMENT("pose refinement");

        project_keypoints(refined_pose, keyframe->kps.kps3d, camera_settings,
                frame->kps.kps2d);

        frame->kps.kps3d = previous_frame->kps.kps3d;
        frame->pose = refined_pose;

        if (keyframe_inserter.keyframe_needed(*frame)) {
            cout << "New keyframe is needed" << endl;
            keyframes.push_back(KeyFrame());
            keyframe = &keyframes.back();
            keyframe_inserter.new_keyframe(*frame, *keyframe);
            frame->pose = keyframe->pose;
            frame->kps = keyframe->kps;
        }

        DepthFilter filter(keyframes, camera_settings);
        filter.update_depth(*frame);

        for (size_t i = 0; i < frame->kps.kps3d.size(); i++) {
            KeyPointInformation &info = frame->kps.info[i];
            KeyFrame &keyframe = keyframes[info.keyframe_id];
            KeyPoint3d &kp3d = keyframe.kps.kps3d[info.keypoint_index];

            kp3d.x = info.seed.kf.statePost.at<float>(0);
            kp3d.y = info.seed.kf.statePost.at<float>(1);
            kp3d.z = info.seed.kf.statePost.at<float>(2);
            frame->kps.kps3d[i] = kp3d;
        }
    }

    trajectory.push_back(frame->pose);

    if (!previous_frame.empty()) {
        previous_frame->kps.kps2d.clear();
        previous_frame->kps.kps3d.clear();
        previous_frame->kps.info.clear();
        previous_frame->stereo_image.left.clear();
        previous_frame->stereo_image.right.clear();
        previous_frame.release();
    }
}

void StereoSlam::get_keyframe(KeyFrame &keyframe)
{
    keyframe = *this->keyframe;
}

void StereoSlam::get_frame(Frame &frame)
{
    frame = *this->frame;
}

void StereoSlam::get_keyframes(std::vector<KeyFrame> &keyframes)
{
    keyframes = this->keyframes;
}

void StereoSlam::get_trajectory(std::vector<Pose> &trajectory)
{
    trajectory = this->trajectory;
}

