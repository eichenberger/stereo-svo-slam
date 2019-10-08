#include <vector>

#include "transform_keypoints.hpp"
#include "depth_calculator.hpp"
#include "pose_estimator.hpp"
#include "optical_flow.hpp"
#include "pose_refinement.hpp"

#include "stereo_slam.hpp"

using namespace cv;
using namespace std;

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

    if (previous_frame.empty()) {
        keyframe = new KeyFrame();
        keyframe_inserter.new_keyframe(*frame, *keyframe);
        frame->pose.x = 0;
        frame->pose.y = 0;
        frame->pose.z = 0;
        frame->pose.pitch = 0;
        frame->pose.yaw = 0;
        frame->pose.roll = 0;
    }
    else {
        PoseEstimator estimator(frame->stereo_image, previous_frame->stereo_image,
                previous_frame->kps, camera_settings);

        Pose estimated_pose;
        float cost = estimator.estimate_pose(previous_frame->pose, estimated_pose);

        cout << "Cost after estimation: " << cost << endl;
        cout << "Pose after estimation: " <<
            estimated_pose.x << ", " << estimated_pose.y << ", " << estimated_pose.z << ", " <<
            estimated_pose.pitch << ", " << estimated_pose.yaw << ", " << estimated_pose.roll << endl;

        vector<KeyPoint2d> estimated_kps;
        project_keypoints(estimated_pose, previous_frame->kps.kps3d, camera_settings,
                estimated_kps);


        OpticalFlow optical_flow;
        vector<float> err;

        // TODO: This is probably not a deep copy!
        KeyPoints refined_kps = previous_frame->kps;
        optical_flow.calculate_optical_flow(keyframe->stereo_image,
                keyframe->kps.kps2d, frame->stereo_image, refined_kps.kps2d, err);

        for (size_t i = 0; i < refined_kps.kps2d.size(); i++) {
            if (err[i] > 1000) {
                refined_kps.kps2d[i].x = 10000000;
            }
        }

        Pose refined_pose;
        PoseRefiner refiner(camera_settings);
        refiner.refine_pose(refined_kps, estimated_pose, refined_pose);

        project_keypoints(refined_pose, keyframe->kps.kps3d, camera_settings,
                frame->kps.kps2d);

        frame->kps.kps3d = previous_frame->kps.kps3d;
        frame->kps.info = previous_frame->kps.info;
        frame->pose = refined_pose;

        if (keyframe_inserter.keyframe_needed(*frame)) {
            cout << "New keyframe is needed" << endl;
            keyframe = new KeyFrame;
            keyframe_inserter.new_keyframe(*frame, *keyframe);
            frame->pose = keyframe->pose;
            frame->kps = keyframe->kps;
        }
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
