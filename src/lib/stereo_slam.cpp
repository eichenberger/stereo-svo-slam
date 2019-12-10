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
    camera_settings(camera_settings), keyframe_manager(camera_settings),
    motion(0, 0, 0, 0, 0, 0)
{
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
    float cost = estimator.estimate_pose(previous_frame->pose, estimated_pose);
    END_MEASUREMENT("estimator");

    cout << "Cost after estimation: " << cost << endl;
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
    refiner.refine_pose(keyframe_manager, *frame);

    // Reproject keypoints with refined pose
//    project_keypoints(frame->pose, frame->kps.kps3d, camera_settings,
//            frame->kps.kps2d);

    END_MEASUREMENT("pose refinement");
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

void StereoSlam::new_image(const Mat &left, const Mat &right) {
    // Size window_size = Size(3, 3);
    Ptr<Frame> previous_frame = frame;
    frame = new Frame;

    START_MEASUREMENT();
    createImgPyramid(left, camera_settings.max_pyramid_levels, frame->stereo_image.left);
    createImgPyramid(right, camera_settings.max_pyramid_levels, frame->stereo_image.right);
    END_MEASUREMENT("Create pyramid");

    // Check if this is the first frame
    if (previous_frame.empty()) {
        Pose pose;

        frame->id = 0;
        pose.x = 0;
        pose.y = 0;
        pose.z = 0;
        pose.pitch = 0;
        pose.yaw = 0; // M_PI; //0;
        pose.roll = 0;
        frame->pose.set_pose(pose);

        keyframe = keyframe_manager.create_keyframe(*frame);
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

        Vec6f motion_applied = frame->pose.get_vector() + motion;
        frame->pose.set_vector(motion_applied);

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

            kp3d = updated_kps3d[i];
            frame->kps.kps3d[i] = kp3d;
        }
        END_MEASUREMENT("Filter update");
        project_keypoints(frame->pose, frame->kps.kps3d, camera_settings,
                frame->kps.kps2d);



        if (keyframe_manager.keyframe_needed(*frame)) {
            cout << "New keyframe is needed" << endl;
            START_MEASUREMENT();
            keyframe = keyframe_manager.create_keyframe(*frame);
            END_MEASUREMENT("Create new keyframe");
        }
    }

    trajectory.push_back(frame->pose.get_pose());

    if (!previous_frame.empty()) {
        const Vec6f current_pose(frame->pose.get_vector());
        const Vec6f previous_pose(previous_frame->pose.get_vector());
        motion = current_pose - previous_pose;

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
    this->keyframe_manager.get_keyframes(keyframes);
}

void StereoSlam::get_trajectory(std::vector<Pose> &trajectory)
{
    trajectory = this->trajectory;
}

