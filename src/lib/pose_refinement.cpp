#include <vector>
#include <cassert>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include "pose_refinement.hpp"
#include "transform_keypoints.hpp"
#include "image_comparison.hpp"
#include "optical_flow.hpp"

//#define SUPER_VERBOSE 1

using namespace cv;
using namespace std;

// We could use one of the opencv solver. However, they don't really fit
class PoseRefinerCallback : public MinProblemSolver::Function
{
public:
    PoseRefinerCallback(const vector<KeyPoint2d> &keypoints2d,
                        const vector<KeyPoint3d> &keypoints3d,
                        const vector<KeyPointInformation> &keypoint_information,
                        const CameraSettings &camera_settings);
    //bool compute(InputArray param, OutputArray err, OutputArray J) const;
    double calc(const double *x) const;
    int getDims() const { return 6; }
    void getGradient(const double *x, double *grad);


private:
    const vector<KeyPoint2d> &keypoints2d;
    const vector<KeyPoint3d> &keypoints3d;
    const vector<KeyPointInformation> &keypoint_information;
    const CameraSettings &camera_settings;
};


PoseRefiner::PoseRefiner(const CameraSettings &camera_settings) :
    camera_settings(camera_settings)
{
}

static void exponential_map(const Mat &twist, Mat &pose)
{
    Mat v(3, 1, CV_64F, (void*)twist.ptr(0));
    Mat w(3, 1, CV_64F, (void*)twist.ptr(3));

    twist.copyTo(pose);
    // The angles don't change. See robotics vision and control
    // page 53 for more details
    Mat w_skew = (Mat_<double>(3,3) <<
            0, -w.at<double>(2), w.at<double>(1),
            w.at<double>(2), 0, -w.at<double>(0),
            -w.at<double>(1), w.at<double>(0), 0);
    float _norm = 1.0;
    Mat _eye = Mat::eye(3,3, CV_64F);
    Mat translation = Mat(3,1, CV_64F, pose.ptr<double>(0));

    // Take closed form solution from robotics vision and control page 53
    // Note: This exponential map doesn't give the exact same value as expm
    // from Matlab or Numpy. It is different up to a scaling. It seems that
    // expm uses norm set to 1.0. However, this differs from the closed form
    // solution written in robotics vision and control.
    translation = (_eye*_norm + (1-cos(_norm))*w_skew+(_norm-sin(_norm))*(w_skew*w_skew))*v;
}

float PoseRefiner::refine_pose(KeyFrameManager &keyframe_manager,
            Frame &frame)
{
    OpticalFlow optical_flow(camera_settings);

    map<int, vector<KeyPoint2d>> reference;
    map<int, vector<KeyPoint2d>> active;
    map<int, vector<float>> err;

    // Split the keypoints acording to keyframe where it was first seen
    for (size_t i = 0; i < frame.kps.info.size(); i++) {
        KeyPointInformation &info = frame.kps.info[i];

        KeyFrame *keyframe = keyframe_manager.get_keyframe(info.keyframe_id);

        vector<KeyPoint2d> &ref_keypoints = reference[info.keyframe_id];
        vector<KeyPoint2d> &active_keypoints= active[info.keyframe_id];

        ref_keypoints.push_back(keyframe->kps.kps2d[info.keypoint_index]);
        active_keypoints.push_back(frame.kps.kps2d[i]);
    }

    // Do optical flow with each ref image <-> frame set
    for (auto itr : reference) {
        KeyFrame *keyframe = keyframe_manager.get_keyframe(itr.first);
        vector<KeyPoint2d> &active_keypoints = active[itr.first];
        vector<float> &_err = err[itr.first];

        optical_flow.calculate_optical_flow(keyframe->stereo_image,
                itr.second, frame.stereo_image, active_keypoints, _err);
    }

    // Add refined keypoints back to the original array, start with last entry
    // to allow pop_back
    for (size_t i = frame.kps.kps2d.size(); i > 0; i--) {
        size_t j = i-1;
        KeyPointInformation &info = frame.kps.info[j];
        KeyPoint2d &kp2d = frame.kps.kps2d[j];
        KeyPoint2d &new_estimate = active[info.keyframe_id].back();

        float _err = err[info.keyframe_id].back();
        err[info.keyframe_id].pop_back();

        float diff = (kp2d.x - new_estimate.x)*(kp2d.x - new_estimate.x) +
            (kp2d.y - new_estimate.y)*(kp2d.y - new_estimate.y);
        // don't move more than 9px!
        if (diff > 81 || _err > 1000) {
            info.ignore_during_refinement = true;
        }
        else {
            info.ignore_during_refinement = false;
        }
        kp2d = new_estimate;
        active[info.keyframe_id].pop_back();
    }

    Mat result;
    frame.stereo_image.left[0].copyTo(result);
    cvtColor(result, result,  COLOR_GRAY2RGB);
    for (size_t i = 0; i < frame.kps.kps2d.size(); i++) {
        KeyPointInformation &info = frame.kps.info[i];
        KeyPoint2d &kp2d = frame.kps.kps2d[i];
        Scalar color (info.color.r, info.color.g, info.color.b);
        int msize = 10;
        if (frame.kps.info[i].ignore_during_refinement) {
            msize=5;
        }
        int marker = MARKER_CROSS;
        Point kp = Point(kp2d.x, kp2d.y);
        cv::drawMarker(result, kp, color, marker, msize);
    }

    imshow("bla", result);
    Pose refined_pose;
    float ret = update_pose(frame.kps, frame.pose, refined_pose);
    frame.pose = refined_pose;

    return ret;
}

#if 1
float PoseRefiner::update_pose(const KeyPoints &keypoints,
        const Pose &estimated_pose,
        Pose &refined_pose)
{
    const vector<KeyPoint2d> &keypoints2d = keypoints.kps2d;
    const vector<KeyPoint3d> &keypoints3d = keypoints.kps3d;
    const vector<KeyPointInformation> &keypoint_information = keypoints.info;

    refined_pose = estimated_pose;
    Mat _pose = Mat::zeros(6, 1, CV_64FC1);
    double *x0 = _pose.ptr<double>(0);

    x0[0] = estimated_pose.x;
    x0[1] = estimated_pose.y;
    x0[2] = estimated_pose.z;
    x0[3] = estimated_pose.pitch;
    x0[4] = estimated_pose.yaw;
    x0[5] = estimated_pose.roll;

    Ptr<MinProblemSolver::Function> callback = new PoseRefinerCallback(keypoints2d,
            keypoints3d, keypoint_information, camera_settings);
    Ptr<DownhillSolver> solver = DownhillSolver::create(callback);
    Mat step = Mat::ones(6,1, CV_64F);
    step *= 0.01;
    solver->setInitStep(step);

    solver->minimize(_pose);

    refined_pose.x = x0[0];
    refined_pose.y = x0[1];
    refined_pose.z = x0[2];
    refined_pose.pitch = x0[3];
    refined_pose.yaw = x0[4];
    refined_pose.roll = x0[5];

    cout << "estimated pose: " << estimated_pose.x << "," << estimated_pose.y << ","  << estimated_pose.z << "," << estimated_pose.pitch << ","  << estimated_pose.yaw << "," << estimated_pose.roll << endl;
    cout << "refined pose: " << refined_pose.x << "," << refined_pose.y << ","  << refined_pose.z << "," << refined_pose.pitch << ","  << refined_pose.yaw << "," << refined_pose.roll << endl;

    vector<KeyPoint2d> projected_keypoints2d;
    project_keypoints(refined_pose, keypoints3d, camera_settings, projected_keypoints2d);
    Mat _projected_keypoints2d(2, projected_keypoints2d.size(), CV_32F, &projected_keypoints2d[0].x);
    Mat _keypoints2d(2, keypoints2d.size(), CV_32F, (void*)&keypoints2d[0].x);

    Mat err;
    absdiff(_projected_keypoints2d, _keypoints2d, err);

    cout << "Error after optimization: " << sum(err)[0] << endl;

    return sum(err)[0];
}

#else
float PoseRefiner::update_pose(const KeyPoints &keypoints,
        const Pose &estimated_pose,
        Pose &refined_pose)
{
    const vector<KeyPoint2d> &keypoints2d = keypoints.kps2d;
    const vector<KeyPoint3d> &keypoints3d = keypoints.kps3d;
    const vector<KeyPointInformation> &keypoint_information = keypoints.info;

    Mat x0 = (Mat_<double>(6,1) <<
        estimated_pose.x, estimated_pose.y, estimated_pose.z,
        estimated_pose.pitch, estimated_pose.yaw, estimated_pose.roll);

    size_t maxIter = 50;
    float k = 0.1;

    Ptr<MinProblemSolver::Function> solver_callback = new PoseRefinerCallback(keypoints2d,
            keypoints3d, keypoint_information, camera_settings);

    double prev_cost = solver_callback->calc(x0.ptr<double>(0));
    for (size_t i = 0; i < maxIter; i++) {
        Mat gradient(6,1,CV_64F);
        solver_callback->getGradient(x0.ptr<double>(0),
                gradient.ptr<double>(0));
        for (;i < maxIter; i++) {
            Mat x = x0 + (k*gradient);
            double new_cost = solver_callback->calc(x.ptr<double>(0));
            if (new_cost < prev_cost) {
                x0 = x;
                prev_cost = new_cost;
                break;
            }
            else if (fabs(new_cost - prev_cost) < 0.0001) {
                i = maxIter;
                break;
            }
            else {
                k /= 2;
            }
        }
    }

    refined_pose.x = x0.at<double>(0);
    refined_pose.y = x0.at<double>(1);
    refined_pose.z = x0.at<double>(2);
    refined_pose.pitch = x0.at<double>(3);
    refined_pose.yaw = x0.at<double>(4);
    refined_pose.roll = x0.at<double>(5);

    vector<KeyPoint2d> projected_keypoints2d;
    project_keypoints(refined_pose, keypoints3d, camera_settings, projected_keypoints2d);
    Mat _projected_keypoints2d(2, projected_keypoints2d.size(), CV_32F, &projected_keypoints2d[0].x);
    Mat _keypoints2d(2, keypoints2d.size(), CV_32F, (void*)&keypoints2d[0].x);

    Mat err;
    absdiff(_projected_keypoints2d, _keypoints2d, err);

    cout << "estimated pose: " << estimated_pose.x << "," << estimated_pose.y << ","  << estimated_pose.z << "," << estimated_pose.pitch << ","  << estimated_pose.yaw << "," << estimated_pose.roll << endl;
    cout << "refined pose: " << refined_pose.x << "," << refined_pose.y << ","  << refined_pose.z << "," << refined_pose.pitch << ","  << refined_pose.yaw << "," << refined_pose.roll << endl;

    cout << "Error after optimization: " << sum(err)[0] << endl;

    return prev_cost;

}

#endif

PoseRefinerCallback::PoseRefinerCallback(const vector<KeyPoint2d> &keypoints2d,
        const vector<KeyPoint3d> &keypoints3d,
        const vector<KeyPointInformation> &keypoint_information,
        const CameraSettings &camera_settings):
    keypoints2d(keypoints2d),
    keypoints3d(keypoints3d),
    keypoint_information(keypoint_information),
    camera_settings(camera_settings)
{
}

double PoseRefinerCallback::calc(const double *x) const
{
    Pose pose;
    pose.x = x[0];
    pose.y = x[1];
    pose.z = x[2];
    pose.pitch = x[3];
    pose.yaw = x[4];
    pose.roll = x[5];

    vector<KeyPoint2d> projected_keypoints2d;
    project_keypoints(pose, keypoints3d, camera_settings, projected_keypoints2d);
    Mat _projected_keypoints2d(2, projected_keypoints2d.size(), CV_32F, &projected_keypoints2d[0].x);
    Mat _keypoints2d(2, keypoints2d.size(), CV_32F, (void*)&keypoints2d[0].x);

    Mat diff;
    absdiff(_projected_keypoints2d, _keypoints2d, diff);

    double tot_diff = 0;
    for (size_t i=0; i < keypoint_information.size(); i++) {
        if (!keypoint_information[i].ignore_during_refinement) {
            tot_diff += keypoint_information[i].confidence
                *(diff.at<float>(0, i) + diff.at<float>(1, i));
        }
    }
#ifdef SUPER_VERBOSE
    cout << "pose: " << pose.x << "," << pose.y << ","  << pose.z << "," << pose.pitch << ","  << pose.yaw << "," << pose.roll << endl;
    cout << "Diff: " << tot_diff << endl;
#endif

    return tot_diff;
}

void PoseRefinerCallback::getGradient(const double *x, double *grad)
{
    Pose pose;
    pose.x = x[0];
    pose.y = x[1];
    pose.z = x[2];
    pose.pitch = x[3];
    pose.yaw = x[4];
    pose.roll = x[5];

    vector<KeyPoint2d> projected_keypoints2d;
    project_keypoints(pose, keypoints3d, camera_settings, projected_keypoints2d);

    Mat err = Mat::zeros(6, 1, CV_64F);
    Mat hessian = Mat::zeros(6,6, CV_64F);

    Mat tot_diff = Mat::zeros(2,1, CV_64F);

    for (size_t i = 0; i < keypoints3d.size(); i++) {
        auto x = keypoints3d[i].x;
        auto y = keypoints3d[i].y;
        auto z = keypoints3d[i].z;

        if (keypoint_information[i].ignore_during_refinement)
            continue;

        Mat jacobian = -(Mat_<double>(2,6) <<
                1.0/z, 0, -1.0*x/(z*z), -1.0*x*y/(z*z), 1.0*(1.0+(x*x)/(z*z)), -1.0*y/z,
                0, 1.0/z, -1.0*y/(z*z), -1.0*(1+(y*y)/(z*z)), 1.0*x*y/(z*z), 1.0*x/z);

        Mat diff = keypoint_information[i].confidence*(Mat_<double>(2,1) <<
                keypoints2d[i].x -projected_keypoints2d[i].x,
                keypoints2d[i].y -projected_keypoints2d[i].y);

        if ((fabs(diff.at<double>(0)) > 3.0) ||
                (fabs(diff.at<double>(1)) > 3.0))
            continue;
        tot_diff += diff;
#if 0
        double weight = max(0.1, (2.0 - cv::norm(diff))/2.0);

        Mat transposed_jacobian = jacobian.t();
        hessian += transposed_jacobian*jacobian * weight;
        err += transposed_jacobian * diff * weight;

#else
        Mat transposed_jacobian = jacobian.t();
        hessian += transposed_jacobian*jacobian;
        err += transposed_jacobian * diff;
#endif

    }

    Mat hessian_inv = hessian.inv();

    Mat twist = hessian_inv*err / keypoints2d.size();

    Mat gradient(1, 6, CV_64F);

    exponential_map(twist, gradient);
    double *data = gradient.ptr<double>();

    // double _norm = cv::norm(gradient);
    // gradient /= _norm;

#ifdef SUPER_VERBOSE
    cout << "gradient: " << data[0] << "," << data[1] << "," << data[2] << ","<< data[3] << ","<< data[4] << ","<< data[5] << "," << endl;
    cout << "Total diff: " << tot_diff << endl;
#endif

    memcpy(grad, data, 6*sizeof(double));
}
