#include <vector>
#include <cassert>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include "pose_refinement.hpp"
#include "transform_keypoints.hpp"
#include "image_comparison.hpp"

//#define SUPER_VERBOSE 1

using namespace cv;
using namespace std;

// We could use one of the opencv solver. However, they don't really fit
class PoseRefinerCallback : public LMSolver::Callback
{
public:
    PoseRefinerCallback(const vector<KeyPoint2d> &keypoints2d,
                        const vector<KeyPoint3d> &keypoints3d,
                        const CameraSettings &camera_settings);
    bool compute(InputArray param, OutputArray err, OutputArray J) const;

private:
    const vector<KeyPoint2d> &keypoints2d;
    const vector<KeyPoint3d> &keypoints3d;
    const CameraSettings &camera_settings;
};


PoseRefiner::PoseRefiner(const CameraSettings &camera_settings) :
    camera_settings(camera_settings)
{
}

float PoseRefiner::refine_pose(const vector<KeyPoint2d> &keypoints2d,
    const vector<KeyPoint3d> &keypoints3d,
    const Pose &estimated_pose,
    Pose &refined_pose)
{
    refined_pose = estimated_pose;
    _InputOutputArray _pose(_InputArray::FIXED_TYPE, &refined_pose);

    Ptr<LMSolver::Callback> callback = new PoseRefinerCallback(keypoints2d, keypoints3d, camera_settings);
    Ptr<LMSolver> solver = LMSolver::create(callback, 50);

    solver->run(_pose);

    vector<KeyPoint2d> projected_keypoints2d;
    project_keypoints(refined_pose, keypoints3d, camera_settings, projected_keypoints2d);
    Mat _projected_keypoints2d(2, projected_keypoints2d.size(), CV_32F, &projected_keypoints2d[0].x);
    Mat _keypoints2d(2, keypoints2d.size(), CV_32F, (void*)&keypoints2d[0].x);


    Mat err;
    absdiff(_projected_keypoints2d, _keypoints2d, err);

    return sum(err)[0];
}

PoseRefinerCallback::PoseRefinerCallback(const vector<KeyPoint2d> &keypoints2d,
        const vector<KeyPoint3d> &keypoints3d,
        const CameraSettings &camera_settings):
    keypoints2d(keypoints2d),
    keypoints3d(keypoints3d),
    camera_settings(camera_settings)
{
}

bool PoseRefinerCallback::compute(InputArray param,
        OutputArray err, OutputArray J) const
{
    Pose *pose = static_cast<Pose*>(param.getObj());
    vector<KeyPoint2d> projected_keypoints2d;
    project_keypoints(*pose, keypoints3d, camera_settings, projected_keypoints2d);
    Mat _projected_keypoints2d(2, projected_keypoints2d.size(), CV_32F, &projected_keypoints2d[0].x);
    Mat _keypoints2d(2, keypoints2d.size(), CV_32F, (void*)&keypoints2d[0].x);
    err.getMatRef() = _projected_keypoints2d - _keypoints2d;

    return true;
}

//static void exponential_map(const Mat &twist, Mat &pose)
//{
//    Mat v(3, 1, CV_64F, (void*)twist.ptr(0));
//    Mat w(3, 1, CV_64F, (void*)twist.ptr(3));
//
//    twist.copyTo(pose);
//    // The angles don't change. See robotics vision and control
//    // page 53 for more details
//    Mat w_skew = (Mat_<double>(3,3) <<
//            0, -w.at<double>(2), w.at<double>(1),
//            w.at<double>(2), 0, -w.at<double>(0),
//            -w.at<double>(1), w.at<double>(0), 0);
//    float _norm = 1.0;
//    Mat _eye = Mat::eye(3,3, CV_64F);
//    Mat translation = Mat(3,1, CV_64F, pose.ptr<double>(0));
//
//    // Take closed form solution from robotics vision and control page 53
//    // Note: This exponential map doesn't give the exact same value as expm
//    // from Matlab or Numpy. It is different up to a scaling. It seems that
//    // expm uses norm set to 1.0. However, this differs from the closed form
//    // solution written in robotics vision and control.
//    translation = (_eye*_norm + (1-cos(_norm))*w_skew+(_norm-sin(_norm))*(w_skew*w_skew))*v;
//}
