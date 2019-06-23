#include <vector>
#include <cassert>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include "transform_keypoints.hpp"

using namespace cv;
using namespace std;

class PoseEstimatorCallback: public LMSolver::Callback
{
public:
    PoseEstimatorCallback(vector<array<float, 2>> &previous_keypoints,
            vector<array<float, 3>> &keypoints3d, float fx, float fy,
            float cx, float cy);
    virtual bool compute (InputArray param, OutputArray err, OutputArray J) const;

    void set_current_image(const Mat *current_image);
    void set_previous_image(const Mat *previous_image);


private:
    const Mat *current_image;
    const Mat *previous_image;

    vector<array<float,2>> &previous_keypoints;
    vector<array<float,3>> &keypoints3d;
    float fx;
    float fy;
    float cx;
    float cy;
};


class PoseEstimator
{
public:
    PoseEstimator(vector<Mat> &current_image_pyramid,
            vector<Mat> &previous_image_pyramid,
            vector<array<float, 2>> &previous_keypoints,
            vector<array<float, 3>> &keypoints3d,
            float fx, float fy, float cx, float cy);


    void estimate_pose(const vector<float> &pose_guess, vector<float> &pose,
            float &cost);

private:
    vector<Mat> &current_image_pyramid;
    vector<Mat> &previous_image_pyramid;
    Ptr<PoseEstimatorCallback> solver_callback;
};

PoseEstimator::PoseEstimator(vector<Mat> &current_image_pyramid,
            vector<Mat> &previous_image_pyramid,
            vector<array<float, 2>> &previous_keypoints,
            vector<array<float, 3>> &keypoints3d,
            float fx, float fy, float cx, float cy)
    : current_image_pyramid(current_image_pyramid),
    previous_image_pyramid(previous_image_pyramid)
{
    solver_callback = new PoseEstimatorCallback(previous_keypoints,
            keypoints3d, fx, fy, cx, cy);
}

void PoseEstimator::estimate_pose(const vector<float> &pose_guess,
        vector<float> &pose, float &cost)
{
    assert(pose_guess.size() == 6);
    assert(current_image_pyramid.size() > 2);
    assert(previous_image_pyramid.size() == current_image_pyramid.size());

    vector<float> current_pose(pose_guess);

    Ptr<LMSolver> solver = LMSolver::create(solver_callback, 200, 20.0);

    for (int i = 0; i < 3; i++) {
    }
}

PoseEstimatorCallback::PoseEstimatorCallback(
        vector<array<float, 2>> &previous_keypoints,
        vector<array<float, 3>> &keypoints3d, float fx, float fy,
        float cx, float cy):
    previous_keypoints(previous_keypoints), keypoints3d(keypoints3d),
    fx(fx), fy(fy), cx(cx), cy(cy)
{
}

bool PoseEstimatorCallback::compute(InputArray param, OutputArray err,
        OutputArray J) const
{
    vector<float> *pose_guess = static_cast<vector<float>*>(param.getObj());
    vector<array<float, 2>> keypoints2d;

    transform_keypoints(*pose_guess, keypoints3d, fx, fy, cx, cy, keypoints2d);

    return true;
}


void PoseEstimatorCallback::set_current_image(const Mat *current_image)
{
    this->current_image = current_image;
}

void PoseEstimatorCallback::set_previous_image(const Mat *previous_image)
{
    this->previous_image = previous_image;
}
