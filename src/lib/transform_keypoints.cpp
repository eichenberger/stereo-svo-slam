#include <vector>

#include <opencv2/opencv.hpp>

#include "transform_keypoints.hpp"
#include "rotation_matrix.hpp"

using namespace cv;
using namespace std;

void project_keypoints(const struct Pose &pose,
        const vector<KeyPoint3d> &in, const CameraSettings &camera_settings,
        vector<KeyPoint2d> &out)
{

    cv::Mat angles(1, 3, CV_32F, (void*)&pose.pitch);
    cv::Mat rot_mat(3, 3, CV_32F);
    cv::Rodrigues(angles, rot_mat);

    cv::Mat intrinsic(3, 3, CV_32F);
    intrinsic.at<float>(0,0) = camera_settings.fx;
    intrinsic.at<float>(0,1) = 0;
    intrinsic.at<float>(0,2) = camera_settings.cx;
    intrinsic.at<float>(1,0) = 0;
    intrinsic.at<float>(1,1) = camera_settings.fy;
    intrinsic.at<float>(1,2) = camera_settings.cy;
    intrinsic.at<float>(2,0) = 0;
    intrinsic.at<float>(2,1) = 0;
    intrinsic.at<float>(2,2) = 1;

    out.clear();
    out.resize(in.size());

    // Use matrix instead of vector for easier calculation
    const cv::Mat _in(in.size(), 3, CV_32F, (void*)&in[0].x);
    cv::Mat _out(3, out.size(), CV_32F);


    _out = intrinsic*rot_mat*_in.t();

    cv::Mat translation(3,1, CV_32F, (void*)&pose.x);
//#pragma omp parallel for
    for (int i=0; i < _out.cols; i++) {
        // - because it's the inverse transformation
        _out.col(i) -= intrinsic*translation;
    }

    float *_x = _out.ptr<float>(0);
    float *_y = _out.ptr<float>(1);
    float *_s = _out.ptr<float>(2);

//#pragma omp parallel for
    for (int i=0; i < _out.cols; i++) {
        out[i].x = (*_x)/(*_s);
        out[i].y = (*_y)/(*_s);
        _x++;_y++;_s++;
    }
}

void transform_keypoints_inverse(const struct Pose &pose,
        const vector<KeyPoint3d> &in, vector<KeyPoint3d> &out)
{
    cv::Mat angles(1, 3, CV_32F, (void*)&pose.pitch);
    cv::Mat rot_mat(3, 3, CV_32F);
    cv::Rodrigues(angles, rot_mat);

    out.clear();
    out.resize(in.size());

    // Use matrix instead of vector for easier calculation
    const cv::Mat _in(3, in.size(), CV_32F, (void*)&in[0].x);
    cv::Mat _out(3, out.size(), CV_32F, (void*)&out[0].x);


    _out = rot_mat.t()*_in;

    cv::Vec3f translation(&pose.x);
//#pragma omp parallel for
    for (int i=0; i < _out.cols; i++) {
        // - because it's the inverse transformation
        _out.col(i) -= translation;
    }
}
