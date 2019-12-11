#include <opencv2/opencv.hpp>

#include "optical_flow.hpp"

using namespace std;
using namespace cv;


OpticalFlow::OpticalFlow(const CameraSettings &camera_settings) :
    camera_settings(camera_settings)
{
}

void OpticalFlow::calculate_optical_flow(const StereoImage &previous_stereo_image_pyr,
        const vector<KeyPoint2d> &previous_keypoints2d,
        const StereoImage &current_stereo_image_pyr,
        vector<KeyPoint2d> &current_keypoints2d,
        vector<float> &err) {

    Size patch_size(camera_settings.window_size_opt_flow,
            camera_settings.window_size_opt_flow);

    const vector<Mat> &previous_image = previous_stereo_image_pyr.opt_flow;
    const vector<Mat> &current_image = current_stereo_image_pyr.opt_flow;

    vector<Point2f> _previous_keypoints2d;
    _previous_keypoints2d.resize(previous_keypoints2d.size());
    for (size_t i = 0; i < previous_keypoints2d.size(); i++) {
        _previous_keypoints2d[i].x = previous_keypoints2d[i].x;
        _previous_keypoints2d[i].y = previous_keypoints2d[i].y;
    }

    vector<Point2f> _current_keypoints2d;
    _current_keypoints2d.resize(current_keypoints2d.size());
    for (size_t i = 0; i < current_keypoints2d.size(); i++) {
        _current_keypoints2d[i].x = current_keypoints2d[i].x;
        _current_keypoints2d[i].y = current_keypoints2d[i].y;
    }

    vector<uchar> status;
    cv::calcOpticalFlowPyrLK(previous_image, current_image, _previous_keypoints2d,
            _current_keypoints2d, status, err, patch_size, 2,
            TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
            OPTFLOW_USE_INITIAL_FLOW);

    for (size_t i = 0; i < status.size(); i++) {
        if (status[i] == 0) {
            err[i] = std::numeric_limits<float>::infinity();
        }
    }

    for (size_t i = 0; i < current_keypoints2d.size(); i++) {
        current_keypoints2d[i].x = _current_keypoints2d[i].x;
        current_keypoints2d[i].y = _current_keypoints2d[i].y;
    }
}
