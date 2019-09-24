#include <opencv2/opencv.hpp>

#include "optical_flow.hpp"

using namespace std;
using namespace cv;


OpticalFlow::OpticalFlow() {
}

void OpticalFlow::calculate_optical_flow(const vector<StereoImage> &previous_stereo_image_pyr,
        const vector<KeyPoint2d> &previous_keypoints2d,
        const vector<StereoImage> &current_stereo_image_pyr,
        vector<KeyPoint2d> &current_keypoints2d,
        vector<float> &err) {
    vector<Mat> previous_images;
    vector<Mat> current_images;

    for (auto previous_image: previous_stereo_image_pyr) {
        previous_images.push_back(previous_image.left);
    }
    for (auto current_image: current_stereo_image_pyr) {
        current_images.push_back(current_image.left);
    }

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

    // TODO: use pyramid instead of calculating it each time
    vector<uchar> status;
    cv::calcOpticalFlowPyrLK(previous_images[0], current_images[0], _previous_keypoints2d,
            _current_keypoints2d, status, err, Size(8,8), 4,
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
