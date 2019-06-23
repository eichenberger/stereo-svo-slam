#include <vector>

#include <opencv2/opencv.hpp>

#include "rotation_matrix.hpp"

using namespace cv;
using namespace std;

void transform_keypoints(const vector<float> &pose,
        const vector<array<float, 3>>& keypoints3d,
        double fx, double fy, double cx, double cy,
        vector<array<float, 2>> keypoints2d)
{
    Matx33f rot_matrix;
    rotation_matrix(pose, rot_matrix);

    float _extrinsic[] = {rot_matrix(0,0), rot_matrix(0,1), rot_matrix(0,2), pose[3],
                       rot_matrix(1,0), rot_matrix(1,1), rot_matrix(1,2), pose[4],
                       rot_matrix(2,0), rot_matrix(2,1), rot_matrix(2,2), pose[5]};
    Mat extrinsic(3, 4, CV_32F, _extrinsic);

    keypoints2d.resize(keypoints3d.size());
    Mat _keypoints3d = Mat::ones(4, keypoints3d.size(), CV_32F);
    for (int i = 0; i < keypoints3d.size(); i++) {
        _keypoints3d.at<float>(0, i) = keypoints3d[i][0];
        _keypoints3d.at<float>(1, i) = keypoints3d[i][1];
        _keypoints3d.at<float>(2, i) = keypoints3d[i][2];
    }

    Mat _keypoints2d(3, keypoints3d.size(), CV_32F);

    _keypoints2d = extrinsic.mul(_keypoints3d);

    keypoints2d.resize(keypoints3d.size());
    for (int i = 0; i < keypoints2d.size(); i++) {
        keypoints2d[i][0] = _keypoints2d.at<float>(i, 0)/_keypoints2d.at<float>(i, 2);
        keypoints2d[i][1] = _keypoints2d.at<float>(i, 1)/_keypoints2d.at<float>(i, 2);
    }
}

