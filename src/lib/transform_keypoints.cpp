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


    Mat cameraMatrix = (Mat_<float>(3, 3, CV_32F) <<
    camera_settings.fx,  0,  camera_settings.cx,
     0,  camera_settings.fy, camera_settings.cy,
     0, 0, 1);

    out.clear();
    out.resize(in.size());


    // Use matrix instead of vector for easier calculation
    const Mat _in(in.size(), 3, CV_32F, (void*)&in[0].x);

    Mat distCoeffs = (Mat_<float>(5, 1) <<
            camera_settings.k1, camera_settings.k2,
            camera_settings.p1, camera_settings.p2, camera_settings.k3);

    Mat rvec(1, 3, CV_32F, (void*)&pose.pitch);
    Mat tvec(1, 3, CV_32F, (void*)&pose.x);

    Mat _out(2, out.size(), CV_32F);
    projectPoints(_in, rvec, tvec, cameraMatrix, distCoeffs, _out);
    memcpy(&out[0].x, _out.ptr<float>(), sizeof(float)*2*out.size());
}

void transform_keypoints_inverse(const struct Pose &pose,
        const vector<KeyPoint3d> &in, vector<KeyPoint3d> &out)
{
    Mat angles(1, 3, CV_32F, (void*)&pose.pitch);
    Mat rot_mat(3, 3, CV_32F);
    Rodrigues(angles, rot_mat);

    out.clear();
    out.resize(in.size());

    // Use matrix instead of vector for easier calculation
    const Mat _in(3, in.size(), CV_32F, (void*)&in[0].x);
    Mat _out(3, out.size(), CV_32F, (void*)&out[0].x);


    _out = rot_mat.t()*_in;

    Vec3f translation(&pose.x);
//#pragma omp parallel for
    for (int i=0; i < _out.cols; i++) {
        // - because it's the inverse transformation
        _out.col(i) -= translation;
    }
}
