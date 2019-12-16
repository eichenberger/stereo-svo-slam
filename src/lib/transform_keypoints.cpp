#include <vector>

#include <opencv2/opencv.hpp>

#include "transform_keypoints.hpp"
#include "rotation_matrix.hpp"

using namespace cv;
using namespace std;

void project_keypoints(const PoseManager &pose,
        const vector<KeyPoint3d> &in, const CameraSettings &camera_settings,
        vector<KeyPoint2d> &out)
{
    Mat cameraMatrix = (Mat_<float>(3, 3, CV_32F) <<
    camera_settings.fx,  0,  camera_settings.cx,
     0,  camera_settings.fy, camera_settings.cy,
     0, 0, 1);

    out.clear();
    out.resize(in.size());

    Vec3f _pose (pose.get_translation());


    // We use the form translate firts rotate later
    // However, projectPoints first translates and then rotates which
    // breaks everything
    vector<Point3f> _in(in.size());
#pragma omp parallel for default(none) shared(_in, in, _pose)
    for (size_t i = 0; i < in.size(); i++) {
        _in[i] = Vec3f(&in[i].x) - _pose;
    }

    Mat distCoeffs = (Mat_<float>(5, 1) <<
            camera_settings.k1, camera_settings.k2,
            camera_settings.p1, camera_settings.p2, camera_settings.k3);

    Vec3f rvec(pose.get_angles());
    Vec3f tvec(0, 0, 0);

    Mat _out;
    // The pose calculated by the algorithm is inverted compared to what
    // would be normal therefore, -
    projectPoints(_in, -rvec, tvec, cameraMatrix, distCoeffs, _out);
    memcpy(&out[0].x, _out.ptr<float>(), sizeof(KeyPoint2d)*out.size());
}

//void transform_keypoints_inverse(const struct Pose &pose,
//        const vector<KeyPoint3d> &in, vector<KeyPoint3d> &out)
//{
//    Pose _pose = pose;
//    _pose.x = -pose.x;
//    _pose.y = -pose.y;
//    _pose.z = -pose.z;
//    _pose.pitch = -pose.pitch;
//    _pose.yaw = -pose.yaw;
//    _pose.roll = -pose.roll;
//
//
//    Mat angles(1, 3, CV_32F, (void*)&_pose.pitch);
//    Mat rot_mat(3, 3, CV_32F);
//    Rodrigues(angles, rot_mat);
//
//    out.clear();
//    out.resize(in.size());
//
//    // Use matrix instead of vector for easier calculation
//    const Mat _in(3, in.size(), CV_32F, (void*)&in[0].x);
//    Mat _out(3, out.size(), CV_32F, (void*)&out[0].x);
//
//
//    _out = rot_mat.t()*_in;
//
//    Vec3f translation(&_pose.x);
////#pragma omp parallel for
//    for (int i=0; i < _out.cols; i++) {
//        // - because it's the inverse transformation
//        _out.col(i) -= translation;
//    }
//}
