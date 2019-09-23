#include <vector>
#include <cassert>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include "pose_estimator.hpp"
#include "transform_keypoints.hpp"
#include "image_comparison.hpp"

using namespace cv;
using namespace std;

class PoseEstimatorCallback : public MinProblemSolver::Function
{
public:
    PoseEstimatorCallback(const StereoImage &current_stereo_image,
                          const StereoImage &previous_stereo_image,
                          const KeyPoints &previous_keypoints,
                          const CameraSettings &camera_settings);
    double calc(const double *x) const;
    int getDims() const { return 6; };
    void getGradient(const double *x, double *grad);

private:
    const StereoImage &current_stereo_image;
    const StereoImage &previous_stereo_image;
    const KeyPoints &keypoints;
    const CameraSettings &camera_settings;

    cv::Ptr<Mat> hessian_inv;
    cv::Ptr<vector<Mat>> gradient_times_jacobians;
};


PoseEstimator::PoseEstimator(const StereoImage &current_stereo_image,
    const StereoImage &previous_stereo_image,
    const KeyPoints &previous_keypoints,
    const CameraSettings &camera_settings)
{
    solver_callback = new PoseEstimatorCallback(current_stereo_image,
            previous_stereo_image, previous_keypoints, camera_settings);
}

float PoseEstimator::estimate_pose(const Pose &pose_guess, Pose &pose)
{
    Ptr<MinProblemSolver> solver = ConjGradSolver::create(solver_callback);

    Mat x0 = (Mat_<double>(6,1) <<
        pose_guess.x, pose_guess.y, pose_guess.z,
        pose_guess.roll, pose_guess.pitch, pose_guess.yaw);

    size_t maxIter = 200;
    float k = 1.0;
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
            else if (fabs(new_cost - prev_cost) < 0.01) {
                i = maxIter;
                break;
            }
            else {
                k /= 2;
            }
        }
    }

    pose.x = x0.at<double>(0);
    pose.y = x0.at<double>(1);
    pose.z = x0.at<double>(2);
    pose.roll = x0.at<double>(3);
    pose.pitch = x0.at<double>(4);
    pose.yaw = x0.at<double>(5);

    cout << std::endl;

    return prev_cost;

    //pose.x = x0.at<double>(0);
    //pose.y = x0.at<double>(1);
    //pose.z = x0.at<double>(2);
    //pose.roll = x0.at<double>(3);
    //pose.pitch = x0.at<double>(4);
    //pose.yaw = x0.at<double>(5);

    //return cost;
}

PoseEstimatorCallback::PoseEstimatorCallback(const StereoImage &current_stereo_image,
                          const StereoImage &previous_stereo_image,
                          const KeyPoints &previous_keypoints,
                          const CameraSettings &camera_settings):
    current_stereo_image(current_stereo_image),
    previous_stereo_image(previous_stereo_image),
    keypoints(previous_keypoints),
    camera_settings(camera_settings)
{
}

static inline Pose pose_from_x(const double *x){
    Pose pose;

    // Because of how we use the jacobian angles and pos is exchanged
    pose.x = x[0];
    pose.y = x[1];
    pose.z = x[2];
    pose.roll = x[3];
    pose.pitch = x[4];
    pose.yaw = x[5];

    return pose;
}

double PoseEstimatorCallback::calc(const double *x) const
{
    Pose pose = pose_from_x(x);

    vector<KeyPoint2d> kps2d;
    project_keypoints(pose, keypoints.kps3d, camera_settings, kps2d);

    vector<float> diffs;
    // Difference will always be positive (absdiff)
    get_total_intensity_diff(previous_stereo_image.left, current_stereo_image.left,
            keypoints.kps2d, kps2d, 4, diffs);

    // Use float because maybe it is faster? TODO: Verify
    float diff_sum = 0;
    for (auto diff:diffs) {
        diff_sum += diff;
    }

//    cout << "Diff sum: " << diff_sum << " x: " <<
//        x[0] << "," <<
//        x[1] << "," <<
//        x[2] << "," <<
//        x[3] << "," <<
//        x[4] << "," <<
//        x[5] << "," <<
//        std::endl;

    return diff_sum;
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

void PoseEstimatorCallback::getGradient(const double *x, double *grad)
{
    Pose pose = pose_from_x(x);

    const Mat &current_image = current_stereo_image.left;
    const Mat &previous_image = previous_stereo_image.left;
    //Mat current_image, previous_image;

    //current_stereo_image.left.convertTo(current_image, CV_32F);
    //previous_stereo_image.left.convertTo(previous_image, CV_32F);


    // If we haven't calculated the hessian_inv yet we will do it now.
    // Because of compositional lucas kanade we only do it once
    if (hessian_inv.empty()) {
        hessian_inv = new Mat(6,6, CV_64F);
        Mat hessian = Mat::zeros(6,6, CV_64F);
        gradient_times_jacobians = new vector<Mat>;
        for (size_t i = 0; i < keypoints.kps2d.size(); i++) {
            auto kp2d = keypoints.kps2d[i];
            auto x = keypoints.kps3d[i].x;
            auto y = keypoints.kps3d[i].y;
            auto z = keypoints.kps3d[i].z;

            auto fx = camera_settings.fx;
            auto fy = camera_settings.fy;

            // See A tutorial on SE(3) transformation parameterizations and on-manifold optimization page 58
            // for the jaccobian. Take - because we want to minimize
            Mat jacobian = -(Mat_<double>(2,6) <<
                fx/z, 0, -fx*x/(z*z), -fx*x*y/(z*z), fx*(1+(x*x)/(z*z)), -fx*y/z,
                0, fy/z, -fy*y/(z*z), -fy*(1+(y*y)/(z*z)), fy*x*y/(z*z), fy*x/z);

            //float dx = 0.5f * ((w_ref_tl*ref_img_ptr[1] + w_ref_tr*ref_img_ptr[2] + w_ref_bl*ref_img_ptr[stride+1] + w_ref_br*ref_img_ptr[stride+2])
            //                  -(w_ref_tl*ref_img_ptr[-1] + w_ref_tr*ref_img_ptr[0] + w_ref_bl*ref_img_ptr[stride-1] + w_ref_br*ref_img_ptr[stride]));
            //float dy = 0.5f * ((w_ref_tl*ref_img_ptr[stride] + w_ref_tr*ref_img_ptr[1+stride] + w_ref_bl*ref_img_ptr[stride*2] + w_ref_br*ref_img_ptr[stride*2+1])
            //                  -(w_ref_tl*ref_img_ptr[-stride] + w_ref_tr*ref_img_ptr[1-stride] + w_ref_bl*ref_img_ptr[0] + w_ref_br*ref_img_ptr[1]));

            Mat int1, int2, int3, int4;
            // uint8_t int1, int2, int3, int4;
            // int1 = previous_image.at<uint8_t>(round(kp2d.y), round(kp2d.x-1.0));
            // int2 = previous_image.at<uint8_t>(round(kp2d.y), round(kp2d.x+1.0));
            // int3 = previous_image.at<uint8_t>(round(kp2d.y-1.0), round(kp2d.x));
            // int4 = previous_image.at<uint8_t>(round(kp2d.y+1.0), round(kp2d.x));
            cv::getRectSubPix(previous_image,
                    Size(2,2), Point2f(kp2d.x-1.0,kp2d.y), int1, CV_32F);
            cv::getRectSubPix(previous_image,
                    Size(2,2), Point2f(kp2d.x+1.0,kp2d.y), int2, CV_32F);
            cv::getRectSubPix(previous_image,
                    Size(2,2), Point2f(kp2d.x,kp2d.y-1.0), int3, CV_32F);
            cv::getRectSubPix(previous_image,
                    Size(2,2), Point2f(kp2d.x,kp2d.y+1.0), int4, CV_32F);

            auto diff1 = cv::sum(int2-int1)[0];
            auto diff2 = cv::sum(int4-int3)[0];
            // auto diff1 = int2-int1;
            // auto diff2 = int4-int3;
            Mat _grad = -(Mat_<double>(1,2) <<
                    diff1, diff2);

            //cout << "Gradient at " << kp2d.x << ", " <<kp2d.y << ": ";
            //cout << "(" << diff1 << ", " << diff2 << ") " << endl;

            Mat _grad_times_jac = _grad*jacobian;
            // Store the result of grad*jacobian for further usage
            gradient_times_jacobians->push_back(_grad_times_jac);

            // Calculate the gauss newton hessian_inv (~second derivate)
            hessian += (_grad_times_jac.t()*_grad_times_jac);
        }

        *hessian_inv = hessian.inv();
//        std::cout << "Hessian matrix: " << std::endl;
//        for (size_t i = 0; i < 6; i++) {
//            for (size_t j = 0; j < 6; j++) {
//                std::cout << hessian_inv->at<double>(i, j) << ", ";
//            }
//            std::cout << std::endl;
//        }
    }

    // See ICRA Forster 14 for the whole algorithm
    vector<KeyPoint2d> kps2d;
    project_keypoints(pose, keypoints.kps3d, camera_settings, kps2d);

    Mat residual = Mat::zeros(6,1, CV_64F);
    for (size_t i = 0; i < kps2d.size(); i++) {
        Mat int1, int2;

        cv::getRectSubPix(previous_image, Size(2,2),
                Point2f(keypoints.kps2d[i].x, keypoints.kps2d[i].y), int1, CV_32F);
        cv::getRectSubPix(current_image, Size(2,2),
                Point2f(kps2d[i].x,kps2d[i].y), int2, CV_32F);

        const Mat &_grad_times_jac = (*gradient_times_jacobians)[i];

        auto diff = cv::sum(int2-int1)[0];

        //cout << "Intensity difference template at " <<
        //    keypoints.kps2d[i].x << ", " << keypoints.kps2d[i].y << " image at " <<
        //    kps2d[i].x << ", " << kps2d[i].y << " diff: " << diff << endl;

        Mat residual_kp = _grad_times_jac*diff;
        transpose(residual_kp, residual_kp);
        residual += residual_kp;
    }

    Mat delta_pos = *hessian_inv * residual;
    //cv::solve(*hessian_inv, residual, delta_pos);
    //delta_pos = (Mat_<double>(6,1) <<
    //        0.000469386, -0.00240017, -0.000722628, -0.000622904, -0.000458349, 3.3463e-05);
    Mat pose_gradient(6, 1, CV_64F, grad);
    //cout << "Delta Pos: " <<
    //    delta_pos.at<double>(0) << "," <<
    //    delta_pos.at<double>(1) << "," <<
    //    delta_pos.at<double>(2) << "," <<
    //    delta_pos.at<double>(3) << "," <<
    //    delta_pos.at<double>(4) << "," <<
    //    delta_pos.at<double>(5) << "," <<
    //    std::endl;

    exponential_map(delta_pos, pose_gradient);

    cout << "Current Pose: " <<
        x[0] << "," <<
        x[1] << "," <<
        x[2] << "," <<
        x[3] << "," <<
        x[4] << "," <<
        x[5] << "," <<
        std::endl;

    cout << "Gradient: " <<
        grad[0] << "," <<
        grad[1] << "," <<
        grad[2] << "," <<
        grad[3] << "," <<
        grad[4] << "," <<
        grad[5] << "," <<
        std::endl;

    //double _norm = cv::norm(pose_gradient);
    //cout << "Norm: " << _norm << std::endl;
    //pose_gradient = pose_gradient/_norm;
    //cout << "Gradient normalized: " <<
    //    grad[0] << "," <<
    //    grad[1] << "," <<
    //    grad[2] << "," <<
    //    grad[3] << "," <<
    //    grad[4] << "," <<
    //    grad[5] << "," <<
    //    std::endl;
}

