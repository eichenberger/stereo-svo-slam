#include <vector>
#include <cassert>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include "pose_estimator.hpp"
#include "transform_keypoints.hpp"
#include "image_comparison.hpp"

//#define VERBOSE 1
#define SUPER_VERBOSE 1

using namespace cv;
using namespace std;

// We could use one of the opencv solver. However, they don't really fit
class PoseEstimatorCallback : public MinProblemSolver::Function
{
    friend class PoseEstimator;
public:
    PoseEstimatorCallback(const StereoImage &current_stereo_image,
                          const StereoImage &previous_stereo_image,
                          const KeyPoints &previous_keypoints,
                          const CameraSettings &camera_settings);
    double calc(const double *x) const;
    int getDims() const { return 6; };
    void getGradient(const double *x, double *grad);
    void setLevel(int level);

private:
    const StereoImage &current_stereo_image;
    const StereoImage &previous_stereo_image;
    const KeyPoints &keypoints;
    const CameraSettings &camera_settings;
    CameraSettings level_camera_settings;
    vector<KeyPoint2d> level_keypoints2d;
    int level;

    cv::Ptr<Mat> hessian_inv;
    cv::Ptr<vector<Mat>> gradient_times_jacobians;
};


PoseEstimator::PoseEstimator(const StereoImage &current_stereo_image,
    const StereoImage &previous_stereo_image,
    const KeyPoints &previous_keypoints,
    const CameraSettings &camera_settings) :
    max_levels(camera_settings.max_pyramid_levels),
    min_level(camera_settings.min_pyramid_level_pose_estimation)
{
    solver_callback = new PoseEstimatorCallback(current_stereo_image,
            previous_stereo_image, previous_keypoints, camera_settings);

}


float PoseEstimator::estimate_pose(const Pose &pose_guess, Pose &pose)
{
    float err = 0;
    Pose pose_estimate = pose_guess;
    for (int i = max_levels; i > min_level; i--) {
        Pose new_estimate;
        int current_level = i-1;

        cout << "Current pyramid level: " << current_level << endl;

        err = estimate_pose_at_level(pose_estimate, new_estimate, current_level);
        pose_estimate = new_estimate;
    }

    pose = pose_estimate;

    return err;
}

#if 0

float PoseEstimator::estimate_pose_at_level(const Pose &pose_guess, Pose &pose,
        int level)
{

    Mat x0 = (Mat_<double>(6,1) <<
        pose_guess.x, pose_guess.y, pose_guess.z,
        pose_guess.pitch, pose_guess.yaw, pose_guess.roll);

    solver_callback->setLevel(level);
    Ptr<DownhillSolver> solver = DownhillSolver::create(solver_callback);
    Mat step = Mat::ones(6,1, CV_64F);
    step *= 0.001;
    solver->setInitStep(step);

    solver->minimize(x0);

    pose.x = x0.at<double>(0);
    pose.y = x0.at<double>(1);
    pose.z = x0.at<double>(2);
    pose.pitch = x0.at<double>(3);
    pose.yaw = x0.at<double>(4);
    pose.roll = x0.at<double>(5);

    double err = solver_callback->calc(x0.ptr<double>());

    return err;
}

#else
float PoseEstimator::estimate_pose_at_level(const Pose &pose_guess, Pose &pose,
        int level)
{
    Mat x0 = (Mat_<double>(6,1) <<
        pose_guess.x, pose_guess.y, pose_guess.z,
        pose_guess.pitch, pose_guess.yaw, pose_guess.roll);

    size_t maxIter = 100;
    float k = 1.0;
    solver_callback->setLevel(level);
    double prev_cost = solver_callback->calc(x0.ptr<double>(0));
    for (size_t i = 0; i < maxIter; i++) {
        Mat gradient(6,1,CV_64F);
        solver_callback->getGradient(x0.ptr<double>(0),
                gradient.ptr<double>(0));
        for (;i < maxIter; i++) {
            Mat x = x0 + (k*gradient);
            double new_cost = solver_callback->calc(x.ptr<double>(0));
            if (new_cost < prev_cost) {
#ifdef VERBOSE
                cout << "Better value pev cost: " << prev_cost << " new cost: " << new_cost << endl;
                cout << "x: " <<
                    x.at<double>(0) << "," <<
                    x.at<double>(1) << "," <<
                    x.at<double>(2) << "," <<
                    x.at<double>(3) << "," <<
                    x.at<double>(4) << "," <<
                    x.at<double>(5) << "," <<
                    std::endl;
#endif
                x0 = x;
                prev_cost = new_cost;
                break;
            }
            else if (fabs(new_cost - prev_cost) < 0.01) {
                cout << "Drop out because of small change" << endl;
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
    pose.pitch = x0.at<double>(3);
    pose.yaw = x0.at<double>(4);
    pose.roll = x0.at<double>(5);

    return prev_cost;
}

#endif

PoseEstimatorCallback::PoseEstimatorCallback(const StereoImage &current_stereo_image,
                          const StereoImage &previous_stereo_image,
                          const KeyPoints &previous_keypoints,
                          const CameraSettings &camera_settings):
    current_stereo_image(current_stereo_image),
    previous_stereo_image(previous_stereo_image),
    keypoints(previous_keypoints),
    camera_settings(camera_settings),
    level(0)
{
}

static inline Pose pose_from_x(const double *x){
    Pose pose;

    // Because of how we use the jacobian angles and pos is exchanged
    pose.x = x[0];
    pose.y = x[1];
    pose.z = x[2];
    pose.pitch = x[3];
    pose.yaw = x[4];
    pose.roll = x[5];

    return pose;
}

double PoseEstimatorCallback::calc(const double *x) const
{
    Pose pose = pose_from_x(x);

    vector<KeyPoint2d> kps2d;
    project_keypoints(pose, keypoints.kps3d, level_camera_settings, kps2d);

#ifdef SUPER_VERBOSE
    cout << "keypoints 2d: ";
    for (auto kp:kps2d) {
        cout << kp.x << "x" << kp.y <<", ";
    }
    cout << endl;
#endif

    vector<float> diffs;
    // Difference will always be positive (absdiff)
    get_total_intensity_diff(previous_stereo_image.left[level],
            current_stereo_image.left[level],
            level_keypoints2d, kps2d, level_camera_settings.window_size, diffs);

    // Use float because maybe it is faster? TODO: Verify
    float diff_sum = 0;
    //cout << "Diffs: ";
    //    cout << diff << ", ";
    //cout << endl;
    int i = 0;
    for (auto diff:diffs) {
        if (keypoints.info[i].seed.accepted)
            diff_sum += diff * keypoints.info[i].confidence;
        i++;
    }

#ifdef SUPER_VERBOSE
    cout << "Diff sum: " << diff_sum << " x: " <<
        x[0] << "," <<
        x[1] << "," <<
        x[2] << "," <<
        x[3] << "," <<
        x[4] << "," <<
        x[5] << "," <<
        std::endl;
#endif

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
    Mat translation = Mat(3,1, CV_64F);

    // Take closed form solution from robotics vision and control page 53
    // Note: This exponential map doesn't give the exact same value as expm
    // from Matlab or Numpy. It is different up to a scaling. It seems that
    // expm uses norm set to 1.0. However, this differs from the closed form
    // solution written in robotics vision and control.
    translation = (_eye*_norm + (1-cos(_norm))*w_skew+(_norm-sin(_norm))*(w_skew*w_skew))*v;

    memcpy(pose.ptr<double>(), translation.ptr<double>(), 3*sizeof(double));

}

void PoseEstimatorCallback::getGradient(const double *x, double *grad)
{
    const int PATCH_SIZE=4;
    Pose pose = pose_from_x(x);

    const Mat &current_image = current_stereo_image.left[level];
    const Mat &previous_image = previous_stereo_image.left[level];

    Vec3d angles(&x[3]);
    Matx33d rot_mat;
    Vec3d translation(&x[0]);

    Rodrigues(-angles, rot_mat);

    // If we haven't calculated the hessian_inv yet we will do it now.
    // Because of compositional lucas kanade we only do it once
    if (hessian_inv.empty()) {
        hessian_inv = new Mat(6,6, CV_64F);
        Mat hessian = Mat::zeros(6,6, CV_64F);
        gradient_times_jacobians = new vector<Mat>;
        for (size_t i = 0; i < level_keypoints2d.size(); i++) {
            auto kp2d = level_keypoints2d[i];
            cout << "Keypoint position: " << kp2d.x << ", " << kp2d.y << endl;
            kp2d.x -= PATCH_SIZE/2;
            kp2d.y -= PATCH_SIZE/2;
            Vec3f kp(&keypoints.kps3d[i].x);

            // Neutralize angle
            kp -= translation;
            kp = rot_mat*kp;

            auto x = kp(0);
            auto y = kp(1);
            auto z = kp(2);

            auto fx = level_camera_settings.fx;
            auto fy = level_camera_settings.fy;

            // See A tutorial on SE(3) transformation parameterizations and on-manifold optimization page 58
            // for the jaccobian. Take - because we want to minimize
            Mat jacobian = -(Mat_<double>(2,6) <<
                fx/z, 0, -fx*x/(z*z), -fx*x*y/(z*z), fx*(1+(x*x)/(z*z)), -fx*y/z,
                0, fy/z, -fy*y/(z*z), -fy*(1+(y*y)/(z*z)), fy*x*y/(z*z), fy*x/z);

            cout << "Jacobian: " << endl;
            cout <<
                jacobian.at<double>(0, 0) << "," <<
                jacobian.at<double>(0, 1) << "," <<
                jacobian.at<double>(0, 2) << "," <<
                jacobian.at<double>(0, 3) << "," <<
                jacobian.at<double>(0, 4) << "," <<
                jacobian.at<double>(0, 5) << endl;
            cout <<
                jacobian.at<double>(1, 0) << "," <<
                jacobian.at<double>(1, 1) << "," <<
                jacobian.at<double>(1, 2) << "," <<
                jacobian.at<double>(1, 3) << "," <<
                jacobian.at<double>(1, 4) << "," <<
                jacobian.at<double>(1, 5) << endl;

            for (size_t r = 0; r < PATCH_SIZE; r++)
            {
                for (size_t c = 0; c < PATCH_SIZE; c++) {
                    if (kp2d.x < 0 || kp2d.y < 0 || kp2d.x >= current_image.cols ||
                            kp2d.y >= current_image.rows) {
                        gradient_times_jacobians->push_back(Mat::zeros(1, 6, CV_64F));
                        kp2d.x ++;
                        continue;
                    }
                    Mat int1, int2, int3, int4;

                    const Size patch_size(1,2);
//                    getRectSubPix(previous_grad_x,
//                            patch_size, Point2f(kp2d.x,kp2d.y), grad_x, CV_32F);
//                    getRectSubPix(previous_grad_y,
//                            patch_size, Point2f(kp2d.x,kp2d.y), grad_y, CV_32F);

                    cv::getRectSubPix(previous_image,
                            patch_size, Point2f(kp2d.x+1,kp2d.y), int1, CV_32F);
                    cv::getRectSubPix(previous_image,
                            patch_size, Point2f(kp2d.x-1,kp2d.y), int2, CV_32F);
                    cv::getRectSubPix(previous_image,
                            patch_size, Point2f(kp2d.x,kp2d.y+1), int3, CV_32F);
                    cv::getRectSubPix(previous_image,
                            patch_size, Point2f(kp2d.x,kp2d.y-1), int4, CV_32F);

                    double diff1 = int1.at<float>(0)-int2.at<float>(0);
                    double diff2 = int3.at<float>(0)-int4.at<float>(0);
                    Mat _grad = (Mat_<double>(1,2) <<
                            diff1, diff2);

#if 1
                    cout << "Gradient at " << kp2d.x << ", " <<kp2d.y << ": ";
                    cout << "(" << _grad.at<double> (0) << ", " << _grad.at<double>(1)<< ") " << endl;
#endif

                    Mat _grad_times_jac = _grad*jacobian;
                    // Store the result of grad*jacobian for further usage
                    gradient_times_jacobians->push_back(_grad_times_jac);

                    // Calculate the gauss newton hessian_inv (~second derivate)
                    hessian += (_grad_times_jac.t()*_grad_times_jac);
                    kp2d.x ++;
                }
                kp2d.x -= PATCH_SIZE;
                kp2d.y ++;
            }
        }

        cout << "Jacobian cache:" <<endl;
        for (auto grad: *gradient_times_jacobians) {
            cout << grad.at<double>(0) << "," <<
                grad.at<double>(1) << "," <<
                grad.at<double>(2) << "," <<
                grad.at<double>(3) << "," <<
                grad.at<double>(4) << "," <<
                grad.at<double>(5) << endl;

        }

        *hessian_inv = hessian.inv();
#ifdef SUPER_VERBOSE
        std::cout << "Hessian matrix: " << std::endl;
        for (size_t i = 0; i < 6; i++) {
            for (size_t j = 0; j < 6; j++) {
                std::cout << hessian.at<double>(i, j) << ", ";
            }
            std::cout << std::endl;
        }
#endif
    }

//    namedWindow("previous", WINDOW_GUI_EXPANDED);
//    namedWindow("current", WINDOW_GUI_EXPANDED);
//
//    imshow("previous", previous_image);
//    imshow("current", current_image);
//
//    waitKey(0);

    // See ICRA Forster 14 for the whole algorithm
    vector<KeyPoint2d> kps2d;
    project_keypoints(pose, keypoints.kps3d, level_camera_settings, kps2d);

    Mat residual = Mat::zeros(6,1, CV_64F);
    for (size_t i = 0; i < kps2d.size(); i++) {
        Point2f kp2d(kps2d[i].x-PATCH_SIZE/2, kps2d[i].y-PATCH_SIZE/2);
        Point2f kp2d_ref(level_keypoints2d[i].x-PATCH_SIZE/2, level_keypoints2d[i].y-PATCH_SIZE/2);

        for (size_t r = 0; r < PATCH_SIZE; r++)
        {
            for (size_t c = 0; c < PATCH_SIZE; c++) {
                Mat int1, int2;

//                if (!keypoints.info[i].seed.accepted)
//                    continue;

                cv::getRectSubPix(previous_image, Size(2,2),
                        kp2d_ref, int1, CV_32F);
                cv::getRectSubPix(current_image, Size(2,2),
                        kp2d, int2, CV_32F);

//                int int_prev, int_cur;
//                int_prev = previous_image.at<uint8_t>(kp2d_ref.y, kp2d_ref.x);
//                int_cur = current_image.at<uint8_t>(kp2d.y, kp2d.x);

                const Mat &_grad_times_jac = (*gradient_times_jacobians)[(i*PATCH_SIZE*PATCH_SIZE)+(r*PATCH_SIZE)+c];

                auto diff = int2.at<float>(0) - int1.at<float>(0);

#if SUPER_VERBOSE
                cout << "Intensity difference template at " <<
                     kp2d_ref.x << ", " << kp2d_ref.y << " image at " <<
                     kp2d.x << ", " << kp2d.y << " diff: " << diff << endl;
#endif

                Mat residual_kp = _grad_times_jac*diff;
                transpose(residual_kp, residual_kp);
                residual -= residual_kp;
                kp2d.x++;
                kp2d_ref.x++;
            }
            kp2d.y++;
            kp2d_ref.y++;
            kp2d.x -= PATCH_SIZE;
            kp2d_ref.x -= PATCH_SIZE;
        }

    }

    cout << "Residual: " << residual.at<double>(0) << "," <<
        residual.at<double>(1) << "," <<
        residual.at<double>(2) << "," <<
        residual.at<double>(3) << "," <<
        residual.at<double>(4) << "," <<
        residual.at<double>(5) << endl;
    Mat delta_pos = *hessian_inv * residual;
    Mat pose_gradient(6, 1, CV_64F);
    exponential_map(delta_pos, pose_gradient);

    Vec3d translation_grad(pose_gradient.ptr<double>(0));
    Rodrigues(angles, rot_mat);
    translation_grad = rot_mat*translation_grad;

    grad[0] = translation_grad(0);
    grad[1] = translation_grad(1);
    grad[2] = translation_grad(2);
    grad[3] = pose_gradient.at<double>(3);
    grad[4] = pose_gradient.at<double>(4);
    grad[5] = pose_gradient.at<double>(5);

//    memcpy(grad, pose_gradient.ptr<double>(), 6*sizeof(double));

#ifdef SUPER_VERBOSE
    cout << "Delta Pos: " <<
        delta_pos.at<double>(0) << "," <<
        delta_pos.at<double>(1) << "," <<
        delta_pos.at<double>(2) << "," <<
        delta_pos.at<double>(3) << "," <<
        delta_pos.at<double>(4) << "," <<
        delta_pos.at<double>(5) << "," <<
        std::endl;

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

    double _norm = cv::norm(pose_gradient);
    cout << "Norm: " << _norm << std::endl;
    pose_gradient = pose_gradient/_norm;
    cout << "Gradient normalized: " <<
        pose_gradient.at<double>(0) << "," <<
        pose_gradient.at<double>(1) << "," <<
        pose_gradient.at<double>(2) << "," <<
        pose_gradient.at<double>(3) << "," <<
        pose_gradient.at<double>(4) << "," <<
        pose_gradient.at<double>(5) << "," <<
        std::endl;
#endif
}

void PoseEstimatorCallback::setLevel(int level)
{
    this->level = level;
    int divider = 1 << level;

    level_camera_settings = camera_settings;
    level_camera_settings.fx /= divider;
    level_camera_settings.fy /= divider;
    level_camera_settings.cx /= divider;
    level_camera_settings.cy /= divider;
    level_camera_settings.baseline /= divider;


    level_keypoints2d = keypoints.kps2d;

    if (level == 0)
        return;

    for (size_t i = 0; i < level_keypoints2d.size(); i++) {
        level_keypoints2d[i].x /= divider;
        level_keypoints2d[i].y /= divider;
    }
}
