#include <vector>
#include <cassert>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include "pose_estimator.hpp"
#include "transform_keypoints.hpp"
#include "image_comparison.hpp"
#include "exponential_map.hpp"

//#define VERBOSE 1
//#define SUPER_VERBOSE 1

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
    float do_calc(const PoseManager &pose_manager) const;
    int getDims() const { return 6; };
    void get_gradient(const PoseManager pose, Vec6f &grad);
    void setLevel(int level);
    void reset_hessian();

private:
    void calculate_hessian(const PoseManager pose);

    const StereoImage &current_stereo_image;
    const StereoImage &previous_stereo_image;
    const KeyPoints &keypoints;
    const CameraSettings &camera_settings;
    CameraSettings level_camera_settings;
    vector<KeyPoint2d> level_keypoints2d;
    int level;

    cv::Ptr<Mat> hessian;
    cv::Ptr<vector<Mat>> gradient_times_jacobians;

    const static int PATCH_SIZE;
};

const int PoseEstimatorCallback::PATCH_SIZE = 4;

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


float PoseEstimator::estimate_pose(const PoseManager &pose_manager_guess, PoseManager &pose)
{
    float err = 0;
    PoseManager pose_manager_estimate = pose_manager_guess;
    for (int i = max_levels; i > min_level; i--) {
        PoseManager new_estimate;
        int current_level = i-1;

        err = estimate_pose_at_level(pose_manager_estimate, new_estimate, current_level);
        pose_manager_estimate = new_estimate;
    }

    pose = pose_manager_estimate;

    return err;
}

#if 0
float PoseEstimator::estimate_pose_at_level(const PoseManager &pose_manager_guess, PoseManager &pose,
        int level)
{
    const Pose &pose_guess = pose_manager_guess.get_pose();

    Mat x0 = (Mat_<double>(6,1) <<
        pose_guess.x, pose_guess.y, pose_guess.z,
        pose_guess.pitch, pose_guess.yaw, pose_guess.roll);

    solver_callback->setLevel(level);
    Ptr<DownhillSolver> solver = DownhillSolver::create(solver_callback);
    Mat step = Mat::ones(6,1, CV_64F);
    step *= 0.001;
    solver->setInitStep(step);

    solver->minimize(x0);

    Pose _pose;
    _pose.x = x0.at<double>(0);
    _pose.y = x0.at<double>(1);
    _pose.z = x0.at<double>(2);
    _pose.pitch = x0.at<double>(3);
    _pose.yaw = x0.at<double>(4);
    _pose.roll = x0.at<double>(5);

    pose.set_pose(_pose);

    float err = solver_callback->do_calc(pose);

    return err;
}

#else
float PoseEstimator::estimate_pose_at_level(const PoseManager &pose_manager_guess, PoseManager &pose,
        int level)
{
    size_t maxIter = 50;
    solver_callback->reset_hessian();
    solver_callback->setLevel(level);
    PoseManager x0 = pose_manager_guess;
    float prev_cost = solver_callback->do_calc(x0);
    PoseManager _x;

    for (size_t i = 0; i < maxIter; i++) {
        Vec6f gradient;
        solver_callback->get_gradient(x0, gradient);
        float k = 1.0;
        for (;i < maxIter; i++) {
            Vec6f x = x0.get_vector() + (k*gradient);
            _x.set_vector(x);
            float new_cost = solver_callback->do_calc(_x);
            if (new_cost < prev_cost) {
                x0 = _x;
                prev_cost = new_cost;
                break;
            }
            else if (fabs(new_cost - prev_cost) < 0.1) {
                cout << "Drop out because of small change" << endl;
                i = maxIter;
                break;
            }
            else {
                k /= 2;
            }
        }
    }

    pose = x0;

    cout << "Previous pose: " << pose_manager_guess;
    cout << " New pose: " << pose << endl;

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

float PoseEstimatorCallback::do_calc(const PoseManager &pose_manager) const
{
    vector<KeyPoint2d> kps2d;
    project_keypoints(pose_manager, keypoints.kps3d, level_camera_settings, kps2d);

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

    // Don't use OMP -> SSE
    float diff_sum = 0;
    for (size_t i = 0; i < diffs.size(); i++) {
//        if (keypoints.info[i].ignore_completely)
//            continue;
        diff_sum += diffs[i];
    }

#ifdef SUPER_VERBOSE
    cout << "Diff sum: " << diff_sum <<  std::endl;
#endif

    return diff_sum;


}

double PoseEstimatorCallback::calc(const double *x) const
{
    Pose pose = pose_from_x(x);

    PoseManager manager;
    manager.set_pose(pose);

    return do_calc(manager);
}

void PoseEstimatorCallback::calculate_hessian(const PoseManager pose)
{
    const Mat &current_image = current_stereo_image.left[level];
    const Mat &previous_image = previous_stereo_image.left[level];

    cout << "Calculate hessian" << endl;
    hessian = new Mat(Mat::zeros(6,6, CV_32F));
    gradient_times_jacobians.reset(new vector<Mat>);
    gradient_times_jacobians->resize(level_keypoints2d.size()*PATCH_SIZE*PATCH_SIZE);

#pragma omp parallel for default(none) shared(pose, gradient_times_jacobians, hessian, current_image, previous_image)
    for (size_t i = 0; i < level_keypoints2d.size(); i++) {
        const auto fx = level_camera_settings.fx;
        const auto fy = level_camera_settings.fy;
        // Define insice for to make it thread safe
        Matx33f rot_mat(pose.get_inv_rotation_matrix());
        Vec3f translation(pose.get_translation());
        auto kp2d = level_keypoints2d[i];
//        const auto &info = keypoints.info[i];

//        if (info.ignore_completely) {
//            for (size_t j = 0; j < PATCH_SIZE*PATCH_SIZE; j++)
//                (*gradient_times_jacobians)[i*PATCH_SIZE*PATCH_SIZE + j] = Mat::zeros(1, 6, CV_32F);
//            continue;
//        }

        // cout << "Keypoint position: " << kp2d.x << ", " << kp2d.y << endl;
        kp2d.x -= PATCH_SIZE/2;
        kp2d.y -= PATCH_SIZE/2;
        Vec3f kp(&keypoints.kps3d[i].x);

        // Neutralize angle
        kp -= translation;
        kp = rot_mat*kp;

        auto x = kp(0);
        auto y = kp(1);
        auto z = kp(2);

        // See A tutorial on SE(3) transformation parameterizations and on-manifold optimization page 58
        // for the jaccobian. Take - because we want to minimize
        Mat jacobian = -(Mat_<float>(2,6) <<
            fx/z, 0, -fx*x/(z*z), -fx*x*y/(z*z), fx*(1+(x*x)/(z*z)), -fx*y/z,
            0, fy/z, -fy*y/(z*z), -fy*(1+(y*y)/(z*z)), fy*x*y/(z*z), fy*x/z);

        for (size_t r = 0; r < PATCH_SIZE; r++)
        {
            for (size_t c = 0; c < PATCH_SIZE; c++) {
                if (kp2d.x < 0 || kp2d.y < 0 || kp2d.x >= current_image.cols ||
                        kp2d.y >= current_image.rows) {
                    (*gradient_times_jacobians)[i*PATCH_SIZE*PATCH_SIZE+r*PATCH_SIZE+c] = Mat::zeros(1, 6, CV_32F);

                    kp2d.x ++;
                    continue;
                }
                Mat int1, int2, int3, int4;

                const Size patch_size(2,2);

                getRectSubPix(previous_image,
                    patch_size, Point2f(kp2d.x+1,kp2d.y), int1, CV_32F);
                getRectSubPix(previous_image,
                    patch_size, Point2f(kp2d.x-1,kp2d.y), int2, CV_32F);
                getRectSubPix(previous_image,
                    patch_size, Point2f(kp2d.x,kp2d.y+1), int3, CV_32F);
                getRectSubPix(previous_image,
                    patch_size, Point2f(kp2d.x,kp2d.y-1), int4, CV_32F);


                float diff1 = sum(int1-int2)(0);
                float diff2 = sum(int3-int4)(0);
                Mat _grad = (Mat_<float>(1,2) <<
                        diff1, diff2);

                Mat _grad_times_jac = _grad*jacobian;
                // Store the result of grad*jacobian for further usage
                (*gradient_times_jacobians)[i*PATCH_SIZE*PATCH_SIZE+r*PATCH_SIZE+c] = _grad_times_jac;

                // Calculate the gauss newton hessian_inv (~second derivate)
                *hessian += (_grad_times_jac.t()*_grad_times_jac);
                kp2d.x ++;
            }
            kp2d.x -= PATCH_SIZE;
            kp2d.y ++;
        }
    }

#ifdef SUPER_VERBOSE
    std::cout << "Hessian matrix: " << std::endl;
    for (size_t i = 0; i < 6; i++) {
        for (size_t j = 0; j < 6; j++) {
            std::cout << hessian->at<float>(i, j) << ", ";
        }
        std::cout << std::endl;
    }
#endif

}

void PoseEstimatorCallback::get_gradient(const PoseManager pose, Vec6f &grad)
{
    const int PATCH_SIZE=4;

    const Mat &current_image = current_stereo_image.left[level];
    const Mat &previous_image = previous_stereo_image.left[level];

    // If we haven't calculated the hessian_inv yet we will do it now.
    // Because of compositional lucas kanade we only do it once
    if (hessian.empty()) {
        calculate_hessian(pose);
    }

    // See ICRA Forster 14 for the whole algorithm
    vector<KeyPoint2d> kps2d;
    project_keypoints(pose, keypoints.kps3d, level_camera_settings, kps2d);

    cout << "pose for projection: " << pose << endl;

    vector<float> diffs(kps2d.size()*PATCH_SIZE*PATCH_SIZE);
#pragma omp parallel for default(none) shared(diffs, kps2d, previous_image, current_image)
    for (size_t i = 0; i < kps2d.size(); i++) {
        // For OMP
        vector<float>::iterator diff = diffs.begin() + i*PATCH_SIZE*PATCH_SIZE;
//        auto &info = keypoints.info[i];
//        if (info.ignore_completely) {
//            for (size_t j = 0; j < PATCH_SIZE*PATCH_SIZE; j++, diff++)
//                *diff=0;
//            continue;
//        }

        Point2f kp2d(kps2d[i].x-PATCH_SIZE/2, kps2d[i].y-PATCH_SIZE/2);
        Point2f kp2d_ref(level_keypoints2d[i].x-PATCH_SIZE/2, level_keypoints2d[i].y-PATCH_SIZE/2);
        for (size_t r = 0; r < PATCH_SIZE; r++)
        {
            for (size_t c = 0; c < PATCH_SIZE; c++, diff++ , kp2d.x++, kp2d_ref.x++) {
                Mat int1, int2;

                cv::getRectSubPix(previous_image, Size(2,2),
                        kp2d_ref, int1, CV_32F);
                cv::getRectSubPix(current_image, Size(2,2),
                        kp2d, int2, CV_32F);

                *diff = sum(int2-int1)(0);
            }
            kp2d.y++;
            kp2d_ref.y++;
            kp2d.x -= PATCH_SIZE;
            kp2d_ref.x -= PATCH_SIZE;
        }

    }

    Mat residual = Mat::zeros(1,6, CV_32F);
#pragma omp parallel for default(none) shared(diffs, gradient_times_jacobians, kps2d, residual)
    for (size_t i = 0; i < kps2d.size(); i++) {
        // For OMP
        size_t j = i*PATCH_SIZE*PATCH_SIZE;
        vector<float>::iterator diff = diffs.begin() + i*PATCH_SIZE*PATCH_SIZE;
        for (size_t r = 0; r < PATCH_SIZE; r++)
        {
            for (size_t c = 0; c < PATCH_SIZE; c++, diff++, j++) {
                Mat &_grad_times_jac = (*gradient_times_jacobians)[j];
                Mat residual_kp = _grad_times_jac*(*diff);
                residual -= residual_kp;
            }
        }
    }
    transpose(residual, residual);

    Mat delta_pos;
    solve(*hessian, residual, delta_pos, DECOMP_SVD);
    Mat pose_gradient(6, 1, CV_32F);
    exponential_map(delta_pos, pose_gradient);

    Vec3f translation_grad(pose_gradient.ptr<float>(0));
    Vec3f rotation_gradient(pose_gradient.ptr<float>(3));
    Matx33f rot_mat(pose.get_rotation_matrix());
    translation_grad = rot_mat*translation_grad;
    rotation_gradient = rot_mat*rotation_gradient;

    grad[0] = translation_grad(0);
    grad[1] = translation_grad(1);
    grad[2] = translation_grad(2);
    grad[3] = rotation_gradient(0);
    grad[4] = rotation_gradient(1);
    grad[5] = rotation_gradient(2);

//    memcpy(grad, pose_gradient.ptr<double>(), 6*sizeof(double));

#ifdef SUPER_VERBOSE
    cout << "Delta Pos: " <<
        delta_pos.at<float>(0) << "," <<
        delta_pos.at<float>(1) << "," <<
        delta_pos.at<float>(2) << "," <<
        delta_pos.at<float>(3) << "," <<
        delta_pos.at<float>(4) << "," <<
        delta_pos.at<float>(5) << "," <<
        std::endl;

    cout << "Current Pose: " << pose << endl;

    cout << "Gradient: " <<
        grad[0] << "," <<
        grad[1] << "," <<
        grad[2] << "," <<
        grad[3] << "," <<
        grad[4] << "," <<
        grad[5] << "," <<
        std::endl;

    float _norm = cv::norm(pose_gradient);
    cout << "Norm: " << _norm << std::endl;
    pose_gradient = pose_gradient/_norm;
    cout << "Gradient normalized: " <<
        pose_gradient.at<float>(0) << "," <<
        pose_gradient.at<float>(1) << "," <<
        pose_gradient.at<float>(2) << "," <<
        pose_gradient.at<float>(3) << "," <<
        pose_gradient.at<float>(4) << "," <<
        pose_gradient.at<float>(5) << "," <<
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

void PoseEstimatorCallback::reset_hessian()
{
    hessian.reset();
    cout << "Reset hessian empty: " << hessian.empty() << endl;
}
