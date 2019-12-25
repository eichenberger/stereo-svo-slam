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

#define PRINT_TIME_TRACE

#ifdef PRINT_TIME_TRACE
static TickMeter tick_meter;
#define START_MEASUREMENT() tick_meter.reset(); tick_meter.start()

#define END_MEASUREMENT(_name) tick_meter.stop();\
    cout << "ESTIMATOR: " << _name << " took: " << tick_meter.getTimeMilli() << "ms" << endl

#else
#define START_MEASUREMENT()
#define END_MEASUREMENT(_name)
#endif


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
    Matx66f inv_hessian;
    cv::Ptr<vector<Matx16f>> gradient_times_jacobians;

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

static float get_patch_sum(const Mat &image,const Point2f &center)
{
    Point2f start = center;
    Point ip;

    // Even if center is e.g. 2 we would take center 2.5 because we want
    // to have the center in the middle of the pixel
    start.x -= 0.5f;
    start.y -= 0.5f;

    ip.x = floor(start.x);
    ip.y = floor(start.y);

    // how much do we count the last pixel
    float x2 = start.x - ip.x;
    float y2 = start.y - ip.y;

    // how much do we count the first pixel
    float x1 = 1.0 - x2;
    float y1 = 1.0 - y2;

    const uint8_t *src1 = image.ptr<uint8_t>(ip.y, ip.x);
    const uint8_t *src2 = image.ptr<uint8_t>(ip.y+1, ip.x);
    const uint8_t *src3 = image.ptr<uint8_t>(ip.y+2, ip.x);

    float intensity = x1*y1*(*src1) + y1*(*(src1+1)) + x2*y1*(*(src1+2)) + \
                      x1*(*src2)    + *(src2+1)      + x2*(*(src2+2)) + \
                      x1*y2*(*src3) + y2*(*(src3+1)) + x2*y2*(*(src3+2));

    return intensity;

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
        float k = 10.0;
        for (;i < maxIter; i++) {
            Vec6f x = x0.get_vector() + (k*gradient);
            _x.set_vector(x);
            float new_cost = solver_callback->do_calc(_x);
            if (new_cost < prev_cost) {
                x0 = _x;
                prev_cost = new_cost;
                break;
            }
            else if (fabs(new_cost - prev_cost) < 1.0) {
                // Change below 1px
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

    gradient_times_jacobians.reset(new vector<Matx16f>);
    gradient_times_jacobians->resize(level_keypoints2d.size()*PATCH_SIZE*PATCH_SIZE);

#pragma omp parallel for shared(gradient_times_jacobians, current_image, previous_image)
    for (size_t i = 0; i < level_keypoints2d.size(); i++) {
        // For OMP
        const auto fx = level_camera_settings.fx;
        const auto fy = level_camera_settings.fy;
        const Matx33f rot_mat(pose.get_inv_rotation_matrix());
        const Vec3f translation(pose.get_translation());
        auto kp2d = level_keypoints2d[i];

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
        Matx<float, 2,6> jacobian (-fx/z, 0, fx*x/(z*z), fx*x*y/(z*z), -fx*(1+(x*x)/(z*z)), fx*y/z,
            0, -fy/z, fy*y/(z*z), fy*(1+(y*y)/(z*z)), -fy*x*y/(z*z), -fy*x/z);

        vector<Matx16f>::iterator it = gradient_times_jacobians->begin() + i*PATCH_SIZE*PATCH_SIZE;

        for (size_t r = 0; r < PATCH_SIZE; r++)
        {
            for (size_t c = 0; c < PATCH_SIZE; c++, it++) {
                if ((kp2d.x-1.5) < 0 || (kp2d.y-1.5) < 0 || (kp2d.x+2.5) >= current_image.cols ||
                        (kp2d.y+2.5) >= current_image.rows) {
                    *it = Matx16f::zeros();
                    kp2d.x ++;
                    continue;
                }

#if 0
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
#else
                float int1, int2, int3, int4;
                int1 = get_patch_sum(previous_image, Point2f(kp2d.x+1,kp2d.y));
                int2 = get_patch_sum(previous_image, Point2f(kp2d.x-1,kp2d.y));
                int3 = get_patch_sum(previous_image, Point2f(kp2d.x,kp2d.y+1));
                int4 = get_patch_sum(previous_image, Point2f(kp2d.x,kp2d.y-1));
                float diff1 = int1-int2;
                float diff2 = int3-int4;
#endif
                Matx12f _grad (diff1, diff2);

                Matx<float, 1, 6> _grad_times_jac = _grad*jacobian;
                // Store the result of grad*jacobian for further usage
                *it = _grad_times_jac;

                // Calculate the gauss newton hessian_inv (~second derivate)
                kp2d.x ++;
            }
            kp2d.x -= PATCH_SIZE;
            kp2d.y ++;
        }
    }


    Matx66f hessian;
    // avoid problems with omp -> do it outside of omp loop
    for (auto it: *gradient_times_jacobians) {
        hessian += (it.t()*it);
    }

    inv_hessian = hessian.inv(DECOMP_SVD);
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

    START_MEASUREMENT();
    vector<float> diffs(kps2d.size()*PATCH_SIZE*PATCH_SIZE);
#pragma omp parallel for default(none) shared(diffs, kps2d, previous_image, current_image)
    for (size_t i = 0; i < kps2d.size(); i++) {
        // For OMP
        vector<float>::iterator diff = diffs.begin() + i*PATCH_SIZE*PATCH_SIZE;

        Point2f kp2d(kps2d[i].x-PATCH_SIZE/2, kps2d[i].y-PATCH_SIZE/2);
        Point2f kp2d_ref(level_keypoints2d[i].x-PATCH_SIZE/2, level_keypoints2d[i].y-PATCH_SIZE/2);
        for (size_t r = 0; r < PATCH_SIZE; r++)
        {
            for (size_t c = 0; c < PATCH_SIZE; c++, diff++ , kp2d.x++, kp2d_ref.x++) {
#if 0
                Mat int1, int2;

                cv::getRectSubPix(previous_image, Size(2,2),
                        kp2d_ref, int1, CV_32F);
                cv::getRectSubPix(current_image, Size(2,2),
                        kp2d, int2, CV_32F);

                *diff = sum(int2-int1)(0);
#else

                if ((kp2d_ref.x-0.5) < 0 || (kp2d.x-0.5) < 0 ||
                        (kp2d_ref.y - 0.5) < 0  || (kp2d.y-0.5) < 0 ||
                        (kp2d_ref.x + 1.5) > previous_image.cols ||
                        (kp2d.x + 1.5) > current_image.cols ||
                        (kp2d_ref.y + 1.5) > previous_image.rows||
                        (kp2d.y + 1.5) > current_image.rows)
                    *diff = 0;
                else {
                    float int1 = get_patch_sum(previous_image, kp2d_ref);
                    float int2 = get_patch_sum(current_image, kp2d);
                    *diff = int2 - int1;
                }
#endif
            }
            kp2d.y++;
            kp2d_ref.y++;
            kp2d.x -= PATCH_SIZE;
            kp2d_ref.x -= PATCH_SIZE;
        }

    }
    END_MEASUREMENT("Calculate diffs");

    START_MEASUREMENT();
    Matx16f _residual;
    // OMP can't reduce matrices, therfore do it manually
    float &r1 = _residual(0,0);
    float &r2 = _residual(0,1);
    float &r3 = _residual(0,2);
    float &r4 = _residual(0,3);
    float &r5 = _residual(0,4);
    float &r6 = _residual(0,5);
#pragma omp parallel for default(none) shared(diffs, gradient_times_jacobians, kps2d) \
    reduction(-: r1, r2, r3, r4, r5, r6)
    for (size_t i = 0; i < kps2d.size(); i++) {
        // For OMP
        size_t j = i*PATCH_SIZE*PATCH_SIZE;
        vector<float>::iterator diff = diffs.begin() + i*PATCH_SIZE*PATCH_SIZE;
        for (size_t r = 0; r < PATCH_SIZE; r++)
        {
            for (size_t c = 0; c < PATCH_SIZE; c++, diff++, j++) {
                Matx16f &_grad_times_jac = (*gradient_times_jacobians)[j];
                Matx16f residual_kp = _grad_times_jac*(*diff);
                r1 -= residual_kp(0);
                r2 -= residual_kp(1);
                r3 -= residual_kp(2);
                r4 -= residual_kp(3);
                r5 -= residual_kp(4);
                r6 -= residual_kp(5);
            }
        }
    }
    Matx61f  residual = _residual.t();
    END_MEASUREMENT("Calculate residuals");

    Mat delta_pos;
    // solve(*hessian, residual, delta_pos, DECOMP_SVD);
    delta_pos = Mat(inv_hessian * residual);
    Mat pose_gradient(6, 1, CV_32F);
    exponential_map(delta_pos, pose_gradient);

    Vec3f translation_grad(pose_gradient.ptr<float>(0));
    Vec3f rotation_gradient(pose_gradient.ptr<float>(3));

    // Here we calculate delta t and delta r from compositional lk
    // Basically we just calculate W(W(x;dp),p) which are two matrix
    // multiplications and then get dt and dr
    // This gives us the aditivie gradient
    Matx33f rot_mat(pose.get_rotation_matrix());
    translation_grad = rot_mat*translation_grad;
    rotation_gradient = rot_mat*rotation_gradient;

    grad[0] = translation_grad(0);
    grad[1] = translation_grad(1);
    grad[2] = translation_grad(2);
    grad[3] = rotation_gradient(0);
    grad[4] = rotation_gradient(1);
    grad[5] = rotation_gradient(2);

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
    hessian.release();
    cout << "Reset hessian empty: " << hessian.empty() << endl;
}
