#include <math.h>
#include <algorithm>

#include<opencv2/opencv.hpp>

#include "transform_keypoints.hpp"
#include "image_comparison.hpp"
#include "convert_pose.hpp"
#include "depth_filter.hpp"

using namespace cv;
using namespace std;

#define PRINT_TIME_TRACE

#ifdef PRINT_TIME_TRACE
static TickMeter tick_meter;
#define START_MEASUREMENT() tick_meter.reset(); tick_meter.start()

#define END_MEASUREMENT(_name) tick_meter.stop();\
    cout << _name << " took: " << tick_meter.getTimeMilli() << "ms" << endl

#else
#define START_MEASUREMENT()
#define END_MEASUREMENT(_name)
#endif



DepthFilter::DepthFilter(const vector<KeyFrame> &keyframes, const CameraSettings &camera_settings) :
    keyframes(keyframes), camera_settings(camera_settings)
{
}

void DepthFilter::update_depth(Frame &frame)
{
    triangulate_points(frame);
}

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
static inline Mat linear_ls_triangulation(KeyPoint2d pt1,       //homogenous image point (u,v,1)
                   Pose pose1,       //camera 1 pose
                   KeyPoint2d pt2,      //homogenous image point in 2nd camera
                   Pose pose2,       //camera 2 pose
                   const CameraSettings &camera_settings
                                   )
{

    Mat camera_matrix = (Mat_<float>(3,3) <<
            camera_settings.fx, 0, camera_settings.cx,
            0, camera_settings.fy, camera_settings.cy,
            0, 0, 1);

    Vec3f angles1(&pose1.pitch);
    Vec3f angles2(&pose2.pitch);

    Mat _projection1(3, 4, CV_32F), _projection2(3, 4, CV_32F);
    cv::Rodrigues(-angles1, _projection1(Rect(0, 0, 3, 3)));
    cv::Rodrigues(-angles2, _projection2(Rect(0, 0, 3, 3)));

    _projection1.at<float>(0,3)  = -pose1.x;
    _projection1.at<float>(1,3)  = -pose1.y;
    _projection1.at<float>(2,3)  = -pose1.z;


    _projection2.at<float>(0,3)  = -pose2.x;
    _projection2.at<float>(1,3)  = -pose2.y;
    _projection2.at<float>(2,3)  = -pose2.z;

    _projection1 = camera_matrix * _projection1;
    _projection2 = camera_matrix * _projection2;

    vector<Point2f> pts1, pts2;
    pts1.push_back(Point2f(pt1.x, pt1.y));
    pts2.push_back(Point2f(pt2.x, pt2.y));

    Mat kps3d;

    triangulatePoints(_projection1, _projection2, pts1, pts2, kps3d);

    return (Mat_<float>(3,1) <<
            kps3d.at<float>(0)/kps3d.at<float>(3),
            kps3d.at<float>(1)/kps3d.at<float>(3),
            kps3d.at<float>(2)/kps3d.at<float>(3));
}

void DepthFilter::triangulate_points(Frame &frame)
{

    for (size_t i = 0; i < frame.kps.kps2d.size(); i++) {
        KeyPointInformation &info  = frame.kps.info[i];
        KeyPoint3d &kp3d = frame.kps.kps3d[i];
        const KeyFrame &keyframe  = keyframes[info.keyframe_id];

        START_MEASUREMENT();
        vector<KeyPoint3d> max_min_kp3d(2);
        move_point_along_ray(keyframe.pose,
                kp3d, -info.seed.z_range, max_min_kp3d[0]);

        move_point_along_ray(keyframe.pose,
                kp3d, info.seed.z_range, max_min_kp3d[1]);

        vector<KeyPoint2d> max_min_kp2d;
        project_keypoints(frame.pose, max_min_kp3d,
                camera_settings, max_min_kp2d);

        END_MEASUREMENT("projection");
        START_MEASUREMENT();
        KeyPoint2d kp2d_ref = keyframe.kps.kps2d[info.keypoint_index];
        KeyPoint2d kp2d;
        float diff;
        if (max_min_kp2d[0].x < max_min_kp2d[1].x)
            diff = match(keyframe, frame, kp2d_ref,
                    max_min_kp2d[0], max_min_kp2d[1], kp2d);
        else
            diff = match(keyframe, frame, kp2d_ref,
                    max_min_kp2d[1], max_min_kp2d[0], kp2d);

        cout << "Match from " << max_min_kp2d[0].x << "x" << max_min_kp2d[0].y
            << " to " << max_min_kp2d[1].x << "x" << max_min_kp2d[1].y <<
            " found : " << kp2d.x << "x" << kp2d.y << " Diff: " << diff << endl;
        cout << "Reference at: " << kp2d_ref.x << "x" << kp2d_ref.y << endl;


        END_MEASUREMENT("match");



        START_MEASUREMENT();
        Mat kp3d_update = linear_ls_triangulation(kp2d_ref, keyframe.pose, kp2d,
                frame.pose, camera_settings);

        END_MEASUREMENT("triangulation");
        cout << "Old kp3d: " << kp3d.x << ", " << kp3d.y << ", " << kp3d.z << endl;
        cout << "New kp3d: " << kp3d_update.at<float>(0) << ", " <<
            kp3d_update.at<float>(1) << ", " <<
            kp3d_update.at<float>(2) << endl;

        START_MEASUREMENT();

        Mat angles(1, 3, CV_32F, (void*)&keyframe.pose.pitch);
        Mat rot_mat(3, 3, CV_32F);
        Rodrigues(angles, rot_mat);

        Mat translation(3, 1, CV_32F, (void*)&keyframe.pose.x);

        Mat kp3d_update_ref = rot_mat * kp3d_update + translation;
        double z = kp3d_update_ref.at<float>(2);
        double px_error_angle = atan(1.0/(2.0*camera_settings.fx))*2.0; // law of chord (sehnensatz)
        KeyPoint3d point3d;
        point3d.x = kp3d_update.at<float>(0);
        point3d.y = kp3d_update.at<float>(1);
        point3d.z = kp3d_update.at<float>(2);
        double tau = compute_tau(frame.pose, point3d, z, px_error_angle);

        cout << "New z: " << z << ", tau: " << tau << endl;

        cout << "Seed before update ID: " << info.keyframe_id << "." <<info.keypoint_index  <<
            " a: " << info.seed.a << ", " <<
            "b: " << info.seed.b << ", " <<
            "mu: " << info.seed.mu << ", " <<
            "sigma2: " << info.seed.sigma2 << endl;

        float tau_inverse = 0.5 * (1.0/max(0.0000001, z-tau) - 1.0/(z+tau));
        update_seed (1.0/z, tau_inverse*tau_inverse, &info.seed);

        cout << "Seed after update ID: " << info.keyframe_id << "." <<info.keypoint_index <<
            " a: " << info.seed.a << ", " <<
            "b: " << info.seed.b << ", " <<
            "mu: " << info.seed.mu << ", " <<
            "z: " << 1.0/info.seed.mu << ", " <<
            "sigma2: " << info.seed.sigma2 << endl;

        info.seed.z_range = std::min<float>(z, 0.01 + 5*sqrt(info.seed.sigma2));

        END_MEASUREMENT("seed update");
    }
}

float DepthFilter::match(const KeyFrame &keyframe, const Frame &current,
            const KeyPoint2d &ref,
            const KeyPoint2d &kp2d_min,
            const KeyPoint2d &kp2d_max,
            KeyPoint2d &matched_kp2d)
{
    float min_diff = numeric_limits<float>::infinity();
    float diff_y = kp2d_max.y-kp2d_min.y;
    float diff_x = kp2d_max.x-kp2d_min.x;
    float a = diff_y/(diff_x+0.0001);
    float m = kp2d_min.y - a * kp2d_min.x;
    float step_size = abs(a) < 1.0 ? 1.0 : fabs(1/a);

    KeyPoint2d kp2d = kp2d_min;

    while (kp2d.x < kp2d_max.x+1.0) {
        vector<float> diffs;
        float diff = get_intensity_diff(current.stereo_image.left[0],
                keyframe.stereo_image.left[0], kp2d, ref,
                camera_settings.window_size_depth_calculator);

        if (diff < min_diff) {
            matched_kp2d = kp2d;
            min_diff = diff;

        }
        kp2d.x += step_size;
        kp2d.y = a*kp2d.x + m;
    }

    return min_diff;
}

void DepthFilter::move_point_along_ray(const Pose &pose, const KeyPoint3d &pt,
        float d, KeyPoint3d &new_pt)
{
    Vec3f C (pose.x, pose.y, pose.z);
    Vec3f kp3d (pt.x, pt.y, pt.z);

    Vec3f dt = kp3d - C;

    normalize(dt, dt);
    // Move in the direction of the vector
    kp3d = kp3d + d*dt;

    new_pt.x = kp3d(0);
    new_pt.y = kp3d(1);
    new_pt.z = kp3d(2);
}

// Calculate the normal probability density distribution
static float pdf(float x, float mu, float var)
{
    float den = sqrt(2*M_PI*var);
    float counter = exp(-(x-mu)*(x-mu)/(2*var));

    return counter/den;
}

// Use to compute real z (1/mu), variance (sigma2), inlier count (a) and outlier count (b)
// x = 1/z, tau from computeTau and seed is previous seed
// see rpg_svo
void DepthFilter::update_seed(const float x, const float tau2, Seed* seed)
{
    float norm_scale = seed->sigma2 + tau2;
    float s2 = 1./(1.0/seed->sigma2 + 1.0/tau2);
    float m = s2*(seed->mu/seed->sigma2 + x/tau2);
    float C1 = seed->a/(seed->a+seed->b) * pdf(x, seed->mu, norm_scale);
    float C2 = seed->b/(seed->a+seed->b) * 1./seed->z_range;
    float normalization_constant = C1 + C2;
    C1 /= normalization_constant;
    C2 /= normalization_constant;
    float f = C1*(seed->a+1.)/(seed->a+seed->b+1.) + C2*seed->a/(seed->a+seed->b+1.);
    float e = C1*(seed->a+1.)*(seed->a+2.)/((seed->a+seed->b+1.)*(seed->a+seed->b+2.))
            + C2*seed->a*(seed->a+1.0f)/((seed->a+seed->b+1.0f)*(seed->a+seed->b+2.0f));

    // update parameters
    float mu_new = C1*m+C2*seed->mu;
    seed->sigma2 = C1*(s2 + m*m) + C2*(seed->sigma2 + seed->mu*seed->mu) - mu_new*mu_new;
    seed->mu = mu_new;
    seed->a = (e-f)/(f-e/f);
    seed->b = seed->a*(1.0f-f)/f;
}

// Use to compute tau
double DepthFilter::compute_tau(const Pose &pose, const KeyPoint3d &point,
        const double z, const double px_error_angle)
{
    Mat t = (Mat_<float>(3,1) <<
            pose.x, pose.y, pose.z);
    Mat f = (Mat_<float>(3,1) <<
            point.x, point.y, point.z);

    normalize(f, f);
    // TODO: Is f really the normalized point vector?
    Mat a = f*z-t;
    double t_norm = norm(t);
    double a_norm = norm(a);
    double norm_dot = f.dot(t)/t_norm;
    double alpha = acos(norm_dot); // dot product
    double beta = acos(a.dot(-t)/(t_norm*a_norm)); // dot product
    double beta_plus = beta + px_error_angle;
    double gamma_plus = M_PI-alpha-beta_plus; // triangle angles sum to PI
    double z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
    return (z_plus - z); // tau
}
