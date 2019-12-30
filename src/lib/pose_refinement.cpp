#include <vector>
#include <cassert>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include "pose_refinement.hpp"
#include "transform_keypoints.hpp"
#include "image_comparison.hpp"
#include "optical_flow.hpp"
#include "exponential_map.hpp"

//#define SUPER_VERBOSE 1

using namespace cv;
using namespace std;

#define PRINT_TIME_TRACE

#ifdef PRINT_TIME_TRACE
static TickMeter tick_meter;
#define START_MEASUREMENT() tick_meter.reset(); tick_meter.start()

#define END_MEASUREMENT(_name) tick_meter.stop();\
    cout << "REFINEMENT: " << _name << " took: " << tick_meter.getTimeMilli() << "ms" << endl

#else
#define START_MEASUREMENT()
#define END_MEASUREMENT(_name)
#endif


// We could use one of the opencv solver. However, they don't really fit
class PoseRefinerCallback : public MinProblemSolver::Function
{
public:
    PoseRefinerCallback(const vector<KeyPoint2d> &keypoints2d,
                        const vector<KeyPoint3d> &keypoints3d,
                        const vector<KeyPointInformation> &keypoint_information,
                        const CameraSettings &camera_settings);
    //bool compute(InputArray param, OutputArray err, OutputArray J) const;
    double calc(const double *x) const;
    float do_calc(const PoseManager &x) const;
    int getDims() const { return 6; }
    void get_gradient(const PoseManager &x, Vec6f &grad);


private:
    const vector<KeyPoint2d> &keypoints2d;
    const vector<KeyPoint3d> &keypoints3d;
    const vector<KeyPointInformation> &keypoint_information;
    const CameraSettings &camera_settings;
};


PoseRefiner::PoseRefiner(const CameraSettings &camera_settings) :
    camera_settings(camera_settings)
{
}

float PoseRefiner::refine_pose(KeyFrameManager &keyframe_manager,
            Frame &frame)
{

    map<int, vector<KeyPoint2d>> reference;
    map<int, vector<KeyPoint2d>> active;
    map<int, vector<float>> err;

    START_MEASUREMENT();
    // Split the keypoints acording to keyframe where it was first seen
    for (size_t i = 0; i < frame.kps.info.size(); i++) {
        KeyPointInformation &info = frame.kps.info[i];

        KeyFrame *keyframe = keyframe_manager.get_keyframe(info.keyframe_id);

        vector<KeyPoint2d> &ref_keypoints = reference[info.keyframe_id];
        vector<KeyPoint2d> &active_keypoints= active[info.keyframe_id];

        ref_keypoints.push_back(keyframe->kps.kps2d[info.keypoint_index]);
        active_keypoints.push_back(frame.kps.kps2d[i]);
    }
    END_MEASUREMENT("Split keypoints");

    START_MEASUREMENT();
    // Do optical flow with each ref image <-> frame set
    // Do a lot of dirty hacks to make omp work...
#if 0
	OpticalFlow optical_flow(camera_settings);
    // No omp because optical flow should alread take care
    for (auto itr : reference) {
		const KeyFrame *keyframe = keyframe_manager.get_keyframe(itr.first);
		vector<KeyPoint2d> &active_keypoints = active[itr.first];
		vector<float> &_err = err[itr.first];
		optical_flow.calculate_optical_flow(keyframe->stereo_image,
				itr.second, frame.stereo_image, active_keypoints, _err);
	}
#else
	// The openmp variant doesn't ouput exactly the same. However, the difference
	// is super small and it's much faster
#pragma omp parallel for default(none) shared(frame, keyframe_manager, reference, active, err)
    for(size_t i = 0; i < reference.size(); i++) {
        OpticalFlow optical_flow(camera_settings);
        // Do stupid hack for openmp
        map<int, vector<KeyPoint2d>>::iterator itr = reference.begin();
        for (size_t j = 0; j < i; j++)
            itr++;
        const KeyFrame *keyframe = keyframe_manager.get_keyframe(itr->first);
        vector<KeyPoint2d> &active_keypoints = active[itr->first];
        vector<float> _err;

        optical_flow.calculate_optical_flow(keyframe->stereo_image,
                itr->second, frame.stereo_image, active_keypoints, _err);

        // Make sure this is atomic
#pragma omp critical
        err[itr->first] = _err;
    }
#endif
    END_MEASUREMENT("Optical flow");

    START_MEASUREMENT();
    // Add refined keypoints back to the original array, start with last entry
    // to allow pop_back
    for (size_t i = frame.kps.kps2d.size(); i > 0; i--) {
        size_t j = i-1;
        KeyPointInformation &info = frame.kps.info[j];
        KeyPoint2d &kp2d = frame.kps.kps2d[j];
        KeyPoint2d &new_estimate = active[info.keyframe_id].back();

        float _err = err[info.keyframe_id].back();
        err[info.keyframe_id].pop_back();

        float diff = (kp2d.x - new_estimate.x)*(kp2d.x - new_estimate.x) +
            (kp2d.y - new_estimate.y)*(kp2d.y - new_estimate.y);
        // Maybe it is occluded now -> ignore it
        if (_err > 20) {
            // cout << "Refine: ignore kp " << kp2d.x << ", " << kp2d.y << endl;
            info.ignore_completely = true;
        }
        // don't move more than 9px!
        else if (diff > 81) {
            info.ignore_during_refinement = true;
        }
        else {
            info.ignore_during_refinement = false;
        }
        kp2d = new_estimate;
        active[info.keyframe_id].pop_back();
    }
    END_MEASUREMENT("Merge keypoints");

#if 0
    Mat result;
    frame.stereo_image.left[0].copyTo(result);
    cvtColor(result, result,  COLOR_GRAY2RGB);
    for (size_t i = 0; i < frame.kps.kps2d.size(); i++) {
        KeyPointInformation &info = frame.kps.info[i];
        KeyPoint2d &kp2d = frame.kps.kps2d[i];
        Scalar color (info.color.r, info.color.g, info.color.b);
        int msize = 10;
        if (frame.kps.info[i].ignore_during_refinement || frame.kps.info[i].ignore_during_refinement) {
            msize=5;
        }
        int marker = MARKER_CROSS;
        Point kp = Point(kp2d.x, kp2d.y);
        cv::drawMarker(result, kp, color, marker, msize);
    }

    imshow("bla", result);
#endif
    PoseManager refined_pose;
    float ret = update_pose(frame.kps, frame.pose, refined_pose);
    frame.pose = refined_pose;

    return ret;
}

#if 0
float PoseRefiner::update_pose(const KeyPoints &keypoints,
        const PoseManager &estimated_pose,
        PoseManager &refined_pose)
{
    const vector<KeyPoint2d> &keypoints2d = keypoints.kps2d;
    const vector<KeyPoint3d> &keypoints3d = keypoints.kps3d;
    const vector<KeyPointInformation> &keypoint_information = keypoints.info;

    refined_pose = estimated_pose;
    Mat _pose = Mat::zeros(6, 1, CV_64FC1);
    double *x0 = _pose.ptr<double>(0);

    Pose pose = estimated_pose.get_pose();

    x0[0] = pose.x;
    x0[1] = pose.y;
    x0[2] = pose.z;
    x0[3] = pose.pitch;
    x0[4] = pose.yaw;
    x0[5] = pose.roll;

    Ptr<MinProblemSolver::Function> callback = new PoseRefinerCallback(keypoints2d,
            keypoints3d, keypoint_information, camera_settings);
    Ptr<DownhillSolver> solver = DownhillSolver::create(callback);
    Mat step = Mat::ones(6,1, CV_64F);
    step *= 0.01;
    solver->setInitStep(step);

    solver->minimize(_pose);

    pose.x = x0[0];
    pose.y = x0[1];
    pose.z = x0[2];
    pose.pitch = x0[3];
    pose.yaw = x0[4];
    pose.roll = x0[5];

    refined_pose.set_pose(pose);

    cout << "estimated pose: " << estimated_pose << endl;
    cout << "refined pose: " << refined_pose << endl;

    vector<KeyPoint2d> projected_keypoints2d;
    project_keypoints(refined_pose, keypoints3d, camera_settings, projected_keypoints2d);
    Mat _projected_keypoints2d(projected_keypoints2d.size(), 2, CV_32F, &projected_keypoints2d[0].x);
    Mat _keypoints2d(keypoints2d.size(), 2, CV_32F, (void*)&keypoints2d[0].x);

    Mat err;
    absdiff(_projected_keypoints2d, _keypoints2d, err);

    cout << "Error after optimization: " << sum(err)[0] << endl;

    return sum(err)[0];
}

#else
float PoseRefiner::update_pose(const KeyPoints &keypoints,
        const PoseManager &estimated_pose,
        PoseManager &refined_pose)
{
    const vector<KeyPoint2d> &keypoints2d = keypoints.kps2d;
    const vector<KeyPoint3d> &keypoints3d = keypoints.kps3d;
    const vector<KeyPointInformation> &keypoint_information = keypoints.info;

    size_t maxIter = 50;

    Ptr<PoseRefinerCallback> solver_callback = new PoseRefinerCallback(keypoints2d,
            keypoints3d, keypoint_information, camera_settings);

    PoseManager x0 = estimated_pose;
    PoseManager _x;

    float prev_cost = solver_callback->do_calc(x0);
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
            else if (fabs(new_cost - prev_cost) < 0.0001) {
                i = maxIter;
                break;
            }
            else {
                k /= 2;
            }
        }
    }

    refined_pose = x0;

    return prev_cost;

}

#endif

PoseRefinerCallback::PoseRefinerCallback(const vector<KeyPoint2d> &keypoints2d,
        const vector<KeyPoint3d> &keypoints3d,
        const vector<KeyPointInformation> &keypoint_information,
        const CameraSettings &camera_settings):
    keypoints2d(keypoints2d),
    keypoints3d(keypoints3d),
    keypoint_information(keypoint_information),
    camera_settings(camera_settings)
{
}

double PoseRefinerCallback::calc(const double *x) const
{
    Pose pose;
    pose.x = x[0];
    pose.y = x[1];
    pose.z = x[2];
    pose.pitch = x[3];
    pose.yaw = x[4];
    pose.roll = x[5];

    PoseManager _pose;
    _pose.set_pose(pose);

    return do_calc(_pose);
}

float PoseRefinerCallback::do_calc(const PoseManager &pose) const
{
    vector<KeyPoint2d> projected_keypoints2d;
    project_keypoints(pose, keypoints3d, camera_settings, projected_keypoints2d);
    Mat _projected_keypoints2d(projected_keypoints2d.size(), 2, CV_32F, &projected_keypoints2d[0].x);
    Mat _keypoints2d(keypoints2d.size(), 2, CV_32F, (void*)&keypoints2d[0].x);

    Mat diff;
    absdiff(_projected_keypoints2d, _keypoints2d, diff);

    float tot_diff = 0;
    // TODO: omp has almost no effect
// #pragma omp parallel for shared(keypoint_information, tot_diff, diff)
    for (size_t i=0; i < keypoint_information.size(); i++) {
        float *_diff = diff.ptr<float>(i);
        if (!keypoint_information[i].ignore_during_refinement && !keypoint_information[i].ignore_completely) {
            tot_diff += *_diff + *(_diff+1);
        }
    }
#ifdef SUPER_VERBOSE
    cout << "pose: " << pose;
    cout << "Diff: " << tot_diff << endl;
#endif

    return tot_diff;
}

void PoseRefinerCallback::get_gradient(const PoseManager &x, Vec6f &grad)
{
    vector<KeyPoint2d> projected_keypoints2d;
    project_keypoints(x, keypoints3d, camera_settings, projected_keypoints2d);

    Matx61f err(Matx61f::zeros());
    Matx66f hessian(Matx66f::zeros());

    Vec2f tot_diff(0,0);

#pragma omp parallel for default(none) \
    shared(keypoints3d, keypoint_information, tot_diff, err, hessian, projected_keypoints2d)
    for (size_t i = 0; i < keypoints3d.size(); i++) {
        const auto fx = camera_settings.fx;
        const auto fy = camera_settings.fy;
        auto x = keypoints3d[i].x;
        auto y = keypoints3d[i].y;
        auto z = keypoints3d[i].z;

        if (keypoint_information[i].ignore_during_refinement ||
                keypoint_information[i].ignore_completely)
            continue;

        Matx<float, 2,6> jacobian (-fx/z, 0, fx*x/(z*z), fx*x*y/(z*z), -fx*(1+(x*x)/(z*z)), fx*y/z,
            0, -fy/z, fy*y/(z*z), fy*(1+(y*y)/(z*z)), -fy*x*y/(z*z), -fy*x/z);

        Vec2f diff (keypoints2d[i].x -projected_keypoints2d[i].x,
                keypoints2d[i].y -projected_keypoints2d[i].y);

        if ((fabs(diff(0)) > 3.0) ||
                (fabs(diff(1)) > 3.0))
            continue;
        tot_diff += diff;
        Matx<float, 6, 2> transposed_jacobian = jacobian.t();
        // It seems that matrices are not atomic even if they are marked as shared
#pragma omp critical
        hessian += transposed_jacobian*jacobian;
#pragma omp critical
        err += transposed_jacobian * diff;
    }

    Matx66f hessian_inv = hessian.inv(DECOMP_SVD);
    Mat twist (hessian_inv*err); //  / keypoints2d.size();

    Mat gradient(1, 6, CV_32F);

    exponential_map(twist, gradient);
    float *data = gradient.ptr<float>();


#ifdef SUPER_VERBOSE
    cout << "gradient: " << data[0] << "," << data[1] << "," << data[2] << ","<< data[3] << ","<< data[4] << ","<< data[5] << "," << endl;
    cout << "Total diff: " << tot_diff << endl;
#endif
    grad = Vec6f(data);
}
