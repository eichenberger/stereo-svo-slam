#ifndef POSE_REFINEMENT_H
#define POSE_REFINEMENT_H

#include <vector>

#include <opencv2/opencv.hpp>

#include "keyframe_manager.hpp"
#include "stereo_slam_types.hpp"

class PoseRefinerCallback;

/*!
 * \brief Do pose refinement based on optical flow (internal use)
 *
 * Handle pose refinement by first doing optical flow and then minimizes
 * the rerpojection error
 */
class PoseRefiner
{
public:
    PoseRefiner(const CameraSettings &camera_settings);

    float refine_pose(KeyFrameManager &keyframe_manager,
            Frame &frame);


private:
    float update_pose(const KeyPoints &keypoints,
            const PoseManager &estimated_pose,
            PoseManager &refined_pose);

    const CameraSettings &camera_settings;
    cv::Ptr<PoseRefinerCallback> solver_callback;
};

#endif
