#ifndef DEPTH_FILTER_HPP
#define DEPTH_FILTER_HPP

#include "keyframe_manager.hpp"

#include "stereo_slam_types.hpp"

/*!
 * \brief Update point cloud, detect outliers (internal use)
 */
class DepthFilter
{
public:
    DepthFilter(KeyFrameManager &keyframe_manager,
            const CameraSettings &camera_settings);

    void update_depth(Frame &frame, std::vector<KeyPoint3d> &updated_kps3d);

private:
    void calculate_disparities(Frame &frame, std::vector<float> &disparity);
    void outlier_check(Frame &frame, const std::vector<float> &disparities);
    void update_kps3d(Frame &frame, std::vector<KeyPoint3d> &updated_kps3d);

    KeyFrameManager &keyframe_manager;
    const CameraSettings &camera_settings;

};



#endif
