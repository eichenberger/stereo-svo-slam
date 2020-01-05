#include "image_input.hpp"

using namespace cv;

ImageInput::~ImageInput()
{
}

ImageInput::ImageInput()
{
}

void ImageInput::read_settings(const std::string &settings)
{
    FileStorage fs(settings, FileStorage::READ);
    camera_settings.fx = fs["Camera1.fx"];
    camera_settings.fy = fs["Camera1.fy"];
    camera_settings.cx = fs["Camera1.cx"];
    camera_settings.cy = fs["Camera1.cy"];
    camera_settings.baseline = fs["Camera.baseline"];
    camera_settings.window_size_pose_estimator = fs["Camera.window_size_pose_estimator"];
    camera_settings.window_size_opt_flow = fs["Camera.window_size_opt_flow"];
    camera_settings.window_size_depth_calculator = fs["Camera.window_size_depth_calculator"];
    camera_settings.max_pyramid_levels = fs["Camera.max_pyramid_levels"];
    camera_settings.min_pyramid_level_pose_estimation = fs["Camera.min_pyramid_level_pose_estimation"];

    camera_settings.k1 = fs["Camera1.k1"];
    camera_settings.k2 = fs["Camera1.k2"];
    camera_settings.k3 = fs["Camera1.k3"];
    camera_settings.p1 = fs["Camera1.p1"];
    camera_settings.p2 = fs["Camera1.p2"];

    camera_settings.grid_width = fs["Camera.grid_width"];
    camera_settings.grid_height = fs["Camera.grid_height"];
    camera_settings.search_x = fs["Camera.search_x"];
    camera_settings.search_y = fs["Camera.search_y"];
}
