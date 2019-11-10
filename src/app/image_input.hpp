#ifndef IMAGE_INPUT_HPP
#define IMAGE_INPUT_HPP

#include <opencv2/opencv.hpp>
#include "stereo_slam_types.hpp"

class ImageInput {
public:
    ImageInput();
    virtual ~ImageInput() = 0;

    virtual void read(cv::Mat &left, cv::Mat &right) = 0;
    virtual void get_camera_settings(CameraSettings &camera_settings) = 0;

protected:
    CameraSettings camera_settings;

    virtual void read_settings(const std::string &settings);
};

#endif
