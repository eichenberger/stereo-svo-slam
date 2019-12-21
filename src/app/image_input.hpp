#ifndef IMAGE_INPUT_HPP
#define IMAGE_INPUT_HPP

#include <opencv2/opencv.hpp>
#include "stereo_slam_types.hpp"

class ImageInput {
public:
    ImageInput();
    virtual ~ImageInput() = 0;

    virtual bool read(cv::Mat &left, cv::Mat &right, float &time_stamp) = 0;
    virtual void get_camera_settings(CameraSettings &camera_settings) = 0;
    uint32_t get_fps() const { return fps; }

protected:
    uint32_t fps;
    CameraSettings camera_settings;

    virtual void read_settings(const std::string &settings);
};

#endif
