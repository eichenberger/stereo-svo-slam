#ifndef IMAGE_INPUT_HPP
#define IMAGE_INPUT_HPP

#include <opencv2/opencv.hpp>
#include "stereo_slam_types.hpp"

/*!
 * \brief Abstract class for image input
 *
 * Abstract class used for all kind of input image sources
 */
class ImageInput {
public:
    ImageInput();
    virtual ~ImageInput() = 0;

    virtual bool read(cv::Mat &left, cv::Mat &right, float &time_stamp) = 0;    //!< Read a new image from the camera including a timestamp
    virtual void get_camera_settings(CameraSettings &camera_settings) = 0;  //!< Get the camera settings from the yaml file
    uint32_t get_fps() const { return fps; }    //!< Get the frame rate of the camera

protected:
    uint32_t fps;
    CameraSettings camera_settings;

    virtual void read_settings(const std::string &settings);
};

#endif
