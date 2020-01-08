#ifndef VIDEO_INPUT_HPP
#define VIDEO_INPUT_HPP

#include <iostream>
#include <fstream>

#include "image_input.hpp"
/*!
 * \brief Class for Video File input
 *
 * Video Files need to have the images aligned horizontally
 * | video left | video right |
 */

class VideoInput: public ImageInput
{
public:
    VideoInput(const std::string &video_path, const std::string &settings);


    virtual bool read(cv::Mat &left, cv::Mat &right, float &time_stamp);
    virtual void get_camera_settings(CameraSettings &camera_settings);
    void jump_to(int frame_number);

private:
    cv::Ptr<cv::VideoCapture> cap;
    float time_stamp;
};


#endif
