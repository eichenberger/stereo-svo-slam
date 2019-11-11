#ifndef VIDEO_INPUT_HPP
#define VIDEO_INPUT_HPP

#include <iostream>
#include <fstream>

#include "image_input.hpp"

class VideoInput: public ImageInput
{
public:
    VideoInput(const std::string &video_path, const std::string &settings);


    virtual void read(cv::Mat &left, cv::Mat &right);
    virtual void get_camera_settings(CameraSettings &camera_settings);
    void jump_to(int frame_number);

private:
    cv::Ptr<cv::VideoCapture> cap;
};


#endif
