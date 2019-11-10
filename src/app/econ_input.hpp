#ifndef ECON_INPUT_HPP
#define ECON_INPUT_HPP

#include <iostream>
#include <fstream>

#include "image_input.hpp"

class EconInput: public ImageInput
{
public:
    EconInput(const std::string &camera_path, const std::string &hidraw_path,
            const std::string &settings, int exposure);


    virtual void read(cv::Mat &left, cv::Mat &right);
    virtual void get_camera_settings(CameraSettings &camera_settings);

private:
    cv::Ptr<cv::VideoCapture> cap;

    void set_manual_exposure(const std::string &hidraw, int value);

};


#endif
