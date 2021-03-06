#ifndef EUROC_INPUT_HPP
#define EUROC_INPUT_HPP

#include <iostream>
#include <fstream>

#include "image_input.hpp"

/*!
 * \brief Class for euroc video input
 *
 * This class accepts a path to the euroc mav0 folder and then starts to
 * read the images from there.
 */
class EurocInput: public ImageInput
{
public:
    /*!
     * \brief Create the EurocInput object
     *
     * @param[i] image_path Where to find the images (mav0 folder)
     * @param[i] settings The settings file (.yaml)
     */
    EurocInput(const std::string &image_path, const std::string &settings);


    virtual bool read(cv::Mat &left, cv::Mat &right, float &time_stamp);
    virtual void get_camera_settings(CameraSettings &camera_settings);
    void jump_to(int frame_number); //!< Jump to a specific frame

private:
    size_t read_count;
    cv::Ptr<cv::VideoCapture> cap;
    void load_images(std::string image_path);

    std::vector<float> timestamps;
    std::vector<std::string> left_images;
    std::vector<std::string> right_images;
    cv::Mat M1l,M2l,M1r,M2r;
};


#endif
