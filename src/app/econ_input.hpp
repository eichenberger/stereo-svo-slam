#ifndef ECON_INPUT_HPP
#define ECON_INPUT_HPP

#include <iostream>
#include <fstream>

#include "image_input.hpp"

struct ImuData{
    float acceleration_x;
    float acceleration_y;
    float acceleration_z;
    float gyro_x;
    float gyro_y;
    float gyro_z;
};

/*!
 * \brief Class for Econ Tara video input
 *
 * This class accepts a path to the video file, the hidraw device to control
 * exposer etc, the YAML settings file and the
 */
class EconInput: public ImageInput
{
public:
    /*!
     * \brief Create the EconInput object
     *
     * @param[i] camera_path The video device e.g. /dev/video0
     * @param[i] hidraw_path The hidraw path to control the camera (exposure) e.g. /dev/hidraw0
     * @param[i] settings The settings file (.yaml)
     * @param[i] hidraw_imu_path The hidraw path where we can find the IMU data e.g. /dev/hidraw1
     */
    EconInput(const std::string &camera_path, const std::string &hidraw_path,
            const std::string &settings, const std::string &hidraw_imu_path = "");

    virtual bool read(cv::Mat &left, cv::Mat &right, float &time_stamp);
    virtual void get_camera_settings(CameraSettings &camera_settings);
    virtual bool set_manual_exposure(int exposure);
    virtual bool configure_imu();
    virtual bool get_imu_data(ImuData &imu_data);
    virtual bool set_hdr(bool hdr);
    virtual bool read_temperature(float &temperature);
    float get_freqency() { return frequency; }
    void calibrate_imu();
    bool imu_available();

private:
    const std::string &hidraw;
    const std::string &hidraw_imu;
    cv::Ptr<cv::VideoCapture> cap;
    float gyro_sensitivity;
    float acc_sensitivity;
    float frequency;
    float time_stamp;
    ImuData imu_calibration;

    std::fstream fhidraw;
    std::ifstream fhidraw_imu;

};

std::ostream& operator<<(std::ostream& os, const ImuData& imu);

#endif
