#ifndef ECON_INPUT_HPP
#define ECON_INPUT_HPP

#include <iostream>
#include <fstream>

#include "image_input.hpp"

/*!
 * \brief Data received from IMU
 */
struct ImuData{
    float acceleration_x;   //!< Acceleration in x direction
    float acceleration_y;   //!< Acceleration in y direction
    float acceleration_z;   //!< Acceleration in z direction
    float gyro_x;           //!< Angle velocity around x axis
    float gyro_y;           //!< Angle velocity around y axis
    float gyro_z;           //!< Angle velocity around z axis
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
    virtual bool set_manual_exposure(int exposure); //!< Set the exposure value (1 = auto exposure -> 30000)
    virtual bool configure_imu();   //!< Configure IMU (e.g. frequency, resolution, etc.)
    virtual bool get_imu_data(ImuData &imu_data); //!< Read IMU data
    virtual bool set_hdr(bool hdr); //!< Set HDR mode to on or off
    virtual bool read_temperature(float &temperature);  //!< Read camera temperature
    float get_freqency() { return frequency; }
    void calibrate_imu();   //!< Calibrate the IMU
    bool imu_available();   //!< Check if IMU is available

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
