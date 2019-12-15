#include <unistd.h>

#include "econ_input.hpp"

using namespace std;
using namespace cv;

#define ARRAY_SIZE(x) (sizeof(x)/sizeof(x[0]))

#define BUFFER_LENGTH				65

#define HIDRAW_CAMERA_CONTROL_STEREO        0x78

#define HIDRAW_SEND_IMU_VAL_BUFF            0x06
#define HIDRAW_CONTROL_IMU_VAL				0x05
#define HIDRAW_SET_SUCCESS                  0x01
#define HIDRAW_GET_SUCCESS                  0x01
#define HIDRAW_GET_IMU_TEMP_DATA			0x0D

#define HIDRAW_IMU_NUM_OF_VAL				0xFF

#define HIDRAW_IMU_ACC_VAL                  0xFE
#define HIDRAW_IMU_GYRO_VAL                 0xFD

EconInput::EconInput(const std::string &camera_path, const std::string &hidraw_path,
        const std::string &hidraw_imu_path, const std::string &settings) :
    hidraw(hidraw_path), hidraw_imu(hidraw_imu_path)
{
    read_settings(settings);

    cap = new VideoCapture(camera_path);

    cap->set(CAP_PROP_FRAME_WIDTH, 752);
    cap->set(CAP_PROP_FRAME_HEIGHT, 480);
}

bool EconInput::set_manual_exposure(int exposure)
{
    int MAX_EXPOSURE = 300000;
    cout << "Set expsure to: " << exposure << endl;
    if (exposure >= MAX_EXPOSURE) {
        cout << "Exposure must be less than" << MAX_EXPOSURE << "is " << exposure << ")" << endl;
        return false;
    }

    uint8_t buffer[] = {
        0x00,
        HIDRAW_CAMERA_CONTROL_STEREO,
        0x02,
        (uint8_t)((exposure >> 24)&0xFF),
        (uint8_t)((exposure >> 16)&0xFF),
        (uint8_t)((exposure>>8)&0xFF),
        (uint8_t)(exposure&0xFF)};


    ofstream f;
    f.open(hidraw, ios::binary);
    if (!f.is_open()) {
        cout << "Set exposure: Can't open hidraw device: " << hidraw << endl;
        return false;
    }
    f.write((char*)buffer, ARRAY_SIZE(buffer));
    f.flush();
    f.close();
    return true;
}

void EconInput::get_camera_settings(CameraSettings &camera_settings)
{
    camera_settings = this->camera_settings;
}

bool EconInput::read(cv::Mat &left, cv::Mat &right)
{
    Mat image;
    if (!cap->read(image))
        return false;

    extractChannel(image, right, 1);
    extractChannel(image, left, 2);

    return true;
}

bool EconInput::configure_imu()
{
    uint8_t buffer[] = {
        0x00,
        HIDRAW_CAMERA_CONTROL_STEREO,  // Stereo config
        0x05,  // IMU config
        0x01,  // Update enable
        0x07,  // Enable all accelerormeter axis
        0x04,  // Accelerometer at 104 Hz
        0x00,  // 2G senstivity

        0x00,
        0x00,

        0x07, // Gyro all axes enable
        0x00,
        0x00 // 250 DGPS
    };

    acc_sensitivity = 0.00061;
    gyro_sensitivity = 0.00875;

    ofstream f;
    f.open(hidraw, ios::binary);
    if (!f.is_open()) {
        cout << "Configure IMU: Can't open hidraw device: " << hidraw << endl;
        return false;
    }
    f.write((char*)buffer, ARRAY_SIZE(buffer));

    memset(buffer, 0, sizeof(buffer));

	buffer[1] = HIDRAW_CAMERA_CONTROL_STEREO;
	buffer[2] = HIDRAW_CONTROL_IMU_VAL;
	buffer[3] = 0x01;
	buffer[6] = HIDRAW_IMU_NUM_OF_VAL;
	buffer[7] = 0x00;//(INT8)((lIMUInput.IMU_NUM_OF_VALUES & 0xFF00) >> 8);
	buffer[8] = 0x00;//(INT8)(lIMUInput.IMU_NUM_OF_VALUES & 0xFF);
    f.write((char*)buffer, ARRAY_SIZE(buffer));

    f.flush();
    f.close();
    return true;
}


bool EconInput::get_imu_data(ImuData &imu_data)
{
    ifstream imu;
    imu.open(hidraw_imu, ios::binary |ios::in);
    if (!imu.is_open()) {
        cout << "Get IMU Data: Can't open hidraw_imu device: " << hidraw_imu << endl;
        return false;
    }

    uint8_t read_buffer[64];
    memset(read_buffer, 0, ARRAY_SIZE(read_buffer));
    imu.read((char*)&read_buffer[0], ARRAY_SIZE(read_buffer));

    if (read_buffer[0] != HIDRAW_CAMERA_CONTROL_STEREO) {
        cout << "Didn't receive correct command: " << read_buffer[0] <<
            " instead of " << HIDRAW_CAMERA_CONTROL_STEREO << endl;
        return false;
    }

    if (read_buffer[1] != HIDRAW_SEND_IMU_VAL_BUFF) {
        cout << "Didn't receive correct command: " << read_buffer[1] <<
            " instead of " << HIDRAW_SEND_IMU_VAL_BUFF << endl;
        return false;
    }

    if (read_buffer[48] != HIDRAW_SET_SUCCESS) {
        cout << "Didn't receive succeed: " << read_buffer[48] <<
            " instead of " << HIDRAW_SET_SUCCESS << endl;
        return false;
    }

    if (read_buffer[4] != HIDRAW_IMU_ACC_VAL) {
        cout << "No Acceleration data received" << endl;
        return false;
    }
    imu_data.acceleration_x = (((int16_t)((read_buffer[6]) | (read_buffer[5]<<8))) * acc_sensitivity);
    imu_data.acceleration_y = (((int16_t)((read_buffer[8]) | (read_buffer[7]<<8))) * acc_sensitivity);
    imu_data.acceleration_z = (((int16_t)((read_buffer[10]) | (read_buffer[9]<<8))) * acc_sensitivity);

    if (read_buffer[15] != HIDRAW_IMU_GYRO_VAL) {
        cout << "No Gyro data received" << endl;
        return false;
    }
    imu_data.gyro_x = (((int16_t)((read_buffer[17]) | (read_buffer[16]<<8))) * gyro_sensitivity);
    imu_data.gyro_y = (((int16_t)((read_buffer[19]) | (read_buffer[18]<<8))) * gyro_sensitivity);
    imu_data.gyro_z = (((int16_t)((read_buffer[21]) | (read_buffer[20]<<8))) * gyro_sensitivity);

    return true;
}

bool EconInput::read_temperature(float &temperature)
{
    uint8_t buffer[] {
        0,
        HIDRAW_CAMERA_CONTROL_STEREO,
        HIDRAW_GET_IMU_TEMP_DATA
    };

    fstream f;
    f.open(hidraw, ios::binary | ios::in | ios::out);
    if (!f.is_open()) {
        cout << "Get IMU Data: Can't open hidraw device: " << hidraw << endl;
        return false;
    }
    f.write((char*)buffer, sizeof(buffer));

    uint8_t read_buffer[7];
    memset(read_buffer, 0, ARRAY_SIZE(read_buffer));
    f.read((char*)&read_buffer, ARRAY_SIZE(read_buffer));

    f.close();

    if (read_buffer[0] != HIDRAW_CAMERA_CONTROL_STEREO) {
        cout << "Didn't receive correct command: " << read_buffer[0] <<
            " instead of " << HIDRAW_CAMERA_CONTROL_STEREO << endl;
        return false;
    }

    if (read_buffer[1] != HIDRAW_GET_IMU_TEMP_DATA) {
        cout << "Didn't receive correct command: " << read_buffer[1] <<
            " instead of " << HIDRAW_GET_IMU_TEMP_DATA << endl;
        return false;
    }

    if (read_buffer[6] != HIDRAW_GET_SUCCESS) {
        cout << "Didn't receive succeed: " << read_buffer[6] <<
            " instead of " << HIDRAW_GET_SUCCESS << endl;
        return false;
    }

    temperature = (read_buffer[2]) + (0.5*read_buffer[3]);
    return true;
}
