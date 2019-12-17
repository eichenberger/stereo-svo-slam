#include <unistd.h>

#include "econ_input.hpp"

using namespace std;
using namespace cv;

#define ARRAY_SIZE(x) (sizeof(x)/sizeof(x[0]))

#define BUFFER_LENGTH                       65

#define HIDRAW_CAMERA_CONTROL_STEREO        0x78

#define HIDRAW_SET_IMU_CONFIG               0x04
#define HIDRAW_SEND_IMU_VAL_BUFF            0x06
#define HIDRAW_CONTROL_IMU_VAL              0x05
#define HIDRAW_SET_SUCCESS                  0x01
#define HIDRAW_GET_SUCCESS                  0x01
#define HIDRAW_GET_IMU_TEMP_DATA            0x0D
#define HIDRAW_SET_HDR_MODE_STEREO          0x0E

#define HIDRAW_IMU_NUM_OF_VAL               0xFF

#define HIDRAW_IMU_ACC_VAL                  0xFE
#define HIDRAW_IMU_GYRO_VAL                 0xFD

EconInput::EconInput(const std::string &camera_path, const std::string &hidraw_path, const std::string &settings,
        const std::string &hidraw_imu_path) :
    hidraw(hidraw_path), hidraw_imu(hidraw_imu_path)
{
    read_settings(settings);

    cap = new VideoCapture(camera_path);

    cap->set(CAP_PROP_FRAME_WIDTH, 752);
    cap->set(CAP_PROP_FRAME_HEIGHT, 480);
    fhidraw.open(hidraw, ios::binary |ios::in | ios::out);
    if (!fhidraw.is_open())
        cout << "Can't open hidraw device: " << hidraw << endl;
    if (hidraw_imu != "") {
        fhidraw_imu.open(hidraw_imu, ios::binary |ios::in);
        if (!fhidraw_imu.is_open())
            cout << "Can't open hidraw imu device: " << hidraw_imu << endl;
    }

    memset(&imu_calibration.acceleration_x, 0, sizeof(ImuData));
}

bool EconInput::set_manual_exposure(int exposure)
{
    int MAX_EXPOSURE = 300000;
    cout << "Set expsure to: " << exposure << endl;
    if (exposure >= MAX_EXPOSURE) {
        cout << "Exposure must be less than" << MAX_EXPOSURE << "is " << exposure << ")" << endl;
        return false;
    }

    if (!fhidraw.is_open()) {
        cout << "Set manual exposure: hidraw not open" << endl;
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


    fhidraw.write((char*)buffer, ARRAY_SIZE(buffer));
    fhidraw.flush();
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

    // This values are for Econ Tara REV B has an LSM6DS33 sensor, REV A (LSM6DS0)
    // In therory it should be a REV A however, the frequency more looks like REB B....
    uint8_t buffer[] = {
        0x00,
        HIDRAW_CAMERA_CONTROL_STEREO,  // Stereo config
        HIDRAW_SET_IMU_CONFIG,  // IMU config
        0x03,  // Acclerometer and Gyro enable
        0x00,
        0x00,
        0x07,  // Enable all accelerormeter axis
        0x04,  // Read at 104Hz
        0x00,  // 2G senstivity

        0x00,
        0x00,

        0x07, // Gyro all axes enable
        0x00,
        0x04 // 245 DGPS
    };

    frequency = 104.0;

    // Acceleration is in milli g
    acc_sensitivity = (0.061*9.81)/1000;
    // 8.75 mdps
    gyro_sensitivity = 0.00875;

    if (!fhidraw.is_open()) {
        cout << "Configure imu: hidraw not open" << endl;
        return false;
    }

    fhidraw.write((char*)buffer, ARRAY_SIZE(buffer));
    fhidraw.flush();

    // Configure how many values shoudl be sent
    buffer[1] = HIDRAW_CAMERA_CONTROL_STEREO;
    buffer[2] = HIDRAW_CONTROL_IMU_VAL;
    buffer[3] = 0x01;
    buffer[6] = HIDRAW_IMU_NUM_OF_VAL;
    buffer[7] = 0x00;//(INT8)((lIMUInput.IMU_NUM_OF_VALUES & 0xFF00) >> 8);
    buffer[8] = 0x00;//(INT8)(lIMUInput.IMU_NUM_OF_VALUES & 0xFF);
    fhidraw.write((char*)buffer, 9);
    fhidraw.flush();

    // Enable continus send
    buffer[1] = HIDRAW_CAMERA_CONTROL_STEREO;
    buffer[2] = HIDRAW_SEND_IMU_VAL_BUFF;
    fhidraw.write((char*)buffer, 3);
    fhidraw.flush();

    return true;
}


bool EconInput::get_imu_data(ImuData &imu_data)
{
    if (!fhidraw_imu.is_open()) {
        cout << "Get IMU Data: hidraw imu not open"  << endl;
        return false;
    }

    uint8_t read_buffer[64];
    memset(read_buffer, 0, sizeof(read_buffer));
    fhidraw_imu.read((char*)&read_buffer[0], ARRAY_SIZE(read_buffer));

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
    imu_data.acceleration_y = (((int16_t)((read_buffer[6]) | (read_buffer[5]<<8))) * acc_sensitivity) -
        imu_calibration.acceleration_y;
    imu_data.acceleration_x = (((int16_t)((read_buffer[8]) | (read_buffer[7]<<8))) * acc_sensitivity) -
        imu_calibration.acceleration_x;
    imu_data.acceleration_z = (((int16_t)((read_buffer[10]) | (read_buffer[9]<<8))) * acc_sensitivity) -
        imu_calibration.acceleration_z;

    if (read_buffer[15] != HIDRAW_IMU_GYRO_VAL) {
        cout << "No Gyro data received" << endl;
        return false;
    }
    imu_data.gyro_y = (((int16_t)((read_buffer[17]) | (read_buffer[16]<<8))) * gyro_sensitivity);
    imu_data.gyro_x = (((int16_t)((read_buffer[19]) | (read_buffer[18]<<8))) * gyro_sensitivity);
    imu_data.gyro_z = (((int16_t)((read_buffer[21]) | (read_buffer[20]<<8))) * gyro_sensitivity);

    // We need radians/s not degrees/s
    imu_data.gyro_x *= -1;
    imu_data.gyro_y *= -1;
    imu_data.gyro_z *= -1;

    imu_data.gyro_x -= imu_calibration.gyro_x;
    imu_data.gyro_y -= imu_calibration.gyro_y;
    imu_data.gyro_z -= imu_calibration.gyro_z;



    return true;
}

bool EconInput::read_temperature(float &temperature)
{
    uint8_t buffer[] {
        0,
        HIDRAW_CAMERA_CONTROL_STEREO,
        HIDRAW_GET_IMU_TEMP_DATA
    };

    fhidraw.write((char*)buffer, sizeof(buffer));

    uint8_t read_buffer[7];
    memset(read_buffer, 0, ARRAY_SIZE(read_buffer));
    fhidraw.read((char*)&read_buffer, ARRAY_SIZE(read_buffer));

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

bool EconInput::set_hdr(bool hdr)
{
    uint8_t _hdr = hdr ? 1 : 0;
    uint8_t buffer[] {
        0,
        HIDRAW_CAMERA_CONTROL_STEREO,
        HIDRAW_SET_HDR_MODE_STEREO,
        _hdr
    };

    if (!fhidraw.is_open()) {
        cout << "Set HDR: hidraw device not open" << endl;
        return false;
    }
    fhidraw.write((char*)buffer, sizeof(buffer));

    return true;
}

void EconInput::calibrate_imu()
{
    Vec6f calib(0,0,0,0,0,0);

    TickMeter tick_meter;
#define CALIB_LOOPS 200
    for (size_t i = 0; i < CALIB_LOOPS; i++) {
        ImuData cur_data;
        get_imu_data(cur_data);
        Vec6f current(&cur_data.acceleration_x);
        calib += current;
    }
    cout << "calibration took: " << tick_meter.getTimeMilli() << "ms" << endl;

    imu_calibration.acceleration_x = calib(0)/CALIB_LOOPS;
    imu_calibration.acceleration_y = calib(1)/CALIB_LOOPS;
    imu_calibration.acceleration_z = calib(2)/CALIB_LOOPS;
    imu_calibration.gyro_x = calib(3)/CALIB_LOOPS;
    imu_calibration.gyro_y = calib(4)/CALIB_LOOPS;
    imu_calibration.gyro_z = calib(5)/CALIB_LOOPS;

    // This is a super inaccurate calibration! We assume that camera is
    // sill and in rest position
    imu_calibration.acceleration_x = imu_calibration.acceleration_x - 9.81;
    imu_calibration.acceleration_y = imu_calibration.acceleration_y - 0;
    imu_calibration.acceleration_z = imu_calibration.acceleration_z - 0;

    cout << "Calibration: " << imu_calibration << endl;
}

bool EconInput::imu_available() {
    return fhidraw_imu.is_open();
}

std::ostream& operator<<(std::ostream& os, const ImuData& imu) {
    os << imu.acceleration_x << "," << imu.acceleration_y << "," << imu.acceleration_z <<
        "," << imu.gyro_x << "," << imu.gyro_y << "," << imu.gyro_z;
    return os;
}
