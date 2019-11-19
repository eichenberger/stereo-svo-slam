#include "econ_input.hpp"

using namespace std;
using namespace cv;


EconInput::EconInput(const std::string &camera_path, const std::string &hidraw_path,
            const std::string &settings, int exposure)
{
    set_manual_exposure(hidraw_path, exposure);
    read_settings(settings);

    cap = new VideoCapture(camera_path);

    cap->set(CAP_PROP_FRAME_WIDTH, 752);
    cap->set(CAP_PROP_FRAME_HEIGHT, 480);
}

void EconInput::set_manual_exposure(const std::string &hidraw, int value)
{
    int MAX_EXPOSURE = 300000;
    cout << "Set expsure to: " << value << endl;
    if (value >= MAX_EXPOSURE) {
        cout << "Exposure must be less than" << MAX_EXPOSURE << "is " << value << ")" << endl;
        return;
    }

#define BUFFER_SIZE 6
    char buffer[BUFFER_SIZE] = {
        0x78, 0x02,
        (char)((value >> 24)&0xFF),
        (char)((value >> 16)&0xFF),
        (char)((value>>8)&0xFF),
        (char)(value&0xFF)};


    ofstream f;
    f.open(hidraw, ios::binary);
    if (!f.is_open()) {
        cout << "Can't open hidraw device: " << hidraw;
        return;
    }
    f.write(buffer, BUFFER_SIZE);
    f.flush();
    f.close();
}

void EconInput::get_camera_settings(CameraSettings &camera_settings)
{
    camera_settings = this->camera_settings;
}

void EconInput::read(cv::Mat &left, cv::Mat &right)
{
    Mat image;
    cap->read(image);

    extractChannel(image, right, 1);
    extractChannel(image, left, 2);
}
