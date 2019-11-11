#include "video_input.hpp"

using namespace std;
using namespace cv;


VideoInput::VideoInput(const std::string &video_path,
            const std::string &settings)
{
    read_settings(settings);

    cap = new VideoCapture(video_path);

    cap->set(CAP_PROP_FRAME_WIDTH, camera_settings.image_width);
    cap->set(CAP_PROP_FRAME_HEIGHT, camera_settings.image_height);
}

void VideoInput::get_camera_settings(CameraSettings &camera_settings)
{
    camera_settings = this->camera_settings;
}

void VideoInput::read(cv::Mat &left, cv::Mat &right)
{
    Mat image;
    cap->read(image);

    if (image.channels() == 3) {
        cvtColor(image, image, cv::COLOR_BGR2GRAY);
    }

    right = image(Rect(0, 0, camera_settings.image_width, camera_settings.image_height));
    left = image(Rect(camera_settings.image_width, 0, camera_settings.image_width, camera_settings.image_height));
}

void VideoInput::jump_to(int frame_number)
{
    cap->set(cv::CAP_PROP_POS_FRAMES, frame_number);
}
