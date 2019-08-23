#include <opencv2/opencv.hpp>

using namespace cv;


class StereoSlam
{
public:
    StereoSlam();

    void new_image(Mat &left, Mat &right);
};
