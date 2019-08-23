#include <vector>

#include <opencv2/opencv.hpp>

#include "stereo_slam_types.hpp"

using namespace cv;
using namespace std;

class Match
{
public:
    uint32_t x;
    uint32_t y;
    uint32_t err;
};

class DepthCalculator
{
public:
    DepthCalculator(){}
    void calculate_depth(const struct StereoImage &stereo_image,
            const struct CameraSettings &camera_settings,
            struct KeyPoints &keypoints);


private:
    Match match(Mat &roi, Mat &templ);
};

