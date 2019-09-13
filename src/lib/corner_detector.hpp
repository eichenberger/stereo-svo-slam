#include <vector>
#include <array>

#include <opencv2/opencv.hpp>

#include "stereo_slam_types.hpp"

using namespace cv;
using namespace std;

class CornerDetector
{
public:
    CornerDetector();

    void detect_keypoints(const Mat &image,
            int grid_width, int grid_height,
            vector<KeyPoint2d> &keypoints,
            vector<KeyPointInformation> &kp_info,
            int level);

};


