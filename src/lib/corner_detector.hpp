#include <vector>
#include <array>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class CornerDetector
{
public:
    CornerDetector(uint32_t margin);

    void detect_keypoints(Mat &image, uint32_t split_count,
            vector<Point> &keypoints);

private:
    uint32_t margin;

};


