#include <vector>

#include <opencv2/opencv.hpp>

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
    DepthCalculator(float baseline,
            float fx, float fy, float cx, float cy,
            int window_size, int search_x, int search_y, int margin);
    void calculate_depth(Mat &left, Mat &right, int split_count,
            vector<array<float, 2>> &keypoints2d,
            vector<array<float, 3>> &keypoints3d,
            vector<uint32_t> &err);

private:
    Match match(Mat &roi, Mat &templ);

    float baseline;
    float fx;
    float fy;
    float cx;
    float cy;

    int half_window_size;
    int search_x;
    int search_y;
    int margin;
};

