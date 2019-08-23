#include <vector>
#include <array>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include "corner_detector.hpp"

CornerDetector::CornerDetector()
{
}

void CornerDetector::detect_keypoints(const Mat &image,
        int grid_width, int grid_height,
        vector<Point> &keypoints)
{
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
    vector<KeyPoint> _keypoints;

    detector->detect(image, _keypoints);

    Mat edge;
    Sobel(image, edge, -1, 1, 0);

    auto top = 0;
    auto bottom = grid_height;
    while (true) {
        auto left = 0;
        auto right = grid_width;

        while (true) {
            right += grid_width;
            if (right > image.cols)
                break;

            left += grid_width;

            KeyPoint candidate(0,0,1,-1,-1);
            for (auto keypoint : _keypoints) {
                if (keypoint.pt.x  < left || keypoint.pt.x >= right ||
                        keypoint.pt.y < top || keypoint.pt.y >= bottom)
                    continue;
                if (candidate.response < keypoint.response) {
                    candidate.response = keypoint.response;
                    candidate.pt.x = keypoint.pt.x;
                    candidate.pt.y = keypoint.pt.y;
                }
            }

            if (candidate.response == -1) {
                for (int k = left; k < right; k++) {
                    for (int l = top; l < bottom; l++) {
                        uint8_t response = edge.at<uint8_t>(l,k);
                        if (candidate.response < response) {
                            candidate.response = response;
                            candidate.pt.x = k;
                            candidate.pt.y = l;
                        }
                    }
                }
            }
            keypoints.push_back(candidate.pt);
        }

        bottom += grid_height;
        if (bottom > image.rows)
            break;
        top += grid_height;
    }
}

