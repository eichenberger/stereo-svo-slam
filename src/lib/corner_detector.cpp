#include <vector>
#include <array>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include "corner_detector.hpp"

CornerDetector::CornerDetector(uint32_t margin) : margin(margin)
{
}

void CornerDetector::detect_keypoints(Mat &image, uint32_t split_count,
        vector<Point> &keypoints)
{
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
    vector<KeyPoint> _keypoints;

    detector->detect(image, _keypoints);

    auto sub_width = (image.cols - 2*margin)/split_count;
    auto sub_height = (image.rows - 2*margin)/split_count;

    Mat edge;
    Sobel(image, edge, -1, 1, 0);


    for (uint32_t i = 0; i < split_count; i++) {
        for (uint32_t j = 0; j < split_count; j++) {
            auto left = margin + i * sub_width;
            auto right = margin + (i+1)*sub_width - 1;
            auto top = margin + j * sub_height;
            auto bottom = margin + (j+1) * sub_height - 1;

            KeyPoint candidate(0,0,1,-1,-1);
            for (auto keypoint : _keypoints) {
                if (keypoint.pt.x  < left || keypoint.pt.x > right ||
                        keypoint.pt.y < top || keypoint.pt.y > bottom)
                    continue;
                if (candidate.response < keypoint.response) {
                    candidate.response = keypoint.response;
                    candidate.pt.x = keypoint.pt.x;
                    candidate.pt.y = keypoint.pt.y;
                }
            }

            if (candidate.response == -1) {
                for (uint32_t k = left; k < right; k++) {
                    for (uint32_t l = top; l < bottom; l++) {
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
    }
}

