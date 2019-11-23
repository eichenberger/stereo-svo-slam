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
        vector<KeyPoint2d> &keypoints,
        vector<KeyPointInformation> &kp_info,
        int level)
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
            if (right > image.cols)
                break;

            // First see if we find a FAST corner
            KeyPoint2d candidate;
            KeyPointInformation info;
            info.score = -1;
            for (auto keypoint : _keypoints) {
                if (keypoint.pt.x  < left || keypoint.pt.x >= right ||
                        keypoint.pt.y < top || keypoint.pt.y >= bottom)
                    continue;
                if (info.score < keypoint.response) {
                    info.score = keypoint.response;
                    candidate.x = keypoint.pt.x;
                    candidate.y = keypoint.pt.y;
                    info.type = KP_FAST;
                }
            }
            // If not we try to find an edgelet
            if (info.score < 0) {
                for (int k = left; k < right; k++) {
                    for (int l = top; l < bottom; l++) {
                        uint8_t response = edge.at<uint8_t>(l,k);
                        if (info.score < response) {
                            info.score = response;
                            candidate.x = k;
                            candidate.y = l;
                            info.type = KP_EDGELET;
                        }
                    }
                }
            }
            info.level = level;
            keypoints.push_back(candidate);
            kp_info.push_back(info);

            left += grid_width;
            right += grid_width;
        }

        bottom += grid_height;
        if (bottom > image.rows)
            break;
        top += grid_height;
    }
}

