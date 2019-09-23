// main() provided by Catch in file 020-TestCase-1.cpp.
#include <iostream>
#include "catch.hpp"

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

static void exponential_map(const Mat &twist, Mat &pose)
{
    Mat v(3, 1, CV_64F, (void*)twist.ptr(0));
    Mat w(3, 1, CV_64F, (void*)twist.ptr(3));

    twist.copyTo(pose);
    // The angles don't change. See robotics vision and control
    // page 53 for more details
    Mat w_skew = (Mat_<double>(3,3) <<
            0, -w.at<double>(2), w.at<double>(1),
            w.at<double>(2), 0, -w.at<double>(0),
            -w.at<double>(1), w.at<double>(0), 0);
    float _norm = 1.0;
    Mat _eye = Mat::eye(3,3, CV_64F);
    Mat translation = pose(Rect(0,3,1, 3));

    // Take closed form solution from robotics vision and control page 53
    // Note: This exponential map doesn't give the exact same value as expm
    // from Matlab or Numpy. It is different up to a scaling. It seems that
    // expm uses norm set to 1.0. However, this differs from the closed form
    // solution written in robotics vision and control.
    translation = (_eye*_norm + (1-cos(_norm))*w_skew+(_norm-sin(_norm))*(w_skew*w_skew))*v;
}


TEST_CASE( "Check for angles < pi()", "[multi-file:1]" ) {
   Mat twist = (Mat_<double>(6,1) <<
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6);

   Mat pose;
   exponential_map(twist, pose);

   REQUIRE(pose.at<double>(0) == 0.1);
   REQUIRE(pose.at<double>(1) == 0.2);
   REQUIRE(pose.at<double>(2) == 0.3);
   REQUIRE(fabs(pose.at<double>(3) - 0.12187591059875308) < 0.01);
   REQUIRE(fabs(pose.at<double>(4) - 0.173369312443241) < 0.01);
   REQUIRE(fabs(pose.at<double>(5) - 0.30760829923146377) < 0.01);
}

