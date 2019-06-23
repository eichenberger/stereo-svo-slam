// 020-TestCase-2.cpp

// main() provided by Catch in file 020-TestCase-1.cpp.

#include "catch.hpp"

#include <opencv2/opencv.hpp>
#include "corner_detector.hpp"

using namespace cv;

static vector<Point> keypoints;
static int split_count = 16;
static int margin = 10;
static int width;
static int height;

static void setup()
{
   if (keypoints.size() != 0)
      return;

   auto image = imread("testimage0.png");
   CornerDetector corner_detector(margin);
   corner_detector.detect_keypoints(image, split_count, keypoints);

   width = image.cols - 2*margin;
   height = image.rows - 2*margin;

}

TEST_CASE( "Corner Detector finds right amount of corners", "[multi-file:1]" ) {
   setup();

   REQUIRE(keypoints.size() == split_count*split_count);

}

TEST_CASE( "Corner Detector distribute corners", "[multi-file:1]" ) {
   setup();

   int box_width = width/split_count;
   int box_height = height/split_count;

   bool test_box [split_count][split_count];

   for (int i = 0; i < split_count; i++) {
      for (int j = 0; j < split_count; j++) {
         test_box[i][j] = false;
      }
   }

   for (auto kp: keypoints) {
      int x = (kp.x - margin)/box_width;
      int y = (kp.y - margin)/box_height;
      REQUIRE(test_box[y][x] == false);
      test_box[y][x] = true;
   }

   for (int i = 0; i < split_count; i++) {
      for (int j = 0; j < split_count; j++) {
         REQUIRE(test_box[i][j] == true);
      }
   }

}
