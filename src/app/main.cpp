#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "stereo_slam_types.hpp"
#include "stereo_slam.hpp"


using namespace cv;
using namespace std;

static void draw_frame(KeyFrame &keyframe, Frame &frame)
{
    const int SCALE = 2;
    Size image_size = Size(SCALE*keyframe.stereo_image.left[0].cols,
			SCALE*keyframe.stereo_image.left[0].rows);
	Mat left_kf;
	Mat left;
	cv::resize(keyframe.stereo_image.left[0], left_kf, image_size);
	cv::resize(frame.stereo_image.left[0], left, image_size);

    cvtColor(left_kf, left_kf, COLOR_GRAY2RGB);
    cvtColor(left, left,  COLOR_GRAY2RGB);

	vector<KeyPoint2d> &kf_kps = keyframe.kps.kps2d;
    vector<KeyPointInformation> &info = keyframe.kps.info;
    for (size_t i = 0; i < kf_kps.size(); i++) {
        Point kp = Point(SCALE*kf_kps[i].x, SCALE*kf_kps[i].y);
        Scalar color (info[i].color.r, info[i].color.g, info[i].color.b);

		int marker = info[i].type == KP_FAST ? MARKER_CROSS : MARKER_SQUARE;

		cv::drawMarker(left_kf, kp, color, marker, 10);

		KeyPoint3d &kp3d = keyframe.kps.kps3d[i];
		stringstream text;
		text << fixed << setprecision(1) <<
			info[i].keyframe_id << ":" <<
			kp3d.x  << "," <<
			kp3d.y << ","  <<
			kp3d.z;
        putText(left_kf, text.str(), kp, FONT_HERSHEY_PLAIN, 0.8, Scalar(0,0,0));
	}

    string text = "ID: " + to_string(keyframe.id);
    putText(left_kf, text, Point(20,20), FONT_HERSHEY_PLAIN, 1, Scalar(255,0,0));

	vector<KeyPoint2d> &kps = frame.kps.kps2d;
    info = frame.kps.info;
    for (size_t i = 0; i < kps.size(); i++) {
        Point kp = Point(SCALE*kps[i].x, SCALE*kps[i].y);
        Scalar color (info[i].color.r, info[i].color.g, info[i].color.b);

		int marker = info[i].type == KP_FAST ? MARKER_CROSS : MARKER_SQUARE;

		cv::drawMarker(left, kp, color, marker, 10);

		KeyPoint3d &kp3d = keyframe.kps.kps3d[i];
		stringstream text;
		text << fixed << setprecision(1) <<
			info[i].keyframe_id << ":" <<
			kp3d.x  << "," <<
			kp3d.y << ","  <<
			kp3d.z;
        putText(left, text.str(), kp, FONT_HERSHEY_PLAIN, 0.8, Scalar(0,0,0));
	}

    text = "ID: " + to_string(frame.id);
    putText(left, text, Point(20,20), FONT_HERSHEY_PLAIN, 1, Scalar(255,0,0));

	Mat result(left.rows, left.cols*2, CV_8UC3);
	//result(Rect(0, 0, left.cols, left.rows)) = left_kf;
	//result(Rect(left.cols, 0, left.cols, left.rows)) = left;
	left_kf.copyTo(result(Rect(0, 0, left.cols, left.rows)));
	left.copyTo(result(Rect(left.cols, 0, left.cols, left.rows)));

    namedWindow("result", WINDOW_GUI_EXPANDED);
    imshow("result", result);
    int key = waitKey(1);
    if (key == 's')
        key = waitKey(0);

    if (key == 'q')
        throw("end");
}

void set_manual_exposure(const char *hidraw, int value)
{
    int MAX_EXPOSURE = 300000;
    if (value >= MAX_EXPOSURE) {
        cout << "Exposure must be less than" << MAX_EXPOSURE << "is " << value << ")";
        return;
	}

	ofstream f;
	f.open(hidraw, ios::binary);
	f << 0x78 << 0x02 <<
		(char)((value >> 24)&0xFF) <<
		(char)((value >> 16)&0xFF) <<
		(char)((value>>8)&0xFF) <<
		(char)(value&0xFF);
	f.close();
}

void set_auto_exposure(const char *hidraw)
{
    set_manual_exposure(hidraw, 1);
}


int main(int argc, char **argv)
{
	char *camera = argv[1];
	char *config = argv[2];
	char *hidraw = argv[3];

	FileStorage fs(config, FileStorage::READ);
	CameraSettings camera_settings;
	camera_settings.fx = fs["Camera.fx"];
	camera_settings.fy = fs["Camera.fy"];
	camera_settings.cx = fs["Camera.cx"];
	camera_settings.cy = fs["Camera.cy"];
	camera_settings.baseline = fs["Camera.bf"];

	Mat distortion;
	fs["LEFT.D"] >> distortion;
	camera_settings.k1 = distortion.at<double>(0);
	camera_settings.k2 = distortion.at<double>(1);
	camera_settings.k3 = distortion.at<double>(4);
	camera_settings.p1 = distortion.at<double>(2);
	camera_settings.p2 = distortion.at<double>(3);

	camera_settings.grid_width = 40;
	camera_settings.grid_height = 30;
	camera_settings.search_x = 30;
	camera_settings.search_y = 6;
    camera_settings.window_size = 4;
    camera_settings.window_size_opt_flow = 8;
    camera_settings.max_pyramid_levels = 3;

	StereoSlam slam(camera_settings);

	set_manual_exposure(hidraw, 20000);

	VideoCapture cap(camera);
	cap.set(CAP_PROP_FRAME_WIDTH, 752);
	cap.set(CAP_PROP_FRAME_HEIGHT, 480);
	Mat image;

	while (1) {
		cap.read(image);

		Mat gray_r, gray_l;
		extractChannel(image, gray_r, 1);
		extractChannel(image, gray_l, 2);

		slam.new_image(gray_l, gray_r);
		Frame frame;
		slam.get_frame(frame);
		KeyFrame keyframe;
		slam.get_keyframe(keyframe);
		draw_frame(keyframe, frame);
		cout << "Current pose: " << frame.pose.x << "," <<
			frame.pose.y << "," <<
			frame.pose.z << "," <<
			frame.pose.pitch << "," <<
			frame.pose.yaw << "," <<
			frame.pose.roll << "," <<
			endl;
	}
}
