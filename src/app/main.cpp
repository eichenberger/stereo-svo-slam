#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/opencv.hpp>

#include <QtWidgets/QApplication>
#include <QtCore/QTimer>

#include "stereo_slam_types.hpp"
#include "stereo_slam.hpp"

#include "svo_slam_backend.hpp"
#include "websocketserver.hpp"

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

        int marker_size = info[i].confidence*8 + 2;

        cv::drawMarker(left, kp, color, marker, marker_size);

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
    cout << "Set expsure to: " << value << endl;
    if (value >= MAX_EXPOSURE) {
        cout << "Exposure must be less than" << MAX_EXPOSURE << "is " << value << ")" << endl;
        return;
    }

#define BUFFER_SIZE 6
    char buffer[BUFFER_SIZE] = {
        0x78, 0x02,
        (char)((value >> 24)&0xFF),
        (char)((value >> 16)&0xFF),
        (char)((value>>8)&0xFF),
        (char)(value&0xFF)};


    ofstream f;
    f.open(hidraw, ios::binary);
    f.write(buffer, BUFFER_SIZE);
    f.flush();
    f.close();
}

void set_auto_exposure(const char *hidraw)
{
    set_manual_exposure(hidraw, 1);
}

static void read_image(VideoCapture &cap, StereoSlam *slam)
{
    Mat image;
    cap.read(image);

    Mat gray_r, gray_l;
    extractChannel(image, gray_r, 1);
    extractChannel(image, gray_l, 2);

    slam->new_image(gray_l, gray_r);
    Frame frame;
    slam->get_frame(frame);
    KeyFrame keyframe;
    slam->get_keyframe(keyframe);
    draw_frame(keyframe, frame);
    cout << "Current pose: " << frame.pose.x << "," <<
        frame.pose.y << "," <<
        frame.pose.z << "," <<
        frame.pose.pitch << "," <<
        frame.pose.yaw << "," <<
        frame.pose.roll << "," <<
        endl;

}

int main(int argc, char **argv)
{
    char *camera = argv[1];
    char *config = argv[2];
    char *hidraw = argv[3];
    char *exposure = argv[4];

    set_manual_exposure(hidraw, atoi(exposure));

    FileStorage fs(config, FileStorage::READ);
    CameraSettings camera_settings;
    camera_settings.fx = fs["Camera1.fx"];
    camera_settings.fy = fs["Camera1.fy"];
    camera_settings.cx = fs["Camera1.cx"];
    camera_settings.cy = fs["Camera1.cy"];
    camera_settings.baseline = fs["Camera.baseline"];
    camera_settings.window_size = fs["Camera.window_size"];
    camera_settings.window_size_opt_flow = fs["Camera.window_size_opt_flow"];
    camera_settings.window_size_depth_calculator = fs["Camera.window_size_depth_calculator"];
    camera_settings.max_pyramid_levels = fs["Camera.max_pyramid_levels"];

    camera_settings.dist_window_k0 = fs["Camera.dist_window_k0"];
    camera_settings.dist_window_k1 = fs["Camera.dist_window_k1"];
    camera_settings.dist_window_k2 = fs["Camera.dist_window_k2"];
    camera_settings.dist_window_k3 = fs["Camera.dist_window_k3"];
    camera_settings.cost_k0 = fs["Camera.cost_k0"];
    camera_settings.cost_k1 = fs["Camera.cost_k1"];



    camera_settings.k1 = fs["Camera1.k1"];
    camera_settings.k2 = fs["Camera1.k2"];
    camera_settings.k3 = fs["Camera1.k3"];
    camera_settings.p1 = fs["Camera1.p1"];
    camera_settings.p2 = fs["Camera1.p2"];

    camera_settings.grid_width = fs["Camera.grid_width"];
    camera_settings.grid_height = fs["Camera.grid_height"];
    camera_settings.search_x = fs["Camera.search_x"];
    camera_settings.search_y = fs["Camera.search_y"];

    StereoSlam slam(camera_settings);

    QApplication app(argc, argv);

    VideoCapture cap(camera);
    cap.set(CAP_PROP_FRAME_WIDTH, 752);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);


    QTimer timer;
    timer.setInterval(1.0/30.0*1000.0);

    QObject::connect(&timer, &QTimer::timeout,
            std::bind(&read_image, cap, &slam));
    timer.start();

    SvoSlamBackend backend(&slam);
    WebSocketServer server("svo", 8001, backend);

    app.exec();
}
