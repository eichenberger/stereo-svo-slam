#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include <QtWidgets/QApplication>
#include <QtCore/QTimer>
#include <QtCore/QCommandLineParser>
#include <QtCore/QCommandLineOption>
#include <QtCore/QFile>
#include <QtCore/QTextStream>
#include <QtCore/QThread>
#include <QtCore/QMutex>
#include <QtCore/QSemaphore>

#include "slam_app.hpp"

#include "svo_slam_backend.hpp"
#include "websocketserver.hpp"

using namespace cv;
using namespace std;

#define PRINT_TIME_TRACE

#ifdef PRINT_TIME_TRACE
static TickMeter tick_meter;
#define START_MEASUREMENT() tick_meter.reset(); tick_meter.start()

#define END_MEASUREMENT(_name) tick_meter.stop();\
    cout << _name << " took: " << tick_meter.getTimeMilli() << "ms" << endl

#else
#define START_MEASUREMENT()
#define END_MEASUREMENT(_name)
#endif

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

        stringstream text;
        text << fixed << setprecision(1) << info[i].keyframe_id << ":" << i;
        putText(left_kf, text.str(), kp, FONT_HERSHEY_PLAIN, 0.8, Scalar(0,0,0));
    }

    string text = "ID: " + to_string(keyframe.id);
    putText(left_kf, text, Point(20,20), FONT_HERSHEY_PLAIN, 1, Scalar(255,0,0));

    vector<KeyPoint2d> &kps = frame.kps.kps2d;
    vector<KeyPoint3d> &kps3d = frame.kps.kps3d;
    info = frame.kps.info;
    for (size_t i = 0; i < kps.size(); i++) {
        if (info[i].ignore_completely)
            continue;
        Point kp = Point(SCALE*kps[i].x, SCALE*kps[i].y);
        Scalar color (info[i].color.r, info[i].color.g, info[i].color.b);

        int marker = info[i].type == KP_FAST ? MARKER_CROSS : MARKER_SQUARE;

        int marker_size = info[i].ignore_temporary ? 10 : 20;

        cv::drawMarker(left, kp, color, marker, marker_size);

        //KeyPoint3d &kp3d = frame.kps.kps3d[i];
        stringstream text;

        float x = kps3d[i].x;
        float y = kps3d[i].y;
        float z = kps3d[i].z;
        char ignore = info[i].ignore_completely ? '-' : '+';
        text << fixed << setprecision(3) <<
            info[i].keyframe_id << ":" << info[i].keypoint_index << ":" << ignore << ":" <<
            x  << "," <<
            y << ","  <<
            z << "; " << info[i].inlier_count << ", " << info[i].outlier_count;
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
        QApplication::quit();
}

static void process_image(SlamApp *app) {
    if (!app->process_image())
        return;

    StereoSlam *slam = app->slam;
    Frame frame;
    slam->get_frame(frame);
    KeyFrame keyframe;
    slam->get_keyframe(keyframe);
    draw_frame(keyframe, frame);
    cout << "Current pose: " << frame.pose << endl;
}

int main(int argc, char **argv)
{
    QApplication app(argc, argv);

    QCommandLineParser parser;
    QStringList arguments = app.arguments();
    parser.setApplicationDescription("SVO stereo SLAM application");
    parser.addHelpOption();
    parser.addPositionalArgument("camera", "The camera type to use can be econ, video or euroc");
    parser.addOptions({
            {{"v", "video"}, "Path to camera or video (/dev/videoX, video.mov)", "video"},
            {{"s", "settings"}, "Path to the settings file (Econ.yaml)", "settings"},
            {{"r", "hidraw"}, "econ: HID device to control the camera (/dev/hidrawX)", "hidraw"},
            {{"i", "hidrawimu"}, "econ: HID device to control the imu (/dev/hidrawX)", "hidrawimu"},
            {{"e", "exposure"}, "econ: The exposure for the camera 1-30000", "exposure"},
            {{"t", "trajectory"}, "File to store trajectory", "trajectory"},
            {{"m", "move"}, "video: skip first n frames", "move"},
            {{"d", "hdr"}, "econ: Use HDR video"},
            });


    parser.process(arguments);
    arguments.pop_back();

    if (parser.positionalArguments().size() != 1) {
        cout << "You need to specify a camera type!" << endl;
        cout << parser.helpText().toStdString() << endl;
        return -1;
    }

    QString camera_type = parser.positionalArguments().at(0);
    if (camera_type == "econ") {
        if (!parser.isSet("video") ||
                !parser.isSet("hidraw") ||
                !parser.isSet("settings") ||
                !parser.isSet("exposure")) {
            cout << "Please set all inputs for econ" << endl;
            cout << parser.helpText().toStdString() << endl;
            return -1;
        }
    }
    else if (camera_type == "video" || camera_type == "euroc") {
        if (!parser.isSet("video") ||
                !parser.isSet("settings")) {
            cout << "Please set all inputs for video" << endl;
            cout << parser.helpText().toStdString() << endl;
            return -1;
        }
    }
    else {
        cout << "Unknown camera type " << camera_type.toStdString() << endl;
        return -2;
    }

    SlamApp slam_app;
    if (!slam_app.initialize(camera_type,
            parser.value("video"),
            parser.value("settings"),
            parser.value("trajectory"),
            parser.value("hidraw"),
            parser.value("exposure").toInt(),
            parser.isSet("hdr"),
            parser.value("move").toInt(),
            parser.value("hidrawimu"))) {
        cout << "Can't initialize slam app" << endl;
        return -3;
    }

    slam_app.start();

    SvoSlamBackend backend(slam_app.slam);
    WebSocketServer server("svo", 8001, backend);

    QTimer timer;
    timer.setInterval(1.0/60.0*1000.0);
    QObject::connect(&timer, &QTimer::timeout,
            std::bind(process_image, &slam_app));

    timer.start();

    app.exec();

    slam_app.stop();

    return 0;
}
