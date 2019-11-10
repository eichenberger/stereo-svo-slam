#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/opencv.hpp>

#include <QtWidgets/QApplication>
#include <QtCore/QTimer>
#include <QtCore/QCommandLineParser>
#include <QtCore/QCommandLineOption>
#include <QtCore/QFile>
#include <QtCore/QTextStream>

#include "stereo_slam_types.hpp"
#include "stereo_slam.hpp"

#include "svo_slam_backend.hpp"
#include "websocketserver.hpp"

#include "econ_input.hpp"
#include "video_input.hpp"

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

        int marker_size = info[i].confidence*16 + 4;

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
        QApplication::quit();
}

static void read_image(ImageInput *input, StereoSlam *slam)
{
    Mat image;

    Mat gray_r, gray_l;
    input->read(gray_l, gray_r);

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
    QApplication app(argc, argv);

    QCommandLineParser parser;
    QStringList arguments = app.arguments();
    parser.setApplicationDescription("SVO stereo SLAM application");
    parser.addHelpOption();
    parser.addPositionalArgument("camera", "The camera type to use can be econ, video or blender");
    parser.addOptions({
            {{"v", "video"}, "Path to camera or video (/dev/videoX, video.mov)", "video"},
            {{"s", "settings"}, "Path to the settings file (Econ.yaml)", "settings"},
            {{"r", "hidraw"}, "econ: HID device to control the camera (/dev/hidrawX)", "hidraw"},
            {{"e", "exposure"}, "econ: The exposure for the camera 1-30000", "exposure"},
            {{"t", "trajectory"}, "File to store trajectory", "trajectory"},
            });


    parser.process(arguments);
    arguments.pop_back();

    QString camera_type = parser.positionalArguments().at(0);
    ImageInput *input;
    if (camera_type == "econ") {
        if (!parser.isSet("video") ||
                !parser.isSet("hidraw") ||
                !parser.isSet("settings") ||
                !parser.isSet("exposure")) {
            cout << "Please set all inputs for econ" << endl;
            cout << parser.helpText().toStdString() << endl;
            return -1;
        }

        EconInput *econ = new EconInput(parser.value("video").toStdString(),
                parser.value("hidraw").toStdString(),
                parser.value("settings").toStdString(),
                parser.value("exposure").toInt());
        input = econ;
    }
    else if (camera_type == "video") {
        if (!parser.isSet("video") ||
                !parser.isSet("settings")) {
            cout << "Please set all inputs for video" << endl;
            cout << parser.helpText().toStdString() << endl;
            return -1;
        }
        VideoInput *video = new VideoInput(parser.value("video").toStdString(),
                parser.value("settings").toStdString());
        input = video;

    }
    else {
        cout << "Unknown camera type " << camera_type.toStdString() << endl;
        return -2;
    }

    CameraSettings camera_settings;
    input->get_camera_settings(camera_settings);

    StereoSlam slam(camera_settings);

    QCommandLineOption showProgressOption("p", QCoreApplication::translate("main", "Show progress during copy"));
    parser.addOption(showProgressOption);

    QTimer timer;
    timer.setInterval(1.0/30.0*1000.0);

    QObject::connect(&timer, &QTimer::timeout,
            std::bind(&read_image, input, &slam));
    timer.start();

    SvoSlamBackend backend(&slam);
    WebSocketServer server("svo", 8001, backend);

    app.exec();

    if (parser.isSet("trajectory")) {
        QFile trajectory_file(parser.value("trajectory"));
        trajectory_file.open(QIODevice::WriteOnly | QIODevice::Text);
        vector<Pose> trajectory;
        slam.get_trajectory(trajectory);
        QTextStream trajectory_stream(&trajectory_file);
        for (auto pose: trajectory) {
            trajectory_stream << pose.x << "," << pose.y << "," << pose.z << "," <<
                pose.pitch << "," << pose.yaw << "," << pose.roll << endl;
        }
        trajectory_file.close();
    }

    delete input;
}
