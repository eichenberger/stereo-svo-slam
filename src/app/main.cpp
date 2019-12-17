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
#include <QtCore/QThread>
#include <QtCore/QMutex>

#include "stereo_slam_types.hpp"
#include "stereo_slam.hpp"

#include "svo_slam_backend.hpp"
#include "websocketserver.hpp"

#include "econ_input.hpp"
#include "video_input.hpp"

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

        int marker_size = info[i].confidence*16 + 4;

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

static bool running = true;

static QMutex imu_data_lock;
static vector<ImuData> imu_data;
static void read_imu_data(EconInput *econ)
{
    Mat image;

    while (running) {
        cout << "read imu data" << endl;
        ImuData _imu_data;
        econ->get_imu_data(_imu_data);

        imu_data_lock.lock();
        imu_data.push_back(_imu_data);
        imu_data_lock.unlock();

        QThread::msleep(5);
    }

}

static void update_pose_from_imu(StereoSlam *slam)
{
    Frame frame;
    slam->get_frame(frame);

    float f = 104.0;

    imu_data_lock.lock();
    vector<ImuData> _imu_data = imu_data;
    imu_data.clear();
    imu_data_lock.unlock();

    Vec6f pose_variance(10.0, 10.0, 10.0, 10.0, 10.0, 10.0);
    Vec6f speed_variance(100.0, 100.0, 100.0, 0.1, 0.1, 0.1);
    double measurement_time = frame.time_stamp;
    float y_angle = frame.pose.get_pose().yaw;
    for (size_t i = 0; i < _imu_data.size(); i++) {
        ImuData &imu_data = _imu_data[i];

        cout << "IMU Data: " << imu_data << endl;
        Vec6f speed(0,0,0,imu_data.gyro_x/180.0*M_PI, imu_data.gyro_y/180.0*M_PI, imu_data.gyro_z/180.0*M_PI);
        y_angle += imu_data.gyro_y/180.0*M_PI/104.0;

        measurement_time += 1.0/f;
        slam->update_pose(frame.pose.get_pose(), speed, pose_variance, speed_variance, measurement_time);
    }

    cout << "Angle withouth KF: " << y_angle << endl;
}

static QMutex image_lock;
static Mat gray_r, gray_l;
static bool image_avalilable = false;

static void read_image(ImageInput *input)
{
    Mat image;

    while (running) {
        cout << "read image" << endl;
        Mat _gray_r, _gray_l;
        if (!input->read(_gray_l, _gray_r)) {
            QApplication::quit();
            return;
        }
        image_lock.lock();
        gray_r = _gray_r.clone();
        gray_l = _gray_l.clone();
        image_avalilable = true;
        image_lock.unlock();
        QThread::msleep(10);
    }

}

static void process_image(bool use_imu, StereoSlam *slam)
{
    Mat _gray_r, _gray_l;

    image_lock.lock();
    if (!image_avalilable) {
        image_lock.unlock();
        return;
    }
    _gray_r = gray_r.clone();
    _gray_l = gray_l.clone();
    image_avalilable = false;
    image_lock.unlock();

    cout << "Process image" << endl;
    START_MEASUREMENT();
    slam->new_image(_gray_l, _gray_r);
    cout << "End process image" << endl;
    END_MEASUREMENT("Stereo SLAM");

    Frame frame;
    slam->get_frame(frame);
    KeyFrame keyframe;
    slam->get_keyframe(keyframe);
    draw_frame(keyframe, frame);
    cout << "Current pose: " << frame.pose << endl;
    if (use_imu)
        update_pose_from_imu(slam);

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
            {{"i", "hidrawimu"}, "econ: HID device to control the imu (/dev/hidrawX)", "hidrawimu"},
            {{"e", "exposure"}, "econ: The exposure for the camera 1-30000", "exposure"},
            {{"t", "trajectory"}, "File to store trajectory", "trajectory"},
            {{"m", "move"}, "video: skip first n frames", "move"},
            {{"d", "hdr"}, "econ: Use HDR video"},
            });


    parser.process(arguments);
    arguments.pop_back();

    QString camera_type = parser.positionalArguments().at(0);
    ImageInput *input;
    bool imu_available = false;
    QThread* read_imu_thread = nullptr;
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
                parser.value("hidrawimu").toStdString());
        econ->set_hdr(parser.isSet("hdr"));
        econ->set_manual_exposure(parser.value("exposure").toInt());
        if (econ->imu_available()) {
            econ->configure_imu();
            econ->calibrate_imu();
            read_imu_thread = QThread::create(read_imu_data, econ);
            imu_available = true;
        }
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
        if (parser.isSet("move")) {
            video->jump_to(parser.value("move").toInt());
        }
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
    timer.setInterval(1.0/60.0*1000.0);

    QObject::connect(&timer, &QTimer::timeout,
            std::bind(&process_image, imu_available, &slam));
    timer.start();

    SvoSlamBackend backend(&slam);
    WebSocketServer server("svo", 8001, backend);

    QThread* read_image_thread = QThread::create(read_image, input);

    read_image_thread->start();
    if (read_imu_thread)
        read_imu_thread->start();


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

    running = false;
    read_image_thread->wait(100);
    if (read_imu_thread)
        read_imu_thread->wait(100);

    delete input;
    return 0;
}
