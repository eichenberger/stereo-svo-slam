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

static void read_data(EconInput *econ)
{
    static float angle_x = 0, angle_y = 0, angle_z = 0;
    float temperature;
    econ->read_temperature(temperature);

    cout << "Temperature: " << temperature << endl;

    ImuData imu_data;
    econ->get_imu_data(imu_data);
    cout << "IMU Data: " << imu_data<< endl;


    float f = econ->get_freqency();
    angle_x += imu_data.gyro_x/f;
    angle_y += imu_data.gyro_y/f;
    angle_z += imu_data.gyro_z/f;

    cout << "Angle: " << angle_x << ", " <<
        angle_y << ", " <<
        angle_z << endl;

    cout << endl;
}


int main(int argc, char **argv)
{
    QApplication app(argc, argv);

    QCommandLineParser parser;
    QStringList arguments = app.arguments();
    parser.setApplicationDescription("IMU test");
    parser.addHelpOption();
    parser.addOptions({
            {{"v", "video"}, "econ: video input", "vdieo"},
            {{"s", "settings"}, "econ: Settings", "settings"},
            {{"r", "hidraw"}, "econ: HID device to control the camera (/dev/hidrawX)", "hidraw"},
            {{"i", "hidrawimu"}, "econ: HID device to control the IMU (/dev/hidrawX)", "hidrawimu"},
            });


    parser.process(arguments);
    arguments.pop_back();

    if (!parser.isSet("hidraw") ||
        !parser.isSet("video") ||
            !parser.isSet("video")) {
        cout << "Hidraw or video not specified" << endl;
        cout << parser.helpText().toStdString() << endl;
        return -1;
    }

    EconInput *econ = new EconInput(parser.value("video").toStdString(),
            parser.value("hidraw").toStdString(),
            parser.value("settings").toStdString(),
            parser.value("hidrawimu").toStdString());
    econ->configure_imu();
    econ->calibrate_imu();

    QTimer timer;
    timer.setInterval(5);

    QObject::connect(&timer, &QTimer::timeout,
            std::bind(&read_data, econ));
    timer.start();

    app.exec();
}
