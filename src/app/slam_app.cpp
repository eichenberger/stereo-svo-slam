#include <QtCore/QThread>
#include <QtWidgets/QApplication>
#include <QtCore/QFile>
#include <QtCore/QTextStream>

#include "slam_app.hpp"

#include "econ_input.hpp"
#include "video_input.hpp"
#include "euroc_input.hpp"

using namespace std;
using namespace cv;

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


SlamApp::SlamApp() :
    input(nullptr), imu_available(false), realtime(false),
    read_imu_thread(nullptr), images_available(1),
    images_read(0), time_stamp(0.0), running(false)
{
}

SlamApp::~SlamApp()
{
    delete slam;
    delete input;
}
bool SlamApp::initialize(const QString &camera_type,
            const QString &video,
            const QString &settings,
            const QString &trajectory_file,
            const QString &hidraw_settings,
            int exposure,
            bool hdr,
            int move,
            const QString &hidraw_imu)
{
    if (camera_type == "econ") {
        EconInput *econ = new EconInput(video.toStdString(),
                hidraw_settings.toStdString(),
                settings.toStdString(),
                hidraw_imu.toStdString());
        econ->set_manual_exposure(exposure);
        econ->set_hdr(hdr);
        if (econ->imu_available()) {
            econ->configure_imu();
            econ->calibrate_imu();
            imu_available = true;
        }
        realtime = true;
        input = econ;
    }
    else if (camera_type == "video") {
        VideoInput *video_input = new VideoInput(video.toStdString(),
                settings.toStdString());
        if (move)
            video_input->jump_to(move);

        input = video_input;
    }
    else if (camera_type == "euroc") {
        EurocInput *euroc_input = new EurocInput(video.toStdString(),
                settings.toStdString());
        if (move)
            euroc_input->jump_to(move);

        input = euroc_input;
    }
    else {
        cout << "Unknown camera type " << camera_type.toStdString() << endl;
        return false;
    }


    this->trajectory_file = trajectory_file;
    return true;
}

void SlamApp::read_imu_data()
{
    Mat image;
    EconInput *econ = dynamic_cast<EconInput*>(input);

    while (running) {
        ImuData _imu_data;
        econ->get_imu_data(_imu_data);

        imu_data_lock.lock();
        imu_data.push_back(_imu_data);
        imu_data_lock.unlock();

        QThread::msleep(5);
    }

}

void SlamApp::update_pose_from_imu(StereoSlam *slam, float dt)
{
    Frame frame;
    // No frame yet
    if (!slam->get_frame(frame))
        return;

    float f = 104.0;

    imu_data_lock.lock();
    vector<ImuData> _imu_data = imu_data;
    imu_data.clear();
    imu_data_lock.unlock();

    Vec6f pose_variance(1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0);
    Vec6f speed_variance(100.0, 100.0, 100.0, 0.1, 0.1, 0.1);
    Pose filtered_pose = frame.pose.get_pose();
    float y_angle = frame.pose.get_pose().yaw;
    for (size_t i = 0; i < std::min<size_t>(_imu_data.size(), f*dt); i++) {
        ImuData &imu_data = _imu_data[i];

        Vec6f speed(0,0,0,imu_data.gyro_x/180.0*M_PI, imu_data.gyro_y/180.0*M_PI, imu_data.gyro_z/180.0*M_PI);
        y_angle += imu_data.gyro_y/180.0*M_PI/104.0;

        // frae pose is wrong. We need kf.statePost
        filtered_pose = slam->update_pose(filtered_pose, speed, pose_variance, speed_variance, 1.0/f);
    }
}

void SlamApp::read_image()
{
    Mat image;

    while (running) {
        Mat _gray_r, _gray_l;
        float _time_stamp;
        if (!realtime)
            images_available.acquire(1);
        if (!input->read(_gray_l, _gray_r, _time_stamp)) {
            cout << "No video data received" << endl;
            QApplication::quit();
            return;
        }
        image_lock.lock();
        gray_r = _gray_r.clone();
        gray_l = _gray_l.clone();
        time_stamp = _time_stamp;
        images_read ++;
        image_lock.unlock();
    }
}

bool SlamApp::process_image()
{
    Mat _gray_r, _gray_l;
    float _time_stamp;
    int _images_read = 0;

    image_lock.lock();
    if (!images_read) {
        image_lock.unlock();
        return false;
    }
    _gray_r = gray_r.clone();
    _gray_l = gray_l.clone();
    _time_stamp = time_stamp;
    _images_read = images_read;
    images_read = 0;
    image_lock.unlock();
    if (images_available.available() == 0)
        images_available.release(1);

    if (imu_available)
        update_pose_from_imu(slam, _images_read/30.0);

    cout << "Process image" << endl;
    START_MEASUREMENT();
    slam->new_image(_gray_l, _gray_r, _time_stamp);
    cout << "End process image" << endl;
    END_MEASUREMENT("Stereo SLAM");

    return true;

}

bool SlamApp::start()
{
    CameraSettings camera_settings;
    input->get_camera_settings(camera_settings);

    slam = new StereoSlam(camera_settings);

    running = true;

    read_image_thread = QThread::create([](SlamApp *app) {
            app->read_image();}, this);

    read_image_thread->start();
    if (imu_available) {
        read_imu_thread = QThread::create(
                [](SlamApp *app) {app->read_imu_data();}, this);
        read_imu_thread->start();
    }

    return true;
}

bool SlamApp::stop()
{
    running = false;
    if (!trajectory_file.isEmpty()) {
        QFile ftrajectory(trajectory_file);
        ftrajectory.open(QIODevice::WriteOnly | QIODevice::Text);
        vector<Pose> trajectory;
        slam->get_trajectory(trajectory);
        QTextStream trajectory_stream(&ftrajectory);
        for (auto pose: trajectory) {
            trajectory_stream << "0," << pose.x << "," << pose.y << "," << pose.z << "," <<
                pose.pitch << "," << pose.yaw << "," << pose.roll << endl;
        }
        ftrajectory.close();
    }

    read_image_thread->wait(100);
    if (read_imu_thread)
        read_imu_thread->wait(100);

    return true;
}

