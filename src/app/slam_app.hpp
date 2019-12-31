#ifndef SLAM_APP_H
#define SLAM_APP_H

#include <vector>

#include <QtCore/QString>
#include <QtCore/QTimer>
#include <QtCore/QMutex>
#include <QtCore/QSemaphore>

#include <opencv2/opencv.hpp>

#include "stereo_slam.hpp"
#include "stereo_slam_types.hpp"

#include "image_input.hpp"
#include "econ_input.hpp"

class SlamApp {
public:
    SlamApp();
    ~SlamApp();

    bool initialize(const QString &camera_type,
            const QString &video,
            const QString &settings,
            const QString &trajectory_file = QString(),
            const QString &hidraw_settings = QString(),
            int exposure = 1,
            bool hdr = false,
            int move = 0,
            const QString &hidraw_imu = QString());

    bool start();
    bool stop();

    void read_imu_data();
    void read_image();
    bool process_image();

    StereoSlam *slam;
private:
    void update_pose_from_imu(StereoSlam *slam, float dt);
    ImageInput *input;
    bool imu_available;
    bool realtime;
    QString trajectory_file;
    QTimer timer;

    QThread* read_image_thread;
    QThread *read_imu_thread;
    QMutex image_lock;
    QSemaphore images_available;
    cv::Mat gray_r, gray_l;
    int images_read;
    float time_stamp;
    bool running;
    QMutex imu_data_lock;
    std::vector<ImuData> imu_data;
};

#endif
