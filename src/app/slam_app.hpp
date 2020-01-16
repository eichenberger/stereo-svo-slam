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

/*!
 * \brief Class which wraps camera handling, and StereoSlam
 *
 * This class allows us to have a shared code base between e.g. ar-app and app.
 *
 * Example:
 * \code{.cpp}
 * SlamApp slam_app;
 * slam_app.initialize("econ", "/dev/video0", "Econ.yaml", "trajectory_out.csv",
 *          "/dev/hidraw0", 10000, false, 0, "/dev/hidraw1");
 * slam_app.start();
 * while (slam_app.process_image());
 *
 * vector<Pose> trajectory;
 * slam_app.slam->get_trajectory(trajectory);
 * \endcode
 */

class SlamApp {
public:
    SlamApp();
    ~SlamApp();
    /*!
     * \brief Initalize the SLAM Applicaiton
     *
     * @param[in] camera_type Can be econ, euroc or video
     * @param[in] video The path to the video file or device
     * @param[in] trajecotry_file Where to store the trajectory (empty-> don't save)
     * @param[in] hidraw_settings Path to hidraw device for Settings
     * @param[in] expusre Exposure for econ camera
     * @param[in] hdr Enable/Disable HDR for econ
     * @param[in] move Skip n frames for video or EuRoC input
     * @param[in] hidraw_imu The hidraw device to receive IMU data from
     */
    bool initialize(const QString &camera_type,
            const QString &video,
            const QString &settings,
            const QString &trajectory_file = QString(),
            const QString &hidraw_settings = QString(),
            int exposure = 1,
            bool hdr = false,
            int move = 0,
            const QString &hidraw_imu = QString());

    bool start();   //!< Start the SLAM App and all its threads
    bool stop();    //!< Stop the SLAM App and all its threads

    void read_imu_data();   //!< Read data from IMU
    void read_image();  //!< Read image from camera
    bool process_image();   //!< Process image

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
    std::vector<float> time_stamps;
};

#endif
