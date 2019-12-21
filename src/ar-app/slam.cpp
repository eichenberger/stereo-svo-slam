#include "slam.h"

#include <QtMath>

Slam::Slam(CameraSettings &cameraSettings, QObject *parent) : QObject(parent), slam(new StereoSlam(cameraSettings))
{
    previous_pose.x = 0;
    previous_pose.y = 0;
    previous_pose.z = 0;
    previous_pose.pitch = 0;
    previous_pose.yaw = 0;
    previous_pose.roll= 0;
}

Slam::~Slam()
{
    delete slam;
}

void Slam::new_image(cv::Mat &left, cv::Mat &right, float time_stamp)
{

    slam->new_image(left, right, time_stamp);
    Frame frame;
    slam->get_frame(frame);


    Pose _pose = frame.pose.get_pose();

    // OpenGL uses a different coordinate system than we do. Everything besides z is mirrored...
    QVector3D position(-_pose.x, -_pose.y, _pose.z);
    QVector3D rotation((_pose.pitch)/M_PI*180.0,
                       (_pose.yaw)/M_PI*180.0,
                       (_pose.roll)/M_PI*180.0);

    previous_pose = _pose;
    emit pose(position, rotation);
}
