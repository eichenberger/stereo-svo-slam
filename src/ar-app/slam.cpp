#include "slam.h"

#include <QtMath>

Slam::Slam(QObject *parent) : QObject(parent)
{
    previous_pose.x = 0;
    previous_pose.y = 0;
    previous_pose.z = 0;
    previous_pose.rx = 0;
    previous_pose.ry = 0;
    previous_pose.rz= 0;
}

Slam::~Slam()
{
}

bool Slam::process_image()
{

    if (!slam_app.process_image())
        return false;

    StereoSlam *slam = slam_app.slam;
    Frame frame;
    slam->get_frame(frame);

    Pose _pose = frame.pose.get_pose();

    // OpenGL uses a different coordinate system than we do. Everything besides z is mirrored...
    QVector3D position(_pose.x, _pose.y, _pose.z);
    QVector3D rotation((_pose.rx)/M_PI*180.0,
                       (_pose.ry)/M_PI*180.0,
                       (_pose.rz)/M_PI*180.0);

    previous_pose = _pose;
    emit pose(position, rotation);
    return true;
}
