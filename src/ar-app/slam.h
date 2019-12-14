#ifndef SLAM_H
#define SLAM_H

#include <QObject>
#include <QVector3D>


#include <stereo_slam_types.hpp>
#include <stereo_slam.hpp>

#include <opencv2/opencv.hpp>

class Slam : public QObject
{
    Q_OBJECT
public:
    explicit Slam(CameraSettings &cameraSettings, QObject *parent = nullptr);
    ~Slam();

    void new_image(cv::Mat &left, cv::Mat &right);

signals:
    void pose(QVector3D position, QVector3D rotation);

private:
    StereoSlam *slam;
    Pose previous_pose;
};

#endif // SLAM_H
