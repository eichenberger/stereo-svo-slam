#ifndef SLAM_H
#define SLAM_H

#include <QObject>
#include <QVector3D>


#include <stereo_slam_types.hpp>
#include <stereo_slam.hpp>

#include <opencv2/opencv.hpp>

#include "slam_app.hpp"

class Slam : public QObject
{
    Q_OBJECT
public:
    explicit Slam(QObject *parent = nullptr);
    ~Slam();

    bool process_image();
    SlamApp slam_app;
signals:
    void pose(QVector3D position, QVector3D rotation);

private:
    Pose previous_pose;
};

#endif // SLAM_H
