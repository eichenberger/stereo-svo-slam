#include <vector>

#include <QtCore/QJsonObject>
#include <QtCore/QJsonArray>
#include <QtCore/QJsonValue>
#include <QtCore/QJsonDocument>
#include <QtCore/QVector>

#include "svo_slam_backend.hpp"

using namespace std;

SvoSlamBackend::SvoSlamBackend(StereoSlam *slam) :
    slam(slam)
{
}

void SvoSlamBackend::text_message_received(QWebSocket &socket,
            const QString &message)
{
    QString url = socket.resourceName();
    QRegularExpression re("^.*//.*/");
    url.replace(re, "");
    qDebug() << "URL: " << url;
    QJsonDocument *json = NULL;
    if (url == "keyframes") {
        if (message == "get") {
            QJsonArray array;
            vector<KeyFrame> keyframes;
            slam->get_keyframes(keyframes);
            for (auto keyframe:keyframes) {
                cv::Vec3f translation = keyframe.pose.get_translation();
                cv::Vec3f angles = keyframe.pose.get_robot_angles();

                QJsonObject pose;
                pose["x"] = translation(0);
                pose["y"] = translation(1);
                pose["z"] = translation(2);
                pose["rx"] = angles(0);
                pose["ry"] = angles(1);
                pose["rz"] = angles(2);
                QJsonArray kps;
                for (auto kp:keyframe.kps.kps3d) {
                    QJsonObject _kp;
                    _kp["x"] = kp.x;
                    _kp["y"] = kp.y;
                    _kp["z"] = kp.z;
                    kps.append(_kp);
                }

                QJsonArray colors;
                for (auto info :keyframe.kps.info) {
                    QJsonObject color;
                    color["r"] = info.color.r;
                    color["g"] = info.color.g;
                    color["b"] = info.color.b;
                    colors.append(color);
                }
                QJsonObject _keyframe;
                _keyframe["pose"] = pose;
                _keyframe["keypoints"] = kps;
                _keyframe["colors"] = colors;
                array.append(_keyframe);
            }
            json = new QJsonDocument(array);
        }
    }
    else if (url == "pose") {
        Frame frame;
        slam->get_frame(frame);
        QJsonObject pose;
        cv::Vec3f translation = frame.pose.get_translation();
        cv::Vec3f angles = frame.pose.get_robot_angles();
        pose["x"] = translation(0);
        pose["y"] = translation(1);
        pose["z"] = translation(2);
        pose["rx"] = angles(0);
        pose["ry"] = angles(1);
        pose["rz"] = angles(2);

        QJsonObject top_object;
        top_object["pose"] = pose;

        json = new QJsonDocument(top_object);
    }
    else if (url == "trajectory") {
        vector<Pose> _trajectory;
        slam->get_trajectory(_trajectory);
        QJsonArray trajectory;
        for (size_t i = 0; i < _trajectory.size(); i++) {
            trajectory.append(_trajectory[i].x);
            trajectory.append(_trajectory[i].y);
            trajectory.append(_trajectory[i].z);
            trajectory.append(_trajectory[i].rx);
            trajectory.append(_trajectory[i].ry);
            trajectory.append(_trajectory[i].rz);
        }

        QJsonObject top_object;
        top_object["trajectory"] = trajectory;
        json = new QJsonDocument(top_object);
    }

    if (json != NULL) {
        QString json_string = json->toJson(QJsonDocument::Compact);
        //qDebug() << json_string;
        socket.sendTextMessage(json_string);
        delete json;
    }
}

