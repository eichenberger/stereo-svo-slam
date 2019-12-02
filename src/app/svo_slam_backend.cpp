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
                Pose _pose = keyframe.pose.get_pose();
                QJsonObject pose;
                pose["x"] = _pose.x;
                pose["y"] = _pose.y;
                pose["z"] = _pose.z;
                pose["pitch"] = _pose.pitch;
                pose["yaw"] = _pose.yaw;
                pose["roll"] = _pose.roll;
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
        Pose _pose = frame.pose.get_pose();
        pose["x"]  = _pose.x;
        pose["y"]  = _pose.y;
        pose["z"]  = _pose.z;
        pose["pitch"]  = _pose.pitch;
        pose["yaw"]  = _pose.yaw;
        pose["roll"]  = _pose.roll;

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
            trajectory.append(_trajectory[i].pitch);
            trajectory.append(_trajectory[i].yaw);
            trajectory.append(_trajectory[i].roll);
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

