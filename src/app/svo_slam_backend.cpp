#include <vector>

#include <QtCore/QJsonObject>
#include <QtCore/QJsonArray>
#include <QtCore/QJsonValue>
#include <QtCore/QJsonDocument>

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
                QJsonObject pose;
                pose["x"] = keyframe.pose.x;
                pose["y"] = keyframe.pose.y;
                pose["z"] = keyframe.pose.z;
                pose["pitch"] = keyframe.pose.pitch;
                pose["yaw"] = keyframe.pose.yaw;
                pose["roll"] = keyframe.pose.roll;
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
        pose["x"]  = frame.pose.x;
        pose["y"]  = frame.pose.y;
        pose["z"]  = frame.pose.z;
        pose["pitch"]  = frame.pose.pitch;
        pose["yaw"]  = frame.pose.yaw;
        pose["roll"]  = frame.pose.roll;

        QJsonObject top_object;
        top_object["pose"] = pose;

        json = new QJsonDocument(top_object);
    }

    if (json != NULL) {
        QString json_string = json->toJson(QJsonDocument::Compact);
        qDebug() << json_string;
        socket.sendTextMessage(json_string);
        delete json;
    }
}

