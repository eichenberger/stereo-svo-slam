#ifndef SVO_SLAM_BACKEND_H
#define SVO_SLAM_BACKEND_H

#include "websocketserver.hpp"
#include "stereo_slam.hpp"

class SvoSlamBackend : public WebSocketObserver {
public:
    SvoSlamBackend(StereoSlam *slam);

    void text_message_received(QWebSocket &socket,
            const QString &message);


private:
    StereoSlam *slam;
};

#endif
