#ifndef SVO_SLAM_BACKEND_H
#define SVO_SLAM_BACKEND_H

#include "websocketserver.hpp"
#include "stereo_slam.hpp"

/*!
 * \brief SVO SLAM Backend for Qt Viewer
 *
 * This Class provides the data from StereoSlam to a WebSocket interface. It
 * expects a SLAM object and must be register at the WebSocket server.
 */
class SvoSlamBackend : public WebSocketObserver {
public:
    SvoSlamBackend(StereoSlam *slam);

    /*!
     * \brief Read text message and geenerate answer
     *
     * This function reacts on messages from a client and
     * generates an answer
     */
    void text_message_received(QWebSocket &socket,
            const QString &message);


private:
    StereoSlam *slam;
};

#endif
