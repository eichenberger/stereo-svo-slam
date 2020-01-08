#ifndef WEBSOCKETSERVER_HPP
#define WEBSOCKETSERVER_HPP

#include <QtWebSockets/QtWebSockets>

/*!
 * \brief WebSocket observer class
 *
 * Abstract class that should be inherited to write a WebSocketServer
 * observer. If a meassage is received text_message_received is called
 * by WebSocketServer
 */
class WebSocketObserver {
public:
    WebSocketObserver() {}
    /*!
     * \brief Receive text message
     *
     * Receive a text message. We can use socket to send back an answer.
     * @param[in] socket The socket where a message was received from
     * @param[in] message The message we received
     */
    virtual void text_message_received(QWebSocket &socket,
            const QString &message) = 0;

};

/*!
 * \brief Wrapper Class for QWebSocketServer
 *
 * This class provides an easy to use weboscket server
 * implementation. Every class interested in messages from the WebSocketServer
 * should implement WebSocketObserver and register itself in the WebSocketerServer.
 */
class WebSocketServer {
public:

    /*!
     * \brief Create WebSocketServer
     *
     * Create a websocker server and register an observer
     *
     * @param[in] server_name The name of the server (can be anything)
     * @param[in] port The port to listen on
     * @param[in] observer An observer that is called when a message arrives
     * @param[in] mode If SSL should be used or not
     */
    WebSocketServer(const char* server_name, int port,
        WebSocketObserver &observer,
        QWebSocketServer::SslMode mode = QWebSocketServer::NonSecureMode);
    ~WebSocketServer();

private:
    void new_websocket_connection();
    void text_message_recieved(const QString &message);

    WebSocketObserver &observer;
    QWebSocketServer server;
};

#endif
