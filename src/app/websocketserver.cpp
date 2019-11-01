#include <vector>

#include <QtCore/QObject>
#include <QtCore/QList>
#include <QtNetwork/QHostAddress>

#include "websocketserver.hpp"

using namespace std;

class WebSocketHandler {
public:
    WebSocketHandler(QWebSocket *websocket, WebSocketObserver &observer) :
        websocket(websocket), observer(observer)
    {}
    ~WebSocketHandler(){}

    void text_message_recieved(const QString &message);
    void client_disconnected();


private:
    QWebSocket *websocket;
    WebSocketObserver &observer;

};

WebSocketServer::WebSocketServer(const char* server_name, int port,
        WebSocketObserver &observer,
        QWebSocketServer::SslMode mode) :
    observer(observer), server(QString(server_name), mode)
{
    QObject::connect(&server, &QWebSocketServer::newConnection,
        std::bind( &WebSocketServer::new_websocket_connection, this));
    server.listen(QHostAddress::Any, port);
}

WebSocketServer::~WebSocketServer()
{
    server.close();
}

void WebSocketServer::new_websocket_connection()
{
    QWebSocket *pSocket = server.nextPendingConnection();
    WebSocketHandler *handler = new WebSocketHandler(pSocket, observer);

    qDebug() << "Web socket client connected";

    QObject::connect(pSocket, &QWebSocket::textMessageReceived,
            std::bind(&WebSocketHandler::text_message_recieved, handler, placeholders::_1));
    QObject::connect(pSocket, &QWebSocket::disconnected,
            std::bind(&WebSocketHandler::client_disconnected, handler));
}

void WebSocketHandler::text_message_recieved(const QString &message)
{
    qDebug() << "Message recieved: " << message;
    observer.text_message_received(*websocket, message);
}

void WebSocketHandler::client_disconnected()
{
    qDebug() << "Websocket client disconnected";
    delete websocket;
    delete this;
}
