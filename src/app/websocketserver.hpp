#ifndef WEBSOCKETSERVER_HPP
#define WEBSOCKETSERVER_HPP

#include <QtWebSockets/QtWebSockets>

class WebSocketObserver {
public:
    WebSocketObserver() {}
    virtual void text_message_received(QWebSocket &socket,
            const QString &message) = 0;

};

class WebSocketServer {
public:
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
