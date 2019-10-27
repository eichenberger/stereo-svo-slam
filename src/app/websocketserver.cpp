#include <vector>

#include <QtWebSockets/QtWebSockets>
#include <QtCore/QObject>
#include <QtCore/QList>
#include <QtNetwork/QHostAddress>

using namespace std;

class WebSocketServer {
public:
	WebSocketServer(const char* server_name, int port,
			QWebSocketServer::SslMode mode = QWebSocketServer::NonSecureMode);
	~WebSocketServer();


private:
	void new_websocket_connection();

	void text_message_recieved(const QString &message);

	QWebSocketServer server;

};

class WebSocketHandler {
public:
	WebSocketHandler(QWebSocket *websocket) : websocket(websocket)
	{}
	~WebSocketHandler(){}

	void text_message_recieved(const QString &message);
	void client_disconnected();


private:
	QWebSocket *websocket;

};

WebSocketServer::WebSocketServer(const char* server_name, int port,
		QWebSocketServer::SslMode mode) :
	server(QString(server_name), mode)
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
	WebSocketHandler *handler = new WebSocketHandler(pSocket);

	QObject::connect(pSocket, &QWebSocket::textMessageReceived,
			std::bind(&WebSocketHandler::text_message_recieved, handler, placeholders::_1));
	QObject::connect(pSocket, &QWebSocket::disconnected,
			std::bind(&WebSocketHandler::client_disconnected, handler));
}

void WebSocketHandler::text_message_recieved(const QString &message)
{
	qDebug() << "Message recieved: " << message;
}

void WebSocketHandler::client_disconnected()
{
	qDebug() << "Websocket client disconnected";
	delete websocket;
	delete this;
}
