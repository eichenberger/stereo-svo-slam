import QtQuick 2.0
import QtWebSockets 1.1

WebSocket {
    property var keyframes
    url: "ws://localhost:8001/keyframes"
    onTextMessageReceived: {
        this.keyframes = JSON.parse(message);
    }
    onStatusChanged: {
        console.log("Status changed: " + status);
        if (status == WebSocket.Error) {
             console.log("Error: " + errorString);
         } else if (status === WebSocket.Closed) {
             console.log("\nSocket closed");
             this.active = false;
         }
    }

    function getKeyFrames() {
        sendTextMessage("get");
    }
}
