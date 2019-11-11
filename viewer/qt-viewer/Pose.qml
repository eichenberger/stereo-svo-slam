import QtQuick 2.0
import QtWebSockets 1.1

WebSocket {
    property var pose: {"x": 0.0, "y": 0.0, "z": 0.0, "pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    url: "ws://localhost:8001/pose"
    onTextMessageReceived: {
        this.pose = JSON.parse(message)['pose'];
    }
    onStatusChanged: {
        console.log("Status changed: " + status);
        if (status == WebSocket.Error) {
            console.log("Error: " + errorString)
         } else if (status === WebSocket.Closed) {
            console.log("\nSocket closed");
            this.active = false;
         }
    }

    function getPose() {
        sendTextMessage("get");
    }

}
