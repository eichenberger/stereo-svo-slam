import QtQuick 2.0
import QtWebSockets 1.1

WebSocket {
    property var trajectory
    url: "ws://localhost:8001/trajectory"
    onTextMessageReceived: {
        this.trajectory = JSON.parse(message)['trajectory'];
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

    function getTrajectory() {
        sendTextMessage("get");
    }

}
