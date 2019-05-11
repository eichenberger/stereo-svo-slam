import QtQuick 2.0
import QtQuick.Controls 2.3
import QtQuick.Controls.Styles 1.4

Item {

    property var keyframes: keyframes.keyframes
    signal reset()
    Column {
        anchors.fill: parent

        KeyFrames {
            id: keyframes
        }

        Button {
            width: parent.width
            text: qsTr("Connect")
            onClicked: keyframes.active = true
        }

        Button {
            width: parent.width
            text: qsTr("Keyframes")
            onClicked: keyframes.getKeyFrames()
        }

        Button {
            width: parent.width
            text: qsTr("Reset Camera")
            onClicked: reset()
        }
    }

}
