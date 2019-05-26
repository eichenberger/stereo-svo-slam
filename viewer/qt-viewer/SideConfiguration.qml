import QtQuick 2.0
import QtQuick.Controls 2.3
import QtQuick.Controls.Styles 1.4
import QtQuick.Scene3D 2.0

Item {

    property var keyframes: keyframes.keyframes
    property var pose: pose.pose
    signal setTop()
    signal setSide()
    signal setFront()
    signal reset()
    Column {
        anchors.fill: parent

        KeyFrames {
            id: keyframes
        }

        Pose {
            id: pose
        }

        Button {
            width: parent.width
            text: qsTr("Connect")
            onClicked: {keyframes.active = true; pose.active = true;}
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

        Button {
            width: parent.width
            text: qsTr("Current Pose")
            onClicked: pose.getPose()
        }

        Button {
            width: parent.width
            text: qsTr("Top View")
            onClicked: setTop()
        }

        Button {
            width: parent.width
            text: qsTr("Side View")
            onClicked: setSide()
        }

        Button {
            width: parent.width
            text: qsTr("Front View")
            onClicked: setFront()
        }
    }

}
