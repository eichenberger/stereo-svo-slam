import QtQuick 2.0
import QtQuick.Scene3D 2.0
import QtMultimedia 5.0

import OpenCVImageProvider 1.0

Item {
    anchors.fill: parent

    VideoOutput {
        height: parent.height
        width: 752.0/480.0*parent.height
        id: video
        fillMode: VideoOutput.PreserveAspectFit
        source: mediaplayer

        Scene3D {
            id: scene3d
            width: parent.width
            height: parent.height
            focus: true
            aspects: ["input", "logic"]
            cameraAspectRatioMode: Scene3D.AutomaticAspectRatio

            AnimatedEntity {
                id: mainEntity

            }
        }
    }
    Connections {
        target: slam
        onPose: {mainEntity.cameraPosition = position; mainEntity.cameraRotation = rotation}
    }

}
