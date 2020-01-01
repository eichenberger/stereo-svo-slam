import QtQuick 2.0

import Qt3D.Core 2.0
import Qt3D.Render 2.0
import Qt3D.Input 2.0
import Qt3D.Extras 2.0

Entity {
    property var cameraPosition: Qt.vector3d( 0.0, 0.0, 0.0 )
    property var cameraRotation: Qt.vector3d( 0.0, 0.0, 0.0 )

    Entity {
        Camera {
            id: camera
            projectionType: CameraLens.FrustumProjection
            nearPlane : 0.01
            farPlane : 20.0
            // See http://kgeorge.github.io/2014/03/08/calculating-opengl-perspective-matrix-from-opencv-intrinsic-matrix for explaination
            right: (0.01*365.86)/743.804
            left: -right
            top: (0.01*235.7)/743.804
            bottom: -top
            position: cameraPosition
            upVector: Qt.vector3d( 0.0, -1.0, 0.0 )
            viewCenter: Qt.vector3d( 0.0, 0.0, 1.0 )
        }

        components: [
            Transform {
                translation: cameraPosition
                rotationX: cameraRotation.x
                rotationY: cameraRotation.y
                rotationZ: cameraRotation.z
            }
        ]
    }

    FirstPersonCameraController { camera: camera }

    components: [
        RenderSettings {
            activeFrameGraph: ForwardRenderer {
                camera: camera
                clearColor: "transparent"
            }
        },
        InputSettings { }
    ]


    ObjectEntity {
        id: groupedObject
        components: [
            Transform {
                id: test
                translation: Qt.vector3d(0.0, 0.0, 1.0)
            }
        ]
    }

}
