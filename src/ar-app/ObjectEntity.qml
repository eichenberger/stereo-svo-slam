import QtQuick 2.0
import Qt3D.Core 2.0
import Qt3D.Render 2.0
import Qt3D.Input 2.0
import Qt3D.Extras 2.0

Entity {
    Entity {
        id: tux
        components: [
            Transform {
                scale3D: Qt.vector3d(0.001, 0.001, 0.001)
                rotation: fromEulerAngles(0,180, 180)
            },
            SceneLoader {
                source: "qrc:/tux.dae"
            }
        ]
    }
    // Spot Light (sweeping)
    Entity {
        components: [
            SpotLight {
                localDirection: Qt.vector3d(0.0, 0.0, 0.0)
                color: "white"
                intensity: 0.3
            },
            Transform {
                id: spotLightTransform
                translation: Qt.vector3d(0.0, -2.0, -2.0)
            }
        ]
    }

    // 2 x Directional Lights (steady)
    Entity {
        components: [
            DirectionalLight {
                worldDirection: Qt.vector3d(0.0, 0.0, 2.0).normalized();
                color: "#fbf9ce"
                intensity: 0.4
            }
        ]
    }

}
