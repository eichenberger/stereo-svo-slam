import QtQuick 2.0
import Qt3D.Core 2.0
import Qt3D.Render 2.0
import Qt3D.Input 2.0
import Qt3D.Extras 2.0

Entity {

    Entity {
        id: torusEntity
        components: [
            Transform {
                id: torusTransform
                scale3D: Qt.vector3d(1, 0.8, 0.2)
                rotation: fromAxisAndAngle(Qt.vector3d(1, 0, 0), 45)
            },
            PhongMaterial {
                id: torusMaterial
                ambient: "#880000"
                shininess: 1.0
                diffuse: "#0000FF"
                specular: "#00FF00"

            },
            TorusMesh {
                id: torusMesh
                radius: 0.2
                minorRadius: 0.1
                rings: 100
                slices: 20
            }
        ]
    }

    Entity {
        id: sphereEntity
        components: [
            SphereMesh {
                id: sphereMesh
                radius: 0.1
            },
            PhongMaterial {
                id: sphereMaterial
                ambient: "#880000"
                shininess: 1.0
                diffuse: "#0000FF"
                specular: "#00FF00"

            },
            Transform {
                id: sphereTransform
                property real userAngle: 0.0
                matrix: {
                    var m = Qt.matrix4x4();
                    m.rotate(userAngle, Qt.vector3d(0, 1, 0))
                    m.translate(Qt.vector3d(0.5, 0, 0));
                    return m;
                }
            }
        ]
        NumberAnimation {
            target: sphereTransform
            property: "userAngle"
            duration: 10000
            from: 0
            to: 360

            loops: Animation.Infinite
            running: true
        }
    }
}
