import QtQuick 2.0
import QtQuick.Scene3D 2.0
import Qt3D.Core 2.0
import Qt3D.Render 2.0
import Qt3D.Input 2.0
import Qt3D.Extras 2.12

Scene3D {
    id: scene3d
    aspects: ["input", "logic"]
    property var keyframes
    property var trajectory
    property var currentPose
    property var trajectoryVertecies: new Float32Array(0)
    property var pointVertices: new Float32Array(0)
    property var pointColors: new Float32Array(0)
    property int showNKeyframes: 0

    function reset() {
        camera.position = Qt.vector3d( 0.0, 0.0, -1.0 );
        camera.upVector = Qt.vector3d( 0.0, -1.0, 0.0 );
        camera.viewCenter = Qt.vector3d( 0.0, 0.0, 0.0 );
    }

    function setTop() {
        camera.position = Qt.vector3d( 0.0, -5.0, 0.0 );
        camera.upVector = Qt.vector3d( 0.0, 0.0, 1.0 );
        camera.viewCenter = Qt.vector3d( 0.0, 0.0, 0.0 );
    }
    function setSide() {
        camera.position = Qt.vector3d( -5.0, 0.0, 0.0 );
        camera.upVector = Qt.vector3d( 0.0, -1.0, 0.0 );
        camera.viewCenter = Qt.vector3d( 0.0, 0.0, 0.0 );
    }
    function setFront() {
        camera.position = Qt.vector3d( 0.0, 0.0, -1.0 );
        camera.upVector = Qt.vector3d( 0.0, -1.0, 0.0 );
        camera.viewCenter = Qt.vector3d( 0.0, 0.0, 0.0 );
    }

    function showKeyframes() {
        if (keyframes === undefined)
            return;
        var floats_per_keyframe = keyframes[0].keypoints.length*3;
        var _points = new Float32Array(keyframes.length*floats_per_keyframe);
        var _keyframePoses = new Float32Array(keyframes.length*6);
        var _pointColors = new Float32Array(keyframes.length*floats_per_keyframe);
        var nKeyframes = keyframes.length
        if (showNKeyframes > 0 && showNKeyframes < keyframes.length)
            nKeyframes = showNKeyframes
        for (var i=0; i < nKeyframes; i++) {
            for (var j=0; j < keyframes[i].keypoints.length; j++) {
                _points[floats_per_keyframe*i+j*3+0] = keyframes[i].keypoints[j].x;
                _points[floats_per_keyframe*i+j*3+1] = keyframes[i].keypoints[j].y;
                _points[floats_per_keyframe*i+j*3+2] = keyframes[i].keypoints[j].z;

                _pointColors[floats_per_keyframe*i+j*3+0] = keyframes[i].colors[j].b/255.0;
                _pointColors[floats_per_keyframe*i+j*3+1] = keyframes[i].colors[j].g/255.0;
                _pointColors[floats_per_keyframe*i+j*3+2] = keyframes[i].colors[j].r/255.0;
            }

            _keyframePoses[i*6+0] = keyframes[i].pose.x;
            _keyframePoses[i*6+1] = keyframes[i].pose.y;
            _keyframePoses[i*6+2] = keyframes[i].pose.z;
            _keyframePoses[i*6+3] = keyframes[i].pose.pitch;
            _keyframePoses[i*6+4] = keyframes[i].pose.yaw;
            _keyframePoses[i*6+5] = keyframes[i].pose.roll;


        }

        pointVertices = _points;
        pointColors = _pointColors;
        keyframeCollection.poses = _keyframePoses;
    }

    onKeyframesChanged: showKeyframes()
    onShowNKeyframesChanged: showKeyframes()


    function showTrajectory() {
        if (trajectory === undefined)
            return;

        trajectoryVertecies = new Float32Array(trajectory);
    }
    onTrajectoryChanged: showTrajectory()

    Entity {
        id: sceneRoot

        Camera {
            id: camera
            projectionType: CameraLens.PerspectiveProjection
            fieldOfView: 45
            nearPlane : 0.1
            farPlane : 1000.0
            position: Qt.vector3d( 0.0, 0.0, -5.0 )
            upVector: Qt.vector3d( 0.0, -1.0, 0.0 )
            viewCenter: Qt.vector3d( 0.0, 0.0, 0.0 )
        }

        // This is the most natural controller I found
        OrbitCameraController {
            id: controller
            camera: camera
            linearSpeed: 100
        }

        Transform {
            id: cameraTransformation
            matrix: camera.viewMatrix
        }

        // We need to make sure points are rendered with a fixed size
        components: [
            RenderSettings {
                activeFrameGraph: Viewport {
                    RenderSurfaceSelector {
                        CameraSelector {
                            id : cameraSelector
                            camera: camera
                            FrustumCulling {
                                // We need a clear buffer, if not then points would be painted more than once
                                ClearBuffers {
                                    buffers : ClearBuffers.ColorDepthBuffer
                                    clearColor: "white"
                                    NoDraw {}
                                }
                                LayerFilter {
                                    layers: pointLayer
                                    RenderStateSet {
                                        // Set fixed point size
                                        renderStates: [
                                            PointSize { sizeMode: PointSize.Fixed; value: 5.0 }
                                        ]
                                    }
                                }
                            }
                        }
                    }
                }
            },
            // Event Source will be set by the Qt3DQuickWindow
            InputSettings {}
        ]

        // We will set the color of each vertex
        PerVertexColorMaterial {
            id: pointCloudMaterial
        }

        // Make sure pointLayer properties are applied to points (see Entity)
        Layer {
            id: pointLayer
        }

        GeometryRenderer{
            id: pointcloudRenderer

            primitiveType: GeometryRenderer.Points

            geometry: Geometry {
                Attribute {
                    attributeType: Attribute.VertexAttribute
                    vertexBaseType: Attribute.Float
                    vertexSize: 3
                    count: pointVertices.length/3
                    byteOffset: 0
                    byteStride: 3 * 4 // 1 vertex (=3 coordinates) * sizeof(float)
                    name: defaultPositionAttributeName
                    buffer: vertexBuffer
                }

                Attribute {
                    attributeType: Attribute.VertexAttribute
                    vertexBaseType: Attribute.Float
                    vertexSize: 3
                    count: 3
                    byteOffset: 0
                    byteStride: 3 * 4 // 1 vertex (=3 colors) * sizeof(float)
                    name: defaultColorAttributeName
                    buffer: colorBuffer
                }
            }
            // This can be the point cloud
            Buffer {
                id: vertexBuffer
                type: Buffer.VertexBuffer
                data: pointVertices
            }

            // This are the colors of each point
            Buffer {
                id: colorBuffer
                type: Buffer.VertexBuffer
                data: pointColors
            }
        }

        Entity {
            components: [pointcloudRenderer, pointCloudMaterial, pointLayer]
        }

        // We will set the color of each vertex
        PhongMaterial {
            id: trajectoryMaterial
            ambient: "red"
        }

        GeometryRenderer{
            id: trajectoryRenderer

            primitiveType: GeometryRenderer.LineStrip

            geometry: Geometry {
                Attribute {
                    attributeType: Attribute.VertexAttribute
                    vertexBaseType: Attribute.Float
                    vertexSize: 3
                    count: trajectoryVertecies.length/6
                    byteOffset: 0
                    byteStride: 6 * 4 // 1 vertex (=3 coordinates + 3 angles) * sizeof(float)
                    name: defaultPositionAttributeName
                    buffer: trjacetoryVertexBuffer
                }
            }
            // This can be the point cloud
            Buffer {
                id: trjacetoryVertexBuffer
                type: Buffer.VertexBuffer
                data: trajectoryVertecies
            }
        }

        Entity {
            components: [trajectoryRenderer, trajectoryMaterial, pointLayer]
        }

        GeometryRenderer{
            id: keyframeRenderer

            primitiveType: GeometryRenderer.Lines

            geometry: Geometry {
                Attribute {
                    attributeType: Attribute.VertexAttribute
                    vertexBaseType: Attribute.Float
                    vertexSize: 3
                    count: 5
                    byteOffset: 0
                    byteStride: 3 * 4 // 1 vertex (=3 coordinates) * sizeof(float)
                    name: defaultPositionAttributeName
                    buffer: keyframeBuffer
                }
                Attribute {
                      attributeType: Attribute.IndexAttribute
                      vertexBaseType: Attribute.UnsignedInt
                      vertexSize: 1
                      count: 16
                      byteOffset: 0
                      byteStride: 1 * 4 // 1 index * sizeof(Uint32)
                      buffer: indexBuffer
                }
            }
            // This can be the point cloud
            Buffer {
                property real width: 0.1
                property real height: 0.08
                property real depth: 0.07
                id: keyframeBuffer
                type: Buffer.VertexBuffer
                data: new Float32Array([
                    0,0,0,
                    -width, height, depth,
                    -width, -height, depth,
                    width, -height, depth,
                    width, height, depth
                ])

            }

            Buffer {
                id: indexBuffer
                type: Buffer.IndexBuffer
                data: new Uint32Array([
                    0,1,
                    0,2,
                    0,3,
                    0,4,
                    1,2,
                    2,3,
                    3,4,
                    4,1
                ])
            }
        }

        NodeInstantiator {
            id: keyframeCollection

            property var poses: new Float32Array()
            model: poses.length/6

            Entity {
                Transform {
                    id: keyframeTransformation
                    translation: Qt.vector3d(keyframeCollection.poses[index*6+0], keyframeCollection.poses[index*6+1], keyframeCollection.poses[index*6+2])
                    rotation: fromEulerAngles(Qt.vector3d(180.0*keyframeCollection.poses[index*6+3]/Math.PI, 180.0*keyframeCollection.poses[index*6+4]/Math.PI, 180.0*keyframeCollection.poses[index*6+5]/Math.PI))
                }

                PhongMaterial {
                    id: wireframeMaterial
                    ambient: "blue"

                }

                components: [keyframeTransformation, keyframeRenderer, wireframeMaterial, pointLayer]
            }
        }

        Entity {
            PhongMaterial {
                id: poseMaterial
                ambient: "green"

            }
            Transform {
                id: poseTransformation

                property var currentPose: scene3d.currentPose

                translation: Qt.vector3d(currentPose.x, currentPose.y, currentPose.z)
                rotation: fromEulerAngles(Qt.vector3d(180.0*currentPose.pitch/Math.PI, 180.0*currentPose.yaw/Math.PI, 180.0*currentPose.roll/Math.PI))
            }

            components: [poseTransformation, keyframeRenderer, poseMaterial, pointLayer]
        }
    }
}
