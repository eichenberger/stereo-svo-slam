import QtQuick 2.0
import QtQuick.Scene3D 2.0
import Qt3D.Core 2.0
import Qt3D.Render 2.0
import Qt3D.Input 2.0
import Qt3D.Extras 2.0

Scene3D {
    id: scene3d
    aspects: ["input", "logic"]
    property var keyframes
    property var pointVertices: new Float32Array(0)
    property var pointColors: new Float32Array(0)
    property var keyframeVertices: new Float32Array(0)
    property var keyframeIndexes: new Uint32Array(0)
    property var originalViewMatrix

    function reset() {
        camera.position = Qt.vector3d( 0.0, 0.0, -1.0 );
        camera.upVector = Qt.vector3d( 0.0, -1.0, 0.0 );
        camera.viewCenter = Qt.vector3d( 0.0, 0.0, 0.0 );
    }

    onKeyframesChanged: {
        if (keyframes === undefined)
            return;
        var floats_per_keyframe = keyframes[0]['keypoints'][0].length*3;
        var _points = new Float32Array(keyframes.length*floats_per_keyframe);
        var _keyframes = new Float32Array(keyframes.length*3*5);
        var _keyframeIndexes = new Uint32Array(keyframes.length*16);
        var _pointColors = new Float32Array(keyframes.length*floats_per_keyframe);;
        for (var i=0; i < keyframes.length; i++) {
            for (var j=0; j < keyframes[i]['keypoints'][0].length; j++) {
                _points[floats_per_keyframe*i+j*3+0] = keyframes[i]['keypoints'][0][j];
                _points[floats_per_keyframe*i+j*3+1] = keyframes[i]['keypoints'][1][j];
                _points[floats_per_keyframe*i+j*3+2] = keyframes[i]['keypoints'][2][j];

                _pointColors[floats_per_keyframe*i+j*3+0] = keyframes[i]['colors'][j]/255.0;
                _pointColors[floats_per_keyframe*i+j*3+1] = keyframes[i]['colors'][j]/255.0;
                _pointColors[floats_per_keyframe*i+j*3+2] = keyframes[i]['colors'][j]/255.0;
            }


            // Create camera with lines
            var cam_size = 0.1;
            var cam_depth = 0.08;

            _keyframes[i*3*5+0] = keyframes[i]['pose'][0];
            _keyframes[i*3*5+1] = keyframes[i]['pose'][1];
            _keyframes[i*3*5+2] = keyframes[i]['pose'][2];

            _keyframes[i*3*5+3] = keyframes[i]['pose'][0]-cam_size;
            _keyframes[i*3*5+4] = keyframes[i]['pose'][1]+cam_size;
            _keyframes[i*3*5+5] = keyframes[i]['pose'][2]+cam_depth;

            _keyframes[i*3*5+6] = keyframes[i]['pose'][0]-cam_size;
            _keyframes[i*3*5+7] = keyframes[i]['pose'][1]-cam_size;
            _keyframes[i*3*5+8] = keyframes[i]['pose'][2]+cam_depth;

            _keyframes[i*3*5+9] = keyframes[i]['pose'][0]+cam_size;
            _keyframes[i*3*5+10] = keyframes[i]['pose'][1]-cam_size;
            _keyframes[i*3*5+11] = keyframes[i]['pose'][2]+cam_depth;

            _keyframes[i*3*5+12] = keyframes[i]['pose'][0]+cam_size;
            _keyframes[i*3*5+13] = keyframes[i]['pose'][1]+cam_size;
            _keyframes[i*3*5+14] = keyframes[i]['pose'][2]+cam_depth;

            _keyframeIndexes[i*16+0] = i*5+0;
            _keyframeIndexes[i*16+1] = i*5+1;

            _keyframeIndexes[i*16+2] = i*5+0;
            _keyframeIndexes[i*16+3] = i*5+2;

            _keyframeIndexes[i*16+4] = i*5+0;
            _keyframeIndexes[i*16+5] = i*5+3;

            _keyframeIndexes[i*16+6] = i*5+0;
            _keyframeIndexes[i*16+7] = i*5+4;

            _keyframeIndexes[i*16+8] = i*5+1;
            _keyframeIndexes[i*16+9] = i*5+2;

            _keyframeIndexes[i*16+10] = i*5+2;
            _keyframeIndexes[i*16+11] = i*5+3;

            _keyframeIndexes[i*16+12] = i*5+3;
            _keyframeIndexes[i*16+13] = i*5+4;

            _keyframeIndexes[i*16+14] = i*5+4;
            _keyframeIndexes[i*16+15] = i*5+1;
        }

        pointVertices = _points;
        keyframeVertices = _keyframes;
        keyframeIndexes = _keyframeIndexes;
        pointColors = _pointColors;
    }

    Entity {
        id: sceneRoot

        Camera {
            id: camera
            projectionType: CameraLens.PerspectiveProjection
            fieldOfView: 45
            nearPlane : 0.1
            farPlane : 1000.0
            position: Qt.vector3d( 0.0, 0.0, -1.0 )
            upVector: Qt.vector3d( 0.0, -1.0, 0.0 )
            viewCenter: Qt.vector3d( 0.0, 0.0, 0.0 )
        }

        // This is the most natural controller I found
        OrbitCameraController {
            id: controller
            camera: camera
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

        // Now we create a entity that contains, pointcloud, material and layer information
        Entity {
            components: [pointcloudRenderer, pointCloudMaterial, pointLayer]
        }

        GeometryRenderer{
            id: keyframeRenderer

            primitiveType: GeometryRenderer.Lines

            geometry: Geometry {
              Attribute {
                attributeType: Attribute.VertexAttribute
                vertexBaseType: Attribute.Float
                vertexSize: 3
                count: keyframeVertices.length/3
                byteOffset: 0
                byteStride: 3 * 4 // 1 vertex (=3 coordinates) * sizeof(float)
                name: defaultPositionAttributeName
                buffer: keyframeBuffer
              }
              Attribute {
                  attributeType: Attribute.IndexAttribute
                  vertexBaseType: Attribute.UnsignedInt
                  vertexSize: 1
                  count: keyframeIndexes.length
                  byteOffset: 0
                  byteStride: 1 * 4 // 1 index * sizeof(Uint32)
                  buffer: indexBuffer
              }
            }
            // This can be the point cloud
            Buffer {
                id: keyframeBuffer
                type: Buffer.VertexBuffer
                data: keyframeVertices
            }

            Buffer {
                id: indexBuffer
                type: Buffer.IndexBuffer
                data: keyframeIndexes
            }
        }


        PhongMaterial {
            id: wireframeMaterial
            ambient: "blue"

        }

        // Now we create a entity that contains, pointcloud, material and layer information
        Entity {
            components: [keyframeRenderer, wireframeMaterial, pointLayer]
        }
    }
}
