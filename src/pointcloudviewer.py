# -----------------------------------------------------------------------------
# Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
import numpy as np
from glumpy import app, gl, gloo, glm
from glumpy.transforms import OrthographicProjection, Trackball, Position

from math import cos, sin

vertex = """
#version 120
attribute vec3  position;
attribute vec3  colors;
varying vec4  v_bg_color;
void main (void)
{
    v_bg_color = vec4(1.0,0,0,1.0);
    gl_Position = <transform(vec4(position,1.0))>;
    v_bg_color = vec4(colors, 0.5);
    gl_PointSize = 4.0;
}
"""

fragment = """
#version 120
varying vec4  v_bg_color;

void main()
{
    gl_FragColor = v_bg_color;
}
"""

vertex_cam = """
#version 120
attribute vec3  position;
void main (void)
{
    gl_Position = <transform(vec4(position,1.0))>;
}
"""

fragment_cam = """
#version 120

void main()
{
    gl_FragColor = vec4(0,0,0,0.5);
}
"""


class PointCloudViewer():
    def __init__(self):
        window = app.Window(width=800, height=800, color=(1,1,1,1))
        window.dispatch_event('on_draw', self.on_draw)
        self.program = gloo.Program(vertex, fragment)
        self.camera = gloo.Program(vertex_cam, fragment_cam, count=15)

        self.points = None
        self.colors = None
        self.program['transform'] = Trackball(Position('position'))
        self.camera['transform'] = self.program['transform']

        gl.glEnable(gl.GL_DEPTH_TEST)
        window.attach(self.program["transform"])
        window.on_draw = self.on_draw
        self.window = window

    def on_draw(self, dt):
        self.window.clear()
        if self.points is not None:
            self.program.draw(gl.GL_POINTS)
            self.camera.draw(gl.GL_LINE_LOOP)

    def _update_pc(self):
        data = np.empty(self.points.shape[0], dtype=[('position', np.float32, 3),
                                                     ('colors', np.float32, 3)])
        data['position'] = self.points
        data['colors'] = self.colors
        data = data.view(gloo.VertexBuffer)
        self.program.bind(data)

    def add_point(self, point):
        self.add_points(np.array([point]))

    def add_points(self, points, colors):
        points = np.float32(points)
        if self.points is not None:
            self.points = np.append(self.points, points, axis=0)
            self.colors = np.append(self.colors, colors, axis=0)
        else:
            self.points = points
            self.colors = colors
        self._update_pc()


    def set_camera_pose(self, pose):
        camera_size = 0.1
        focus = 1.2
        camera = camera_size * np.mat([
                         [+0, +0, +0     ],
                         [-1, -1, +focus ],
                         [-1, +1, +focus ],

                         [+0, +0, +0     ],
                         [-1, -1, +focus ],
                         [+1, -1, +focus ],

                         [+0, +0, +0     ],
                         [+1, +1, +focus ],
                         [+1, -1, +focus ],

                         [+0, +0, +0     ],
                         [-1, -1, +focus ],
                         [-1, +1, +focus ],

                         [+0, +0, +0     ],
                         [+1, +1, +focus ],
                         [-1, +1, +focus ],
                         ])
        self.camera['position'] = camera

def rot_mat_x(angle):
    return np.mat([[1, 0, 0],
                    [0, cos(angle), -sin(angle)],
                    [0, sin(angle), cos(angle)]])

def rot_mat_y(angle):
    return np.mat([[cos(angle), 0, sin(angle)],
                    [0, 1, 0],
                    [-sin(angle), 0, cos(angle)]])

def rot_mat_z(angle):
    return np.mat([[cos(angle), -sin(angle), 0],
                    [sin(angle), cos(angle), 0],
                    [0, 0, 1]])

def rot_mat(angles):
    rot_x = rot_mat_x(angles[0])
    rot_y = rot_mat_y(angles[1])
    rot_z = rot_mat_z(angles[2])

    rot = np.matmul(rot_x, rot_y)
    rot = np.matmul(rot, rot_z)

    return rot


#pcv = PointCloudViewer()
#
#pose = np.append(rot_mat([0, 0, 0]), [[0],[0],[0]], axis=1)
#
#pcv.set_camera_pose(pose)
#
#@pcv.window.timer(1)
#def timeout(dt):
#    points = np.random.randn(2,3)
#    pcv.add_points(points)
#
##@pcv.window.timer(1)
#def test():
#    points = np.array([
#        [ 0.5, 0.5, 0.5],
#        [-0.5, 0.5, 0.5],
#        [-0.5,-0.5, 0.5],
#        [ 0.5,-0.5, 0.5]
#    ])
#    pcv.add_points(points)
#
#test()
#app.run()
