import cv2
import argparse
import numpy as np
import math
import io
import asyncio
import websockets
import yaml
import json

from slam_accelerator import CameraSettings, StereoSlam

from draw_kps import draw_frame

from econ_utils import set_auto_exposure, set_manual_exposure

class EconInput():
    def __init__(self, camera, hidraw, settings):
        self.cap = cv2.VideoCapture(camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 752)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        set_manual_exposure(hidraw, 20000)

        self.settings = cv2.FileStorage(settings, cv2.FILE_STORAGE_READ)

        l_d = self.settings.getNode('LEFT.D').mat()

        self.camera_settings = CameraSettings()

        self.camera_settings.baseline = self.settings.getNode('Camera.bf').real()
        self.camera_settings.fx = self.settings.getNode('Camera.fx').real()
        self.camera_settings.fy = self.settings.getNode('Camera.fy').real()
        self.camera_settings.cx = self.settings.getNode('Camera.cx').real()
        self.camera_settings.cy = self.settings.getNode('Camera.cy').real()

        self.camera_settings.k1 = l_d[0,0]
        self.camera_settings.k2 = l_d[0,1]
        self.camera_settings.k3 = l_d[0,4]

        self.camera_settings.p1 = l_d[0,2]
        self.camera_settings.p2 = l_d[0,3]

        self.camera_settings.grid_width = 40
        self.camera_settings.grid_height = 30
        self.camera_settings.search_x = 30
        self.camera_settings.search_y = 6


    def read(self):
        ret, image = self.cap.read()
        gray_r = cv2.extractChannel(image, 1);
        gray_l = cv2.extractChannel(image, 2);

        #gray_l = cv2.remap(gray_l, self.m1l, self.m2l, cv2.INTER_LINEAR)
        #gray_r = cv2.remap(gray_r, self.m1r, self.m2r, cv2.INTER_LINEAR)

        return gray_l, gray_r

class BlenderInput():
    def __init__(self, video):
        self.cap = cv2.VideoCapture(video)
        self.frame_nr = 0

        width = 752
        height = 480
        sensor_size = 32
        focal_length = 25
        self.camera_settings = CameraSettings()
        self.camera_settings.cx = width/2
        self.camera_settings.cy = height/2
        self.camera_settings.fx = focal_length/(sensor_size/width)
        self.camera_settings.fy = focal_length/(sensor_size/height)

        self.camera_settings.baseline = 0.06*self.camera_settings.fx

        self.camera_settings.k1 = 0.0
        self.camera_settings.k2 = 0.0
        self.camera_settings.k3 = 0.0

        self.camera_settings.p1 = 0.0
        self.camera_settings.p2 = 0.0

        self.camera_settings.grid_width = 40
        self.camera_settings.grid_height = 30
        self.camera_settings.search_x = 30
        self.camera_settings.search_y = 6
        self.camera_settings.window_size_pose_estimator = 4
        self.camera_settings.window_size_opt_flow = 31
        self.camera_settings.window_size_depth_calculator = 31
        self.camera_settings.max_pyramid_levels = 5
        self.camera_settings.min_pyramid_level_pose_estimation = 2

    def read(self):

#        if self.frame_nr == 75:
#            raise("Hallo")
        self.cap.set(1, self.frame_nr)
        ret, image = self.cap.read()
        self.frame_nr += 1
        image = image[:,:,1]
        width = int(image.shape[1]/2)
        gray_r = image[:,0:width]
        gray_l = image[:,width:2*width]

        return gray_l, gray_r


class VideoInput():
    def __init__(self, left, right, settings):
        self.capl = cv2.VideoCapture(left)
        self.capr = cv2.VideoCapture(right)
        #skip first 10 images
        #self.capl.set(cv2.CAP_PROP_POS_FRAMES, 10)
        #self.capr.set(cv2.CAP_PROP_POS_FRAMES, 10)
        self.settings = cv2.FileStorage(settings, cv2.FILE_STORAGE_READ)

        l_d = self.settings.getNode('LEFT.D').mat()

        self.camera_settings = CameraSettings()

        self.camera_settings.baseline = self.settings.getNode('Camera.bf').real()
        self.camera_settings.fx = self.settings.getNode('Camera.fx').real()
        self.camera_settings.fy = self.settings.getNode('Camera.fy').real()
        self.camera_settings.cx = self.settings.getNode('Camera.cx').real()
        self.camera_settings.cy = self.settings.getNode('Camera.cy').real()

        self.camera_settings.k1 = l_d[0,0]
        self.camera_settings.k2 = l_d[0,1]
        self.camera_settings.k3 = l_d[0,4]

        self.camera_settings.p1 = l_d[0,2]
        self.camera_settings.p2 = l_d[0,3]

        self.camera_settings.grid_width = 40
        self.camera_settings.grid_height = 30
        self.camera_settings.search_x = 30
        self.camera_settings.search_y = 6

    def read(self):
        ret, iml = self.capl.read()
        ret, imr = self.capr.read()

        # This is super slow (why?)
        gray_l = cv2.cvtColor(iml, cv2.COLOR_RGB2GRAY)
        gray_r = cv2.cvtColor(imr, cv2.COLOR_RGB2GRAY)

        # Don't rectify at the moment
        # tm = cv2.TickMeter()
        # tm.start()
        # gray_l = cv2.remap(gray_l, self.m1l, self.m2l, cv2.INTER_LINEAR)
        # gray_r = cv2.remap(gray_r, self.m1r, self.m2r, cv2.INTER_LINEAR)
        # tm.stop()

        #print(f'rectification took: {tm.getTimeMilli()} ms')

        return gray_l, gray_r


def main():
    parser = argparse.ArgumentParser(description='Edge slam test')

    subparsers = parser.add_subparsers()
    econ_parser = subparsers.add_parser(name='econ', description='use econ input')
    econ_parser.add_argument('camera', help='camera to use', type=str)
    econ_parser.add_argument('hidraw', help='camera hidraw device', type=str)
    econ_parser.add_argument('settings', help='settings file', type=str)
    econ_parser.set_defaults(func=lambda args: EconInput(args.camera, args.hidraw, args.settings))

    video_parser = subparsers.add_parser(name='video', description='use video input')
    video_parser.add_argument('left', help='left video input', type=str)
    video_parser.add_argument('right', help='right video input', type=str)
    video_parser.add_argument('settings', help='settings file', type=str)
    video_parser.set_defaults(func=lambda args: VideoInput(args.left, args.right, args.settings))

    video_parser = subparsers.add_parser(name='blender', description='use blender input')
    video_parser.add_argument('video', help='video left and right side by side', type=str)
    video_parser.set_defaults(func=lambda args: BlenderInput(args.video))


    args = parser.parse_args()
    camera = args.func(args)

    # Size of window for depth estimation and pose estimation
    camera.camera_settings.window_size_pose_estimator = 4
    # Size of window for optical flow
    camera.camera_settings.window_size_opt_flow = 8
    camera.camera_settings.max_pyramid_levels = 3

    slam = StereoSlam(camera.camera_settings)

    gray_l, gray_r = camera.read()
    timestamp = cv2.TickMeter()
    timestamp.start()
    async def read_frame():
        print("SLAM started")
        key = 0
        try:
            while key != ord('q'):
                gray_l, gray_r = camera.read()
                tm = cv2.TickMeter()
                tm.start()
                timestamp.stop()
                dt = timestamp.getTimeSec()
                timestamp.start()
                slam.new_image(gray_l.copy(), gray_r.copy(), dt)
                tm.stop()
                print(f"stereo slam took: {tm.getTimeMilli()} ms")

                draw_frame(slam.get_keyframe(), slam.get_frame())

                key = cv2.waitKey(1)
                await asyncio.sleep(0.001)
        except cv2.error:
            print(f'camera read done, wait for exit')

        while key != ord('q'):
            key = cv2.waitKey(1)
            await asyncio.sleep(0.2)

    async def websocketserver(websocket, path):
        async for message in websocket:
            if path == "/keyframes":
                if message == "get":
                    keyframes = []
                    _keyframes = slam.get_keyframes()
                    for keyframe in _keyframes:
                        kps = []
                        i = 0
                        for kp in keyframe.kps.kps3d:
                            kps.append({'x': kp.x, 'y': kp.y, 'z': kp.z})
                            i += 1

                        colors = []
                        for inf in keyframe.kps.info:
                            colors.append(inf.color)
                            i += 1

                        pose = {
                            'x': keyframe.pose.x,
                            'y': keyframe.pose.y,
                            'z': keyframe.pose.z,
                            'rx': keyframe.pose.rx,
                            'ry': keyframe.pose.ry,
                            'rz': keyframe.pose.rz
                        }

                        print(f"Pose: {pose}")

                        keyframes.append({
                            'keypoints': kps,
                            'pose': pose,
                            'colors': colors})
                    encoder = json.JSONEncoder()
                    await websocket.send(encoder.encode(keyframes))
            elif path == "/pose":
                if message == "get":
                    encoder = json.JSONEncoder()
                    frame = slam.get_frame()
                    pose = {
                        'x': frame.pose.x,
                        'y': frame.pose.y,
                        'z': frame.pose.z,
                        'rx': frame.pose.rx,
                        'ry': frame.pose.ry,
                        'rz': frame.pose.rz
                    }

                    pose = {'pose': pose}
                    await websocket.send(encoder.encode(pose))


    async def async_main():
        slam_task = asyncio.create_task(read_frame())
        server_task = websockets.serve(websocketserver, 'localhost', 8001)

        # This yields a WebsocketServer object
        server = await server_task

        # SLAM is the main task
        await slam_task
        server.wait_closed()

    asyncio.run(async_main())

if __name__ == "__main__":
    main()
