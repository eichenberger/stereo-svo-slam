import cv2
import argparse
import numpy as np
import math
import io
import asyncio
import websockets
import yaml
import json

from stereo_slam import StereoSLAM

from econ_utils import set_auto_exposure, set_manual_exposure

class EconInput():
    def __init__(self, camera, hidraw, settings):
        self.cap = cv2.VideoCapture(camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 752)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        set_manual_exposure(hidraw, 20000)

        self.settings = cv2.FileStorage(settings, cv2.FILE_STORAGE_READ)

        l_d = self.settings.getNode('LEFT.D').mat()
        l_k = self.settings.getNode('LEFT.K').mat()
        l_r = self.settings.getNode('LEFT.R').mat()
        l_p = self.settings.getNode('LEFT.P').mat()
        l_width = int(self.settings.getNode('LEFT.width').real())
        l_height = int(self.settings.getNode('LEFT.height').real())

        r_d = self.settings.getNode('RIGHT.D').mat()
        r_k = self.settings.getNode('RIGHT.K').mat()
        r_r = self.settings.getNode('RIGHT.R').mat()
        r_p = self.settings.getNode('RIGHT.P').mat()
        r_width = int(self.settings.getNode('RIGHT.width').real())
        r_height = int(self.settings.getNode('RIGHT.height').real())

        self.m1l, self.m2l = cv2.initUndistortRectifyMap(l_k, l_d, l_r, l_p, (l_width, l_height), cv2.CV_32F)
        self.m1r, self.m2r = cv2.initUndistortRectifyMap(r_k, r_d, r_r, r_p, (r_width, r_height), cv2.CV_32F)

        self.baseline = self.settings.getNode('Camera.bf')
        self.fx = self.settings.getNode('Camera.fx')
        self.fy = self.settings.getNode('Camera.fy')
        self.cx = self.settings.getNode('Camera.cx')
        self.cy = self.settings.getNode('Camera.cy')


    def read(self):
        ret, image = self.cap.read()
        gray_r = cv2.extractChannel(image, 1);
        gray_l = cv2.extractChannel(image, 2);

        gray_l = cv2.remap(gray_l, self.m1l, self.m2l, cv2.INTER_LINEAR)
        gray_r = cv2.remap(gray_r, self.m1r, self.m2r, cv2.INTER_LINEAR)

        return gray_l, gray_r

class BlenderInput():
    def __init__(self, video):
        self.cap = cv2.VideoCapture(video)
        self.frame_nr = 1

        width = 752
        height = 480
        sensor_size = 32
        focal_length = 25
        self.cx = width/2
        self.cy = height/2
        self.fx = focal_length/(sensor_size/width)
        self.fy = focal_length/(sensor_size/height)

        self.baseline = 0.06*self.fx

    def read(self):
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
        l_k = self.settings.getNode('LEFT.K').mat()
        l_r = self.settings.getNode('LEFT.R').mat()
        l_p = self.settings.getNode('LEFT.P').mat()
        l_width = int(self.settings.getNode('LEFT.width').real())
        l_height = int(self.settings.getNode('LEFT.height').real())

        r_d = self.settings.getNode('RIGHT.D').mat()
        r_k = self.settings.getNode('RIGHT.K').mat()
        r_r = self.settings.getNode('RIGHT.R').mat()
        r_p = self.settings.getNode('RIGHT.P').mat()
        r_width = int(self.settings.getNode('RIGHT.width').real())
        r_height = int(self.settings.getNode('RIGHT.height').real())

        self.m1l, self.m2l = cv2.initUndistortRectifyMap(l_k, l_d, l_r, l_p, (l_width, l_height), cv2.CV_32F)
        self.m1r, self.m2r = cv2.initUndistortRectifyMap(r_k, r_d, r_r, r_p, (r_width, r_height), cv2.CV_32F)

        self.baseline = self.settings.getNode('Camera.bf').real()
        self.fx = self.settings.getNode('Camera.fx').real()
        self.fy = self.settings.getNode('Camera.fy').real()
        self.cx = self.settings.getNode('Camera.cx').real()
        self.cy = self.settings.getNode('Camera.cy').real()

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

    slam = StereoSLAM(camera.baseline, camera.fx, camera.fy, camera.cx, camera.cy)

    gray_l, gray_r = camera.read()
    async def read_frame():
        print("SLAM started")
        key = 0
        try:
            while key != ord('q'):
                gray_l, gray_r = camera.read()
                tm = cv2.TickMeter()
                tm.start()
                slam.new_image(gray_l, gray_r)
                tm.stop()
                print(f"processing took: {tm.getTimeMilli()} ms")
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
                    for keyframe in slam.mapping.keyframes:
                        x = keyframe.keypoints2d[0].astype(np.uint16)
                        y = keyframe.keypoints2d[1].astype(np.uint16)
                        # colors = keyframe.left[y, x]
                        keyframes.append({
                            'keypoints': keyframe.keypoints3d[0].tolist(),
                            'pose': keyframe.pose.tolist(),
                            'colors': keyframe.colors.tolist()})
                    encoder = json.JSONEncoder()
                    await websocket.send(encoder.encode(keyframes))
            elif path == "/pose":
                if message == "get":
                    encoder = json.JSONEncoder()
                    pose = {'pose': slam.pose.tolist()}
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
